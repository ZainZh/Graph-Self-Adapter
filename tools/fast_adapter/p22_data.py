"""Complete physical demonstrations; only RGB and proprio feed the policies."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.geometry_insert import GeometryInsert,geometry,initial_position,domain
from prove.fast_adapter.insert_adapter import PROBE_ACTIONS
from prove.fast_adapter.lila_frontend import ThreeFrontends
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL
from tools.fast_adapter.p18_run import save_json,digest

R=Path('evaluations/fast_adapter_p22_geometry_v01')

class RGBEncoder:
    def __init__(self,dino=DEFAULT_MODEL):
        self.encoder=ThreeFrontends(dino,microbatch=16,selected='lila')
    def encode(self,rgb):
        # LiLa's full-image branch does not use this dummy argument.
        return self.encoder.encode(rgb,np.zeros((len(rgb),2,4),np.float32))[:,0]

def collect(shard,n,dino):
    torch.set_num_threads(2)
    out=R/'bank'/f'shard{shard}';out.mkdir(parents=True,exist_ok=False)
    encoder=RGBEncoder(dino); features=[]; pending=[]; positions=[]; nominals=[]
    teachers=[]; actions=[]; domain_rows=[]; trajectory_success=[]; examples=[]
    began=time.time()
    def encode():
        features.append(encoder.encode(np.stack(pending)).cpu().half());pending.clear()
    for i in range(n):
        sid=shard*n+i;cfg=geometry(2201000+sid)
        env=GeometryInsert(cfg);M,b,fr=domain(2202000+sid);env.set_domain(M,b,fr)
        ps=[];ns=[];ts=[];aa=[]
        def observe():
            rgb=env.rgb();pending.append(rgb);ps.append(env.state())
            if i<2:examples.append(rgb)
            if len(pending)==16:encode()
        env.reset(initial_position(2203000+sid));observe()
        for a in PROBE_ACTIONS:env.step(a);observe()
        env.reset(initial_position(2204000+sid));observe()
        rng=np.random.default_rng(2205000+sid)
        for t in range(48):
            ns.append(env.expert_nominal());ts.append(env.expert_action())
            a=np.clip(ts[-1]+rng.uniform(-.08,.08,3),-1,1)
            aa.append(a);env.step(a);observe()
        positions.append(ps);nominals.append(ns);teachers.append(ts);actions.append(aa)
        trajectory_success.append(env.streak>=3)
        domain_rows.append({'id':sid,'geometry':cfg,'M':M.tolist(),'bias':b.tolist(),'friction':fr})
        env.close()
        if (i+1)%16==0:print(json.dumps({'shard':shard,'domains':i+1,'seconds':time.time()-began}),flush=True)
    if pending:encode()
    torch.save(torch.cat(features).reshape(n,55,49152),out/'FEATURES.pt')
    meta={k:torch.tensor(np.asarray(v),dtype=torch.float32) for k,v in
          [('proprio',positions),('nominal_label',nominals),('expert_label',teachers),('query_action',actions)]}
    torch.save(meta,out/'META.pt');save_json(out/'EVALUATOR_ONLY.json',domain_rows)
    np.savez_compressed(out/'EXAMPLE_RGB.npz',rgb=np.stack(examples))
    save_json(out/'RECEIPT.json',{'domains':n,'frames_per_domain':55,'images':n*55,
      'demonstration_successes':sum(trajectory_success),'seconds':time.time()-began,
      'runtime_inputs':['RGB','controlled_object_xyz'],'labels':['nominal_action','expert_action'],
      'artifacts':[{'path':str(p),'sha256':digest(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    print('COLLECT_COMPLETE',shard,flush=True)

def merge():
    torch.set_num_threads(4);out=R/'bank'
    parts=[torch.load(out/f'shard{i}/FEATURES.pt',weights_only=True) for i in range(3)]
    # Balanced fixed sampling of training frames only, before any dev/final images.
    sample=torch.cat([x[:,::6].flatten(0,1) for x in parts]).float().cuda()
    mu=sample.mean(0);centered=sample-mu;torch.manual_seed(22022)
    _,sv,basis=torch.pca_lowrank(centered,q=80,center=False,niter=4)
    basis=basis[:,:64];sd=(sv[:64].square()/(len(sample)-1)).clamp_min(1e-6).sqrt()
    pca={'mean':mu.cpu(),'basis':basis.cpu(),'sd':sd.cpu(),
         'retained_variance':float(sv[:64].square().sum()/centered.square().sum()),
         'fit':'training-only, every sixth frame per domain'}
    torch.save(pca,out/'PCA.pt');del sample,centered
    values=[]
    for part in parts:
        values.append(torch.cat([((x.float().cuda()-mu)@basis/sd).cpu() for x in part.split(8)]))
    feat=torch.cat(values)
    metas=[torch.load(out/f'shard{i}/META.pt',weights_only=True) for i in range(3)]
    bank={k:torch.cat([x[k] for x in metas]) for k in metas[0]}
    bank.update(support_visual=feat[:,:6],query_visual=feat[:,6:])
    torch.save(bank,out/'TRAIN.pt')
    save_json(out/'MERGE.json',{'domains':len(feat),'pca_retained_variance':pca['retained_variance'],
      'pca_sha256':digest(out/'PCA.pt'),'train_sha256':digest(out/'TRAIN.pt'),
      'test_or_dev_images_used':False})
    print('MERGE_COMPLETE',len(feat),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','merge']);p.add_argument('--shard',type=int)
    p.add_argument('--domains',type=int,default=128);p.add_argument('--dino',default=DEFAULT_MODEL)
    a=p.parse_args()
    if a.mode=='collect':collect(a.shard,a.domains,a.dino)
    else:merge()
