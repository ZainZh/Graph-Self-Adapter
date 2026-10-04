"""Finite training refresh and fresh development, with actual fp32 inputs saved."""
import argparse,json,time
import numpy as np
import torch
from prove.fast_adapter.geometry_insert import geometry
from prove.fast_adapter.insertion_teacher import reference_action
from tools.fast_adapter.p24_rollout import SceneBatch,visual_assets,score
from tools.fast_adapter.p26_common import R,PARENT,SEEDS,ARMS,load_adapter,load_base,adaptation_hash,weight_hash
from tools.fast_adapter.p25_common import select_feedback,transitions
from tools.fast_adapter.p18_run import save_json,digest


def trajectory(scene,base,resets,steps,net=None,z=None,reference=False,bounded=False):
    scene.reset(resets);previous=torch.zeros(len(resets),3,device='cuda')
    ps,vel,vs,aa,bb,rgbs=[],[],[],[],[],[]
    for t in range(steps+1):
        ps.append(np.stack([e.state() for e in scene.envs]));vel.append(np.stack([e.data.qvel.copy() for e in scene.envs]))
        if not reference or bounded:
            v,p=scene.observe();vs.append(v.cpu())
            if t in (0,steps//2,steps):rgbs.append(scene.last_rgb[:2].copy())
            with torch.no_grad():ba=base(v,p)
        if t==steps:break
        if reference:
            action=np.stack([reference_action(e) for e in scene.envs])
            if bounded:
                b=ba.cpu().numpy();action=np.clip(action,np.maximum(-1,b-.35),np.minimum(1,b+.35))
        else:
            with torch.no_grad():a=ba if net is None else net.action(v,p,previous,ba,z)
            action=a.cpu().numpy();previous=a
        if not reference or bounded:bb.append(ba.cpu())
        aa.append(action);scene.step(action)
    return dict(positions=np.stack(ps,1),velocities=np.stack(vel,1),actions=np.stack(aa,1),
        visual=torch.stack(vs,1) if vs else None,base=torch.stack(bb,1) if bb else None,
        expert_label=None,audit_rgb=np.stack(rgbs) if rgbs else None,audit_rgb_steps=[0,steps//2,steps])


def feedback(raw):
    p=torch.tensor(raw['positions'],dtype=torch.float32);a=torch.tensor(raw['actions']);v=raw['visual']
    idx=select_feedback(p,a)
    return idx,transitions(v,p,a,idx)


def collect(dino):
    spec=json.loads((R/'PROTOCOL.json').read_text())['learning'];ids=np.array(spec['refresh_domain_ids'])
    assert len(ids)==32
    out=R/'training_refresh';out.mkdir(exist_ok=False)
    encoder,pca=visual_assets(dino);base=load_base();bh=weight_hash(base)
    scene=SceneBatch([geometry(2201000+int(i)) for i in ids],2202000+ids,encoder,pca)
    began=time.time();support=trajectory(scene,base,26100000+ids,32)
    idx,fb=feedback(support);support.update(training_ids=ids.tolist(),selected_indices=idx,feedback=fb)
    torch.save(support,out/'SUPPORT.pt');save_json(out/'SUPPORT_SEAL.json',{'sha256':digest(out/'SUPPORT.pt'),'queries_not_started':True})
    summaries=[]
    for seed in SEEDS:
        net=load_adapter(seed,'relabel',stage='a');h=weight_hash(net)
        z=net.feedback_path({k:v.cuda() for k,v in fb.items()})[0][-1].detach()
        raw=trajectory(scene,base,26200000+1000*seed+ids,48,net,z)
        raw.update(training_ids=ids.tolist(),seed=seed,z=z.cpu(),scope='TRAIN only; no development')
        path=out/f'query_seed{seed}_RAW.pt';torch.save(raw,path)
        metric=score(scene,raw);metric.update(raw_sha256=digest(path),model_sha256=h)
        assert weight_hash(net)==h and weight_hash(base)==bh
        summaries.append({'seed':seed,**metric});print('TRAIN_REFRESH_COMPLETE',seed,flush=True)
    scene.close();save_json(out/'COMPLETE.json',{'queries':64,'supports':32,'physical_actions':4096,'seconds':time.time()-began,
        'source_models_frozen_before_collection':digest(R/'STAGE_A_FREEZE.json'),'summaries':summaries})


def evaluate(st,dino):
    assert (R/'MODEL_FREEZE.json').exists()
    spec=json.loads((R/'PROTOCOL.json').read_text())['development'];n=spec['conditions_each'];ss=spec['seed']+1000*('id','position','scale').index(st)
    out=R/'development'/st;out.mkdir(parents=True,exist_ok=False)
    encoder,pca=visual_assets(dino);base=load_base();bh=weight_hash(base)
    scene=SceneBatch([geometry(ss+i,st) for i in range(n)],ss+100000+np.arange(n),encoder,pca);began=time.time()
    support=trajectory(scene,base,ss+200000+np.arange(n),32);idx,fb=feedback(support)
    support.update(seed_base=ss,selected_indices=idx,feedback=fb);torch.save(support,out/'SUPPORT_SEALED.pt')
    save_json(out/'SUPPORT_SEAL.json',{'sha256':digest(out/'SUPPORT_SEALED.pt'),'queries_not_started':True})
    metrics={};weights={'base':bh};latent_hashes={}
    def run(arm,net=None,z=None,reference=False,bounded=False):
        raw=trajectory(scene,base,ss+300000+np.arange(n),48,net,z,reference,bounded)
        raw.update(seed_base=ss,stratum=st,arm=arm);path=out/f'{arm}_RAW.pt';torch.save(raw,path)
        raw_hash=digest(path);metric=score(scene,raw);metric['raw_sha256']=raw_hash;metrics[arm]=metric
        print(json.dumps({'stratum':st,'arm':arm,'successes':metric['successes'],'n':n}),flush=True)
    run('base');run('repaired_oracle',reference=True);run('bounded_repaired_oracle',reference=True,bounded=True)
    for seed in SEEDS:
        net=load_adapter(seed,'parent');path,loss=net.feedback_path({k:v.cuda() for k,v in fb.items()})
        mean=torch.load(PARENT/'models'/f'seed{seed}'/'physical.pt',weights_only=True,map_location='cpu')['training_mean_z']
        latent={'correct':path[-1].detach(),'swapped':path[-1].detach().roll(1,0),'mean':mean.cuda().expand(n,-1),'k0':net.initial(n).detach()}
        lp=out/f'LATENTS_seed{seed}.pt';torch.save({'path':torch.stack(path,1).cpu(),'loss':loss.cpu(),'latents':{k:z.cpu() for k,z in latent.items()}},lp)
        latent_hashes[str(seed)]=digest(lp);ah=adaptation_hash(net)
        for arm in ('parent',*ARMS):
            candidate=net if arm=='parent' else load_adapter(seed,arm)
            assert adaptation_hash(candidate)==ah;h=weight_hash(candidate);weights[f'{arm}_seed{seed}']=h
            for mode in (('correct','swapped','mean') if arm=='refresh' else ('correct',)):
                run(f'{arm}_{mode}_seed{seed}',candidate,latent[mode])
            assert weight_hash(candidate)==h
    assert weight_hash(base)==bh;scene.close()
    save_json(out/'RESULT.json',{'metrics':metrics,'weights':weights,'latent_records':latent_hashes,'weights_unchanged':True,
        'adaptation_modules_unchanged':True,'latents_fixed_during_query':True,'slot_geometry_not_runtime_input':True,
        'k0_aliases':'All heads anchored at frozen z0, exactly base','seconds':time.time()-began})
    print('EVAL_COMPLETE',st,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','evaluate']);p.add_argument('--stratum');p.add_argument('--dino',default='assets/dinov3');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    if a.mode=='collect':collect(a.dino)
    else:evaluate(a.stratum,a.dino)
