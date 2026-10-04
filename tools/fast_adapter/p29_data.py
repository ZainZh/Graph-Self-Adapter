import argparse,time
from pathlib import Path
import numpy as np
import torch
from tools.fast_adapter.p29_common import *

def collect_block(task,configs,seeds,horizon,mode,visual,base=None,net=None,z=None,labels=True):
    envs=[ObjectEnv(c) for c in configs];n=len(envs)
    ps=[np.stack([e.reset(int(s)) for e,s in zip(envs,seeds)])];features=[];actions=[];teachers=[];states=[];audit_rgb=[];raw=[]
    previous=torch.zeros(n,4,device='cuda');began=time.time()
    for t in range(horizon+1):
        images=np.stack([e.rgb() for e in envs]);v=visual.encode(images)
        if isinstance(v,dict):raw.append({k:x.detach().cpu().half() for k,x in v.items()})
        else:features.append(v.detach().cpu().float())
        states.append({k:np.stack([e.state()[k] for e in envs]) for k in ('qpos','qvel','mocap_pos','mocap_quat')})
        if t in (0,horizon//2,horizon):audit_rgb.append(images[:2])
        if t==horizon:break
        if labels:
            teacher=np.stack([e.teacher() for e in envs]);teachers.append(teacher)
        if mode=='expert':action=teacher.copy()
        else:
            with torch.no_grad():
                pp=torch.as_tensor(ps[-1],device='cuda');ba=base(v,pp,previous)
                action=(ba if net is None else net.action(v,pp,previous,ba,z)).cpu().numpy()
        ps.append(np.stack([e.step(a) for e,a in zip(envs,action)]));actions.append(action);previous=torch.as_tensor(action,device='cuda')
    metrics=[e.metrics for e in envs]
    for e in envs:e.close()
    vals={'proprio':torch.tensor(np.stack(ps,1),dtype=torch.float32),'actions':torch.tensor(np.stack(actions,1),dtype=torch.float32),
          'reset_seeds':list(map(int,seeds)),'mode':mode,'states':{k:torch.tensor(np.stack([s[k] for s in states],1)) for k in states[0]},
          'audit_rgb':torch.tensor(np.stack(audit_rgb,1)),'seconds':time.time()-began}
    if raw:vals.update({k:torch.stack([v[k] for v in raw],1) for k in raw[0]})
    else:vals['visual']=torch.stack(features,1)
    if labels:vals['teacher']=torch.tensor(np.stack(teachers,1),dtype=torch.float32)
    return vals,metrics

def collect(task,stage,dino,count=None):
    torch.set_num_threads(2);out=task_root(task)/'data'/stage;out.mkdir(parents=True,exist_ok=False)
    pca=None if stage=='demos' else torch.load(task_root(task)/'data/PCA.pt',weights_only=True)
    visual=Visual(dino,task,pca);base=None if stage=='demos' else load_base(task,'base_bc' if stage=='dagger' else 'base_final')
    count=count or (96 if stage=='demos' else 64)
    origin={'demos':2910000,'dagger':2920000,'meta':2930000}[stage]+TASKS.index(task)*10000
    roles=('support','expert','base') if stage=='meta' else (stage,);receipt=[]
    for ri,role in enumerate(roles):
        for off in range(0,count,BLOCK):
            ids=np.arange(off,min(off+BLOCK,count));cfgs=[object_config(task,origin+500000+int(i),'train' if stage=='meta' else 'canonical') for i in ids]
            seeds=origin+ri*100000+ids;horizon=SUPPORT if role=='support' else HORIZON;mode='expert' if role in ('demos','expert') else 'base'
            blob,metrics=collect_block(task,cfgs,seeds,horizon,mode,visual,base,labels=role!='support');p=out/f'{role}_{off:03d}.pt';torch.save(blob,p)
            rec={'path':str(p),'sha256':digest(p),'n':len(ids),'successes':sum(any(x['success'] for x in m) for m in metrics),'seconds':blob['seconds']};receipt.append(rec)
            save_json(out/f'{role}_{off:03d}_EVALUATOR.json',{'source_sha256':rec['sha256'],'configs':cfgs,'metrics':metrics})
            print(task,stage,role,off,rec['successes'],f"{blob['seconds']:.1f}s",flush=True)
    save_json(out/'COMPLETE.json',{'task':task,'stage':stage,'files':receipt,'test_data_used':False})

def merge(task):
    torch.set_num_threads(4);root=task_root(task)/'data';paths=sorted((root/'demos').glob('demos_*.pt'))
    parts=[]
    for p in paths:
        b=torch.load(p,weights_only=True);f=b['features'][:,::8].reshape(-1,49152);m=b['presence'][:,::8].reshape(-1)>0;parts.append(f[m])
    sample=torch.cat(parts).float().cuda();mean=sample.mean(0);centered=sample-mean;torch.manual_seed(29029)
    _,sv,basis=torch.pca_lowrank(centered,q=80,center=False,niter=4);basis=basis[:,:64];sd=(sv[:64].square()/(len(sample)-1)).clamp_min(1e-5).sqrt()
    pca={'mean':mean.cpu(),'basis':basis.cpu(),'sd':sd.cpu(),'fit':'present RGB-derived actor/object crops in canonical training demonstrations only','retained_variance':float(sv[:64].square().sum()/centered.square().sum())};torch.save(pca,root/'PCA.pt')
    del sample,centered,parts
    banks=[]
    for p in paths:
        b=torch.load(p,weights_only=True);proj=[]
        for f in b['features'].split(2):proj.append(((f.float().cuda()-mean)@basis/sd).cpu())
        v=torch.cat(proj)*b['presence'][...,None].float();v=torch.cat((v,b['boxes'].float(),b['presence'][...,None].float()),-1).flatten(-2)
        banks.append({'visual':v,**{k:b[k] for k in ('proprio','actions','teacher')}})
    torch.save({k:torch.cat([b[k] for b in banks]) for k in banks[0]},root/'DEMOS.pt')
    save_json(root/'PCA_SEAL.json',{'sha256':digest(root/'PCA.pt'),'demo_sha256':digest(root/'DEMOS.pt'),'sources':[{'path':str(p),'sha256':digest(p)} for p in paths],'test_images_used':False,'retained_variance':pca['retained_variance']})
    print(task,'PCA',pca['retained_variance'],flush=True)
def load_stage(task,stage,role=None):
    root=task_root(task)/'data'
    if stage=='demos':return torch.load(root/'DEMOS.pt',weights_only=True)
    banks=[torch.load(p,weights_only=True) for p in sorted((root/stage).glob(f'{role or stage}_[0-9]*.pt'))]
    keys=['visual','proprio','actions']+(['teacher'] if 'teacher' in banks[0] else [])
    return {k:torch.cat([b[k] for b in banks]) for k in keys}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','merge']);p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--stage',choices=['demos','dagger','meta']);p.add_argument('--dino',default='assets/dinov3');p.add_argument('--count',type=int);a=p.parse_args()
    if a.mode=='collect':collect(a.task,a.stage,a.dino,a.count)
    else:merge(a.task)
