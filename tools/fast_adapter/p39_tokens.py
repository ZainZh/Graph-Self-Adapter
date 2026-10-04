"""P39 bounded collection, training and sealed physical evaluation."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.planar_contact import PlanarPush, configuration, controller, projected_force, STEPS
from prove.fast_adapter.planar_tokens import (VisualTokens,FeatureEffect,PlanarAdapter,PRIOR,
    KINDS,relation,desired_force)

ROOT=Path('evaluations/fast_adapter_p39_tokens_v01')

def js(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cfg(seed,split):
    c=configuration(seed,'resistance' if split=='joint' else split)
    if split=='joint':
        geometry=configuration(seed+90000,'geometry')
        for k in ('radius','direction','lateral'):c[k]=geometry[k]
        c['split']=split
    return c
def probes(obs,t):
    d=obs['positions'][2]-obs['positions'][1];d=d/max(np.linalg.norm(d),1e-6)
    return d*(.3,.75,.45)[t]


def collect(shard):
    out=ROOT/'data'/f'shard{shard}';out.mkdir(parents=True,exist_ok=False)
    vision=VisualTokens();features={'global':[],'local':[]};records=[]
    for i in range(40):
        c=cfg(394000+40*shard+i,'train');rng=np.random.default_rng(c['seed']+13)
        env=PlanarPush(c);images=[];masks=[];trace={k:[] for k in ('positions','radii','proprio','actions','force','qpos','qvel')}
        try:
            for t in range(STEPS+1):
                obs,rgb,mask=env.public();images.append(rgb);masks.append(mask)
                for k in ('positions','radii','proprio'):trace[k].append(obs[k].copy())
                state=env.state()
                for k in ('qpos','qvel'):trace[k].append(state[k])
                if t==STEPS:break
                a=probes(obs,t) if t<3 else controller(obs)+rng.uniform(-.06,.06,2)*(t<30)
                a=np.clip(a,-1.5,1.5);trace['actions'].append(a);trace['force'].append(projected_force(obs,a));env.step(a)
            f=vision.encode(images,masks)
            features['global'].append(f['global_'].cpu().half());features['local'].append(f['local'].cpu().half())
            records.append(dict(config=c,**{k:np.array(v) for k,v in trace.items()}))
        finally:env.close()
        if (i+1)%5==0:print('collection',shard,i+1,flush=True)
    torch.save({k:torch.stack(v) for k,v in features.items()},out/'FEATURES.pt')
    np.save(out/'RECORDS.npy',np.array(records,dtype=object),allow_pickle=True)
    js(out/'DONE.json',dict(episodes=40,features=sha(out/'FEATURES.pt'),records=sha(out/'RECORDS.npy')))


def merge():
    out=ROOT/'data';pca={};data={};torch.manual_seed(39001)
    for kind in ('global','local'):
        x=torch.cat([torch.load(out/f'shard{i}/FEATURES.pt',weights_only=True)[kind] for i in range(4)]).float().cuda()
        sample=x[:,::4].flatten(0,1);mean=sample.mean(0)
        _,sv,basis=torch.pca_lowrank(sample-mean,q=20,center=False,niter=3)
        scale=(sv[:16].square()/(len(sample)-1)).clamp_min(1e-5).sqrt();basis=basis[:,:16]
        pca[kind]=dict(mean=mean.cpu(),basis=basis.cpu(),scale=scale.cpu(),variance=float(sv[:16].square().sum()/(sample-mean).square().sum()))
        data[kind]=((x-mean)@basis/scale).cpu();del x,sample
    records=[]
    for i in range(4):records.extend(np.load(out/f'shard{i}/RECORDS.npy',allow_pickle=True).tolist())
    for k in ('positions','radii','proprio','actions','force'):
        data[k]=torch.tensor(np.stack([r[k] for r in records]),dtype=torch.float32)
    data['labels']=torch.tensor([np.stack((20/(np.array(r['config']['damping'])+2),-20*(np.array(r['config']['friction'])+.005)/(np.array(r['config']['damping'])+2)),-1) for r in records],dtype=torch.float32)
    torch.save(pca,out/'PCA.pt');torch.save(data,out/'TRAIN.pt')
    js(out/'MERGE.json',dict(episodes=len(records),pca=sha(out/'PCA.pt'),train=sha(out/'TRAIN.pt'),variance={k:v['variance'] for k,v in pca.items()}))


def bank():
    d={k:v.cuda() for k,v in torch.load(ROOT/'data/TRAIN.pt',weights_only=True).items()}
    d['relation']=relation(d['positions']);d['relation_df']=d['relation'][:,1:]-d['relation'][:,:-1]
    d['actor_motion']=(d['proprio'][:,1:,:2]-d['proprio'][:,:-1,:2])/.002
    d['object_motion']=(d['positions'][:,1:,1]-d['positions'][:,:-1,1])/.002
    for k in ('global','local'):d[k+'_df']=d[k][:,1:]-d[k][:,:-1]
    return d


def train_effect():
    out=ROOT/'effects';out.mkdir(parents=True,exist_ok=False);d=bank()
    for k in ('global','local'):
        torch.manual_seed(39002);n=FeatureEffect().cuda();opt=torch.optim.Adam(n.parameters(),lr=.001)
        logs=[]
        for step in range(1,4001):
            ids=torch.randint(160,(256,),device='cuda');ts=torch.randint(STEPS,(256,),device='cuda')
            movement=torch.cat((d['actor_motion'][ids,ts],d['object_motion'][ids,ts]),-1)
            j=n(d[k][ids,ts]);pred=torch.einsum('bdc,bc->bd',j,movement)
            loss=(pred-d[k+'_df'][ids,ts]).square().mean()+.0001*j.square().mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%1000==0:logs.append(dict(step=step,loss=float(loss)));print(k,logs[-1],flush=True)
        torch.save(n.cpu().state_dict(),out/(k+'.pt'));js(out/(k+'.json'),dict(log=logs,hash=sha(out/(k+'.pt'))))


def effect_models():
    result={}
    for k in ('global','local'):
        n=FeatureEffect().cuda();n.load_state_dict(torch.load(ROOT/'effects'/(k+'.pt'),weights_only=True));result[k]=n.eval().requires_grad_(False)
    return result


def train(kind,seed):
    out=ROOT/'models'/kind/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);d=bank();heads=effect_models()
    with torch.no_grad():
        for k,n in heads.items():d[k+'_j']=n(d[k][:,:-1])
    n=PlanarAdapter(kind).cuda();opt=torch.optim.Adam(n.parameters(),lr=.0005 if kind=='history' else .003)
    started=time.time();log=[]
    for step in range(1,1501):
        ids=torch.randint(160,(16,),device='cuda');z=n.initial(16);losses=[]
        for t in range(STEPS):
            lo=max(0,t-3)
            w={k:v[ids,lo:t+1] for k,v in d.items() if k not in ('labels','object_motion')}
            z,_=n.update(z,w,meta=True)
            theta=torch.empty(16,device='cuda').uniform_(-1.1,1.1)
            dist=torch.empty(16,1,device='cuda').uniform_(.002,.04)
            err=torch.stack((theta.cos(),theta.sin()),-1)*dist
            loss=(desired_force(err,n.response(z))-desired_force(err,d['labels'][ids])).square().mean()
            losses.append(loss)
        loss=torch.stack(losses).mean();assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),5);opt.step()
        if step==1 or step%250==0:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.time()-started);log.append(row);js(out/'PROGRESS.json',row);print(kind,seed,row,flush=True)
    torch.save(n.cpu().state_dict(),out/'model.pt')
    js(out/'TRAIN.json',dict(kind=kind,seed=seed,updates=1500,log=log,model_hash=sha(out/'model.pt'),data_hash=sha(ROOT/'data/TRAIN.pt')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','merge','effects','train']);p.add_argument('--shard',type=int);p.add_argument('--kind',choices=KINDS);p.add_argument('--seed',type=int,default=11)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    if a.mode=='collect':collect(a.shard)
    elif a.mode=='merge':merge()
    elif a.mode=='effects':train_effect()
    else:train(a.kind,a.seed)
