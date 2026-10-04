"""P39 runtime sealing, independent task labels, and fixed causal controls."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.planar_contact import PlanarPush,controller,projected_force,STEPS
from prove.fast_adapter.planar_tokens import VisualTokens,PlanarAdapter,PRIOR,relation
from tools.fast_adapter.p39_tokens import ROOT,js,sha,cfg,probes,bank,effect_models

SPLITS=('train','resistance','geometry','joint')

def load(kind,seed):
    n=PlanarAdapter(kind).cuda();n.load_state_dict(torch.load(ROOT/'models'/kind/f'seed{seed}/model.pt',weights_only=True))
    return n.eval().requires_grad_(False)

def means(kind,seed):
    p=ROOT/'models'/kind/f'seed{seed}'/'TRAIN_MEAN.pt'
    if p.exists():return torch.load(p,weights_only=True).cuda()
    n=load(kind,seed);d=bank();heads=effect_models()
    with torch.no_grad():
        for k,h in heads.items():d[k+'_j']=h(d[k][:,:-1])
    z=n.initial(160)
    for t in range(STEPS):
        w={k:v[:,max(0,t-3):t+1] for k,v in d.items() if k not in ('labels','object_motion')}
        z,_=n.update(z,w)
    mean=z.detach().mean(0);torch.save(mean.cpu(),p)
    js(p.with_suffix('.json'),dict(source='160 training trajectories only; endpoint latent mean',sha256=sha(p)))
    return mean

def vision_model():
    pca=torch.load(ROOT/'data/PCA.pt',weights_only=True)
    pca={k:{a:b.cuda() if hasattr(b,'cuda') else b for a,b in v.items()} for k,v in pca.items()}
    return VisualTokens(pca)

def configurations(phase,split):
    if split=='paired':
        assert phase=='final'
        from tools.fast_adapter.p39_paired import pairs
        return pairs(398000,32)
    n=8 if phase=='dev' else 32
    first=(395000 if phase=='dev' else 397000)+200*SPLITS.index(split)
    return [cfg(first+i,split) for i in range(n)]

def windows(records,heads):
    """Window from already completed transitions only."""
    pos=records['positions'];p=records['proprio']
    w={k:records[k][:,:-1] for k in ('global','local','relation')}
    w['actor_motion']=(p[:,1:,:2]-p[:,:-1,:2])/.002
    w['relation_df']=records['relation'][:,1:]-records['relation'][:,:-1]
    w['force']=records['force']
    with torch.no_grad():
        for k in ('global','local'):
            w[k+'_df']=records[k][:,1:]-records[k][:,:-1]
            w[k+'_j']=heads[k](w[k])
    return w

def run_trial(phase,split,kind,seed,arm,vision=None,heads=None,donor=None):
    configs=configurations(phase,split);nitems=len(configs)
    out=ROOT/phase/kind/f'seed{seed}'/split;out.mkdir(parents=True,exist_ok=True)
    if (out/(arm+'_RUNTIME.pt')).exists():raise RuntimeError('Refusing overwrite')
    n=load(kind,seed) if kind!='references' else None
    z=n.initial(nitems) if n is not None else None
    if arm=='mean':z=means(kind,seed)[None].expand(nitems,-1).clone()
    envs=[PlanarPush(c) for c in configs];alive=np.ones(nitems,dtype=bool);last=[None]*nitems
    records={k:[] for k in ('positions','radii','proprio','global','local','relation','force','actions','z','response','qpos','qvel','valid')}
    rgb_hashes=[];mask_hashes=[]
    try:
        for t in range(STEPS+1):
            obs=[];images=[];masks=[]
            for i,e in enumerate(envs):
                try:
                    current=e.public()
                except RuntimeError as err:
                    if last[i] is None:raise
                    alive[i]=False;current=last[i]
                last[i]=current;o,rgb,m=current;obs.append(o);images.append(rgb);masks.append(m)
            pos=torch.tensor(np.stack([o['positions'] for o in obs]),device='cuda');rel=relation(pos)
            proprio=torch.tensor(np.stack([o['proprio'] for o in obs]),device='cuda')
            need=arm=='base' or (kind in ('global','local','fused','history') and (arm=='continuous' or (arm=='first3' and t<=3)))
            if need:
                f=vision.encode(images,masks);glob=f['global_'];local=f['local']
            else:glob=local=torch.zeros(nitems,16,device='cuda')
            current={'positions':pos,'radii':torch.tensor(np.stack([o['radii'] for o in obs]),device='cuda'),'proprio':proprio,
                     'global':glob,'local':local,'relation':rel,'valid':torch.tensor(alive,device='cuda')}
            for k,v in current.items():records[k].append(v.detach())
            if t and n is not None and arm in ('continuous','first3','donor') and (arm!='first3' or t<=3):
                if arm=='donor':
                    # Fixed independent source trajectories; no coupled feedback loops.
                    ids=(torch.arange(nitems,device='cuda')^1) if split=='paired' else torch.arange(nitems,device='cuda').roll(1)
                    w={k:v[ids,max(0,t-4):t] for k,v in donor.items()}
                else:
                    short={k:torch.stack(v[max(0,t-4):t+1],1) for k,v in records.items() if k in ('positions','proprio','global','local','relation','force')}
                    w=windows(short,heads)
                nz,_=n.update(z,w);z=torch.where(torch.tensor(alive,device='cuda')[:,None],nz,z)
            if n is not None:e=n.response(z).detach().cpu().numpy()
            elif arm=='oracle':
                e=np.array([np.stack((20/(np.array(c['damping'])+2),-20*(np.array(c['friction'])+.005)/(np.array(c['damping'])+2)),-1) for c in configs])
            else:e=np.broadcast_to(np.array(PRIOR),(nitems,2,2))
            records['response'].append(torch.tensor(e,device='cuda'))
            if z is not None:records['z'].append(z.detach())
            states=[en.state() for en in envs]
            for k in ('qpos','qvel'):records[k].append(torch.tensor(np.stack([s[k] for s in states]),device='cuda'))
            rgb_hashes.append([__import__('hashlib').sha256(x.tobytes()).hexdigest() for x in images])
            mask_hashes.append([__import__('hashlib').sha256(x.tobytes()).hexdigest() for x in masks])
            if t==STEPS:break
            actions=np.stack([probes(o,t) if t<3 else controller(o,ep) for o,ep in zip(obs,e)])
            actions[~alive]=0
            records['actions'].append(torch.tensor(actions,device='cuda'))
            records['force'].append(torch.tensor(np.stack([projected_force(o,a) for o,a in zip(obs,actions)]),device='cuda',dtype=torch.float32))
            for en,a in zip(envs,actions):en.step(a)
        raw={k:torch.stack(v,1).cpu() for k,v in records.items() if v}
        raw.update(rgb_hashes=rgb_hashes,mask_hashes=mask_hashes)
        p=out/(arm+'_RUNTIME.pt');torch.save(raw,p)
        js(out/(arm+'_SEAL.json'),dict(time=time.time(),sha256=sha(p),phase=phase,kind=kind,seed=seed,split=split,arm=arm,
             model_hash=sha(ROOT/'models'/kind/f'seed{seed}/model.pt') if n is not None else None,
             oracle=arm=='oracle',observation='RGB448 + simulator role masks + calibration + tool proprioception',online='z only after completed actions; rolling4'))
        # Only after sealing control and latent trajectories are task labels read.
        rows=[]
        for i,(en,c) in enumerate(zip(envs,configs)):
            s=en.state();err=float(np.linalg.norm(s['object']-en.goal));speed=float(np.linalg.norm(s['qvel'][2:]))
            rows.append(dict(seed=c['seed'],error=err,speed=speed,observation_valid=bool(alive[i]),success=bool(alive[i] and err<.0025 and speed<.008)))
        js(out/(arm+'_EVALUATOR.json'),dict(configs=configs,rows=rows))
        result=dict(phase=phase,split=split,kind=kind,seed=seed,arm=arm,n=nitems,successes=sum(r['success'] for r in rows),mean_error=float(np.mean([r['error'] for r in rows])),rows=rows)
        js(out/(arm+'_RESULT.json'),result);print({k:v for k,v in result.items() if k!='rows'},flush=True)
        return raw
    finally:
        for en in envs:en.close()

def main(kind,seed,phase,only_split=None):
    heads=effect_models();vision=vision_model() if kind in ('references','global','local','fused','history') else None
    for split in ([only_split] if only_split else SPLITS+(('paired',) if phase=='final' else ())):
        if kind=='references':
            for arm in ('base','oracle'):run_trial(phase,split,kind,0,arm,vision,heads)
        else:
            r=torch.load(ROOT/phase/'references'/'seed0'/split/'base_RUNTIME.pt',weights_only=True)
            d={k:v.cuda() for k,v in r.items() if isinstance(v,torch.Tensor)};donor=windows(d,heads)
            for arm in ('continuous','first3','mean','donor'):
                run_trial(phase,split,kind,seed,arm,vision,heads,donor)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--seed',type=int,default=11);p.add_argument('--phase',choices=['dev','final'],default='dev');p.add_argument('--split',choices=SPLITS+('paired',))
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cudnn.enabled=False;main(a.kind,a.seed,a.phase,a.split)
