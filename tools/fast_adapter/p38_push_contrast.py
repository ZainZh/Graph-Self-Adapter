"""New, predeclared counterfactual batch using unchanged P38 contact models.

Each pair has identical visible geometry but opposite hidden resistance.
The donor arm reads the partner's valid (action, consequence) transitions.
Training-mean latent is computed only from the existing training trajectories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.fast_adapter import p38_push as common
from prove.fast_adapter.near_contact_push import NearPush, configuration
from prove.fast_adapter.contact_identification import PRIOR, force, ridge_response

R=Path('evaluations/fast_adapter_p38_causal_v01/push_contrast')


def training_mean(kind,seed,n,jnet):
    dest=R/'means'/kind/f'seed{seed}.pt'
    if dest.exists():return torch.load(dest,weights_only=True).cuda()
    b=common.bank();z=n.initial(96,'cuda')
    with torch.no_grad():j=jnet(b['visual'][:,:-1])
    for t in range(30):
        lo=max(0,t-3)
        z,_=n.update(z,b['actions'][:,lo:t+1],b['motion'][:,lo:t+1],b['df'][:,lo:t+1],
                    b['visual'][:,lo:t+1],j[:,lo:t+1])
    mean=z.detach().mean(0)
    dest.parent.mkdir(parents=True,exist_ok=True);torch.save(mean.cpu(),dest)
    common.save_json(dest.with_suffix('.json'),{'source':'96 training episodes,30 updates; same frozen model',
                      'source_hash':common.sha(common.ROOT/'data/TRAIN.pt'),'mean_hash':common.sha(dest)})
    return mean


def configs(phase):
    pairs=4 if phase=='dev' else 32
    result=[]
    start=407000 if phase=='dev' else 417000
    for i in range(pairs):
        base=configuration(start+i,'train');rng=np.random.default_rng(start+10000+i)
        for high in (False,True):
            c=dict(base);c['seed']=start+2*i+int(high);c['split']='contrast_high' if high else 'contrast_low'
            c['damping']=float(rng.uniform(7.,9.) if high else rng.uniform(2.5,3.5))
            c['friction']=float(rng.uniform(.18,.22) if high else rng.uniform(.02,.04))
            result.append(c)
    return result


def evaluate(kind,seed,phase):
    cfgs=configs(phase);count=len(cfgs)
    n=None if kind=='references' else common.load(kind,seed)
    jnet=common.jacobian_model()
    mean=None if n is None else training_mean(kind,seed,n,jnet)
    vision=common.Vision(compressed=True) if kind in ('appearance','combined','history') else None
    out=R/phase/kind/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    arms=('base','least_squares','oracle') if n is None else ('continuous','first3','donor','train_mean')
    for arm in arms:
        envs=[NearPush(c) for c in cfgs]
        z=n.initial(count,'cuda') if n is not None else None
        if arm=='train_mean':z=mean.expand(count,-1).clone()
        fields={k:[] for k in ('positions','visual','encoder','actions','response','z','qpos','qvel','time')}
        hashes=[];window={k:[] for k in ('a','y','df','f','j')}
        prev=None
        for t in range(31):
            public=[e.public() for e in envs]
            p=torch.tensor(np.stack([v[0]['positions'] for v in public]),device='cuda')
            use=vision is not None and arm!='train_mean' and (arm!='first3' or t<=3)
            f=vision.encode([v[2] for v in public]) if use else torch.zeros(count,16,device='cuda')
            with torch.no_grad():j=jnet(f)
            if t:
                oldp,oldf,olda,oldj=prev
                values=(olda,(p[:,1]-oldp[:,1])/.002,f-oldf,oldf,oldj)
                for k,v in zip(window,values):window[k].append(v)
                w={k:torch.stack(v[-4:],1) for k,v in window.items()}
                if n is not None and arm!='train_mean' and (arm!='first3' or t<=3):
                    if arm=='donor':
                        permutation=torch.arange(count,device='cuda')^1
                        w={k:v[permutation] for k,v in w.items()}
                    z,_=n.update(z,w['a'],w['y'],w['df'],w['f'],w['j'])
                if arm=='least_squares':
                    # Same four-transition window as the gradient model.
                    estimate=ridge_response(w['a'],w['y'])
            if n is not None:
                with torch.no_grad():estimate=n.estimate(z)
            elif arm=='oracle':
                estimate=torch.tensor([[20/(c['damping']+.2),-20*(c['friction']+.01)/(c['damping']+.2)] for c in cfgs],device='cuda')
            elif arm=='base' or not t:
                estimate=torch.tensor(PRIOR,device='cuda').expand(count,-1)
            fields['positions'].append(p.cpu());fields['visual'].append(f.cpu())
            fields['encoder'].append(torch.tensor(np.stack([v[1] for v in public])))
            fields['response'].append(estimate.detach().cpu())
            if z is not None:fields['z'].append(z.detach().cpu())
            states=[e.state() for e in envs]
            for k in ('qpos','qvel','time'):fields[k].append(torch.tensor(np.stack([s[k] for s in states])))
            hashes.append([hashlib.sha256(v[2].tobytes()).hexdigest() for v in public])
            if t==30:break
            with torch.no_grad():a=force(p[:,2]-p[:,1],estimate)
            if t<3:a=torch.full_like(a,common.PROBES[t])
            fields['actions'].append(a.cpu())
            for e,command in zip(envs,a.cpu().tolist()):e.step(command)
            prev=(p,f,a,j)
        raw={k:torch.stack(v,1) for k,v in fields.items() if v};raw['rgb_sha256']=hashes
        path=out/(arm+'_RUNTIME.pt');torch.save(raw,path)
        common.save_json(out/(arm+'_SEAL.json'),{'time':time.time(),'runtime_hash':common.sha(path),
              'pairing':'even/odd same visible geometry,opposite hidden resistance','total_actions':30,
              'updates':3 if arm=='first3' else (0 if arm in ('train_mean','base','oracle') else 30)})
        rows=[]
        for e,c in zip(envs,cfgs):
            s=e.state();err=abs(s['object_x']-e.goal)
            rows.append({'seed':c['seed'],'pair':c['seed']//2,'high':c['split']=='contrast_high',
                         'error':float(err),'speed':float(s['qvel'][1]),'success':bool(err<.0015 and abs(s['qvel'][1])<.005)})
            e.close()
        common.save_json(out/(arm+'_EVALUATOR.json'),{'configs':cfgs,'rows':rows})
        common.save_json(out/(arm+'_RESULT.json'),{'kind':kind,'seed':seed,'phase':phase,'split':'contrast',
           'arm':arm,'n':count,'successes':sum(r['success'] for r in rows),'rows':rows})
        print(kind,seed,phase,arm,sum(r['success'] for r in rows),'/',count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=common.KINDS+('references',));p.add_argument('--seed',type=int,default=11);p.add_argument('--phase',choices=['dev','final'],default='dev');a=p.parse_args()
    torch.set_num_threads(2);evaluate(a.kind,a.seed,a.phase)
