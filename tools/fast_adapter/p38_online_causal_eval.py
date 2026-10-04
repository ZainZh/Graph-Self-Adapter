"""Causal controls for continuously-trained numerical models, fixed action budget."""
import argparse
import json
import time
from pathlib import Path
import torch
from prove.fast_adapter.causal_mapping import Identification, nominal, draw_domains, effect, fit_ls
from tools.fast_adapter import p38_mapping as common

R=Path('evaluations/fast_adapter_p38_causal_v01/online_mapping')


def run(model,actor,e,initial,goal,mode,g,mean=None):
    x=initial.clone();n=len(x);z=model.initial(n)
    if mode=='train_mean':z=mean.expand(n,-1)
    raw={k:[] for k in ('state','action','z','response')};hist=[[],[],[],[]]
    for t in range(8):
        with torch.no_grad():
            response=model.response(z)
            if mode=='oracle':response=e
            elif mode=='least_squares' and t:
                response=fit_ls(torch.stack(hist[2][-4:],1),torch.stack(hist[3][-4:],1))
            a=actor(goal-x,response);next_x=x+effect(e,a)
        raw['state'].append(x.detach().cpu());raw['action'].append(a.cpu())
        raw['z'].append(z.detach().cpu());raw['response'].append(response.detach().cpu())
        observation=(next_x-x)+.005*torch.randn(x.shape,generator=g,device='cuda')
        for values,value in zip(hist,(x,goal,a,observation)):values.append(value)
        if mode in ('continuous','donor','zero_feedback') and t<7:
            fields=[torch.stack(v[-4:],1) for v in hist]
            if mode=='donor':fields=[v.roll(1,0) for v in fields]
            if mode=='zero_feedback':fields[3]=torch.zeros_like(fields[3])
            z,_=model.update(z,*fields)
        x=next_x
    raw['state'].append(x.detach().cpu())
    raw={k:torch.stack(v,1) for k,v in raw.items()};raw['goal']=goal.cpu()
    return raw


def evaluate(kind,seed,phase):
    model=Identification(kind).cuda();source=R/'mapping'/kind/f'seed{seed}/model.pt'
    model.load_state_dict(torch.load(source,weights_only=True));model.eval().requires_grad_(False)
    actor=common.controller()
    # Mean is from independently sampled training-distribution rollouts, never final.
    g=common.generator(482000+seed);e=draw_domains(256,g,'cuda')
    x=common.uniform((256,2),g,-.6,.6);goal=common.uniform((256,2),g,-.6,.6)
    m=run(model,actor,e,x,goal,'continuous',g)['z'][:,-1].mean(0).cuda()
    out=R/'causal'/phase/kind/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    torch.save(m.cpu(),out/'TRAIN_MEAN.pt')
    for code,split in enumerate(('native','id','ood')):
        n=128 if phase=='dev' else 512
        gen_seed=(483000 if phase=='dev' else 499000)+code
        g=common.generator(gen_seed);e=draw_domains(n,g,'cuda',split)
        x=common.uniform((n,2),g,-.6,.6);goal=common.uniform((n,2),g,-.6,.6)
        raw={mode:run(model,actor,e,x,goal,mode,common.generator(gen_seed+1000),m)
             for mode in ('frozen','continuous','donor','zero_feedback','train_mean','least_squares','oracle')}
        dest=out/split;dest.mkdir()
        torch.save(raw,dest/'RUNTIME.pt')
        common.save_json(dest/'SEAL.json',{'time':time.time(),'runtime_hash':common.sha(dest/'RUNTIME.pt'),
             'model_hash':common.sha(source),'n':n,'physical_actions_all_arms':8,'support':0,'latent_updates':7})
        torch.save(e.cpu(),dest/'EVALUATOR.pt')
        rows={}
        for mode,tr in raw.items():
            errors=(tr['state']-tr['goal'][:,None]).norm(dim=-1)
            rows[mode]={'success':(errors[:,-1]<.03).tolist(),'error':errors[:,-1].tolist(),
                        'error_curve':errors.mean(0).tolist()}
        common.save_json(dest/'RESULT.json',{'kind':kind,'seed':seed,'phase':phase,'split':split,'n':n,'rows':rows})
        print(kind,seed,phase,split,{k:sum(v['success'])/n for k,v in rows.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['structured','effect','history'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',choices=['dev','final'],default='dev');a=p.parse_args()
    torch.set_num_threads(2);evaluate(a.kind,a.seed,a.phase)
