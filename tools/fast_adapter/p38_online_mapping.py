"""Separately frozen P38 follow-up: train for continuous execution updates.

The original episodic-trained checkpoints/results remain unchanged. The only
two selected architectures are predeclared structured6 and learned-effect32;
the frozen learned controller is shared byte-for-byte with the first wave.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from prove.fast_adapter.causal_mapping import Identification, draw_domains, effect
from tools.fast_adapter import p38_mapping as common

R=Path('evaluations/fast_adapter_p38_causal_v01/online_mapping')


def train(kind,seed):
    out=R/'mapping'/kind/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);g=common.generator(480000+seed)
    n=Identification(kind).cuda();actor=common.controller()
    opt=torch.optim.Adam(n.parameters(),lr=.0007);log=[];began=time.time()
    for step in range(1,4001):
        e=draw_domains(64,g,'cuda');x=common.uniform((64,2),g,-.6,.6);goal=common.uniform((64,2),g,-.6,.6)
        z=n.initial(64);history=[[],[],[],[]];losses=[]
        for t in range(8):
            a=actor(goal-x,n.response(z));new_x=x+effect(e,a)
            losses.append((new_x-goal).square().mean())
            if t<7:
                delta=(new_x-x)+.005*torch.randn(x.shape,generator=g,device='cuda')
                for values,value in zip(history,(x,goal,a,delta)):values.append(value)
                fields=[torch.stack(v[-4:],1) for v in history]
                z,_=n.update(z,*fields,meta=True)
            x=new_x
        loss=torch.stack(losses).mean()
        assert torch.isfinite(loss),(kind,seed,step)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),5.);opt.step()
        if step==1 or step%500==0:
            row={'step':step,'rollout_loss':float(loss.detach()),'last_error':float((x-goal).norm(dim=-1).mean().detach()),
                 'rate':float(n.rate.mean().detach()),'seconds':time.time()-began};log.append(row)
            common.save_json(out/'PROGRESS.json',row);print(kind,seed,row,flush=True)
    torch.save(n.cpu().state_dict(),out/'model.pt')
    common.save_json(out/'TRAIN.json',{'kind':kind,'seed':seed,'steps':4000,'unroll':8,
           'feedback_window':4,'support_actions':0,'log':log,'model_hash':common.sha(out/'model.pt')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','eval']);p.add_argument('--kind',choices=['structured','effect']);p.add_argument('--seed',type=int,required=True);p.add_argument('--phase',choices=['dev','final'],default='dev');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    if a.mode=='train':train(a.kind,a.seed)
    else:
        original_generator=common.generator
        common.generator=lambda seed:original_generator(seed+100000)
        common.ROOT=R
        for split in ('native','id','ood'):common.evaluate(a.kind,a.seed,split,a.phase)
