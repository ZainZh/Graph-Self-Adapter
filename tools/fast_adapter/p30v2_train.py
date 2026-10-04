"""Full200-step chronological meta-training; carry z across TBPTT boundaries.

Observed states/actions are offline physical trajectories. This is supervised
meta-learning, not differentiation through a simulator or on-policy RL.
"""
import argparse
import json
import math
import time
import torch
from prove.fast_adapter.continuous_adapter import ObjectAdapter, feedback
from prove.fast_adapter.continuous_state import ContinuousState
from tools.fast_adapter.p30v2_common import *


def train(task, seed):
    torch.set_num_threads(2); torch.backends.cudnn.enabled=False; torch.manual_seed(seed)
    cfg=json.loads((ROOT/'PROTOCOL.json').read_text())['training']
    out=ROOT/task/'models'/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    b=load_bank(task); base=base_model(task); bh=state_hash(base)
    previous=torch.cat((torch.zeros_like(b['actions'][:,:1]),b['actions'][:,:-1]),1)
    with torch.no_grad():ba=base(b['visual'][:,:-1],b['proprio'][:,:-1],previous)
    if task==TASKS[1]:ba[...,3]=.9
    nets={'gradient':ObjectAdapter().cuda(),'history':ObjectAdapter(history=True).cuda()}
    common=nets['gradient'].state_dict(); hs=nets['history'].state_dict()
    hs.update({k:v.clone() for k,v in common.items()});nets['history'].load_state_dict(hs)
    optim={k:torch.optim.Adam(n.parameters(),lr=cfg['learning_rate']) for k,n in nets.items()}
    begin=time.time();updates={k:0 for k in nets}
    sources=[ROOT/'PROTOCOL.json',bank_path(task),base_path(task),Path(__file__),
             Path('prove/fast_adapter/continuous_adapter.py'),Path('prove/fast_adapter/continuous_state.py')]
    dump(out/'TRAIN_SOURCE_FREEZE.json',{'before_training':True,'bindings':[{'path':str(p),'sha256':digest(p)} for p in sources]})
    with (out/'TRAIN.jsonl').open('x') as log:
        for iteration in range(cfg['trajectory_batches']):
            ids=torch.randint(len(b['actions']),(cfg['batch_episodes'],),device='cuda')
            learners={k:ContinuousState(n,len(ids),mode=k,meta=True) for k,n in nets.items()}
            losses={k:[] for k in nets}
            for start in range(0,b['actions'].shape[1],cfg['truncate_every']):
                stop=min(start+cfg['truncate_every'],b['actions'].shape[1])
                for kind,learner in learners.items():
                    net=nets[kind];obj=0.
                    for t in range(start,stop):
                        v,p,prev=b['visual'][ids,t],b['proprio'][ids,t],previous[ids,t]
                        nv,np_=b['visual'][ids,t+1],b['proprio'][ids,t+1]
                        a=learner.action(v,p,prev,ba[ids,t])
                        # Held tool has fixed gripper; it cannot learn a release correction.
                        dims=3 if task==TASKS[1] else 4
                        action_loss=((a[...,:dims]-b['teacher'][ids,t,:dims])/.5).square().mean()
                        u=net.encode(v,p,prev); y,w=feedback(v,p,nv,np_)
                        prediction=net.prediction(u,b['actions'][ids,t],learner.z)
                        predloss=((prediction-y).square()*w).sum()/w.sum().clamp_min(1)
                        obj=obj+action_loss+cfg['prediction_loss_weight']*predloss
                        learner.observe(v,p,prev,b['actions'][ids,t],nv,np_)
                    loss=obj/(stop-start)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite meta loss')
                    opt=optim[kind]
                    for g in opt.param_groups:g['lr']=3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*iteration/cfg['trajectory_batches']))
                    opt.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
                    if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite outer gradient')
                    opt.step();updates[kind]+=1;losses[kind].append(float(loss.detach()))
                    # z and all recent physical feedback persist; only the graph is cut.
                    learner.detach_segment()
            if iteration==0 or (iteration+1)%20==0:
                row={'trajectory_batch':iteration+1,'seconds':time.time()-begin,
                     'outer_updates':dict(updates),'inner_steps_per_trajectory':{k:x.updates for k,x in learners.items()},
                     'loss':{k:sum(xs)/len(xs) for k,xs in losses.items()},
                     'z_rms':{k:float(x.z.detach().square().mean().sqrt()) for k,x in learners.items()}}
                log.write(json.dumps(row)+'\n');log.flush();print(task,seed,row,flush=True)
    assert state_hash(base)==bh
    # Time/episode-balanced average; uses TRAINING records only.
    for kind,net in nets.items():
        net.eval().requires_grad_(False);sums=torch.zeros(32,device='cuda');count=0
        for offset in range(0,len(b['actions']),8):
            ids=torch.arange(offset,min(offset+8,len(b['actions'])),device='cuda')
            learner=ContinuousState(net,len(ids),mode=kind)
            for t in range(b['actions'].shape[1]):
                learner.observe(b['visual'][ids,t],b['proprio'][ids,t],previous[ids,t],b['actions'][ids,t],
                                b['visual'][ids,t+1],b['proprio'][ids,t+1])
                sums+=learner.z.sum(0);count+=len(ids)
        mean=(sums/count).detach().cpu()
        path=out/f'{kind}.pt';torch.save({'state':net.cpu().state_dict(),'seed':seed,'mean_z':mean,
                 'horizon':200,'truncate_every':20,'z_reset_at_segment':False,'outer_updates':updates[kind],
                 'base_hash':bh,'training_bank_sha256':digest(bank_path(task))},path)
    dump(out/'COMPLETE.json',{'status':'trained_full200_step_sequences','task':task,'seed':seed,
         'trajectory_batches':cfg['trajectory_batches'],'outer_updates':updates,'seconds':time.time()-begin,
         'base_hash':bh,'models':[{'path':str(out/f'{k}.pt'),'sha256':digest(out/f'{k}.pt')} for k in nets]})
    print('TRAIN_COMPLETE',task,seed,time.time()-begin,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();train(a.task,a.seed)
