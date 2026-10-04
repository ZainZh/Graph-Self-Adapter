"""P34 isolated training; no expert targets and no edits to previous experiments."""
import argparse,time
import numpy as np
import torch
from tools.fast_adapter.p34_common import *
from prove.fast_adapter.role_dynamics import RoleDynamics,observed_effect

def encode(net,b,ids,times):
    prev=(times-1).clamp_min(0)
    previous=b['actions'][ids,prev]*(times>0)[...,None]
    velocity=(b['boxes'][ids,times]-b['boxes'][ids,prev])*(times>0)[...,None,None]
    return net.encode(b['tokens'][ids,times],b['boxes'][ids,times],b['proprio'][ids,times],previous,velocity)

def targets(b):
    return observed_effect(b['tokens'][:,:-H],b['boxes'][:,:-H],b['proprio'][:,:-H],
                           b['tokens'][:,H:],b['boxes'][:,H:],b['proprio'][:,H:])

def plans(b):return b['actions'].unfold(1,H,1).transpose(-1,-2).contiguous()

def train(task,seed):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    cfg=protocol()['learning'];out=ROOT/task/'models'/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    b=load_data(task);stats=torch.load(stats_path(task),weights_only=True)
    yy,ww=targets(b);aa=plans(b);yy=yy/stats['effect_sd'].cuda()
    candidates=torch.where(b['object_ids']<48)[0];nets={};opts={}
    for f in FAMILIES:
        torch.manual_seed(seed);n=RoleDynamics(f,task).cuda();n.set_stats(stats)
        nets[f]=n;opts[f]=torch.optim.Adam(n.parameters(),lr=cfg['learning_rate'])
    counts={f:sum(p.numel() for p in n.parameters()) for f,n in nets.items()}
    assert max(counts.values())/min(counts.values())<1.03,counts
    common=('role_project','proprio_project','goal_visual','goal_semantic','roles','types','readout','core','mean','gate','basis','z0','raw_alpha')
    shared={k:v for k,v in nets['flat'].state_dict().items() if k.split('.')[0] in common}
    for n in nets.values():n.load_state_dict(shared,strict=False)
    dump(out/'SOURCE_FREEZE.json',{'before_training':True,'parameter_counts':counts,
          'shared_initialization':list(common),
          'bindings':bindings(source_paths(task)+[Path(__file__),ROOT/'DATA_RECEIPT.json'])})
    gen=torch.Generator(device='cuda').manual_seed(seed+333);begin=time.time();log=(out/'TRAIN.jsonl').open('x')
    def report(row):
        row.update(seconds=time.time()-begin);log.write(json.dumps(row)+'\n');log.flush();print('TRAIN',task,seed,row,flush=True)
    for k in range(cfg['warmup_updates']):
        ids=candidates[torch.randint(len(candidates),(128,),device='cuda',generator=gen)]
        ts=torch.randint(201-H,(128,),device='cuda',generator=gen)
        for f,n in nets.items():
            u=encode(n,b,ids,ts);mu,g=n.coefficients(u,aa[ids,ts]);pred=n.predict_coeff(mu,g,n.z0)
            loss=n.loss(pred,yy[ids,ts],ww[ids,ts]).mean()
            opts[f].zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),5);opts[f].step()
        if k==0 or (k+1)%200==0:report({'phase':'warmup','updates':k+1,'last_loss':float(loss.detach())})
    for iteration in range(cfg['meta_trajectory_batches']):
        ids=candidates[torch.randint(len(candidates),(8,),device='cuda',generator=gen)]
        zs={f:n.z0[None].expand(8,-1) for f,n in nets.items()};losses={f:[] for f in nets}
        for start in range(0,200,20):
            lo=max(0,start-H-7);hi=min(start+20,201-H)
            times=torch.arange(lo,hi,device='cuda')[None].expand(8,-1)
            for f,n in nets.items():
                u=encode(n,b,ids[:,None],times);mu,g=n.coefficients(u,aa[ids[:,None],times])
                z=zs[f];loss=0.;number=0
                for t in range(start,start+20):
                    if t>=H:
                        a=max(0,t-H-7);e=t-H+1;sl=slice(a-lo,e-lo)
                        z,_=n.update(z,mu[:,sl],g[:,sl],yy[ids,a:e],ww[ids,a:e])
                    if t<201-H:
                        pred=n.predict_coeff(mu[:,t-lo],g[:,t-lo],z)
                        loss=loss+n.loss(pred,yy[ids,t],ww[ids,t]).mean();number+=1
                loss=loss/number;opts[f].zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(n.parameters(),5)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite meta gradient')
                opts[f].step();zs[f]=z.detach().requires_grad_(True);losses[f].append(float(loss.detach()))
        if iteration==0 or (iteration+1)%10==0:
            report({'phase':'meta','batches':iteration+1,'loss':{f:float(np.mean(x)) for f,x in losses.items()},
                    'alpha':{f:float(n.alpha.detach()) for f,n in nets.items()},
                    'z_delta_rms':{f:float((z-nets[f].z0).detach().square().mean().sqrt()) for f,z in zs.items()}})
    for f,n in nets.items():torch.save({'state':n.cpu().state_dict(),'family':f,'task':task,'seed':seed,'warmup':600,'meta_outer_updates':1000},out/f'{f}.pt')
    log.close();dump(out/'COMPLETE.json',{'seconds':time.time()-begin,'parameter_counts':counts,
                 'no_expert_action_targets':True,'bindings':bindings([out/f'{f}.pt' for f in nets])})
    print('TRAIN_COMPLETE',task,seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--task',choices=TASKS,required=True);a=p.parse_args();train(a.task,a.seed)
