"""Same chronological meta-training budget across the four visual input arms."""
import argparse,time
import numpy as np
import torch
from tools.fast_adapter.p35_common import *
from prove.fast_adapter.multiview_role_dynamics import MultiViewRoleDynamics
from prove.fast_adapter.role_dynamics import observed_effect

def encode(net,b,ids,times):
    prev=(times-1).clamp_min(0);a=b['actions'][ids,prev]*(times>0)[...,None]
    v=(b['boxes'][ids,times]-b['boxes'][ids,prev])*(times>0)[...,None,None,None]
    return net.encode(b['tokens'][ids,times],b['boxes'][ids,times],b['proprio'][ids,times],a,v)
def targets(b):
    t=b['target_tokens'];bb=b['target_boxes'];p=b['proprio']
    return observed_effect(t[:,:-4],bb[:,:-4],p[:,:-4],t[:,4:],bb[:,4:],p[:,4:])
def plans(b):return b['actions'].unfold(1,H,1).transpose(-1,-2).contiguous()

def train(task,view,seed):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;cfg=protocol()['learning']
    out=ROOT/task/'models'/view/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    b=load_data(task,view);stats=torch.load(stats_path(task,view),weights_only=True)
    yy,ww=targets(b);yy/=stats['effect_sd'].cuda();aa=plans(b);candidates=torch.where(b['object_ids']<24)[0]
    nets={};opts={}
    for f in FAMILIES:
        torch.manual_seed(seed);n=MultiViewRoleDynamics(f,task).cuda();n.set_stats(stats)
        nets[f]=n;opts[f]=torch.optim.Adam(n.parameters(),lr=cfg['learning_rate'])
    common=('role_project','proprio_project','goal_visual','goal_semantic','roles','types','readout','core','mean','gate','basis','z0','raw_alpha')
    shared={k:v for k,v in nets['flat'].state_dict().items() if k.split('.')[0] in common}
    for n in nets.values():n.load_state_dict(shared,strict=False)
    counts={f:sum(p.numel() for p in n.parameters()) for f,n in nets.items()};assert max(counts.values())/min(counts.values())<1.03
    dump(out/'SOURCE_FREEZE.json',{'before_training':True,'parameter_counts':counts,'shared_initialization':list(common),
         'bindings':bindings(source_paths(task,view)+[Path(__file__)])})
    gen=torch.Generator(device='cuda').manual_seed(seed+333);began=time.time();log=(out/'TRAIN.jsonl').open('x')
    def report(row):
        row['seconds']=time.time()-began;log.write(json.dumps(row)+'\n');log.flush();print('P35_TRAIN',task,view,seed,row,flush=True)
    for k in range(cfg['warmup_updates']):
        ids=candidates[torch.randint(len(candidates),(128,),device='cuda',generator=gen)]
        ts=torch.randint(197,(128,),device='cuda',generator=gen)
        for f,n in nets.items():
            u=encode(n,b,ids,ts);mu,g=n.coefficients(u,aa[ids,ts]);pred=n.predict_coeff(mu,g,n.z0)
            loss=n.loss(pred,yy[ids,ts],ww[ids,ts]).mean();opts[f].zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(n.parameters(),5);opts[f].step()
        if k==0 or (k+1)%200==0:report({'phase':'warmup','updates':k+1,'last_loss':float(loss.detach())})
    for iteration in range(cfg['meta_trajectory_batches']):
        ids=candidates[torch.randint(len(candidates),(8,),device='cuda',generator=gen)]
        zs={f:n.z0[None].expand(8,-1) for f,n in nets.items()};losses={f:[] for f in nets}
        for start in range(0,200,20):
            lo=max(0,start-11);hi=min(start+20,197);times=torch.arange(lo,hi,device='cuda')[None].expand(8,-1)
            for f,n in nets.items():
                u=encode(n,b,ids[:,None],times);mu,g=n.coefficients(u,aa[ids[:,None],times]);z=zs[f];loss=0.;number=0
                for t in range(start,start+20):
                    if t>=4:
                        a=max(0,t-11);e=t-3;sl=slice(a-lo,e-lo)
                        z,_=n.update(z,mu[:,sl],g[:,sl],yy[ids,a:e],ww[ids,a:e])
                    if t<197:
                        pred=n.predict_coeff(mu[:,t-lo],g[:,t-lo],z);loss=loss+n.loss(pred,yy[ids,t],ww[ids,t]).mean();number+=1
                loss/=number;opts[f].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(n.parameters(),5)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite gradient')
                opts[f].step();zs[f]=z.detach().requires_grad_(True);losses[f].append(float(loss.detach()))
        if iteration==0 or (iteration+1)%20==0:report({'phase':'meta','batches':iteration+1,
             'loss':{f:float(np.mean(v)) for f,v in losses.items()},'alpha':{f:float(n.alpha.detach()) for f,n in nets.items()}})
    for f,n in nets.items():torch.save({'state':n.cpu().state_dict(),'family':f,'task':task,'view':view,'seed':seed,
                      'warmup':600,'outer_updates':1000},out/f'{f}.pt')
    log.close();dump(out/'COMPLETE.json',{'seconds':time.time()-began,'parameter_counts':counts,'bindings':bindings([out/f'{f}.pt' for f in nets])})
    print('P35_TRAIN_COMPLETE',task,view,seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--view',choices=VIEWS,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args();train(a.task,a.view,a.seed)
