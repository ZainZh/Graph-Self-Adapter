"""P31 equal-budget four-cell factorial; inherited predictive objective."""
import argparse
import math
import time
import torch
from tools.fast_adapter.p31v2_common import *
from prove.fast_adapter.continuous_adapter import feedback
from prove.fast_adapter.interaction_stage_adapter import StageState


def train(seed, data_kind):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.manual_seed(seed)
    cfg=protocol()['training'];out=ROOT/'models'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    names=[f'{data_kind}_full',f'{data_kind}_stage']
    if any((out/f'{n}.pt').exists() for n in names):raise FileExistsError('No checkpoint overwrite')
    assert json.loads((ROOT/'OBSERVER_QUALIFICATION.json').read_text())['passed']
    b={k:v.cuda() for k,v in torch.load(ROOT/'data'/f'{data_kind}.pt',weights_only=True).items()}
    previous=torch.cat((torch.zeros_like(b['actions'][:,:1]),b['actions'][:,:-1]),1)
    base=base_model(TASK);bh=state_hash(base);obs=observer()
    with torch.no_grad():
        ba=base(b['visual'][:,:-1],b['proprio'][:,:-1],previous)
        gate=obs.active(b['visual'][:,:-1],b['proprio'][:,:-1],previous)
    # Re-seed AFTER infrastructure/network loading for exactly matched initialization.
    torch.manual_seed(seed);nets={n:ObjectAdapter().cuda() for n in names}
    init={k:v.clone() for k,v in nets[names[0]].state_dict().items()}
    nets[names[1]].load_state_dict(init)
    generator=torch.Generator(device='cuda').manual_seed(seed+1000)
    optim={n:torch.optim.Adam(x.parameters(),lr=cfg['learning_rate']) for n,x in nets.items()}
    dump(out/f'{data_kind}_SOURCE_FREEZE.json',{'before_training':True,'initial_state_hash':state_hash(nets[names[0]]),
         'bindings':bindings(source_paths()+[ROOT/'observer.pt',ROOT/'data'/f'{data_kind}.pt'])})
    begin=time.time()
    with (out/f'{data_kind}_TRAIN.jsonl').open('x') as log:
        for iteration in range(cfg['trajectory_batches']):
            ids=torch.randint(len(b['actions']),(cfg['batch_episodes'],),generator=generator,device='cuda')
            learners={n:StageState(net,len(ids),stage=n.endswith('_stage'),meta=True) for n,net in nets.items()}
            losses={n:[] for n in names}
            for start in range(0,200,20):
                for name,learner in learners.items():
                    net=nets[name];obj=0.
                    for t in range(start,start+20):
                        v,p,prev=b['visual'][ids,t],b['proprio'][ids,t],previous[ids,t]
                        nv,np_=b['visual'][ids,t+1],b['proprio'][ids,t+1]
                        active=gate[ids,t]
                        a=learner.action(v,p,prev,ba[ids,t],active)
                        actionloss=((a-b['teacher'][ids,t])/.5).square().mean()
                        u=net.encode(v,p,prev);y,w=feedback(v,p,nv,np_)
                        prediction=net.prediction(u,b['actions'][ids,t],learner.z)
                        predloss=((prediction-y).square()*w).sum()/w.sum().clamp_min(1)
                        obj=obj+actionloss+.1*predloss
                        learner.observe(v,p,prev,b['actions'][ids,t],nv,np_,active)
                    loss=obj/20
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite outer loss')
                    opt=optim[name]
                    for g in opt.param_groups:g['lr']=3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*iteration/cfg['trajectory_batches']))
                    opt.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5)
                    if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite outer gradient')
                    opt.step();learner.detach_segment();losses[name].append(float(loss.detach()))
            if iteration==0 or (iteration+1)%20==0:
                row={'batch':iteration+1,'seconds':time.time()-begin,
                     'loss':{k:sum(v)/len(v) for k,v in losses.items()},
                     'z_rms':{k:float(v.z.detach().square().mean().sqrt()) for k,v in learners.items()},
                     'mean_updates':{k:float(v.counts.float().mean()) for k,v in learners.items()}}
                log.write(json.dumps(row)+'\n');log.flush();print('TRAIN',seed,data_kind,row,flush=True)
    assert state_hash(base)==bh
    for name,net in nets.items():
        torch.save({'state':net.cpu().state_dict(),'seed':seed,'variant':name,'outer_updates':2400,
              'initial_state_hash':state_hash_from_init(init),'base_hash':bh,'training_bank_sha256':digest(ROOT/'data'/f'{data_kind}.pt')},out/f'{name}.pt')
    dump(out/f'{data_kind}_COMPLETE.json',{'trained':names,'seconds':time.time()-begin,'outer_updates_each':2400,
         'base_unchanged':True,'bindings':bindings([out/f'{n}.pt' for n in names])})
    print('TRAIN_COMPLETE',seed,data_kind,time.time()-begin,flush=True)


def state_hash_from_init(state):
    n=ObjectAdapter();n.load_state_dict(state);return state_hash(n)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--data',choices=['old','own'],required=True)
    a=p.parse_args();train(a.seed,a.data)
