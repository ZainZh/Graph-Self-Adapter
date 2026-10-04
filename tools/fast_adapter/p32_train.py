"""Equal-budget chronological meta-training; learned energy is inner-only."""
import argparse
import math
import time
import torch
from tools.fast_adapter.p32_common import *
from prove.fast_adapter.continuous_adapter import feedback
from prove.fast_adapter.learned_interaction_loss import LossState


def train(seed):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    cfg=protocol()['training'];out=ROOT/'models'/f'seed{seed}'
    out.mkdir(parents=True,exist_ok=False)
    assert json.loads((ROOT/'QUALIFICATION.json').read_text())['passed']
    bank={k:v.cuda() for k,v in torch.load(P31/'data/own.pt',weights_only=True).items()}
    prevs=torch.cat((torch.zeros_like(bank['actions'][:,:1]),bank['actions'][:,:-1]),1)
    base=base_model(TASK);obs=observer();before={'base':state_hash(base),'observer':state_hash(obs)}
    with torch.no_grad():
        baselines=base(bank['visual'][:,:-1],bank['proprio'][:,:-1],prevs)
        gates=obs.active(bank['visual'][:,:-1],bank['proprio'][:,:-1],prevs)
    nets,counts=make_pair(seed)
    dump(out/'SOURCE_FREEZE.json',{'before_training':True,'bindings':bindings(source_paths()),
         'common_encoder_actor_prior_initialization_identical':True,'parameter_counts':counts,
         'seed':seed,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
    opts={k:torch.optim.Adam(n.parameters(),lr=cfg['learning_rate']) for k,n in nets.items()}
    gen=torch.Generator(device='cuda').manual_seed(seed+1000);begin=time.time();updates=0
    with (out/'TRAIN.jsonl').open('x') as log:
        for iteration in range(cfg['trajectory_batches']):
            ids=torch.randint(len(bank['actions']),(cfg['batch_episodes'],),generator=gen,device='cuda')
            learners={k:LossState(n,len(ids),meta=True) for k,n in nets.items()};losses={k:[] for k in nets}
            energy_grad_seen=False
            for start in range(0,200,20):
                for family,learner in learners.items():
                    net=nets[family];obj=0.
                    for t in range(start,start+20):
                        v,p,prev=bank['visual'][ids,t],bank['proprio'][ids,t],prevs[ids,t]
                        nv,np_=bank['visual'][ids,t+1],bank['proprio'][ids,t+1]
                        action=learner.action(v,p,prev,baselines[ids,t],gates[ids,t])
                        actionloss=((action-bank['teacher'][ids,t])/.5).square().mean()
                        u=net.encode(v,p,prev);y,w=feedback(v,p,nv,np_)
                        # Same auxiliary in both arms. Stop latent gradients here:
                        # the learned energy must be shaped by ACTION supervision,
                        # never by encouraging the latent to fit the auxiliary.
                        prediction=net.prediction(u,bank['actions'][ids,t],learner.z.detach())
                        predloss=((prediction-y).square()*w).sum()/w.sum().clamp_min(1)
                        obj=obj+actionloss+.1*predloss
                        learner.observe(v,p,prev,bank['actions'][ids,t],nv,np_,gates[ids,t])
                    loss=obj/20
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite outer loss')
                    opt=opts[family]
                    for g in opt.param_groups:g['lr']=3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*iteration/cfg['trajectory_batches']))
                    opt.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5)
                    if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite outer gradient')
                    if family=='learned':
                        energy_grad_seen |= any(p.grad is not None and bool(p.grad.abs().max()>0) for p in net.energy.parameters())
                    opt.step();learner.detach_segment();losses[family].append(float(loss.detach()))
                updates+=1
            if iteration==0 or (iteration+1)%20==0:
                row={'batch':iteration+1,'seconds':time.time()-begin,'outer_updates_each':updates,
                     'loss':{k:sum(v)/len(v) for k,v in losses.items()},
                     'z_rms':{k:float(v.z.detach().square().mean().sqrt()) for k,v in learners.items()},
                     'alpha':{k:float(n.alpha.detach()) for k,n in nets.items()},
                     'learned_energy_received_action_meta_gradient':energy_grad_seen}
                log.write(json.dumps(row)+'\n');log.flush();print('TRAIN',seed,row,flush=True)
    assert updates==2400 and before=={'base':state_hash(base),'observer':state_hash(obs)}
    for family,net in nets.items():
        torch.save({'state':net.cpu().state_dict(),'seed':seed,'family':family,'outer_updates':updates,
                    'data_sha256':digest(P31/'data/own.pt'),'parameter_count':counts[family]},out/f'{family}.pt')
    dump(out/'COMPLETE.json',{'models':list(nets),'seconds':time.time()-begin,'outer_updates_each':updates,
                            'frozen_parents_unchanged':True,'parameter_counts':counts,
                            'bindings':bindings([out/f'{n}.pt' for n in nets])})
    print('TRAIN_COMPLETE',seed,time.time()-begin,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);train(p.parse_args().seed)
