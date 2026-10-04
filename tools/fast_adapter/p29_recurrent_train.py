"""P29 hammer candidate3: supervised causal base, same latent adapter comparison."""
import argparse,json,math,time
import torch
from prove.fast_adapter.object_sequence_skill import ObjectSequenceSkill
from prove.fast_adapter.object_interaction_adapter import ObjectAdapter,feedback
from tools.fast_adapter.p29_common import *
from tools.fast_adapter.p29_data import load_stage,collect_block
from tools.fast_adapter.p29_train import support_inputs,previous,gpu
TASK='hammer-block'

def train_base():
    torch.set_num_threads(2);torch.manual_seed(2933);root=task_root(TASK);out=root/'candidates/base_v03';out.mkdir(parents=True,exist_ok=True)
    b=gpu(load_stage(TASK,'demos'));prev=previous(b);net=ObjectSequenceSkill().cuda();opt=torch.optim.Adam(net.parameters(),lr=.0005);began=time.time()
    with (out/'TRAIN.jsonl').open('x') as log:
        for step in range(1,8001):
            d=torch.randint(len(b['actions']),(16,),device='cuda');a=net.forward_sequence(b['visual'][d,:-1],b['proprio'][d,:-1],prev[d,:-1])
            loss=(a-b['teacher'][d]).square().mean()
            for g in opt.param_groups:g['lr']=.00003+.5*(.0005-.00003)*(1+math.cos(math.pi*step/8000))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            if step==1 or step%1000==0:
                row={'step':step,'loss':float(loss.detach()),'seconds':time.time()-began};log.write(json.dumps(row)+'\n');log.flush();print(row,flush=True)
    path=out/'base.pt';torch.save({'state':net.cpu().state_dict(),'updates':8000,'training_episodes':96,'architecture':'causal GRU64 with RGB/proprio input; no runtime gradient'},path)
    save_json(out/'TRAIN_COMPLETE.json',{'path':str(path),'sha256':digest(path),'parameter_count':sum(p.numel() for p in net.parameters()),'seconds':time.time()-began})

def qualify_base(dino='assets/dinov3'):
    out=task_root(TASK)/'candidates/base_v03';net=ObjectSequenceSkill().cuda();net.load_state_dict(torch.load(out/'base.pt',weights_only=True)['state']);net.eval().requires_grad_(False)
    visual=Visual(dino,TASK,torch.load(task_root(TASK)/'data/PCA.pt',weights_only=True))
    cfgs=[object_config(TASK,2940000+i,'canonical') for i in range(8)];ids=[2942000+i for i in range(8)]
    net.reset_rollout();b,m=collect_block(TASK,cfgs,ids,HORIZON,'base',visual,net,labels=False)
    p=out/'QUALIFICATION.pt';torch.save(b,p);success=[bool(any(x['success'] for x in q)) for q in m]
    save_json(out/'BASE_GATE.json',{'passed':sum(success)>=6,'successes':sum(success),'n':8,'success':success,'source_sha256':digest(p),'configs':cfgs,'metrics':m,'same_opened_development_resets':True})
    print('RECURRENT_BASE_GATE',sum(success),'/8',flush=True)

def adapt_train(task,seed,round_name='r1',steps=14000):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.manual_seed(seed)
    out=task_root(task)/'models'/round_name/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    support=support_inputs(gpu(load_stage(task,'meta','support')));qs=[gpu(load_stage(task,'meta',role)) for role in ('expert','base')]
    base=load_base(task);basehash=weight_hash(base)
    # Causal base predictions use full observed prefixes, never random-frame GRU resets.
    for q in qs:
        q['previous']=previous(q)
        with torch.no_grad():q['base_actions']=base(q['visual'][:,:-1],q['proprio'][:,:-1],q['previous'][:,:-1])
    nets={'fast':ObjectAdapter().cuda(),'history':ObjectAdapter(history=True).cuda(),'robot':ObjectAdapter(feedback_mode='robot').cuda()}
    shared=nets['fast'].state_dict()
    for name in ('history','robot'):
        state=nets[name].state_dict();state.update({k:v.clone() for k,v in shared.items()});nets[name].load_state_dict(state)
    opts={k:torch.optim.Adam(n.parameters(),lr=.0003) for k,n in nets.items()};began=time.time()
    with (out/'TRAIN.jsonl').open('x') as log:
        for step in range(1,steps+1):
            ids=torch.randint(len(support['actions']),(24,),device='cuda');q=qs[step%2];t=torch.randint(HORIZON,(24,8),device='cuda');d=ids[:,None]
            v,p,prev=q['visual'][d,t],q['proprio'][d,t],q['previous'][d,t];ba=q['base_actions'][d,t]
            s={k:x[ids] for k,x in support.items()};K=(1,2,3,5)[int(torch.randint(4,()).item())]
            target,weights=feedback(v,p,q['visual'][d,t+1],q['proprio'][d,t+1]);row={'step':step,'K':K,'seconds':time.time()-began}
            for name,net in nets.items():
                z,_=net.adapt(s,K,meta=True);zz=z[:,None].expand(-1,8,-1);a=net.action(v,p,prev,ba,zz)
                actloss=((a-q['teacher'][d,t])/.5).square().mean();pred=net.prediction(net.encode(v,p,prev),q['actions'][d,t],zz)
                w=weights.clone()
                if name=='robot':w[...,:24]=0
                predloss=((pred-target).square()*w).sum()/w.sum().clamp_min(1);loss=actloss+.1*predloss;assert torch.isfinite(loss)
                for g in opts[name].param_groups:g['lr']=.00003+.5*(.0003-.00003)*(1+math.cos(math.pi*step/steps))
                opts[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5);assert torch.isfinite(norm);opts[name].step()
                if step==1 or step%1000==0:row[name]={'action_loss':float(actloss.detach()),'prediction_loss':float(predloss.detach()),'alpha':float(net.alpha.detach())}
            if step==1 or step%1000==0:log.write(json.dumps(row)+'\n');log.flush();print(task,seed,row,flush=True)
    assert weight_hash(base)==basehash
    for name,net in nets.items():
        net.eval().requires_grad_(False);zs=[]
        for off in range(0,len(support['actions']),8):zs.append(net.adapt({k:x[off:off+8] for k,x in support.items()},3)[0].cpu())
        torch.save({'state':net.cpu().state_dict(),'seed':seed,'steps':steps,'round':round_name,'training_z':torch.cat(zs),'mean_z':torch.cat(zs).mean(0),'feedback':net.feedback_mode},out/f'{name}.pt')
    save_json(out/'COMPLETE.json',{'task':task,'seed':seed,'steps_each':steps,'variants':list(nets),'seconds':time.time()-began,'base_weight_hash':basehash,'causal_base_prefix_preserved':True,'models':[{'path':str(out/f'{n}.pt'),'sha256':digest(out/f'{n}.pt')} for n in nets]})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train_base','qualify_base','adapt']);p.add_argument('--task',default=TASK);p.add_argument('--seed',type=int);p.add_argument('--round',default='r1');p.add_argument('--steps',type=int,default=14000);a=p.parse_args()
    if a.mode=='adapt':adapt_train(a.task,a.seed,a.round,a.steps)
    else:globals()[a.mode]()
