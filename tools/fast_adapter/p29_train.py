import argparse,json,math,time
from pathlib import Path
import torch
from prove.fast_adapter.object_interaction_adapter import ObjectSkill,ObjectAdapter,feedback
from tools.fast_adapter.p29_common import *
from tools.fast_adapter.p29_data import load_stage

def gpu(b):return {k:v.cuda() for k,v in b.items()}
def previous(b):return torch.cat((torch.zeros_like(b['actions'][:,:1]),b['actions']),1)
def support_inputs(b):
    return {'visual':b['visual'][:,:-1],'next_visual':b['visual'][:,1:],'proprio':b['proprio'][:,:-1],
            'next_proprio':b['proprio'][:,1:],'previous':previous(b)[:,:-1],'actions':b['actions']}
def base_train(task,stage):
    torch.set_num_threads(2);torch.manual_seed(2911);root=task_root(task);out=root/'models';out.mkdir(parents=True,exist_ok=True)
    banks=[gpu(load_stage(task,'demos'))];net=ObjectSkill().cuda()
    if stage=='dagger':
        net.load_state_dict(torch.load(out/'base_bc.pt',weights_only=True)['state']);banks.append(gpu(load_stage(task,'dagger')))
    prevs=[previous(b) for b in banks];total=12000 if stage=='bc' else 6000;opt=torch.optim.Adam(net.parameters(),lr=.0006);began=time.time()
    with (out/f'BASE_{stage}.jsonl').open('x') as log:
        for step in range(1,total+1):
            bi=step%len(banks);b=banks[bi];d=torch.randint(len(b['actions']),(512,),device='cuda');t=torch.randint(b['actions'].shape[1],(512,),device='cuda')
            a=net(b['visual'][d,t],b['proprio'][d,t],prevs[bi][d,t]);loss=(a-b['teacher'][d,t]).square().mean()
            for g in opt.param_groups:g['lr']=.00003+.5*(.0006-.00003)*(1+math.cos(math.pi*step/total))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
            if step==1 or step%1000==0:
                row={'step':step,'loss':float(loss.detach()),'seconds':time.time()-began};log.write(json.dumps(row)+'\n');log.flush();print(task,stage,row,flush=True)
    p=out/('base_bc.pt' if stage=='bc' else 'base_final.pt');torch.save({'state':net.cpu().state_dict(),'seed':2911,'updates_this_stage':total,'visual_contract':'2*(PCA64+RGBbox6+presence1), robot4,previous4'},p)
    save_json(out/f'BASE_{stage}_COMPLETE.json',{'path':str(p),'sha256':digest(p),'unique_training_episodes':sum(len(b['actions']) for b in banks),'updates':total,'seconds':time.time()-began,'parameter_count':sum(p.numel() for p in net.parameters())})

def adapt_train(task,seed,round_name='r1',steps=14000):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.manual_seed(seed)
    out=task_root(task)/'models'/round_name/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    support=support_inputs(gpu(load_stage(task,'meta','support')));qs=[gpu(load_stage(task,'meta',role)) for role in ('expert','base')]
    for q in qs:q['previous']=previous(q)
    base=load_base(task);basehash=weight_hash(base)
    nets={'fast':ObjectAdapter().cuda(),'history':ObjectAdapter(history=True).cuda(),'robot':ObjectAdapter(feedback_mode='robot').cuda()}
    shared=nets['fast'].state_dict()
    for name in ('history','robot'):
        state=nets[name].state_dict();state.update({k:v.clone() for k,v in shared.items()});nets[name].load_state_dict(state)
    opts={k:torch.optim.Adam(n.parameters(),lr=.0003) for k,n in nets.items()};began=time.time()
    with (out/'TRAIN.jsonl').open('x') as log:
        for step in range(1,steps+1):
            ids=torch.randint(len(support['actions']),(24,),device='cuda');q=qs[step%2];t=torch.randint(q['actions'].shape[1],(24,8),device='cuda');d=ids[:,None]
            v,p,prev=q['visual'][d,t],q['proprio'][d,t],q['previous'][d,t]
            with torch.no_grad():ba=base(v,p,prev)
            s={k:x[ids] for k,x in support.items()};K=(1,2,3,5)[int(torch.randint(4,()).item())]
            target,weights=feedback(v,p,q['visual'][d,t+1],q['proprio'][d,t+1]);row={'step':step,'K':K,'seconds':time.time()-began}
            for name,net in nets.items():
                z,_=net.adapt(s,K,meta=True);zz=z[:,None].expand(-1,8,-1);a=net.action(v,p,prev,ba,zz)
                actloss=((a-q['teacher'][d,t])/.5).square().mean();pred=net.prediction(net.encode(v,p,prev),q['actions'][d,t],zz)
                w=weights.clone()
                if name=='robot':w[...,:24]=0
                predloss=((pred-target).square()*w).sum()/w.sum().clamp_min(1)
                loss=actloss+.1*predloss;assert torch.isfinite(loss)
                for g in opts[name].param_groups:g['lr']=.00003+.5*(.0003-.00003)*(1+math.cos(math.pi*step/steps))
                opts[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5);assert torch.isfinite(norm);opts[name].step()
                if step==1 or step%1000==0:row[name]={'action_loss':float(actloss.detach()),'prediction_loss':float(predloss.detach()),'alpha':float(net.alpha.detach()),'z_delta_rms':float((z-net.z0).detach().square().mean().sqrt())}
            if step==1 or step%1000==0:log.write(json.dumps(row)+'\n');log.flush();print(task,seed,row,flush=True)
    assert weight_hash(base)==basehash
    for name,net in nets.items():
        net.eval().requires_grad_(False);zs=[]
        for off in range(0,len(support['actions']),8):zs.append(net.adapt({k:x[off:off+8] for k,x in support.items()},3)[0].cpu())
        torch.save({'state':net.cpu().state_dict(),'seed':seed,'steps':steps,'round':round_name,'training_z':torch.cat(zs),'mean_z':torch.cat(zs).mean(0),'feedback':net.feedback_mode},out/f'{name}.pt')
    save_json(out/'COMPLETE.json',{'task':task,'seed':seed,'steps_each':steps,'variants':list(nets),'seconds':time.time()-began,'base_weight_hash':basehash,'models':[{'path':str(out/f'{n}.pt'),'sha256':digest(out/f'{n}.pt')} for n in nets]})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['base','adapt']);p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--stage',choices=['bc','dagger']);p.add_argument('--seed',type=int);p.add_argument('--round',default='r1');p.add_argument('--steps',type=int,default=14000);a=p.parse_args()
    if a.mode=='base':base_train(a.task,a.stage)
    else:adapt_train(a.task,a.seed,a.round,a.steps)
