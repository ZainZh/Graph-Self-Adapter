"""Bounded tool revision based only on completed development r1 outcomes.
Canonical replay anchors preserve the existing policy; no final examples are used.
"""
import argparse,json,math,time
import numpy as np
import torch
from prove.fast_adapter.object_tool_adapter_v2 import ObjectToolAdapterV2 as ObjectAdapterV2
from prove.fast_adapter.object_interaction_adapter import feedback
from tools.fast_adapter.p29_common import *
from tools.fast_adapter.p29_data import collect_block,load_stage
from tools.fast_adapter.p29_train import gpu,previous,support_inputs
TASK='hammer-block'

def collect(dino='assets/dinov3'):
    out=task_root(TASK)/'data/canonical_anchors_r2';out.mkdir(parents=True,exist_ok=False);base=load_base(TASK);visual=Visual(dino,TASK,torch.load(task_root(TASK)/'data/PCA.pt',weights_only=True));records=[]
    for role,delta,length in [('support',0,SUPPORT),('query',10000,HORIZON)]:
        for off in (0,8):
            ids=np.arange(off,off+8);cfgs=[object_config(TASK,int(2975000+i),'canonical') for i in ids];seeds=2976000+delta+ids
            b,m=collect_block(TASK,cfgs,seeds,length,'base',visual,base,labels=False);p=out/f'{role}_{off:03d}.pt';torch.save(b,p);h=digest(p)
            save_json(p.with_suffix('.json'),{'configs':cfgs,'metrics':m,'sha256':h});records.append({'path':str(p),'sha256':h})
            print('ANCHOR',role,off,flush=True)
    save_json(out/'COMPLETE.json',{'time':time.time(),'records':records,'all16_included_no_success_filter':True,'new_training_resets_only':True})
def anchors():
    root=task_root(TASK)/'data/canonical_anchors_r2';out={}
    for role in ('support','query'):
        bs=[torch.load(root/f'{role}_{i:03d}.pt',weights_only=True) for i in (0,8)];out[role]=gpu({k:torch.cat([b[k] for b in bs]) for k in ('visual','proprio','actions')})
    out['query']['previous']=previous(out['query']);out['support']=support_inputs(out['support']);return out

def train(seed):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.manual_seed(seed)
    out=task_root(TASK)/'models/r2'/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    support=support_inputs(gpu(load_stage(TASK,'meta','support')));qs=[gpu(load_stage(TASK,'meta',r)) for r in ('expert','base')]
    for q in qs:q['previous']=previous(q)
    anchor=anchors();base=load_base(TASK);basehash=weight_hash(base)
    for q in qs+[anchor['query']]:
        with torch.no_grad():q['base_actions']=base(q['visual'][:,:-1],q['proprio'][:,:-1],q['previous'][:,:-1])
    nets={'fast':ObjectAdapterV2().cuda(),'history':ObjectAdapterV2(history=True).cuda(),'robot':ObjectAdapterV2(feedback_mode='robot').cuda()}
    shared=nets['fast'].state_dict()
    for name in ('history','robot'):
        state=nets[name].state_dict();state.update({k:v.clone() for k,v in shared.items()});nets[name].load_state_dict(state)
    opts={k:torch.optim.Adam(n.parameters(),lr=.0003) for k,n in nets.items()};total=14000;began=time.time()
    with (out/'TRAIN.jsonl').open('x') as log:
        for step in range(1,total+1):
            ids=torch.randint(len(support['actions']),(24,),device='cuda');q=qs[step%2];t=torch.randint(HORIZON,(24,8),device='cuda');d=ids[:,None]
            v,p,prev=q['visual'][d,t],q['proprio'][d,t],q['previous'][d,t]
            ba=q['base_actions'][d,t]
            s={k:x[ids] for k,x in support.items()};K=(1,2,3,5)[int(torch.randint(4,()).item())];target,w=feedback(v,p,q['visual'][d,t+1],q['proprio'][d,t+1])
            ai=torch.randint(16,(8,),device='cuda');at=torch.randint(HORIZON,(8,8),device='cuda');ad=ai[:,None];aq=anchor['query'];av,ap,apr=aq['visual'][ad,at],aq['proprio'][ad,at],aq['previous'][ad,at]
            ab=aq['base_actions'][ad,at]
            ass={k:x[ai] for k,x in anchor['support'].items()};row={'step':step,'seconds':time.time()-began,'K':K}
            for name,net in nets.items():
                z,_=net.adapt(s,K,meta=True);a=net.action(v,p,prev,ba,z[:,None].expand(-1,8,-1));actloss=((a-q['teacher'][d,t])/.5).square().mean()
                pred=net.prediction(net.encode(v,p,prev),q['actions'][d,t],z[:,None].expand(-1,8,-1));ww=w.clone()
                if name=='robot':ww[...,:24]=0
                predloss=((pred-target).square()*ww).sum()/ww.sum().clamp_min(1)
                az,_=net.adapt(ass,3,meta=True);aa=net.action(av,ap,apr,ab,az[:,None].expand(-1,8,-1));preserve=((aa-ab)/.5).square().mean()
                loss=actloss+.1*predloss+1.0*preserve;assert torch.isfinite(loss)
                for g in opts[name].param_groups:g['lr']=.00003+.5*(.0003-.00003)*(1+math.cos(math.pi*step/total))
                opts[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5);assert torch.isfinite(norm);opts[name].step()
                if step==1 or step%1000==0:row[name]={'action_loss':float(actloss.detach()),'prediction_loss':float(predloss.detach()),'preservation_loss':float(preserve.detach()),'alpha':float(net.alpha.detach())}
            if step==1 or step%1000==0:log.write(json.dumps(row)+'\n');log.flush();print(TASK,'r2',seed,row,flush=True)
    assert weight_hash(base)==basehash
    for name,net in nets.items():
        net.eval().requires_grad_(False);zs=[]
        for off in range(0,64,8):zs.append(net.adapt({k:x[off:off+8] for k,x in support.items()},3)[0].cpu())
        torch.save({'state':net.cpu().state_dict(),'seed':seed,'steps':total,'round':'r2','training_z':torch.cat(zs),'mean_z':torch.cat(zs).mean(0),'feedback':net.feedback_mode},out/f'{name}.pt')
    save_json(out/'COMPLETE.json',{'seed':seed,'steps_each':total,'seconds':time.time()-began,'base_weight_hash':basehash,'new_training_anchors':32,'models':[{'path':str(out/f'{v}.pt'),'sha256':digest(out/f'{v}.pt')} for v in nets]})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','train']);p.add_argument('--seed',type=int);p.add_argument('--dino',default='assets/dinov3');a=p.parse_args()
    if a.mode=='collect':collect(a.dino)
    else:train(a.seed)
