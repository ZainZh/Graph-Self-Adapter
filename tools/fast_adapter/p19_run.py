"""P19 structured mechanism qualification; all paths/protocols explicit."""
from __future__ import annotations
import argparse,hashlib,json,time,os
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.structured_response import *
from tools.fast_adapter.p18_run import physics_data,save_json,digest
from tools.fast_adapter.p18_experiment import bounded_axis_probe_actions,P18HiddenShift,step_p18_shift,projected_oracle_action
from prove.fast_adapter.visual_alignment import FrozenDinoROI,appearance,visible,DEFAULT_MODEL

ROOT=Path('evaluations/fast_adapter_p19_structured_v01')
OLD=Path('evaluations/fast_adapter_p18_visual_v01')
SEEDS=[11,22,33,44,55]
KINDS=['numeric','dino_geometry']
A=bounded_axis_probe_actions()

def true_z(shift):
    return torch.cat([shift.M,shift.b[...,None]/SCALE],-1)

def observations(features,readout):
    x=features.flatten(-2).float()
    xx=(x-readout['mu'])/readout['sd']
    xx=torch.cat([xx,torch.ones_like(xx[...,:1])],-1)
    relative=xx@readout['weight']
    return relative[:,:-1]-relative[:,1:]

def package(shift,support,goal,qx,qg,features,readout,seed):
    gen=torch.Generator().manual_seed(seed)
    qa=torch.rand(len(support),4,2,generator=gen)*.5-.25
    response=torch.einsum('bij,bqj->bqi',shift.M,qa)+shift.b[:,None]
    return {'numeric':support[:,1:]-support[:,:-1], 'dino_geometry':observations(features,readout),
            'true_z':true_z(shift),'qx':qx,'qg':qg,'query_actions':qa,'query_response':response}

def save_dataset(data,path):
    torch.save(data,path)
    save_json(path.with_suffix('.json'),{'file':str(path),'sha256':digest(path),'domains':len(data['qx']),'keys':list(data)})

def prepare():
    started=time.time();bank=torch.load(OLD/'train_bank/bank.pt',weights_only=False,map_location='cpu')
    observer=torch.load(OLD/'preflight/DIAGNOSTIC_GEOMETRY_READOUT.pt',weights_only=False,map_location='cpu')
    torch.save(observer,ROOT/'geometry_readout.pt')
    shift,support,goal,qx,qg=physics_data(bank['n'],bank['seed'],'train',train=True)
    for name,lo,hi in [('train',4096,12288),('qualification',12288,12544)]:
        ss=P18HiddenShift(shift.M[lo:hi],shift.b[lo:hi])
        data=package(ss,support[lo:hi],goal[lo:hi],qx[lo:hi],qg[lo:hi],bank['support'][lo:hi],observer,2026092307+lo)
        save_dataset(data,ROOT/(name+'.pt'))
    save_json(ROOT/'DATA_RECEIPT.json',{'seconds':time.time()-started,'source_bank_sha256':digest(OLD/'train_bank/bank.pt'),'source_readout_sha256':digest(OLD/'preflight/DIAGNOSTIC_GEOMETRY_READOUT.pt'),'readout_sha256':digest(ROOT/'geometry_readout.pt'),'opened_data_only':True})
    print('PREPARED',flush=True)

def build(family):return MetaGradient() if family=='meta_gradient' else History()

def train(kind,family,seed,device):
    start=time.time();torch.manual_seed(seed)
    data={k:v.to(device) for k,v in torch.load(ROOT/'train.pt',weights_only=False).items()}
    model=build(family).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    rng=torch.Generator().manual_seed(seed+93000);actions=A.to(device);logs=[]
    for step in range(1500):
        ids=torch.randint(len(data['qx']),(128,),generator=rng).to(device)
        k=[1,2,3,5][step%4]
        z=model(data[kind][ids,:k],actions[:k],steps=3,create_graph=True)
        qa=data['query_actions'][ids]
        x=torch.cat([qa/SCALE,torch.ones_like(qa[...,:1])],-1)
        predicted=torch.einsum('bij,bqj->bqi',z,x)*SCALE
        response_loss=((predicted-data['query_response'][ids])/SCALE).square().mean()
        qx=data['qx'][ids].flatten(0,1);qg=data['qg'][ids].flatten(0,1)
        act,_=control(qx,qg,z.repeat_interleave(4,0))
        truth=data['true_z'][ids].repeat_interleave(4,0)
        target,_=control(qx,qg,truth)
        action_loss=((act-target)/.15).square().mean()
        loss=response_loss+action_loss
        if not torch.isfinite(loss):raise FloatingPointError((kind,family,seed,step))
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        optimizer.step()
        if step%250==0 or step==1499:
            logs.append({'step':step+1,'loss':float(loss.detach()),'response_loss':float(response_loss.detach()),'action_loss':float(action_loss.detach()),'grad_norm':float(grad)})
    out=ROOT/'train'/kind/family/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),out/'model.pt')
    info={'kind':kind,'family':family,'seed':seed,'seconds':time.time()-start,'updates':1500,'parameters':sum(x.numel() for x in model.parameters()),'logs':logs,'checkpoint_sha256':digest(out/'model.pt')}
    if family=='meta_gradient':info.update(z0=model.z0.detach().cpu().tolist(),rate=model.rate.detach().cpu().tolist())
    save_json(out/'TRAIN.json',info);print(json.dumps({k:v for k,v in info.items() if k in ['kind','family','seed','seconds']}),flush=True)

def load_model(kind,family,seed):
    model=build(family)
    model.load_state_dict(torch.load(ROOT/'train'/kind/family/f'seed{seed}'/'model.pt',weights_only=True,map_location='cpu'))
    model.requires_grad_(False);model.eval();return model

def trajectory(data,z,base=False):
    n,q,_=data['qx'].shape
    state=data['qx'].flatten(0,1).clone();goal=data['qg'].flatten(0,1)
    truth=data['true_z'].repeat_interleave(q,0)
    est=z.detach().repeat_interleave(q,0)
    shifted=P18HiddenShift(truth[...,:2],truth[...,2]*SCALE)
    failed=torch.zeros(n*q,dtype=torch.bool);frames=[state.clone()];invalid=torch.zeros(n*q,dtype=torch.bool)
    first=None
    for step in range(12):
        if base:act=(.5*(goal-state)).clamp(-1,1)
        else:
            act,bad=control(state,goal,est);invalid|=bad
        if step==0:first=act.clone()
        state=step_p18_shift(state,shifted,act)
        failed|=(state.abs()>1.546875).any(-1);frames.append(state.clone())
    error=(state-goal).norm(dim=-1).reshape(n,q)
    return {'error':error,'success':(error<=.05)&~failed.reshape(n,q),'out_of_frame':failed.reshape(n,q),'fallback':invalid.reshape(n,q),'first_action':first.reshape(n,q,2),'frames':torch.stack(frames).reshape(13,n,q,2)}

def estimate_arms(data,kind,seed):
    meta=load_model(kind,'meta_gradient',seed);history=load_model(kind,'history_mlp',seed)
    d=data[kind];n=len(d);out={}
    for s in [0,1,2,3,5]:
        out[f'gradient_s{s}_u3']=meta(d[:,:s],A[:s],steps=3).detach()
        out[f'history_s{s}']=history(d[:,:s],A[:s]).detach()
    for u in [0,1,3,10,20]:out[f'gradient_s3_u{u}']=meta(d[:,:3],A[:3],steps=u).detach()
    out['gradient_swapped']=out['gradient_s3_u3'].roll(1,0)
    out['gradient_zero_support']=meta(torch.zeros_like(d[:,:3]),A[:3],steps=3).detach()
    out['history_swapped']=out['history_s3'].roll(1,0)
    for u in [3,20]:out[f'fixed_gradient_u{u}']=adapt(nominal_z(n),d[:,:3],A[:3],steps=u)[0].detach()
    out['least_squares']=least_squares(d[:,:3],A[:3])
    # Oracle is inserted ONLY after ordinary predictions are all complete.
    return out

def evaluate_dataset(name,data,kinds):
    target=ROOT/name;target.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for kind in kinds:
      for seed in SEEDS:
        estimates=estimate_arms(data,kind,seed)
        pred_path=target/f'{kind}_seed{seed}_PREDICTIONS.pt';torch.save(estimates,pred_path)
        save_json(pred_path.with_suffix('.json'),{'sha256':digest(pred_path),'runtime_scope':'support displacement and known actions only','before_evaluation':True})
        estimates['base']=nominal_z(len(data['qx']))
        estimates['oracle']=data['true_z'].clone()
        raw={}
        for arm,z in estimates.items():
            val=trajectory(data,z,base=(arm=='base'))
            val['z']=z
            qa=data['query_actions'];x=torch.cat([qa/SCALE,torch.ones_like(qa[...,:1])],-1)
            pred=torch.einsum('bij,bqj->bqi',z,x)*SCALE
            val['response_mse']=(pred-data['query_response']).square().mean((-1,-2))
            raw[arm]=val
        base_action=raw['gradient_s3_u0']['first_action']
        prior=raw['gradient_s3_u0']['z']
        for arm,val in raw.items():
            row={'kind':kind,'seed':seed,'arm':arm,'success':float(val['success'].float().mean()),'error':float(val['error'].mean()),'response_mse':float(val['response_mse'].mean()),'action_change':float((val['first_action']-base_action).norm(dim=-1).mean()),'z_change':float((val['z']-prior).flatten(1).norm(dim=-1).mean()),'fallback_domains':int(val['fallback'].any(1).sum()),'out_of_frame':int(val['out_of_frame'].sum())}
            all_rows.append(row)
        torch.save(raw,target/f'{kind}_seed{seed}_RAW.pt')
    save_json(target/'SUMMARY.json',{'split':name,'domains':len(data['qx']),'query_count':data['qx'].shape[1],'rows':all_rows})
    return all_rows

def qualify():
    data=torch.load(ROOT/'qualification.pt',weights_only=False)
    rows=evaluate_dataset('qualification',data,KINDS);gates={}
    for kind in KINDS:
        by={(r['seed'],r['arm']):r for r in rows if r['kind']==kind}
        deltas=[by[s,'gradient_s3_u3']['success']-by[s,'gradient_s3_u0']['success'] for s in SEEDS]
        swaps=[by[s,'gradient_s3_u3']['success']-by[s,'gradient_swapped']['success'] for s in SEEDS]
        errs=[by[s,'gradient_s3_u3']['error']-by[s,'gradient_s3_u0']['error'] for s in SEEDS]
        gates[kind]={'passed':bool(min(deltas)>0 and np.mean(deltas)>=.05 and np.mean(swaps)>=.05 and np.mean(errs)<0),'success_delta':float(np.mean(deltas)),'swap_delta':float(np.mean(swaps)),'error_delta':float(np.mean(errs)),'per_seed_success_delta':deltas}
    save_json(ROOT/'QUALIFICATION.json',{'opened_training_material_only':True,'gates':gates});print(json.dumps(gates),flush=True)

def freeze():
    files=['prove/fast_adapter/structured_response.py','tools/fast_adapter/p19_run.py','tools/fast_adapter/p18_experiment.py','tools/fast_adapter/p18_run.py','prove/fast_adapter/visual_alignment.py']
    files += [str(p) for p in (ROOT/'train').glob('*/*/seed*/model.pt')]
    files += [str(ROOT/'PROTOCOL.json'),str(ROOT/'geometry_readout.pt')]
    save_json(ROOT/'FINAL_FREEZE.json',{'files':{f:digest(f) for f in files},'time':time.time(),'selection':'all last-step checkpoints; input gate only; no model or K selection'})

def fresh(stratum,model_dir,device):
    q=json.loads((ROOT/'QUALIFICATION.json').read_text());kinds=[k for k,g in q['gates'].items() if g['passed']]
    if not kinds:print('No qualified inputs; no final generated');return
    seal=json.loads((ROOT/'FINAL_FREEZE.json').read_text())
    for f,h in seal['files'].items():
        if Path(f).exists():assert digest(f)==h,f
    seed=2026092341+{'id':0,'mild':10000,'strong':20000}[stratum]
    sh,support,goal,qx,qg=physics_data(256,seed,stratum)
    readout=torch.load(ROOT/'geometry_readout.pt',map_location='cpu',weights_only=False)
    feats=torch.zeros(256,6,2,1024)
    if 'dino_geometry' in kinds:
        enc=FrozenDinoROI(model_dir,device)
        style=appearance(seed+4,256)
        for k in range(6):
            feats[:,k]=enc.observe(support[:,k].numpy(),goal.numpy(),style).cpu().half().float()
        del enc
        if torch.cuda.is_available():torch.cuda.empty_cache()
    data=package(sh,support,goal,qx,qg,feats,readout,seed+9)
    save_dataset(data,ROOT/f'final_{stratum}.pt')
    rows=evaluate_dataset(f'final/{stratum}',data,kinds)
    print(json.dumps({'stratum':stratum,'kinds':kinds,'rows':len(rows)}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train','qualification','freeze','fresh']);p.add_argument('--kind');p.add_argument('--family');p.add_argument('--seed',type=int);p.add_argument('--stratum');p.add_argument('--device',default='cuda');p.add_argument('--dino',default=DEFAULT_MODEL);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if a.mode=='prepare':prepare()
    elif a.mode=='train':train(a.kind,a.family,a.seed,a.device)
    elif a.mode=='qualification':qualify()
    elif a.mode=='freeze':freeze()
    else:fresh(a.stratum,a.dino,a.device)
if __name__=='__main__':main()
