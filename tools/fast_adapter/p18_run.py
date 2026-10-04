"""P18 real-DINO cached training and RGB feedback evaluation.

Caches contain only actually rendered/encoded observations, never interpolated
features. Privileged physics/targets are consumed only by training supervision,
the simulator, scoring and the explicitly labeled oracle.
"""
from __future__ import annotations
import argparse, hashlib, json, time, os
from pathlib import Path
from dataclasses import asdict, replace
import numpy as np
import torch
from tools.fast_adapter.p18_experiment import *
from prove.fast_adapter.visual_alignment import FrozenDinoROI, DEFAULT_MODEL, appearance, visible, render_batch

ROOT = Path('evaluations/fast_adapter_p18_visual_v01')
FAMILIES = ('gradient_z','history','hybrid')
SEEDS = (11,22,33,44,55)
STRATA = ('id','mild','strong')


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def model(family):
    cfg = P18ModelConfig(residual_width=512 if family=='history' else 256)
    return {'gradient_z':build_gradient_system,'history':build_history_system,'hybrid':build_hybrid_system}[family](cfg)


def pack(feat, prev, base=None):
    empty = feat.new_zeros((feat.shape[0],0))
    if base is None: base = feat.new_zeros((feat.shape[0],2))
    return P18FeatureBatch(feat[:,0],feat[:,1],empty,prev,empty,base)


def records_from_features(system, feats):
    acts=bounded_axis_probe_actions(device=feats.device)
    u=[]
    for k in range(feats.shape[1]):
        prev=torch.zeros(feats.shape[0],2,device=feats.device) if k==0 else acts[k-1].expand(feats.shape[0],-1)
        u.append(system.project(pack(feats[:,k],prev)))
    return [P18TransitionRecord(u[k],acts[k].expand(feats.shape[0],-1),u[k+1]) for k in range(len(u)-1)]


def physics_data(n, seed, stratum, query_count=4, train=False):
    shift=sample_p18_shifts(n,seed,stratum)
    x,g=sample_p18_positions(n,seed+1)
    support=[x]
    for a in bounded_axis_probe_actions(): support.append(step_p18_shift(support[-1],shift,a.expand(n,-1)))
    qx,qg=sample_p18_positions(n*query_count,seed+2)
    qx=qx.reshape(n,query_count,2); qg=qg.reshape(n,query_count,2)
    if train:
        rng=torch.Generator().manual_seed(seed+3)
        qx[:,2:]=qg[:,2:]+torch.rand((n,query_count-2,2),generator=rng)*.2-.1
    return shift,torch.stack(support,1),g,qx,qg


def build_bank(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    n=args.domains; start=time.time()
    shift,support,g,qx,qg=physics_data(n,args.data_seed,'train',train=True)
    style=appearance(args.data_seed+4,n)
    qstyle=appearance(args.data_seed+5,n*4).reshape(n,4,8)
    encoder=FrozenDinoROI(args.dino, args.device)
    support_features=torch.empty(n,6,2,1024,dtype=torch.float16)
    query_features=torch.empty(n,4,2,1024,dtype=torch.float16)
    for begin in range(0,n,64):
        end=min(begin+64,n); sl=slice(begin,end)
        for k in range(6): support_features[sl,k]=encoder.observe(support[sl,k].numpy(),g[sl].numpy(),style[sl]).cpu().half()
        for q in range(4): query_features[sl,q]=encoder.observe(qx[sl,q].numpy(),qg[sl,q].numpy(),qstyle[sl,q]).cpu().half()
        if begin%1024==0: print(json.dumps({'bank_done':end,'total':n,'seconds':time.time()-start}),flush=True)
    bank={'support':support_features,'query':query_features,'query_base':.5*(qg-qx),'query_oracle':torch.stack([projected_oracle_action(qx[:,q],qg[:,q],shift) for q in range(4)],1),'seed':args.data_seed,'n':n}
    torch.save(bank,out/'bank.pt')
    save_json(out/'BANK.json',{'seed':args.data_seed,'domains':n,'seconds':time.time()-start,'sha256':digest(out/'bank.pt'),'images':n*10,'source':'actual frozen DINOv3 RGB ROI features; float16 cache; no interpolation'})


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device=args.device; system=model(args.family).to(device)
    bank=torch.load(args.bank,map_location='cpu',weights_only=False)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.Adam(system.parameters(),lr=3e-4)
    gen=torch.Generator().manual_seed(400000+args.seed)
    start=time.time(); logs=[]
    for step in range(args.steps):
        ids=torch.randint(bank['n'],(args.batch,),generator=gen)
        feats=bank['support'][ids,:4].float().to(device)
        records=records_from_features(system,feats)
        z,_=prepare_latent_for_arm(system,args.family,records,batch_size=args.batch,k=3,create_graph=True)
        # Four independent states, including two near-goal states, share domain z.
        qfeat=bank['query'][ids].flatten(0,1).float().to(device)
        qbase=bank['query_base'][ids].flatten(0,1).to(device)
        target=bank['query_oracle'][ids].flatten(0,1).to(device)
        query=pack(qfeat,torch.zeros_like(qbase),qbase)
        loss,diag=action_imitation_loss(system,query,z.repeat_interleave(4,0),target)
        if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite train loss step {step}')
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(system.parameters(),10.)
        opt.step()
        if (step+1)%50==0 or step==0:
            row={'step':step+1,'loss':float(loss.detach()),'grad_norm':float(norm),'alpha':float(system.inner_alpha.detach()),'seconds':time.time()-start,**diag}
            logs.append(row);print(json.dumps(row),flush=True)
    torch.save({'state':system.state_dict(),'cfg':asdict(system.cfg),'family':args.family,'seed':args.seed,'steps':args.steps,'bank_sha256':digest(args.bank)},out/'model.pt')
    save_json(out/'TRAIN.json',{'seconds':time.time()-start,'parameters':trainable_parameter_groups(system),'logs':logs,'seed':args.seed,'family':args.family,'checkpoint_sha256':digest(out/'model.pt')})


def rollout(encoder, system, z, shift, qx, qg, style, arm, *, zero_visual=False, retain=False):
    device=encoder.device
    x=qx.clone(); g=qg.clone(); n=x.shape[0]
    prev=torch.zeros(n,2,device=device); failed=~visible(x.numpy()); trajectory=[]
    first_action=None
    with torch.no_grad():
        for t in range(12):
            base=nominal_base_action_from_roi(x,g)
            if arm=='base': action=base
            elif arm=='oracle': action=projected_oracle_action(x,g,shift)
            else:
                features=encoder.observe(x.numpy(),g.numpy(),style)
                if zero_visual: features=torch.zeros_like(features)
                action,_,_=system.act(pack(features,prev,base.to(device)),z)
                action=action.cpu()
            action=action.clamp(-1,1)
            if first_action is None: first_action=action.clone()
            if retain: trajectory.append({'state':x.tolist(),'action':action.tolist()})
            x=step_p18_shift(x,shift,action)
            prev=action.to(device);failed|=~visible(x.numpy())
        error=torch.linalg.vector_norm(x-g,dim=-1)
    success=(error<=.05)&~torch.from_numpy(failed)
    return {'success':success,'error':error,'out_of_frame':torch.from_numpy(failed),'first_action':first_action,'trajectory':trajectory,'final_state':x}


def evaluate(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    system=model(checkpoint['family']).to(args.device);system.load_state_dict(checkpoint['state']);system.freeze_for_eval()
    family=checkpoint['family']; seed=checkpoint['seed']
    encoder=FrozenDinoROI(args.dino,args.device); start=time.time()
    frozen={k:v.detach().clone() for k,v in system.state_dict().items()}
    rows=[]; details=[]
    n=args.domains; q=4
    for si,stratum in enumerate(STRATA):
        stream=args.data_seed+si*10000
        shift,support,g,qx,qg=physics_data(n,stream,stratum)
        style=appearance(stream+4,n);qstyle=appearance(stream+5,n*q)
        features=torch.stack([encoder.observe(support[:,k].numpy(),g.numpy(),style) for k in range(6)],1)
        with torch.no_grad(): records=records_from_features(system,features)
        # Independent diagnostic transition, never supplied to adaptation.
        cx,cg=sample_p18_positions(n,stream+6);ca=torch.tensor([.13,-.17]).expand(n,-1)
        cn=step_p18_shift(cx,shift,ca);cs=appearance(stream+7,n)
        cf0=encoder.observe(cx.numpy(),cg.numpy(),cs);cf1=encoder.observe(cn.numpy(),cg.numpy(),cs)
        with torch.no_grad():
            calibration=[P18TransitionRecord(system.project(pack(cf0,torch.zeros_like(ca).to(args.device))),ca.to(args.device),system.project(pack(cf1,ca.to(args.device))))]
        expanded=P18HiddenShift(shift.M.repeat_interleave(q,0),shift.b.repeat_interleave(q,0))
        qxx=qx.flatten(0,1);qgg=qg.flatten(0,1)
        variants=[]
        for k in K_VALUES:
            with torch.enable_grad():
                z,diag=prepare_latent_for_arm(system,family,records,batch_size=n,k=k,create_graph=False,detach_initial=True)
            z=z.detach()
            with torch.no_grad():
                init=system.initial_latent(records[:k],batch_size=n,device=z.device)
                loss=float(prediction_terms(system,records[:k],z,init,system.cfg)['mse'].mean()) if system.dynamics and k else None
            variants.append((family,k,z,False,diag,loss,float(z.sub(init).norm(dim=-1).mean())))
            if k==3:
                variants.append((family+'_swapped',k,z.roll(1,0),False,[],None,None))
                if family=='hybrid': variants.append(('hybrid_no_refinement',k,init.detach(),False,[],None,0.))
                # Replace all support RGB features; actions stay fixed. Query visuals intact.
                with torch.no_grad(): zero_records=records_from_features(system,torch.zeros_like(features))
                zz,_=prepare_latent_for_arm(system,family,zero_records,batch_size=n,k=k,create_graph=False,detach_initial=True)
                variants.append((family+'_zero_support_visual',k,zz.detach(),False,[],None,None))
        if family=='gradient_z':
            variants.extend([('base',0,None,False,[],None,None),('oracle',0,None,False,[],None,None)])
        for arm,k,z,zero,diag,loss,delta in variants:
            with torch.no_grad():
                calibration_loss=float(prediction_terms(system,calibration,z,z,system.cfg)['mse'].mean()) if system.dynamics is not None and z is not None else None
                distance_z0=float((z-system.z0).norm(dim=-1).mean()) if z is not None else None
            res=rollout(encoder,system,None if z is None else z.repeat_interleave(q,0),expanded,qxx,qgg,qstyle,arm,zero_visual=zero)
            for d in range(n):
                sl=slice(d*q,(d+1)*q)
                rows.append({'seed':seed,'family':family,'arm':arm,'k':k,'stratum':stratum,'domain':d,'query_success':res['success'][sl].tolist(),'query_error':res['error'][sl].tolist(),'out_of_frame':res['out_of_frame'][sl].tolist()})
            details.append({'arm':arm,'k':k,'stratum':stratum,'success':float(res['success'].float().mean()),'error':float(res['error'].mean()),'support_prediction_mse':loss,'calibration_prediction_mse':calibration_loss,'z_change':delta,'z_distance_from_z0':distance_z0,'adaptation':diag,'seconds':time.time()-start})
            print(json.dumps(details[-1]),flush=True)
            # Durable progress. Partial files never count as a completed evaluation.
            with open(out/'raw.partial.jsonl','w') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
    for name,value in system.state_dict().items():
        if not torch.equal(value,frozen[name]):raise AssertionError(f'evaluation mutated {name}')
    os.replace(out/'raw.partial.jsonl',out/'raw.jsonl')
    save_json(out/'EVAL.json',{'family':family,'seed':seed,'data_seed':args.data_seed,'domains_per_stratum':n,'query_resets_per_domain':q,'details':details,'seconds':time.time()-start,'all_model_parameters_unchanged':True,'checkpoint_sha256':digest(args.checkpoint),'raw_sha256':digest(out/'raw.jsonl')})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['bank','train','eval']);p.add_argument('--output',required=True)
    p.add_argument('--dino',default=DEFAULT_MODEL);p.add_argument('--device',default='cuda');p.add_argument('--domains',type=int,default=128);p.add_argument('--data-seed',type=int,default=2026092231)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--family',choices=FAMILIES,default='gradient_z');p.add_argument('--steps',type=int,default=1500);p.add_argument('--batch',type=int,default=32);p.add_argument('--bank');p.add_argument('--checkpoint')
    args=p.parse_args();torch.set_num_threads(2)
    if torch.cuda.is_available(): torch.backends.cuda.matmul.allow_tf32=True
    {'bank':build_bank,'train':train,'eval':evaluate}[args.command](args)

if __name__=='__main__': main()
