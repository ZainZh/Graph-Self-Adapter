"""P40 sealed execution followed by a separate physical evaluator.

Development and final use disjoint explicit configurations. This tool refuses
final execution without a source/model/config freeze. No runtime success labels.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.contact_skills import ContactSkill,configuration,STEPS
from prove.fast_adapter.skill_observation import normalize_observation
from prove.fast_adapter.skill_tokens_v02 import (SkillAdapter,SkillControl,MotionEffect,LocalVision,
    observed_motion,observation_coordinates,action_force)
from tools.fast_adapter.p40_train_v03 import ROOT,js,sha,bank
from prove.fast_adapter.skill_visual_reference import DualLocalVision

STRATA=('train','physical','geometry','joint','paired')
FORCES={'cup':(1.5,3.,5.,7.),'hammer':(.25,.45,.7,1.),'insert':(.7,1.5,2.5,4.)}

def configurations(task,phase,split):
    count=8 if phase=='dev' else 16
    start=406000 if phase=='dev' else 409000
    if split!='paired':return [configuration(task,start+100*STRATA.index(split)+i,split) for i in range(count)]
    out=[]
    for i in range(count):
        low=configuration(task,start+500+i,'train');high=copy.deepcopy(low)
        physical=configuration(task,start+500+i,'physical')
        for k in ('mass','friction','resistance','damping'):high[k]=physical[k]
        low.update(pair=i,condition='low',split='paired');high.update(pair=i,condition='high',split='paired')
        out.extend((low,high))
    return out

def load_model(task,kind,seed):
    n=SkillAdapter(task,'local' if kind=='motion' else kind).cuda();p=ROOT/task/'motion'/f'seed{seed}'/'model.pt' if kind=='motion' else ROOT/task/'models_v03'/kind/f'seed{seed}'/'model.pt'
    n.load_state_dict(torch.load(p,weights_only=True));n.eval().requires_grad_(False);return n

def load_head(task,kind='local'):
    p=ROOT/task/('motion/local.pt' if kind=='motion' else 'effects_v03/local.pt')
    n=MotionEffect().cuda();n.load_state_dict(torch.load(p,weights_only=True));return n.eval().requires_grad_(False)

def prepare_means(task):
    out=ROOT/task/'MEANS_v03.pt'
    if out.exists():raise RuntimeError('Mean states already frozen')
    d=bank(task);head=load_head(task)
    with torch.no_grad():d['jacobian']=head(d['local'])
    result={}
    for kind in ('relation','local','fused','history'):
        for seed in (11,22):
            n=load_model(task,kind,seed);z=n.initial(64)
            for t in range(STEPS):z,_=n.update(z,{k:v[:,max(0,t-3):t+1] for k,v in d.items()})
            result[f'{kind}_{seed}']=z.mean(0).detach().cpu()
    torch.save(result,out);js(ROOT/task/'MEANS_v03.json',dict(hash=sha(out),source='Final latent on 64 TRAIN trajectories only; no evaluation trajectories'))

def evaluate_sealed(task,configs,raw_path):
    """Only called after the runtime artifact is written and hashed."""
    raw=torch.load(raw_path,weights_only=True,map_location='cpu');rows=[]
    for i,cfg in enumerate(configs):
        e=ContactSkill(cfg,render=False);tail=[];head_seen=False;start=e.initial_object.copy()
        for t in range(STEPS+1):
            e.data.qpos[:]=raw['qpos'][i,t].numpy();e.data.qvel[:]=raw['qvel'][i,t].numpy();e.mj.mj_forward(e.model,e.data)
            state=e.state();obj=state['object'];pairs={frozenset((int(c.geom1),int(c.geom2))) for c in e.data.contact if c.dist<.001}
            if task=='cup':
                h=int(e.model.geom('part_handle').id)
                contact=all(frozenset((int(e.model.geom(n).id),h)) in pairs for n in ('actor_left','actor_right'))
                progress=float(obj[2]-start[2]);valid=contact and progress>.028
                if t>=STEPS-7:tail.append(valid)
                speed=float(np.linalg.norm(e.data.qvel[:3]))
            elif task=='hammer':
                head_seen|=frozenset((int(e.model.geom('actor_head').id),int(e.model.geom('object_block').id))) in pairs
                progress=float(start[2]-obj[2]);speed=abs(float(e.data.qvel[e.model.joint('object_slide').dofadr[0]]))
            else:
                progress=float(start[2]-obj[2]);speed=abs(float(e.data.qvel[e.model.joint('tool_z').dofadr[0]]))
        visible=not bool(raw['aborted'][i].any())
        if task=='cup':success=all(tail) and len(tail)==8
        elif task=='hammer':success=head_seen and .015<progress<.021 and speed<.008
        else:success=progress>.042 and abs(obj[0]-cfg['offset'])<.0018
        actions=raw['actions'][i,:, -1]
        rows.append(dict(index=i,seed=cfg['seed'],pair=cfg.get('pair'),condition=cfg.get('condition'),success=bool(success and visible),
            progress=progress,speed=speed,observation_valid=visible,head_contact_seen=head_seen if task=='hammer' else None,
            partial_insertion=bool(progress>.025) if task=='insert' else None,force_mean=float(actions.abs().mean()),force_peak=float(actions.abs().max()),force_impulse=float(actions.abs().sum()*.04)))
        e.close()
    return rows

def run(task,phase,split,kind,seed,arm):
    configs=configurations(task,phase,split);count=len(configs)
    if phase=='final':
        freeze=json.loads((ROOT/'FINAL_FREEZE.json').read_text())
        for b in freeze['sources']+freeze['models']:assert sha(b['path'])==b['sha256'],b['path']
        assert configs==freeze['configs'][task][split]
    out=ROOT/task/phase/kind/f'seed{seed}'/split;out.mkdir(parents=True,exist_ok=True)
    raw_path=out/(arm+'_RUNTIME.pt')
    if raw_path.exists():raise RuntimeError(f'Refuse overwriting {raw_path}')
    visual=(kind in ('local','fused','history','motion')) or (kind=='references' and arm=='base')
    data=ROOT/task/('data_v02' if task=='insert' else 'data');vision=None
    if visual:
        pca_path=ROOT/task/'motion/PCA.pt' if kind=='motion' else data/'PCA.pt'
        pca={k:{n:v.cuda() for n,v in p.items()} for k,p in torch.load(pca_path,weights_only=True).items()}
        secondary=None
        if kind=='references' and arm=='base':secondary={k:{n:v.cuda() for n,v in p.items()} for k,p in torch.load(ROOT/task/'motion/PCA.pt',weights_only=True).items()}
        vision=DualLocalVision(pca,secondary)
    head=load_head(task,kind) if visual else None
    n=load_model(task,kind,seed) if kind!='references' else SkillAdapter(task,'relation').cuda().eval().requires_grad_(False)
    z=n.initial(count)
    if arm=='mean':z=torch.load(ROOT/task/('MEANS_motion.pt' if kind=='motion' else 'MEANS_v03.pt'),weights_only=True)[f'{kind}_{seed}'].cuda()[None].repeat(count,1)
    donor=None
    if arm=='donor':
        p=ROOT/task/phase/'references/seed0'/split/'base_RUNTIME.pt';donor=torch.load(p,weights_only=True)
        order=(torch.arange(count)^1) if split=='paired' else torch.arange(count).roll(-1)
        donor={k:v[order].cuda() for k,v in donor['feedback'].items()}
        if kind=='motion':
            donor['local']=donor['motion_local'];donor['df']=donor['motion_df']
        if visual:
            with torch.no_grad():donor['jacobian']=head(donor['local'])
        else:donor['jacobian']=torch.zeros(count,STEPS,16,4,device='cuda')
    envs=[ContactSkill(c) for c in configs];controllers=[SkillControl(task) for _ in configs]
    initial_f=None;initial_secondary=None;secondary_fs=[]
    observed=[];fs=[];raw={k:[] for k in ('qpos','qvel','positions','proprio','visible','actions','latent','parameters','aborted')};hashes=[];mask_hashes=[];feedback=[];losses=[];aborted=np.zeros(count,bool)
    try:
        for t in range(STEPS+1):
            current=[];images=[];masks=[];states=[]
            for i,e in enumerate(envs):
                o,rgb,mask=e.public();o,ok=normalize_observation(task,o);aborted[i]|=not ok
                if not ok:o={k:np.nan_to_num(v) for k,v in o.items()}
                current.append(o);images.append(rgb);masks.append(mask);states.append(e.state())
            for k in ('qpos','qvel'):raw[k].append(np.stack([s[k] for s in states]))
            for k in ('positions','proprio','visible'):raw[k].append(np.stack([o[k] for o in current]))
            raw['aborted'].append(aborted.copy());hashes.append([hashlib.sha256(im.tobytes()).hexdigest() for im in images]);mask_hashes.append([hashlib.sha256(m.tobytes()).hexdigest() for m in masks])
            encoded=vision.encode(images,masks) if visual else {'local':torch.zeros(count,16,device='cuda')}
            f=encoded['local'].float()
            if kind=='motion':
                if initial_f is None:initial_f=f.detach().clone()
                f=f-initial_f
            if 'secondary' in encoded:
                if initial_secondary is None:initial_secondary=encoded['secondary'].detach().clone()
                secondary_fs.append((encoded['secondary']-initial_secondary).detach())
            fs.append(f.detach())
            if t:
                trans=[observed_motion(task,observed[-1][i],current[i]) for i in range(count)]
                w={k:torch.tensor(np.asarray([x[k] for x in trans]),dtype=torch.float32,device='cuda') for k in ('motion','actor_dz','known','relation','relation_df')}
                force=torch.tensor(raw['actions'][-1][:,-1],dtype=torch.float32,device='cuda')
                valid=((w['actor_dz']>.12) if task=='cup' else (force>.03)).float()*torch.tensor(~aborted,device='cuda')
                w.update(force=force,valid=valid,local=fs[-2],df=fs[-1]-fs[-2])
                if secondary_fs:w.update(motion_local=secondary_fs[-2],motion_df=secondary_fs[-1]-secondary_fs[-2])
                with torch.no_grad():w['jacobian']=head(fs[-2]) if visual else torch.zeros(count,16,4,device='cuda')
                feedback.append({k:v.detach() for k,v in w.items()});window={k:torch.stack([x[k] for x in feedback[max(0,t-4):]],1) for k in w}
                if kind!='references' and (arm in ('continuous','donor') or arm=='first12' and t<=12):
                    if donor is not None:window={k:v[:,max(0,t-4):t] for k,v in donor.items()}
                    new_z,loss=n.update(z,window);z=torch.where(torch.tensor(aborted,device='cuda')[:,None],z,new_z)
                    losses.append(None if loss is None else loss.detach().cpu().tolist())
                else:losses.append(None)
            observed.append(current);estimate=n.estimate(z)
            raw['latent'].append(z.detach().cpu().numpy());raw['parameters'].append(estimate.detach().cpu().numpy())
            if t==STEPS:break
            wanted=torch.tensor([min(.6 if task=='hammer' else 1.,max(.08,c.remaining(o)*1000*.20)) for c,o in zip(controllers,current)],device='cuda')
            with torch.no_grad():force=action_force(task,estimate,wanted).cpu().numpy()
            if arm.startswith('fixed'):force[:]=FORCES[task][int(arm[-1])]
            if t<12:force[:]=dict(cup=1.5,hammer=.2,insert=.4)[task]
            actions=[]
            for i,(e,c,o) in enumerate(zip(envs,controllers,current)):
                a=c.action(o,t,force[i]) if not aborted[i] else np.zeros(3 if task=='cup' else 2)
                actions.append(a)
                if not aborted[i]:e.step(a)
            raw['actions'].append(np.stack(actions))
        payload={k:torch.from_numpy(np.stack(v,1)) for k,v in raw.items()}
        payload.update(feedback={k:torch.stack([x[k] for x in feedback],1).detach().cpu() for k in feedback[0]},rgb_hashes=hashes,mask_hashes=mask_hashes,losses=losses)
        torch.save(payload,raw_path)
        js(out/(arm+'_SEAL.json'),dict(time=time.time(),raw=sha(raw_path),task=task,phase=phase,split=split,kind=kind,seed=seed,arm=arm,configs=configs,role='Runtime sealed before evaluator reads true object/contact state'))
    finally:
        for e in envs:e.close()
    rows=evaluate_sealed(task,configs,raw_path)
    result=dict(task=task,phase=phase,split=split,kind=kind,seed=seed,arm=arm,rows=rows,successes=sum(x['success'] for x in rows),total=count,raw_sha256=sha(raw_path))
    js(out/(arm+'_RESULT.json'),result);print({k:v for k,v in result.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--phase',choices=['dev','final'],default='dev');p.add_argument('--split',choices=STRATA,default='train');p.add_argument('--kind',default='relation');p.add_argument('--seed',type=int,default=11);p.add_argument('--arm',default='continuous');p.add_argument('--means',action='store_true');a=p.parse_args();torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    if a.means:prepare_means(a.task)
    else:run(a.task,a.phase,a.split,a.kind,a.seed,a.arm)
