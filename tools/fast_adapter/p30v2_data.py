"""P30-v2 training-only banks. Privileged teachers never enter test-time inputs."""
import argparse
import json
import time
import numpy as np
import torch
from tools.fast_adapter.p30v2_common import *
from tools.fast_adapter.p29_common import Visual
from prove.fast_adapter.object_interaction_env import ObjectEnv, object_config, mujoco
from prove.fast_adapter.held_hammer_env import HeldHammerEnv
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL


def prepare_cup():
    task=TASKS[0];out=ROOT/task;out.mkdir(parents=True,exist_ok=True)
    assert not bank_path(task).exists()
    parts=[];bindings=[];changed=0
    for role in ('expert','base'):
        for path in sorted((P29/task/'data/meta').glob(f'{role}_[0-9]*.pt')):
            b=torch.load(path,weights_only=True)
            part={k:b[k].clone() for k in ('visual','proprio','actions','teacher')}
            if role=='base':
                side=path.with_name(path.stem+'_EVALUATOR.json')
                cfgs=json.loads(side.read_text())['configs']
                bindings.append({'path':str(side),'sha256':digest(side)})
                for j,cfg in enumerate(cfgs):
                    env=ObjectEnv(cfg,render=False);env.reset(b['reset_seeds'][j])
                    for t in range(b['actions'].shape[1]):
                        for key in ('qpos','qvel','mocap_pos','mocap_quat'):
                            getattr(env.env.data,key)[:]=b['states'][key][j,t].numpy()
                        mujoco.mj_forward(env.env.model,env.env.data)
                        left,right,*_=env.env.contacts()
                        if left and right:
                            goal=env.env.data.site('grip_site').xpos.copy();goal[2]=env.env.start_z+.18
                            a=np.r_[np.clip((goal-env.env.tcp_center)*15,-1,1),.8]
                            part['teacher'][j,t]=torch.tensor(a,dtype=torch.float32);changed+=1
                    env.close()
            parts.append(part);bindings.append({'path':str(path),'sha256':digest(path)})
    bank={k:torch.cat([x[k] for x in parts]) for k in parts[0]}
    torch.save(bank,bank_path(task))
    dump(out/'DATA_RECEIPT.json',{'status':'training_only_derived','episodes':len(bank['actions']),
        'horizon':bank['actions'].shape[1],'training_objects':64,'held_frames_relabelled':changed,
        'new_physical_episodes':0,'changed_fields':['teacher_on_bilaterally_held_base_frames'],
        'originals_unchanged':True,'source_bindings':bindings,'output_sha256':digest(bank_path(task))})
    print('CUP_BANK_READY',len(bank['actions']),changed,flush=True)


def collect_block(configs,resets,visual,mode,base=None,net=None,online_mode=None,fixed_z=None):
    """Returns raw evidence and separate metrics. Runtime gets public inputs only."""
    from prove.fast_adapter.continuous_state import ContinuousState
    envs=[HeldHammerEnv(c) if c['task']==TASKS[1] else ObjectEnv(c) for c in configs]
    n=len(envs);learner=ContinuousState(net,n,mode=online_mode,fixed_z=fixed_z) if net is not None else None
    try:
        initial_public=[]
        for e,s,c in zip(envs,resets,configs):
            try:initial_public.append(e.reset(int(s)))
            except Exception:
                if isinstance(e,HeldHammerEnv) and e.preparation is not None:
                    key=hashlib.sha256(json.dumps([c,int(s)],sort_keys=True).encode()).hexdigest()[:16]
                    dump(ROOT/'initialization_failures'/f'{key}.json',
                         {'config':c,'reset_seed':int(s),'preparation_steps':e.preparation['steps'],
                          'metrics':e.preparation['metrics'],'not_silently_excluded':True})
                raise
        pp=np.stack(initial_public)
        initial=[e.state() for e in envs]
        prepared=[dict(steps=e.preparation['steps'],passed=e.preparation['passed']) if isinstance(e,HeldHammerEnv) else None for e in envs]
        images=np.stack([e.rgb() for e in envs]);v=visual.encode(images)
        previous=torch.zeros(n,4,device='cuda');p=torch.as_tensor(pp,device='cuda')
        vs=[v.cpu()];ps=[p.cpu()];aa=[];tt=[];zz=[];states=[initial];rgbs=[images];times=[]
        for t in range(200):
            begin=time.perf_counter()
            if mode in ('expert','base_labeled'):
                teacher=np.stack([e.teacher() for e in envs]);tt.append(torch.from_numpy(teacher))
            if mode=='expert':a=torch.as_tensor(teacher,device='cuda')
            else:
                with torch.no_grad():ba=base(v,p,previous)
                if configs[0]['task']==TASKS[1]:ba[:,3]=.9
                a=ba if learner is None else learner.action(v,p,previous,ba)
            a=a.detach().clone()
            if configs[0]['task']==TASKS[1]:a[:,3]=.9
            if learner is not None:zz.append(learner.z.cpu().clone())
            pp=np.stack([e.step(x) for e,x in zip(envs,a.cpu().numpy())])
            images=np.stack([e.rgb() for e in envs]);nv=visual.encode(images)
            np_=torch.as_tensor(pp,device='cuda')
            if learner is not None:learner.observe(v,p,previous,a,nv,np_)
            times.append(time.perf_counter()-begin)
            if learner is not None and not bool(torch.isfinite(learner.z).all()):raise FloatingPointError('nonfinite runtime latent')
            vs.append(nv.cpu());ps.append(np_.cpu());aa.append(a.cpu())
            states.append([e.state() for e in envs])
            if (t+1) in (100,200):rgbs.append(images)
            v,p,previous=nv,np_,a
        b={'visual':torch.stack(vs,1),'proprio':torch.stack(ps,1),'actions':torch.stack(aa,1),
           'reset_seeds':list(map(int,resets)),
           'states':{k:torch.tensor(np.stack([[s[k] for s in ss] for ss in states],1)) for k in initial[0]},
           'audit_rgb':torch.tensor(np.stack(rgbs)), 'step_seconds':torch.tensor(times),
           'updates':learner.updates if learner is not None else 0}
        if tt:b['teacher']=torch.stack(tt,1)
        if zz:b['latent_before_action']=torch.stack(zz,1);b['final_z']=learner.z.cpu()
        return b,[e.metrics for e in envs],prepared
    finally:
        for env in envs:env.close()


def bc(bank,steps,parent=None):
    net=ObjectSkill().cuda()
    if parent is not None:net.load_state_dict(parent)
    b={k:v.cuda() for k,v in bank.items()}
    previous=torch.cat((torch.zeros_like(b['actions'][:,:1]),b['actions'][:,:-1]),1)
    opt=torch.optim.Adam(net.parameters(),lr=.0005)
    for step in range(steps):
        ids=torch.randint(len(b['actions']),(512,),device='cuda');t=torch.randint(200,(512,),device='cuda')
        a=net(b['visual'][ids,t],b['proprio'][ids,t],previous[ids,t]);loss=(a-b['teacher'][ids,t]).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
        if (step+1)%1000==0:print('HELD_BASE_TRAIN',step+1,float(loss.detach()),flush=True)
    return net.eval().requires_grad_(False)


def prepare_hammer(dino):
    torch.set_num_threads(2);torch.manual_seed(3033)
    task=TASKS[1];out=ROOT/task;out.mkdir(parents=True,exist_ok=False)
    raw=out/'data';raw.mkdir()
    cfg=json.loads((ROOT/'PROTOCOL.json').read_text())['training']['hammer_data']
    visual=Visual(dino,task,torch.load(P29/task/'data/PCA.pt',weights_only=True))
    receipts=[]
    def collect(stage,n,split,mode,base,offset):
        parts=[]
        for start in range(0,n,8):
            ids=np.arange(start,min(start+8,n))
            configs=[object_config(task,cfg['geometry_seed_origin']+offset+int(i),split) for i in ids]
            resets=cfg['reset_seed_origin']+offset+ids
            b,m,prep=collect_block(configs,resets,visual,mode,base)
            path=raw/f'{stage}_{start:03d}.pt';torch.save(b,path)
            dump(path.with_suffix('.json'),{'configs':configs,'preparation':prep,'metrics':m,'sha256':digest(path)})
            receipts.append({'path':str(path),'sha256':digest(path)})
            parts.append({k:b[k] for k in ('visual','proprio','actions','teacher')})
            print('HELD_DATA',stage,start,'success',sum(any(x['success'] for x in xs) for xs in m),flush=True)
        return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}
    demos=collect('canonical_expert',cfg['canonical_demos'],'canonical','expert',None,0)
    base=bc(demos,cfg['base_bc_updates'])
    own=collect('canonical_base',cfg['canonical_on_policy'],'canonical','base_labeled',base,1000)
    mixed={k:torch.cat((demos[k],own[k])) for k in demos}
    base=bc(mixed,cfg['base_dagger_updates'],base.state_dict())
    torch.save({'state':{k:v.cpu() for k,v in base.state_dict().items()},'scope':'held hammer canonical only'},base_path(task))
    expert=collect('meta_expert',cfg['adapter_objects'],'train','expert',None,2000)
    # Same object geometries, distinct resets for the frozen-base replay role.
    parts=[]
    for start in range(0,cfg['adapter_objects'],8):
        ids=np.arange(start,start+8)
        configs=[object_config(task,cfg['geometry_seed_origin']+2000+int(i),'train') for i in ids]
        b,m,prep=collect_block(configs,cfg['reset_seed_origin']+3000+ids,visual,'base_labeled',base)
        path=raw/f'meta_base_{start:03d}.pt';torch.save(b,path)
        dump(path.with_suffix('.json'),{'configs':configs,'preparation':prep,'metrics':m,'sha256':digest(path)})
        receipts.append({'path':str(path),'sha256':digest(path)});parts.append(b)
        print('HELD_DATA meta_base',start,'success',sum(any(x['success'] for x in xs) for xs in m),flush=True)
    bank={k:torch.cat([expert[k]]+[b[k] for b in parts]) for k in ('visual','proprio','actions','teacher')}
    torch.save(bank,bank_path(task))
    dump(out/'DATA_RECEIPT.json',{'training_objects':64,'episodes':len(bank['actions']),
          'raw_files':receipts,'output_sha256':digest(bank_path(task)),'base_sha256':digest(base_path(task))})
    print('HAMMER_BANK_READY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cup','hammer']);p.add_argument('--dino',default=DEFAULT_MODEL);a=p.parse_args()
    if a.mode=='cup':prepare_cup()
    else:prepare_hammer(a.dino)
