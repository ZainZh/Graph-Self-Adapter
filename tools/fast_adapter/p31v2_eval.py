"""Frozen P31 candidates; runtime reads only RGB, proprioception, previous action."""
import argparse
import time
import numpy as np
import torch
from tools.fast_adapter.p31v2_common import *
from tools.fast_adapter.p29_common import Visual
from prove.fast_adapter.interaction_stage_adapter import StageState
from prove.fast_adapter.object_interaction_env import ObjectEnv, object_config


def rollout(configs, resets, visual, base, obs, net, *, stage=False, mode='gradient'):
    envs=[ObjectEnv(c) for c in configs];n=len(envs)
    learner=StageState(net,n,stage=stage,mode=mode) if net is not None else None
    try:
        p=torch.tensor(np.stack([e.reset(s) for e,s in zip(envs,resets)]),device='cuda')
        rgb=np.stack([e.rgb() for e in envs]);v=visual.encode(rgb);previous=torch.zeros(n,4,device='cuda')
        initial=[e.state() for e in envs];states=[initial];vs=[v.cpu()];ps=[p.cpu()]
        actions=[];latents=[];gates=[];probs=[];rgbs=[rgb];counts=[];seconds=[]
        for t in range(200):
            start=time.perf_counter()
            with torch.no_grad():
                ba=base(v,p,previous);prob=obs(v,p,previous).sigmoid();active=(prob>=.5)&(v[:,70]>0)
                if t==0 and net is not None:
                    frozen=net.action(v,p,previous,ba,net.z0[None].expand(n,-1))
                    assert torch.equal(frozen,ba),'z0 must reproduce base exactly'
            a=ba if learner is None else learner.action(v,p,previous,ba,active)
            a=a.detach().clone();gates.append(active.cpu());probs.append(prob.cpu())
            if learner is not None:latents.append(learner.z.cpu().clone());counts.append(learner.counts.cpu().clone())
            pp=np.stack([e.step(x) for e,x in zip(envs,a.cpu().numpy())])
            rgb=np.stack([e.rgb() for e in envs]);nv=visual.encode(rgb);np_=torch.tensor(pp,device='cuda')
            if learner is not None:
                learner.observe(v,p,previous,a,nv,np_,active)
                if not bool(torch.isfinite(learner.z).all()):raise FloatingPointError('Runtime latent nonfinite')
            vs.append(nv.cpu());ps.append(np_.cpu());actions.append(a.cpu());states.append([e.state() for e in envs])
            if t+1 in (100,200):rgbs.append(rgb)
            seconds.append(time.perf_counter()-start);v,p,previous=nv,np_,a
        b={'visual':torch.stack(vs,1),'proprio':torch.stack(ps,1),'actions':torch.stack(actions,1),
           'active':torch.stack(gates,1),'probability':torch.stack(probs,1),'reset_seeds':list(map(int,resets)),
           'states':{k:torch.tensor(np.stack([[s[k] for s in ss] for ss in states],1)) for k in initial[0]},
           'audit_rgb':torch.tensor(np.stack(rgbs)),'step_seconds':torch.tensor(seconds),
           'updates':learner.counts.cpu() if learner is not None else torch.zeros(n,dtype=torch.long)}
        if learner is not None:
            b.update(latent_before_action=torch.stack(latents,1),counts_before_action=torch.stack(counts,1),final_z=learner.z.cpu())
        return b,[e.metrics for e in envs]
    finally:
        for e in envs:e.close()


def evaluate(seed,dino):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    assert json.loads((ROOT/'OBSERVER_QUALIFICATION.json').read_text())['passed']
    cfg=protocol()['evaluation'];out=ROOT/'development'/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    nets={name:adapter(seed,name) for name in protocol()['training']['variants']};base=base_model(TASK);obs=observer()
    visual=Visual(dino,TASK,torch.load(P29/TASK/'data/PCA.pt',weights_only=True))
    freeze=source_paths()+[ROOT/'observer.pt']+[ROOT/'models'/f'seed{seed}'/f'{n}.pt' for n in nets]
    dump(out/'SOURCE_FREEZE.json',{'before_evaluation':True,'bindings':bindings(freeze),
         'cuda_device':torch.cuda.get_device_name(),'seed':seed,'torch':torch.__version__})
    before={n:state_hash(net) for n,net in dict(nets,base=base,observer=obs).items()}
    begin=time.time();rows=[]
    for si,split in enumerate(cfg['strata']):
        for offset in range(0,cfg['conditions_each'],8):
            ids=list(range(offset,min(offset+8,cfg['conditions_each'])))
            configs=[object_config(TASK,cfg['object_seed_origin']+si*1000+i,split) for i in ids]
            resets=[cfg['reset_seed_origin']+si*1000+i for i in ids];initial=None
            for arm in cfg['arms']:
                variant=arm.removesuffix('_first3');net=None if arm=='base' else nets[variant]
                stage='_stage' in arm;mode='first3' if arm.endswith('_first3') else 'gradient'
                b,metrics=rollout(configs,resets,visual,base,obs,net,stage=stage,mode=mode)
                current={k:v[:,0] for k,v in b['states'].items()}
                if initial is None:initial=current
                else:
                    for k in initial:assert torch.equal(initial[k],current[k]),('initial pairing',k)
                path=out/f'{split}_{offset:03d}_{arm}.pt';torch.save(b,path)
                binding={'path':str(path),'sha256':digest(path)};new=[]
                for j,i in enumerate(ids):
                    ms=metrics[j];first=next((t+1 for t,x in enumerate(ms) if x['success']),None)
                    new.append({'task':TASK,'seed':seed,'split':split,'condition':i,'arm':arm,
                         'success':first is not None,'terminal_success':bool(ms[-1]['success']),'first_success':first,
                         'ever_bilateral':any(x['bilateral_handle'] for x in ms),'held_steps':sum(x['bilateral_handle'] for x in ms),
                         'max_lift':max(x['lift'] for x in ms),'updates':int(b['updates'][j]),
                         'eligible_steps':int(b['active'][j].sum()),'source':binding,'source_row':j,
                         'config':configs[j],'reset_seed':resets[j]})
                dump(path.with_suffix('.json'),{'trajectory_sealed_before_summary':binding,'rows':new,'metrics':metrics})
                rows+=new
                print('EVAL',seed,split,offset,arm,sum(r['success'] for r in new),'/',len(new),flush=True)
    after={n:state_hash(net) for n,net in dict(nets,base=base,observer=obs).items()}
    assert before==after
    dump(out/'RESULT.json',{'phase':'development','rows':rows,'episodes':len(rows),'seconds':time.time()-begin,
         'frozen_weights_unchanged':True,'weight_hashes':before,'gpu':torch.cuda.get_device_name()})
    print('EVAL_COMPLETE',seed,len(rows),time.time()-begin,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--dino',default='assets/dinov3');a=p.parse_args()
    evaluate(a.seed,a.dino)
