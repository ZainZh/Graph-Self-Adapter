"""P32 sealed candidate comparison, with coherent independent-object histories."""
import time
import numpy as np
import torch
from tools.fast_adapter.p32_common import *
from tools.fast_adapter.p29_common import Visual
from prove.fast_adapter.learned_interaction_loss import LossState
from prove.fast_adapter.object_interaction_env import ObjectEnv, object_config


def transition(record,t):
    previous=record['actions'][:,t-1] if t else torch.zeros_like(record['actions'][:,0])
    return (record['visual'][:,t],record['proprio'][:,t],previous,record['actions'][:,t],
            record['visual'][:,t+1],record['proprio'][:,t+1])


def rollout(configs,resets,visual,base,obs,net=None,mode='gradient',donor=None):
    envs=[ObjectEnv(c) for c in configs];n=len(envs)
    learner=None if net is None else LossState(net,n,mode=mode)
    try:
        p=torch.tensor(np.stack([e.reset(s) for e,s in zip(envs,resets)]),device='cuda')
        rgb=np.stack([e.rgb() for e in envs]);v=visual.encode(rgb);previous=torch.zeros(n,4,device='cuda')
        initial=[e.state() for e in envs];states=[initial];vs=[v.cpu()];ps=[p.cpu()]
        actions=[];latents=[];counts=[];gates=[];probs=[];rgbs=[rgb];seconds=[];diag=[]
        for t in range(200):
            start=time.perf_counter()
            with torch.no_grad():
                ba=base(v,p,previous);prob=obs(v,p,previous).sigmoid();active=(prob>=.5)&(v[:,70]>0)
                if t==0 and net is not None:
                    assert torch.equal(net.action(v,p,previous,ba,net.z0[None].expand(n,-1)),ba)
            a=(ba if learner is None else learner.action(v,p,previous,ba,active)).detach().clone()
            gates.append(active.cpu());probs.append(prob.cpu())
            if learner is not None:latents.append(learner.z.cpu().clone());counts.append(learner.counts.cpu().clone())
            pp=np.stack([e.step(x) for e,x in zip(envs,a.cpu().numpy())])
            rgb=np.stack([e.rgb() for e in envs]);nv=visual.encode(rgb);np_=torch.tensor(pp,device='cuda')
            if learner is not None:
                before=learner.z.clone()
                if donor is None:learner.observe(v,p,previous,a,nv,np_,active)
                else:learner.observe(*transition(donor,t),active,history_active=donor['active'][:,t])
                if not bool(torch.isfinite(learner.z).all()):raise FloatingPointError('Nonfinite runtime z')
                diag.append({**{k:x.cpu() for k,x in learner.last.items()},
                             'delta_norm':(learner.z-before).norm(dim=-1).cpu()})
            actions.append(a.cpu());vs.append(nv.cpu());ps.append(np_.cpu());states.append([e.state() for e in envs])
            if t+1 in (100,200):rgbs.append(rgb)
            seconds.append(time.perf_counter()-start);v,p,previous=nv,np_,a
        b={'visual':torch.stack(vs,1),'proprio':torch.stack(ps,1),'actions':torch.stack(actions,1),
           'active':torch.stack(gates,1),'probability':torch.stack(probs,1),
           'states':{k:torch.tensor(np.stack([[s[k] for s in ss] for ss in states],1)) for k in initial[0]},
           'audit_rgb':torch.tensor(np.stack(rgbs)),'step_seconds':torch.tensor(seconds),
           'reset_seeds':list(map(int,resets)),
           'updates':learner.counts.cpu() if learner is not None else torch.zeros(n,dtype=torch.long)}
        if learner is not None:
            b.update(latent_before_action=torch.stack(latents,1),counts_before_action=torch.stack(counts,1),final_z=learner.z.cpu())
            b['update_diagnostics']={k:torch.stack([d.get(k,torch.zeros(n,dtype=torch.bool) if k=='updated' else torch.full((n,),float('nan'))) for d in diag],1)
                                    for k in ('updated','inner_loss','grad_norm','valid_transitions','delta_norm')}
        return b,[e.metrics for e in envs]
    finally:
        for e in envs:e.close()


def seal(path,b,metrics,*,seed,split,arm,configs,resets,donor_binding=None):
    torch.save(b,path);binding={'path':str(path),'sha256':digest(path)};rows=[]
    for j,ms in enumerate(metrics):
        first=next((t+1 for t,x in enumerate(ms) if x['success']),None)
        rows.append({'task':TASK,'seed':seed,'split':split,'condition':j,'arm':arm,
              'success':first is not None,'terminal_success':bool(ms[-1]['success']),'first_success':first,
              'ever_bilateral':any(x['bilateral_handle'] for x in ms),'held_steps':sum(x['bilateral_handle'] for x in ms),
              'max_lift':max(x['lift'] for x in ms),'updates':int(b['updates'][j]),'eligible_steps':int(b['active'][j].sum()),
              'source':binding,'source_row':j,'config':configs[j],'reset_seed':resets[j]})
    dump(path.with_suffix('.json'),{'trajectory_sealed_before_summary':binding,'rows':rows,'metrics':metrics,'donor_source':donor_binding})
    return rows


def main():
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    cfg=protocol()['evaluation'];out=ROOT/'development';out.mkdir(exist_ok=False)
    assert json.loads((ROOT/'QUALIFICATION.json').read_text())['passed']
    for seed in protocol()['training']['seeds']:
        for b in json.loads((ROOT/'models'/f'seed{seed}'/'SOURCE_FREEZE.json').read_text())['bindings']:
            assert digest(b['path'])==b['sha256'],b['path']
    nets={(seed,fam):adapter(seed,fam) for seed in (11,22) for fam in ('predictive','learned')}
    base=base_model(TASK);obs=observer();visual=Visual(DINO,TASK,torch.load(P29/TASK/'data/PCA.pt',weights_only=True))
    parents={str(k):n for k,n in nets.items()};parents.update(base=base,observer=obs)
    hashes={k:state_hash(n) for k,n in parents.items()}
    models=[ROOT/'models'/f'seed{s}'/f'{f}.pt' for s,f in nets]
    dump(out/'SOURCE_FREEZE.json',{'before_any_rollout':True,'bindings':bindings(source_paths()+models+[Path(__file__)]),
           'weight_hashes':hashes,'physical_episodes':144,'gpu':torch.cuda.get_device_name()})
    begin=time.time();bases={};settings={};rows=[];maximum_initial_visual_difference=0.
    for si,split in enumerate(cfg['strata']):
        configs=[object_config(TASK,cfg['object_seed_origin']+si*1000+i,split) for i in range(8)]
        resets=[cfg['reset_seed_origin']+si*1000+i for i in range(8)];settings[split]=(configs,resets)
        b,ms=rollout(configs,resets,visual,base,obs)
        path=out/f'base_{split}.pt';rows+=seal(path,b,ms,seed=None,split=split,arm='base',configs=configs,resets=resets)
        bases[split]=b;print('BASE_SEALED',split,8,flush=True)
    for seed in (11,22):
        for split in cfg['strata']:
            configs,resets=settings[split];baseline=bases[split]
            mapping=torch.tensor([(i+1)%8 for i in range(8)])
            donor={k:baseline[k][mapping].cuda() for k in ('visual','proprio','actions','active')}
            donor_binding={'path':str(out/f'base_{split}.pt'),'sha256':digest(out/f'base_{split}.pt'),'recipient_to_donor':mapping.tolist()}
            for arm in ('predictive','learned','learned_first3','learned_donor'):
                fam='predictive' if arm=='predictive' else 'learned'
                b,ms=rollout(configs,resets,visual,base,obs,nets[seed,fam],
                      mode='first3' if arm.endswith('first3') else 'gradient',donor=donor if arm.endswith('donor') else None)
                for k in baseline['states']:assert torch.equal(b['states'][k][:,0],baseline['states'][k][:,0]),('unpaired initial state',k)
                maximum_initial_visual_difference=max(maximum_initial_visual_difference,float((b['visual'][:,0]-baseline['visual'][:,0]).abs().max()))
                path=out/f'seed{seed}_{split}_{arm}.pt'
                rows+=seal(path,b,ms,seed=seed,split=split,arm=arm,configs=configs,resets=resets,donor_binding=donor_binding if arm.endswith('donor') else None)
                print('CANDIDATE_SEALED',seed,split,arm,8,'completed',len(rows),flush=True)
    assert len(rows)==144
    assert hashes=={k:state_hash(n) for k,n in parents.items()}
    dump(out/'RESULT.json',{'phase':'development','physical_episodes':len(rows),'seconds':time.time()-begin,
         'frozen_weights_unchanged':True,'max_initial_visual_difference':maximum_initial_visual_difference,
         'base_shared_between_model_seeds':True,'rows':rows})
    print('EVAL_COMPLETE',len(rows),time.time()-begin,flush=True)


if __name__=='__main__':main()
