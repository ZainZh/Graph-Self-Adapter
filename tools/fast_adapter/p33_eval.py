"""Fixed small development cohort; runtime sees only image features and proprio."""
import time
import numpy as np
import torch
from tools.fast_adapter.p33_common import *
from tools.fast_adapter.p33_visual import RelationVisual
from tools.fast_adapter.p33_runtime import DynamicsState
from prove.fast_adapter.object_interaction_env import object_config
from prove.fast_adapter.held_hammer_env import HeldHammerEnv,PreparationFailed

def collect(configs,resets,visual,base,net=None,mode=None):
    envs=[HeldHammerEnv(c) for c in configs];valid=[];prep=[];initial=[]
    begin=time.time()
    try:
        for j,(e,s) in enumerate(zip(envs,resets)):
            try:
                p=e.reset(s);valid.append(j);initial.append(p)
                prep.append({'passed':True,'steps':e.preparation['steps']})
            except PreparationFailed:
                prep.append({'passed':False,'steps':e.preparation['steps']})
        if not valid:return None,[],prep,valid
        ee=[envs[i] for i in valid];n=len(ee)
        ob=visual.encode(np.stack([e.rgb() for e in ee]));ob['proprio']=torch.tensor(np.stack(initial),device='cuda')
        learner=None if net is None else DynamicsState(net,n,mode)
        previous=torch.zeros(n,4,device='cuda')
        observations=[{k:v.cpu() for k,v in ob.items()}];states=[[e.state() for e in ee]]
        actions=[];latents=[];counts=[];choices=[];scores=[];ds=[];rgbs=[]
        for t in range(200):
            with torch.no_grad():
                ba=base(ob['visual'],ob['proprio'],previous);ba[:,3]=.9
                if learner is None:a=ba;idx=torch.full((n,),13,device='cuda');sc=torch.zeros(n,27,device='cuda')
                else:
                    latents.append(learner.z.cpu().clone());counts.append(learner.updates)
                    a,idx,sc=learner.act(ob,ba)
            a=a.detach().clone();a[:,3]=.9
            pp=np.stack([e.step(x) for e,x in zip(ee,a.cpu().numpy())])
            rgb=np.stack([e.rgb() for e in ee]);no=visual.encode(rgb);no['proprio']=torch.tensor(pp,device='cuda')
            if learner is not None:
                d=learner.observe(a,no)
                ds.append({k:v.cpu() if torch.is_tensor(v) else v for k,v in d.items()})
            actions.append(a.cpu());choices.append(idx.cpu());scores.append(sc.cpu())
            observations.append({k:v.cpu() for k,v in no.items()});states.append([e.state() for e in ee])
            if t in (0,99,199):rgbs.append(rgb)
            ob,previous=no,a
        raw={k:torch.stack([o[k] for o in observations],1) for k in observations[0]}
        raw.update(actions=torch.stack(actions,1),candidate_index=torch.stack(choices,1),scores=torch.stack(scores,1),
                   states={k:torch.tensor(np.stack([[s[k] for s in ss] for ss in states],1)) for k in states[0][0]},
                   audit_rgb=torch.tensor(np.stack(rgbs)),valid_indices=valid,
                   seconds=time.time()-begin,counts_before_action=counts,update_diagnostics=ds)
        if learner is not None:
            raw['latent_before_action']=torch.stack(latents,1);raw['final_z']=learner.z.cpu();raw['updates']=learner.updates
        return raw,[e.metrics for e in ee],prep,valid
    finally:
        for e in envs:e.close()

def main():
    torch.set_num_threads(2);cfg=protocol()['evaluation'];out=ROOT/'development';out.mkdir(exist_ok=False)
    visual=RelationVisual();base=base_model(TASK)
    nets={(s,f):model(s,f) for s in (11,22) for f in ('cross','within')}
    weights={'base':state_hash(base),**{str(k):state_hash(n) for k,n in nets.items()}}
    source=source_paths()+[Path(__file__),Path('tools/fast_adapter/p33_runtime.py'),ROOT/'data/STATS.pt']
    source+=[ROOT/'models'/f'seed{s}'/f'{f}.pt' for s,f in nets]
    dump(out/'SOURCE_FREEZE.json',{'before_new_conditions':True,'weights':weights,'bindings':bindings(source)})
    rows=[];initial={};began=time.time()
    for si,split in enumerate(cfg['strata']):
        configs=[object_config(TASK,cfg['object_seed_origin']+1000*si+i,split) for i in range(8)]
        resets=[cfg['reset_seed_origin']+1000*si+i for i in range(8)]
        jobs=[(None,'base')]+[(s,a) for s in (11,22) for a in cfg['arms'] if a!='base']
        for seed,arm in jobs:
            net=None;mode=None
            if arm!='base':family,mode=arm.split('_',1);net=nets[seed,family]
            raw,metrics,prep,valid=collect(configs,resets,visual,base,net,mode)
            tag=f'{split}_s{seed}_{arm}';p=out/f'{tag}.pt'
            if raw is not None:
                current={k:v[:,0] for k,v in raw['states'].items()};current['visual']=raw['visual'][:,0]
                current['tokens']=raw['tokens'][:,0]
                if split not in initial:initial[split]=(valid,current)
                else:
                    assert valid==initial[split][0]
                    assert all(torch.equal(x,initial[split][1][k]) for k,x in current.items())
                torch.save(raw,p);ref={'path':str(p),'sha256':digest(p)}
            else:ref=None
            new=[]
            for i in range(8):
                j=valid.index(i) if i in valid else None;m=metrics[j] if j is not None else []
                first=next((t+1 for t,x in enumerate(m) if x['success']),None)
                r={'seed':seed,'arm':arm,'split':split,'condition':i,'config':configs[i],'reset_seed':resets[i],
                   'preparation':prep[i],'success':first is not None,'terminal_success':bool(m[-1]['success']) if m else False,
                   'first_success':first,'source':ref,'source_row':j,
                   'head_contact_steps':sum(x['head_target_contact']>0 for x in m),
                   'direct_contact_steps':sum(x['target_direct_contact']>0 for x in m),
                   'max_target_displacement':max((x['target_displacement'] for x in m),default=0),
                   'updates':raw.get('updates',0) if j is not None else 0}
                new.append(r)
            dump(out/f'{tag}.json',{'trajectory_sealed_before_evaluation_summary':ref,'rows':new,'metrics':metrics})
            rows+=new
            print('EVAL',split,seed,arm,sum(x['success'] for x in new),'/8',round(time.time()-began,1),flush=True)
    after={'base':state_hash(base),**{str(k):state_hash(n) for k,n in nets.items()}}
    assert weights==after
    dump(out/'RESULT.json',{'rows':rows,'phase':'development','network_weights_unchanged':True,
                           'paired_initial_inputs_identical':True,'seconds':time.time()-began,'episodes':len(rows)})
    print('EVAL_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
