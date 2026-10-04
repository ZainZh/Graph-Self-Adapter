"""Recompute all decisions, causal latent updates and physical success labels."""
import time
import numpy as np
import torch
from tools.fast_adapter.p33_common import *
from tools.fast_adapter.p33_runtime import DynamicsState
from prove.fast_adapter.relational_dynamics import select_action
from prove.fast_adapter.held_hammer_env import HeldHammerEnv
from prove.fast_adapter.object_interaction_env import mujoco

def main():
    torch.set_num_threads(2);out=ROOT/'development';result=json.loads((out/'RESULT.json').read_text())
    freeze=json.loads((out/'SOURCE_FREEZE.json').read_text());checked=[]
    for entry in [freeze]+[json.loads((ROOT/'models'/f'seed{s}'/'SOURCE_FREEZE.json').read_text()) for s in (11,22)]:
        for x in entry['bindings']:
            assert digest(x['path'])==x['sha256'],x['path'];checked.append(x['path'])
    base=base_model(TASK);nets={(s,f):model(s,f) for s in (11,22) for f in ('cross','within')}
    groups={}
    for r in result['rows']:
        if r['source_row'] is not None:groups.setdefault(r['source']['path'],[]).append(r)
    action_error=latent_error=0.;states_count=0;blocks=[];begin=time.time()
    for path,rows in groups.items():
        assert digest(path)==rows[0]['source']['sha256']
        b=torch.load(path,weights_only=True);n=len(b['actions']);arm=rows[0]['arm'];learner=None
        if arm!='base':
            family,mode=arm.split('_',1);net=nets[rows[0]['seed'],family];learner=DynamicsState(net,n,mode)
        previous=torch.zeros(n,4,device='cuda');choice_changes=torch.zeros(n);nonzero=torch.zeros(n)
        for t in range(200):
            ob={k:b[k][:,t].cuda() for k in ('visual','tokens','boxes','proprio')}
            no={k:b[k][:,t+1].cuda() for k in ('visual','tokens','boxes','proprio')}
            actual=b['actions'][:,t].cuda()
            with torch.no_grad():
                ba=base(ob['visual'],ob['proprio'],previous);ba[:,3]=.9
                if learner is None:a=ba
                else:
                    latent_error=max(latent_error,float((learner.z-b['latent_before_action'][:,t].cuda()).abs().max()))
                    assert learner.updates==b['counts_before_action'][t]
                    a,idx,scores=learner.act(ob,ba)
                    assert torch.equal(idx.cpu(),b['candidate_index'][:,t])
                    assert float((scores-b['scores'][:,t].cuda()).abs().max())<1e-4
                    frozen_a,_,_=select_action(net,learner.contexts[-1],ba,net.z0[None].expand(n,-1))
                    choice_changes+=((a-frozen_a).abs().max(-1).values>1e-6).cpu()
            action_error=max(action_error,float((a-actual).abs().max()))
            if learner is not None:
                d=learner.observe(actual,no);stored=b['update_diagnostics'][t]
                assert d['updated']==stored['updated'] and d['source_end']==t+1
                if d['source_start'] is not None:assert d['source_start']==t-3
                nonzero+=(d['delta_norm'].cpu()>1e-10)
            previous=actual
        if learner is not None:
            expected={'fixed':0,'first3':3,'continuous':197}[mode]
            assert b['updates']==learner.updates==expected
            latent_error=max(latent_error,float((learner.z-b['final_z'].cuda()).abs().max()))
        for r in rows:
            j=r['source_row'];e=HeldHammerEnv(r['config'],render=False);e.reset(r['reset_seed']);success=[]
            try:
                for t in range(1,201):
                    for k,s in b['states'].items():getattr(e.env.data,k)[:]=s[j,t].numpy()
                    mujoco.mj_forward(e.env.model,e.env.data)
                    _,m=e.env.evaluate_state(e.env._get_obs(),b['actions'][j,t-1].numpy())
                    success.append(bool(m['success']));states_count+=1
                assert any(success)==r['success'] and success[-1]==r['terminal_success'],(path,j)
                assert next((t+1 for t,v in enumerate(success) if v),None)==r['first_success']
            finally:e.close()
        blocks.append({'path':path,'seed':rows[0]['seed'],'split':rows[0]['split'],'arm':arm,
                       'nonzero_updates_per_condition':nonzero.tolist(),'decisions_changed_vs_same_state_z0':choice_changes.tolist()})
        print('AUDIT',Path(path).name,flush=True)
    assert action_error<1e-4 and latent_error<1e-4
    dump(ROOT/'AUDIT.json',{'passed':True,'physical_states':states_count,'action_max_error':action_error,
        'latent_max_error':latent_error,'source_files_verified':len(set(checked)),
        'runtime_no_expert_targets':True,'causal_windows_verified':True,'full_RGB_reencoding':False,
        'initial_inputs_exactly_paired':result['paired_initial_inputs_identical'],'blocks':blocks,'seconds':time.time()-begin})
    print('AUDIT_COMPLETE',states_count,action_error,latent_error,flush=True)

if __name__=='__main__':main()
