"""Post-seal P31 replay: causal decisions, stage eligibility and physical success."""
import argparse
import numpy as np
import torch
from tools.fast_adapter.p31_common import *
from prove.fast_adapter.interaction_stage_adapter import StageState
from prove.fast_adapter.continuous_adapter import feedback
from prove.fast_adapter.object_interaction_env import ObjectEnv, mujoco


def audit(seed):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    out=ROOT/'development'/f'seed{seed}';result=json.loads((out/'RESULT.json').read_text())
    for entry in json.loads((out/'SOURCE_FREEZE.json').read_text())['bindings']:
        assert digest(entry['path'])==entry['sha256'],entry['path']
    nets={n:adapter(seed,n) for n in protocol()['training']['variants']};base=base_model(TASK);obs=observer()
    groups={}
    for row in result['rows']:groups.setdefault(row['source']['path'],[]).append(row)
    maximum_action=maximum_z=maximum_frozen=0.;nphysical=0;diagnostics=[]
    for path,rows in groups.items():
        assert digest(path)==rows[0]['source']['sha256']
        b=torch.load(path,weights_only=True);n=len(rows);arm=rows[0]['arm'];stage='_stage' in arm
        net=nets[arm.removesuffix('_first3')] if arm!='base' else None
        learner=StageState(net,n,stage=stage,mode='first3' if arm.endswith('_first3') else 'gradient') if net is not None else None
        prev=torch.zeros(n,4,device='cuda');mse=[];mse0=[];corrections=[]
        for t in range(200):
            v,p=b['visual'][:,t].cuda(),b['proprio'][:,t].cuda();nv,np_=b['visual'][:,t+1].cuda(),b['proprio'][:,t+1].cuda()
            actual=b['actions'][:,t].cuda()
            with torch.no_grad():ba=base(v,p,prev);active=obs.active(v,p,prev)
            assert torch.equal(active.cpu(),b['active'][:,t])
            action=ba if learner is None else learner.action(v,p,prev,ba,active)
            maximum_action=max(maximum_action,float((action-actual).abs().max()))
            if learner is not None:
                maximum_z=max(maximum_z,float((learner.z-b['latent_before_action'][:,t].cuda()).abs().max()))
                assert torch.equal(learner.counts.cpu(),b['counts_before_action'][:,t])
                if stage:assert torch.equal(action[~active],ba[~active]),'off-stage intervention'
                with torch.no_grad():
                    frozen=net.action(v,p,prev,ba,net.z0[None].expand(n,-1))
                    maximum_frozen=max(maximum_frozen,float((frozen-ba).abs().max()))
                    y,w=feedback(v,p,nv,np_);u=net.encode(v,p,prev)
                    for z,dest in ((learner.z,mse),(net.z0[None].expand(n,-1),mse0)):
                        pred=net.prediction(u,actual,z);dest.append((((pred-y).square()*w).sum(-1)/w.sum(-1).clamp_min(1)).cpu())
                corrections.append((action-ba).cpu())
                learner.observe(v,p,prev,actual,nv,np_,active)
            prev=actual
        if learner is not None:assert torch.equal(learner.counts.cpu(),b['updates'])
        if corrections:
            correction=torch.stack(corrections,1)
            diagnostics.append({'path':path,'arm':arm,'split':rows[0]['split'],
                'prediction_mse_current_z':float(torch.stack(mse).mean()),'prediction_mse_z0':float(torch.stack(mse0).mean()),
                'correction_rms_first20':float(correction[:,:20].square().mean().sqrt()),
                'correction_rms_last20':float(correction[:,-20:].square().mean().sqrt()),
                'final_z_distance_rms':float((b['final_z']-net.z0.cpu()).square().mean().sqrt())})
        for row in rows:
            j=row['source_row'];env=ObjectEnv(row['config'],render=False);env.reset(row['reset_seed']);success=[];contacts=[]
            try:
                for t in range(1,201):
                    for key,states in b['states'].items():getattr(env.env.data,key)[:]=states[j,t].numpy()
                    mujoco.mj_forward(env.env.model,env.env.data)
                    _,m=env.env.evaluate_state(env.env._get_obs(),b['actions'][j,t-1].numpy())
                    success.append(bool(m['success']));contacts.append(bool(m['bilateral_handle']));nphysical+=1
                assert any(success)==row['success'] and success[-1]==row['terminal_success']
                assert any(contacts)==row['ever_bilateral'] and sum(contacts)==row['held_steps']
            finally:env.close()
        print('AUDIT_BLOCK',seed,Path(path).name,flush=True)
    assert maximum_action<1e-5 and maximum_z<1e-5 and maximum_frozen==0
    # Equal bank sizes, common expert subset and identical initialization.
    old=torch.load(ROOT/'data/old.pt',weights_only=True);own=torch.load(ROOT/'data/own.pt',weights_only=True)
    assert len(old['actions'])==len(own['actions'])==128
    assert all(torch.equal(old[k][:64],own[k][:64]) for k in FIELDS)
    sources=[json.loads((ROOT/'models'/f'seed{seed}'/f'{k}_SOURCE_FREEZE.json').read_text()) for k in ('old','own')]
    assert sources[0]['initial_state_hash']==sources[1]['initial_state_hash']
    for freeze in sources:
        for entry in freeze['bindings']:assert digest(entry['path'])==entry['sha256'],entry['path']
    dump(out/'AUDIT.json',{'passed':True,'physical_states_replayed':nphysical,'max_action_error':maximum_action,
         'max_latent_error':maximum_z,'max_z0_base_error':maximum_frozen,'off_stage_residual_zero':True,
         'causal_gate_and_update_counts_replayed':True,'shared_experts_and_equal_budget_checked':True,
         'train_initialization_matched':True,'full_visual_reencoding':False,'diagnostics':diagnostics})
    print('AUDIT_COMPLETE',seed,nphysical,maximum_action,maximum_z,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();audit(a.seed)
