"""Replay saved public-input decisions and simulator success, after evaluation."""
import argparse
import json
import time
import numpy as np
import torch
from tools.fast_adapter.p30v2_common import *
from prove.fast_adapter.continuous_adapter import feedback
from prove.fast_adapter.continuous_state import ContinuousState
from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco
from prove.fast_adapter.held_hammer_env import HeldHammerEnv


def audit(task,seed):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    out=ROOT/task/'development'/f'seed{seed}';result=json.loads((out/'RESULT.json').read_text())
    freeze=json.loads((out/'SOURCE_FREEZE.json').read_text())
    for x in freeze['bindings']:assert digest(x['path'])==x['sha256'],x['path']
    nets=adapters(task,seed);base=base_model(task)
    grouped={}
    for row in result['rows']:grouped.setdefault(row['source']['path'],[]).append(row)
    maximum_action_error=maximum_latent_error=0.;states_checked=0;diagnostics={};begin=time.time()
    for path,rows in grouped.items():
        assert digest(path)==rows[0]['source']['sha256']
        b=torch.load(path,weights_only=True);n=len(b['actions']);arm=rows[0]['arm']
        learner=None;net=None;prequential=[]
        if arm!='base':
            net,blob=nets['history' if arm=='history' else 'gradient']
            learner=ContinuousState(net,n,mode='fixed' if arm=='fixed_mean' else arm,
                      fixed_z=blob['mean_z'].cuda()[None] if arm=='fixed_mean' else None)
        prev=torch.zeros(n,4,device='cuda')
        for t in range(200):
            v,p=b['visual'][:,t].cuda(),b['proprio'][:,t].cuda()
            nv,np_=b['visual'][:,t+1].cuda(),b['proprio'][:,t+1].cuda()
            actual=b['actions'][:,t].cuda()
            with torch.no_grad():ba=base(v,p,prev)
            if task==TASKS[1]:ba[:,3]=.9
            action=ba if learner is None else learner.action(v,p,prev,ba)
            if task==TASKS[1]:action=action.clone();action[:,3]=.9
            maximum_action_error=max(maximum_action_error,float((action-actual).abs().max()))
            if learner is not None:
                maximum_latent_error=max(maximum_latent_error,float((learner.z-b['latent_before_action'][:,t].cuda()).abs().max()))
                with torch.no_grad():
                    y,w=feedback(v,p,nv,np_);pred=net.prediction(net.encode(v,p,prev),actual,learner.z)
                    prequential.append((((pred-y).square()*w).sum(-1)/w.sum(-1).clamp_min(1)).cpu())
                learner.observe(v,p,prev,actual,nv,np_)
            prev=actual
        expected={'base':0,'fixed_mean':0,'first3':3,'gradient':200,'history':200}[arm]
        assert b['updates']==expected
        if prequential:diagnostics[path]={'next_transition_mse_before_update':float(torch.stack(prequential).mean()),'updates':expected}
        for row in rows:
            j=row['source_row'];cfg=row['config']
            env=HeldHammerEnv(cfg,render=False) if task==TASKS[1] else ObjectEnv(cfg,render=False)
            env.reset(row['reset_seed']);success=[]
            for t in range(1,201):
                for k,ss in b['states'].items():getattr(env.env.data,k)[:]=ss[j,t].numpy()
                mujoco.mj_forward(env.env.model,env.env.data)
                _,m=env.env.evaluate_state(env.env._get_obs(),b['actions'][j,t-1].numpy())
                success.append(bool(m['success']));states_checked+=1
            assert any(success)==row['success'],(path,j,'success')
            assert success[-1]==row['terminal_success'],(path,j,'terminal')
            env.close()
        print('AUDIT_BLOCK',task,seed,Path(path).name,flush=True)
    assert maximum_action_error<1e-4,maximum_action_error
    assert maximum_latent_error<1e-4,maximum_latent_error
    dump(out/'AUDIT.json',{'status':'passed','physical_states_replayed':states_checked,
         'max_action_error':maximum_action_error,'max_latent_error':maximum_latent_error,
         'threshold':1e-4,'public_input_replay_all_steps':True,'success_replay_all_states':True,
         'full_visual_reencoding':False,'diagnostics':diagnostics,'seconds':time.time()-begin})
    print('AUDIT_COMPLETE',task,seed,states_checked,maximum_action_error,maximum_latent_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--seed',required=True,type=int)
    a=p.parse_args();audit(a.task,a.seed)
