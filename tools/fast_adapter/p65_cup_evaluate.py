"""Root-reviewed P65 single-cell evaluation; P64 weights and evaluator intact."""
import argparse
import copy
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from prove.fast_adapter.p64_cup import configuration,public_pose
from prove.fast_adapter.p64_cup_policy import CupActionPolicy
from prove.fast_adapter.p65_cup import ARMS,RoutedCupController,StageLocalCupRuntime
from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
from prove.fast_adapter.p60_contour import model_state_sha256
from tools.fast_adapter.p45_common import observe_roles,sha,js
from tools.fast_adapter.p47_evaluate import snapshot,evaluate_sealed


def cells():
    return [dict(split=split,seed=seed,arm=arm) for split in ('id','size','position')
        for seed in range(640100,640104) for arm in ARMS]


def validate(path):
    path=Path(path);freeze=json.loads(path.read_text())
    review=json.loads((path.parent/'ROOT_PHYSICAL_REVIEW.json').read_text())
    if (freeze.get('schema')!='prove.p65-cup-local-routing-freeze.v1' or freeze.get('status')!='frozen'
            or freeze.get('steps')!=240 or freeze.get('cells')!=cells()
            or freeze.get('queue')!=dict(first_index_alone=0,max_workers_after_first=2,cpu_threads=1,automatic_retries=0,
                on_program_error='stop_new_submissions_and_await_running')
            or review.get('approved') is not True or review.get('reviewer')!='root' or review.get('freeze_sha256')!=sha(path)):
        raise ValueError('Exact root-reviewed P65 freeze required')
    required=[__file__,'prove/fast_adapter/p65_cup.py',freeze['encoder'],freeze['action_policy'],freeze['plan']]
    resolved={str(Path(p).resolve()) for p in freeze['files']}
    if any(str(Path(p).resolve()) not in resolved for p in required):raise ValueError('Missing required freeze binding')
    for name,digest in freeze['files'].items():
        if sha(name)!=digest:raise ValueError('Frozen source or asset changed: '+name)
    return freeze


def run(freeze_path,index,output,device='cuda'):
    freeze=validate(freeze_path)
    if not 0<=index<len(freeze['cells']):raise ValueError('Cell index outside frozen cohort')
    cell=freeze['cells'][index];out=Path(output);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    js(out/'PRE_RUN.json',dict(index=index,cell=cell,freeze_sha256=sha(freeze_path),runtime_truth=False,
        steps=240,condition_identity='opened P64 condition; development/regression only'))
    env=RobotGeometrySkill(configuration('cup',cell['seed'],cell['split']))
    try:
        runtime=StageLocalCupRuntime(freeze['encoder'],freeze['dino_model'],cell['arm'],device)
        policy=CupActionPolicy().to(device)
        payload=torch.load(freeze['action_policy'],map_location=device,weights_only=True)
        if payload['schema']!='prove.p64-cup-action.v1':raise ValueError('Wrong action checkpoint')
        policy.load_state_dict(payload['state_dict']);policy.eval().requires_grad_(False)
        controller=RoutedCupController(env.start_pos,env.start_R,policy)
        before={name:model_state_sha256(model) for name,model in [('encoder',runtime.encoder),('action',policy)]}
        trace=[];observations=[];images=[]
        for t in range(240):
            observation,rgb,masks=observe_roles(env);public=public_pose(observation);phase_before=controller.phase
            if not controller.adaptation_frozen:
                runtime.update_phase(public,rgb,masks,t,phase_before,adaptation_frozen=False)
            estimate=runtime.estimate();p,R,closed,stage=controller.command(t,estimate,env.proprio())
            adaptation=copy.deepcopy(runtime.records[-1]) if runtime.records else None
            exit_event=None
            if phase_before=='approach' and controller.phase=='close':exit_event=runtime.end_critical(t)
            route=controller.routing_records[-1]
            row=dict(t=t,state=snapshot(env),command_p=p.copy(),command_R=R.copy(),closed=bool(closed),stage=stage,
                phase_before=phase_before,phase_after=controller.phase,command_mode=route['route'],
                estimate=copy.deepcopy(estimate),adaptation=adaptation,latent_exit=copy.deepcopy(exit_event),
                controls_before=env.data.ctrl.copy(),neural_action=copy.deepcopy(route))
            observations.append(dict(t=t,**public))
            if t%4==0:images.append(dict(t=t,rgb=rgb.copy(),masks_packed=np.packbits(masks,axis=-1)))
            env.step(p,R,closed);row['controls_after']=env.data.ctrl.copy();row['robot_tracking']=env.robot_tracking_diagnostics();trace.append(row)
        terminal=snapshot(env)
        if not runtime.critical_closed:runtime.end_critical(240,reason='episode_budget_end')
        np.save(out/'TRACE.npy',np.asarray(trace,dtype=object),allow_pickle=True)
        np.save(out/'OBS.npy',np.asarray(observations,dtype=object),allow_pickle=True)
        np.save(out/'TERMINAL.npy',terminal,allow_pickle=True)
        np.savez_compressed(out/'PUBLIC_IMAGES.npz',t=np.array([r['t'] for r in images]),
            rgb=np.stack([r['rgb'] for r in images]),masks_packed=np.stack([r['masks_packed'] for r in images]))
        np.save(out/'LATENT_LIFECYCLE.npy',np.asarray(runtime.exit_records,dtype=object),allow_pickle=True)
        after={name:model_state_sha256(model) for name,model in [('encoder',runtime.encoder),('action',policy)]}
        js(out/'SEAL.json',dict(files={n:sha(out/n) for n in ('TRACE.npy','OBS.npy','TERMINAL.npy','PRE_RUN.json','PUBLIC_IMAGES.npz','LATENT_LIFECYCLE.npy')},
            weights_before=before,weights_after=after))
        result=evaluate_sealed(env,trace,terminal,out)
        local_violations=sum(bool(r['U']) and r['stage_schedule']['phase_before']!='approach' for r in runtime.records) if cell['arm']=='online_local' else 0
        if before!=after or local_violations:raise RuntimeError('Frozen weight/stage-update invariant failed')
        result.update(index=index,cell=cell,weights_unchanged=before==after,elapsed_seconds=time.monotonic()-started,
            neural_action_steps=sum(r['command_mode']=='neural' for r in trace),code_action_steps=sum(r['command_mode']=='code' for r in trace),
            update_events=sum(bool(r['U']) for r in runtime.records),local_updates_outside_approach=local_violations,
            local_entry_resets=runtime.local_entries,latent_end_reset=not bool(runtime.z.abs().any()),orientation_ood=False,
            final_phase=controller.phase,adaptation_frozen=controller.adaptation_frozen)
        js(out/'RESULT.json',result);return result
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    preexisting=Path(a.output).exists()
    try:print(json.dumps(run(a.freeze,a.index,a.output,a.device),indent=2))
    except Exception as error:
        if not preexisting:
            out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
            js(out/'PROGRAM_ERROR.json',dict(index=a.index,error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
