"""P66 root-reviewed cell; no automatic launch and unchanged physical scoring."""
import os
os.environ['MUJOCO_GL']='egl'
os.environ['PYOPENGL_PLATFORM']='egl'
os.environ['MUJOCO_EGL_DEVICE_ID']='0'
os.environ.setdefault('EGL_DEVICE_ID','0')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
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
from prove.fast_adapter.p65_cup import RoutedCupController
from prove.fast_adapter.p66_cup import ARMS,ReferenceCupRuntime
from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
from prove.fast_adapter.p60_contour import model_state_sha256
from tools.fast_adapter.p45_common import observe_roles,sha,js
from tools.fast_adapter.p47_evaluate import snapshot,evaluate_sealed

QUEUE=dict(first_index_alone=0,max_workers_after_first=2,cpu_threads=1,automatic_retries=0,
    on_program_error='stop_new_submissions_and_await_running')


def cells():
    return [dict(split=s,seed=seed,arm=arm) for s in ('id','size','position') for seed in range(640100,640104) for arm in ARMS]


def validate(path):
    path=Path(path);f=json.loads(path.read_text());r=json.loads((path.parent/'ROOT_PHYSICAL_REVIEW.json').read_text())
    if (f.get('schema')!='prove.p66-cup-reference-freeze.v1' or f.get('status')!='frozen' or f.get('steps')!=240
            or f.get('cells')!=cells() or f.get('queue')!=QUEUE or r.get('approved') is not True
            or r.get('reviewer')!='root' or r.get('freeze_sha256')!=sha(path)):raise ValueError('Exact root P66 review required')
    required=[__file__,'prove/fast_adapter/p66_cup.py','prove/fast_adapter/p65_cup.py',f['encoder'],f['action_policy'],f['plan'],f['qualification']]
    resolved={str(Path(n).resolve()) for n in f['files']}
    if any(str(Path(p).resolve()) not in resolved for p in required):raise ValueError('Missing required frozen file')
    for name,digest in f['files'].items():
        if sha(name)!=digest:raise ValueError('Changed frozen source/asset: '+name)
    return f


def run(freeze_path,index,output,device='cuda'):
    f=validate(freeze_path)
    if not 0<=index<48:raise ValueError('Cell outside frozen48')
    cell=f['cells'][index];out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    js(out/'PRE_RUN.json',dict(index=index,cell=cell,freeze_sha256=sha(freeze_path),runtime_truth=False,steps=240,
        condition_identity='opened P64/P65 development condition'))
    env=None
    try:
        env=RobotGeometrySkill(configuration('cup',cell['seed'],cell['split']))
        runtime=ReferenceCupRuntime(f['encoder'],f['dino_model'],cell['arm'],device)
        policy=CupActionPolicy().to(device);payload=torch.load(f['action_policy'],map_location=device,weights_only=True)
        if payload['schema']!='prove.p64-cup-action.v1':raise ValueError('Wrong action checkpoint')
        policy.load_state_dict(payload['state_dict']);policy.eval().requires_grad_(False)
        controller=RoutedCupController(env.start_pos,env.start_R,policy)
        before={n:model_state_sha256(m) for n,m in [('encoder',runtime.encoder),('action',policy)]}
        trace=[];observations=[];images=[]
        for t in range(240):
            observation,rgb,masks=observe_roles(env);public=public_pose(observation);phase=controller.phase
            if not controller.adaptation_frozen:runtime.update_phase(public,rgb,masks,t,phase)
            estimate=runtime.estimate();p,R,closed,stage=controller.command(t,estimate,env.proprio())
            adaptation=copy.deepcopy(runtime.p66_records[-1]) if runtime.p66_records else None
            exit_event=runtime.end_critical(t) if phase=='approach' and controller.phase=='close' else None
            route=controller.routing_records[-1]
            row=dict(t=t,state=snapshot(env),command_p=p.copy(),command_R=R.copy(),closed=bool(closed),stage=stage,
                phase_before=phase,phase_after=controller.phase,command_mode=route['route'],estimate=copy.deepcopy(estimate),
                adaptation=adaptation,latent_exit=copy.deepcopy(exit_event),controls_before=env.data.ctrl.copy(),neural_action=copy.deepcopy(route))
            observations.append(dict(t=t,**public))
            if t%4==0:images.append(dict(t=t,rgb=rgb.copy(),masks_packed=np.packbits(masks,axis=-1)))
            env.step(p,R,closed);row['controls_after']=env.data.ctrl.copy();row['robot_tracking']=env.robot_tracking_diagnostics();trace.append(row)
        terminal=snapshot(env)
        if not runtime.reference.ended:runtime.end_critical(240,'episode_budget_end')
        np.save(out/'TRACE.npy',np.asarray(trace,dtype=object),allow_pickle=True)
        np.save(out/'OBS.npy',np.asarray(observations,dtype=object),allow_pickle=True)
        np.save(out/'TERMINAL.npy',terminal,allow_pickle=True)
        np.save(out/'LATENT_LIFECYCLE.npy',np.asarray(runtime.reference.exit_records,dtype=object),allow_pickle=True)
        np.savez_compressed(out/'PUBLIC_IMAGES.npz',t=np.array([r['t'] for r in images]),rgb=np.stack([r['rgb'] for r in images]),masks_packed=np.stack([r['masks_packed'] for r in images]))
        after={n:model_state_sha256(m) for n,m in [('encoder',runtime.encoder),('action',policy)]}
        js(out/'SEAL.json',dict(files={n:sha(out/n) for n in ('TRACE.npy','OBS.npy','TERMINAL.npy','PRE_RUN.json','PUBLIC_IMAGES.npz','LATENT_LIFECYCLE.npy')},weights_before=before,weights_after=after))
        result=evaluate_sealed(env,trace,terminal,out);records=runtime.p66_records
        if before!=after:raise RuntimeError('Frozen weights changed')
        result.update(index=index,cell=cell,weights_unchanged=True,elapsed_seconds=time.monotonic()-started,
            neural_action_steps=sum(r['command_mode']=='neural' for r in trace),code_action_steps=sum(r['command_mode']=='code' for r in trace),
            update_events=sum(bool(r['U']) for r in records),entry_events=sum(r['event']=='entry' for r in records),
            continuous_events=sum(r['event']=='continuous' for r in records),reference_time=runtime.reference.ref_time,
            reference_missing_frames=sum(r['qualification_reason']=='no_admitted_historical_reference_hold' for r in records),
            latent_end_reset=not bool(runtime.reference.z.abs().any()),orientation_ood=False,
            final_phase=controller.phase,adaptation_frozen=controller.adaptation_frozen)
        js(out/'RESULT.json',result);return result
    finally:
        if env is not None:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();preexisting=Path(a.output).exists()
    try:print(json.dumps(run(a.freeze,a.index,a.output,a.device),indent=2))
    except Exception as error:
        if not preexisting:
            out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
            js(out/'PROGRAM_ERROR.json',dict(index=a.index,error_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
