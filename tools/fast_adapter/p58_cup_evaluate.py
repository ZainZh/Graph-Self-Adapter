#!/usr/bin/env python3
"""Frozen 32-cell P58 Panda cup development runner."""
from __future__ import annotations

import argparse,copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

from prove.fast_adapter.p58_cup_runtime import load_runtime
from prove.fast_adapter.p47_neural import P47Latent
from prove.fast_adapter.p45_neural import load_model as load_p47_model
from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
from tools.fast_adapter.p45_common import js,make_controller,make_env,observe_roles,sha
from tools.fast_adapter.p47_evaluate import evaluate_sealed,parameters_sha,snapshot


SCHEMA='prove.p58-cup-development-evaluation.v1'
NEW_REPS=('position_only','functional_frame');NEW_ARMS=('fixed','online','direct')
CONDITIONS=tuple(range(455406,455410));STEPS=240
SOURCES=(
    'prove/fast_adapter/p58_geometry_model.py',
    'prove/fast_adapter/p58_cup_functional.py',
    'prove/fast_adapter/p58_cup_runtime.py',
    'prove/task_graph/p57_public_geometry.py',
    'prove/fast_adapter/geometry3d_env.py',
    'prove/fast_adapter/geometry3d_adapter.py',
    'prove/fast_adapter/neural_geometry_adapter.py',
    'prove/fast_adapter/neural_geometry_visual.py',
    'prove/fast_adapter/p45_handle_surfaces.py',
    'prove/fast_adapter/p45_visible_frame.py',
    'prove/fast_adapter/p45_neural.py',
    'prove/fast_adapter/p45_visual.py',
    'prove/fast_adapter/p47_neural.py',
    'prove/fast_adapter/robot_geometry_env_p45.py',  # imported by the frozen P47 evaluator
    'prove/fast_adapter/robot_geometry_env_p45_v02.py',
    'tools/fast_adapter/p45_common.py',
    'tools/fast_adapter/p47_evaluate.py',
    'tools/fast_adapter/p58_cup_evaluate.py',
)
ENVIRONMENT={'module':'prove.fast_adapter.robot_geometry_env_p45_v02',
             'configuration_module':'prove.fast_adapter.robot_geometry_env_p45_v02',
             'controller_module':'prove.fast_adapter.robot_geometry_env_p45_v02','class':'RobotGeometrySkill'}


def cells():
    result=[dict(condition_seed=c,representation=r,arm=a) for c in CONDITIONS for r in NEW_REPS for a in NEW_ARMS]
    result += [dict(condition_seed=c,representation='p47_geometry_reference',arm=a)
               for c in CONDITIONS for a in ('fixed','continuous')]
    return result


def validate(path):
    value=json.loads(Path(path).read_text())
    if (value.get('schema')!=SCHEMA or value.get('status')!='frozen' or value.get('root_authorized') is not True
            or value.get('steps')!=STEPS or value.get('cells')!=cells()):raise ValueError('Invalid P58 cup freeze')
    for name,digest in value['sources'].items():
        if sha(name)!=digest:raise ValueError('Frozen source changed: '+name)
    for item in value['models'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen model changed')
    if sha(value['plan']['path'])!=value['plan']['sha256']:raise ValueError('Frozen plan changed')
    if sha(value['engineering_gate']['path'])!=value['engineering_gate']['sha256']:raise ValueError('Cup gate changed')
    gate=json.loads(Path(value['engineering_gate']['path']).read_text())
    if not (gate.get('passed') is True or gate.get('qualified') is True):raise ValueError('Cup engineering gate failed')
    if value.get('environment')!=ENVIRONMENT or value.get('robot_assets')!=source_assets():raise ValueError('Panda environment/assets changed')
    return value


def freeze(output,position,functional,p47,plan,engineering_gate):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    models={name:dict(path=str(Path(path)),sha256=sha(path)) for name,path in
            [('position_only',position),('functional_frame',functional),('p47_geometry_reference',p47)]}
    value=dict(schema=SCHEMA,status='frozen',root_authorized=True,steps=STEPS,cells=cells(),models=models,
        plan=dict(path=str(Path(plan)),sha256=sha(plan)),sources={name:sha(name) for name in SOURCES},
        engineering_gate=dict(path=str(Path(engineering_gate)),sha256=sha(engineering_gate)),
        environment=ENVIRONMENT,robot_assets=source_assets(),
        input_boundary='visible role masks, camera calibration and EEF only for P58; P47 G retains historical visible corners',
        success='unchanged P45 sustained last10 bilateral contact and lift, with mechanical validity')
    output.parent.mkdir(parents=True,exist_ok=True);js(output,value);return value


def rollout(index,cell,frozen,out,freeze_path):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    model_ref=frozen['models'][cell['representation']]
    js(out/'PRE_RUN.json',dict(index=index,cell=cell,freeze_sha256=sha(freeze_path),model=model_ref,
        steps=STEPS,timestamp=time.time(),runtime_truth=False))
    env=make_env('cup',cell['condition_seed'],'geometry',backend=ENVIRONMENT)
    try:
        controller=make_controller(env,backend=dict(controller_module='prove.fast_adapter.robot_geometry_env_p45_v02'))
        if cell['representation']=='p47_geometry_reference':
            model,_=load_p47_model(model_ref['path']);runtime=P47Latent(model,cell['arm']);new=False
        else:
            runtime=load_runtime(model_ref['path'],cell['arm']);model=runtime.model;new=True
        before=parameters_sha(model);trace=[];observations=[]
        for t in range(STEPS):
            obs,rgb,masks=observe_roles(env)
            if not controller.adaptation_frozen:
                if new:estimate=runtime.update(obs,masks,t)
                else:
                    old=dict(obs,local=np.zeros(48,np.float32));runtime.update(old,t);estimate=runtime.estimate()
            else:estimate=runtime.estimate()
            p,R,closed,stage=controller.command(t,estimate,env.proprio())
            row=dict(t=t,state=snapshot(env),command_p=p.copy(),command_R=R.copy(),closed=bool(closed),stage=stage,
                     estimate=copy.deepcopy(estimate),adaptation=copy.deepcopy(runtime.records[-1]) if runtime.records else None,
                     controls_before=env.data.ctrl.copy())
            observations.append(dict(t=t,corners=copy.deepcopy(obs['corners']),camera_position=obs['camera_position'].copy(),
                camera_rotation=obs['camera_rotation'].copy(),K=obs['K'].copy(),position=obs['position'].copy(),rotation=obs['rotation'].copy()))
            env.step(p,R,closed);row['controls_after']=env.data.ctrl.copy();row['robot_tracking']=env.robot_tracking_diagnostics();trace.append(row)
        terminal=snapshot(env)
        np.save(out/'TRACE.npy',np.asarray(trace,dtype=object),allow_pickle=True)
        np.save(out/'OBS.npy',np.asarray(observations,dtype=object),allow_pickle=True)
        np.save(out/'TERMINAL.npy',terminal,allow_pickle=True)
        after=parameters_sha(model)
        js(out/'SEAL.json',dict(schema='prove.p58-cup-trajectory-seal.v1',files={n:sha(out/n) for n in
            ('TRACE.npy','OBS.npy','TERMINAL.npy','PRE_RUN.json')},neural_parameters_before=before,neural_parameters_after=after))
        result=evaluate_sealed(env,trace,terminal,out)
        result.update(cell=cell,index=index,neural_weights_unchanged=before==after,
            adaptation_updates=getattr(runtime,'update_events',0),elapsed_seconds=time.monotonic()-started)
        js(out/'RESULT.json',result);return result
    finally:env.close()


def run_one(index,freeze_path,output):
    """Run one cell and retain a machine-readable program error on failure."""
    output=Path(output)
    try:
        frozen=validate(freeze_path)
        result=rollout(index,frozen['cells'][index],frozen,output,freeze_path)
        return 0,{k:result[k] for k in ('index','validated_success','outcome_class','neural_weights_unchanged')}
    except Exception as error:
        output.mkdir(parents=True,exist_ok=True)
        payload=dict(schema='prove.p58-cup-cell-error.v1',index=index,outcome_class='program_error',
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            task_success_denominator_included=False,timestamp=time.time())
        js(output/'ERROR.json',payload)
        return 2,{k:payload[k] for k in ('index','outcome_class','error_type','error')}


def cell_output(root,index,cell):
    return Path(root)/f"{index:02d}_{cell['condition_seed']}_{cell['representation']}_{cell['arm']}"


def worker(freeze_path,output_root,worker_index,workers):
    frozen=validate(freeze_path);failures=0
    for index,cell in enumerate(frozen['cells']):
        if index % workers != worker_index:continue
        output=cell_output(output_root,index,cell)
        command=[sys.executable,'-m','tools.fast_adapter.p58_cup_evaluate','run',
                 '--freeze',freeze_path,'--index',str(index),'--output',str(output)]
        completed=subprocess.run(command,env=os.environ.copy())
        failures += completed.returncode != 0
    return int(failures>0)


def queue(freeze_path,output_root,workers):
    if workers!=3:raise ValueError('The frozen development queue uses exactly three workers')
    validate(freeze_path);processes=[]
    for worker_index in range(workers):
        environment=os.environ.copy();environment['MUJOCO_GL']='egl';environment['EGL_DEVICE_ID']=str(worker_index)
        command=[sys.executable,'-m','tools.fast_adapter.p58_cup_evaluate','worker',
                 '--freeze',freeze_path,'--output-root',output_root,
                 '--worker-index',str(worker_index),'--workers',str(workers)]
        processes.append(subprocess.Popen(command,env=environment))
    codes=[process.wait() for process in processes]
    return int(any(codes)),codes


def main():
    global args,started
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--output',required=True);f.add_argument('--position-only',required=True);f.add_argument('--functional-frame',required=True);f.add_argument('--p47',required=True);f.add_argument('--plan',required=True);f.add_argument('--engineering-gate',default='evaluations/fast_adapter_p45_robot_mechanism_v01/engineering/cup_GATE.json')
    r=sub.add_parser('run');r.add_argument('--freeze',required=True);r.add_argument('--index',type=int,required=True);r.add_argument('--output',required=True)
    w=sub.add_parser('worker');w.add_argument('--freeze',required=True);w.add_argument('--output-root',required=True);w.add_argument('--worker-index',type=int,required=True);w.add_argument('--workers',type=int,default=3)
    q=sub.add_parser('queue');q.add_argument('--freeze',required=True);q.add_argument('--output-root',required=True);q.add_argument('--workers',type=int,default=3)
    args=p.parse_args();started=time.monotonic()
    if args.command=='freeze':value=freeze(args.output,args.position_only,args.functional_frame,args.p47,args.plan,args.engineering_gate);print(json.dumps(dict(schema=value['schema'],cells=len(value['cells'])),indent=2))
    elif args.command=='run':
        code,result=run_one(args.index,args.freeze,args.output);print(json.dumps(result,indent=2));raise SystemExit(code)
    elif args.command=='worker':
        raise SystemExit(worker(args.freeze,args.output_root,args.worker_index,args.workers))
    else:
        code,codes=queue(args.freeze,args.output_root,args.workers);print(json.dumps(dict(worker_exit_codes=codes),indent=2));raise SystemExit(code)


if __name__=='__main__':main()
