#!/usr/bin/env python3
"""Bounded original-LIBERO local skill evaluation; prepare never starts physics.

All pose truth stays in initialization/evaluator snapshots. Runtime consumes
only rendered public frames, frozen pretrained features and executed actions.
This local restored-entry pilot cannot establish complete original-task success.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from prove.task_graph.p52_stage_policy import Config,StagePolicy,StageRuntime
from tools.task_graph.collect_libero_p52 import PublicRenderer
from tools.task_graph.collect_libero_demo_gt import verify as verify_environment,sha,dump,snapshot
from tools.task_graph.p52_deployment import source_closure

ROOT=Path(__file__).resolve().parents[2]
ARMS=('bc','fixed','online')


def prepare(environment_binding, entries, checkpoints, dino, lila, vision_python, vision_device):
    """Explicit entries/checkpoints produce a bounded Cartesian scope, no sampling."""
    parent=json.loads(Path(environment_binding).read_text());environment=parent['environment']
    verify_environment(environment)
    if len(entries)!=6:raise ValueError('This pilot requires exactly six explicit validation entries')
    for entry in entries:
        if entry.get('split')!='validation' or 'teacher_smoke' in Path(entry['frame']).parts:
            raise ValueError('Only declared validation entries; no qualification smoke states')
        if sha(entry['frame'])!=entry['frame_sha256']:raise ValueError('Entry state changed')
        if not entry.get('source_kind') or not entry.get('name'):raise ValueError('Entry provenance required')
    weights={arm:dict(path=str(Path(checkpoints[arm]).resolve()),sha256=sha(checkpoints[arm])) for arm in ARMS}
    if weights['fixed']['sha256']!=weights['online']['sha256']:raise ValueError('Fixed/online must share exact checkpoint')
    for arm in ARMS:
        ck=torch.load(weights[arm]['path'],map_location='cpu',weights_only=True)
        if ck['schema']!='p52_stage_skill_v01':raise ValueError('Unexpected model checkpoint')
        model=StagePolicy(Config(**ck['config']));model.load_state_dict(ck['model'],strict=True)
        if model.config.views!=2 or model.config.history!=8:raise ValueError('Require two views and eight-feedback window')
    public=parent['public']
    if public['cameras']!=['agentview','robot0_eye_in_hand'] or public['resolution']!=448:raise ValueError('Unexpected public camera interface')
    entries=[dict(e) for e in entries]
    files=source_closure(ROOT,['tools/task_graph/p52_policy_pilot.py','tools/task_graph/p52_vision_worker.py'])
    assets=[Path(dino)/'model.safetensors',Path(dino)/'config.json',Path(dino)/'preprocessor_config.json',
            Path(lila)/'VISUAL_WEIGHTS.pt',Path(lila)/'models/vla_model_fm.py']
    version_code='import json,importlib.metadata as m;print(json.dumps({n:m.version(n) for n in ("torch","transformers","numpy")}))'
    visual_packages=json.loads(subprocess.check_output([vision_python or sys.executable,'-c',version_code],text=True))
    return dict(schema='p52.local-policy-pilot.v1',environment=environment,public=public,entries=entries,
        checkpoints=weights,source_binding=dict(path=str(Path(environment_binding).resolve()),sha256=sha(environment_binding)),
        sources={p:sha(ROOT/p) for p in files},visual_assets={str(p.resolve()):sha(p) for p in assets},
        packages={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','mujoco')},
        vision=dict(dino=str(Path(dino).resolve()),lila=str(Path(lila).resolve()),python=vision_python,
                    device=vision_device,packages=visual_packages),
        cells=[dict(index=i*len(ARMS)+j,entry=i,arm=arm) for i in range(len(entries)) for j,arm in enumerate(ARMS)],
        max_actions=180,goal_hold_frames=3,automatic_retries=0,initial_restores=1,
        success='native official goal true for 3 consecutive observed frames; local skill only',
        initialization='same real predecessor snapshot and ctrl; restore integrator; reset OSC goal; no later state restore',
        input_permissions='public RGB + visible role masks + proprio only; executed OSC6 feeds observed-transition inner loss',
        stop='native goal streak or 180 actions or implementation error; no hidden truth in action or inner loss',
        hold='policy continues its own actions between successful frames; no teacher hold or teacher target is injected')


def verify(binding):
    if binding['schema']!='p52.local-policy-pilot.v1' or binding['max_actions']!=180 or binding['goal_hold_frames']!=3:
        raise ValueError('Changed execution scope')
    if len(binding['entries'])!=6 or len(binding['cells'])!=18:raise ValueError('Expected six-entry, three-arm scope')
    expected=[dict(index=i*3+j,entry=i,arm=a) for i in range(6) for j,a in enumerate(ARMS)]
    if binding['cells']!=expected:raise ValueError('Cell scope changed')
    verify_environment(binding['environment'])
    for p,h in binding['sources'].items():
        if sha(ROOT/p)!=h:raise ValueError(f'Source changed: {p}')
    for p,h in binding['visual_assets'].items():
        if sha(p)!=h:raise ValueError(f'Visual weights/source changed: {p}')
    for value in binding['checkpoints'].values():
        if sha(value['path'])!=value['sha256']:raise ValueError('Policy checkpoint changed')
    for e in binding['entries']:
        if sha(e['frame'])!=e['frame_sha256']:raise ValueError('Initial state changed')
    if sha(binding['source_binding']['path'])!=binding['source_binding']['sha256']:raise ValueError('Parent binding changed')
    for n,v in binding['packages'].items():
        if importlib.metadata.version(n)!=v:raise ValueError('Runtime package changed')


class Vision:
    def __init__(self,config):
        self.process=None;self.local=None
        if config['python']:
            args=[config['python'],'-m','tools.task_graph.p52_vision_worker','--dino',config['dino'],
                  '--lila',config['lila'],'--device',config['device']]
            self.process=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
            ready=json.loads(self.process.stdout.readline())
            if not ready.get('ready'):raise RuntimeError('Visual worker failed to initialize')
            if ready.get('packages')!=config['packages']:
                self.close();raise ValueError('Actual sidecar package versions differ from frozen prepare')
            self.binding=ready['binding']
        else:
            from prove.task_graph.p52_visual_features import FrozenStageVision
            if {n:importlib.metadata.version(n) for n in ('torch','transformers','numpy')}!=config['packages']:
                raise ValueError('Visual package versions differ from frozen prepare')
            self.local=FrozenStageVision(config['dino'],config['lila'],config['device']);self.binding=self.local.binding
    def encode(self,path):
        if self.process:
            self.process.stdin.write(json.dumps(dict(public_frame=str(path.resolve())))+'\n');self.process.stdin.flush()
            response=json.loads(self.process.stdout.readline());features=response['features']
        else:
            from prove.task_graph.p52_visual_features import public_features
            with np.load(path,allow_pickle=False) as frame:features=public_features(frame,self.local)
        return {k:torch.tensor(v,dtype=torch.float32) for k,v in features.items()}
    def close(self):
        if self.process:
            try:self.process.stdin.write('{"close":true}\n');self.process.stdin.flush();self.process.wait(timeout=20)
            except (BrokenPipeError,subprocess.TimeoutExpired):self.process.kill();self.process.wait()


def validate_features(features,config):
    shapes=dict(raw=(config.views,3,config.raw_dim),boxes=(config.views,3,4),
                visible=(config.views,3),proprio=(config.proprio_dim,))
    if set(features)!=set(shapes):raise ValueError('Unknown or missing public feature keys')
    for key,shape in shapes.items():
        value=features[key]
        if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or tuple(value.shape)!=shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid feature dtype, shape or finiteness: {key}')
    if not ((features['visible']==0)|(features['visible']==1)).all():raise ValueError('Visibility must be boolean-valued')
    if not ((features['boxes']>=0)&(features['boxes']<=1)).all():raise ValueError('Pixel-normalized boxes out of range')


def initialize(environment,entry,output):
    config=output/'libero_config';config.mkdir()
    base=Path(environment['libero_root'])/'libero/libero'
    dump(config/'config.yaml',dict(benchmark_root=str(base),bddl_files=str(base/'bddl_files'),
         init_states=str(base/'init_files'),datasets=str(Path(environment['source']).parent),assets=str(base/'assets')))
    os.environ['LIBERO_CONFIG_PATH']=str(config.resolve());os.environ.setdefault('MUJOCO_GL','egl')
    sys.path.insert(0,environment['libero_root'])
    from libero.libero.envs import TASK_MAPPING
    kwargs=dict(environment['env_kwargs']);kwargs.update(use_camera_obs=False,has_offscreen_renderer=False)
    env=TASK_MAPPING[environment['env_args_original']['problem_name']](**kwargs)
    env.reset();env.reset_from_xml_string(environment['relocated_xml']);env.sim.reset()
    with np.load(entry['frame']) as initial:
        env.sim.set_state_from_flattened(initial['supervisor_sim_state'])
        if 'supervisor_ctrl' not in initial:raise ValueError('Real predecessor ctrl is required')
        env.sim.data.ctrl[:]=initial['supervisor_ctrl']
    env.sim.forward();env._post_process();env._update_observables(force=True)
    for robot in env.robots:
        ids=[env.sim.model.actuator_name2id(name) for name in robot.gripper.actuators]
        ranges=env.sim.model.actuator_ctrlrange[ids]
        robot.gripper.current_action=np.clip((env.sim.data.ctrl[ids]-ranges.mean(1))/(.5*(ranges[:,1]-ranges[:,0])),-1,1)
        robot.controller.update(force=True);robot.controller.reset_goal()
    return env


def run(binding_path,index,output):
    binding=json.loads(binding_path.read_text());verify(binding)
    if index not in range(len(binding['cells'])):raise ValueError('Cell outside frozen scope')
    cell=binding['cells'][index];entry=binding['entries'][cell['entry']]
    output.mkdir(parents=True,exist_ok=False);(output/'public').mkdir();(output/'private').mkdir()
    dump(output/'BINDING.json',binding);dump(output/'CELL.json',cell)
    ck=torch.load(binding['checkpoints'][cell['arm']]['path'],map_location='cpu',weights_only=True)
    model=StagePolicy(Config(**ck['config']));model.load_state_dict(ck['model'],strict=True)
    runtime=StageRuntime(model,online=cell['arm']=='online')
    env=renderer=vision=None;actions=0;streak=0;previous=None;previous_action=None
    try:
        vision=Vision(binding['vision']);dump(output/'VISUAL_BINDING.json',vision.binding)
        env=initialize(binding['environment'],entry,output);renderer=PublicRenderer(env,binding['public'])
        obs=env._get_observations(force_update=True)
        while True:
            path=output/'public'/f'frame_{actions:04d}.npz';np.savez_compressed(path,**renderer.observe(obs))
            features=vision.encode(path);validate_features(features,model.config)
            torch.save(features,output/'public'/f'features_{actions:04d}.pt')
            if previous is None:runtime.enter(features)
            z_before=runtime.z.clone();loss=None
            if previous is not None:loss=runtime.feedback(previous,previous_action,features)
            action=runtime.command(features).numpy()
            if action.shape!=(7,) or not np.isfinite(action).all() or action[6]!=1:raise ValueError('Invalid model action')
            # Private goal is evaluated only after the public-model prediction.
            truth=bool(env._check_success());streak=streak+1 if truth else 0
            stopped=streak>=binding['goal_hold_frames'] or actions>=binding['max_actions']
            record=dict(t=actions,observation=path.name,observation_sha256=sha(path),z_before=z_before.tolist(),
                        z_after=runtime.z.tolist(),inner_loss=loss,updates=runtime.updates,action=action.tolist(),executed=not stopped)
            with (output/'public/ACTIONS.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            private,_=snapshot(env,obs);private={k:v for k,v in private.items() if k.startswith('supervisor_')}
            np.savez_compressed(output/'private'/f'frame_{actions:04d}.npz',**private)
            with (output/'private/GOAL.jsonl').open('a') as f:f.write(json.dumps(dict(t=actions,official_success=truth,goal_streak=streak))+'\n')
            if stopped:break
            previous=features;previous_action=torch.tensor(action[:6],dtype=torch.float32)
            obs,_,_,_=env.step(action.copy());actions+=1
        verify(binding)
        files={str(p.relative_to(output)):sha(p) for folder in ('public','private') for p in sorted((output/folder).iterdir())}
        dump(output/'SEAL.json',dict(files=files,binding_sha256=sha(output/'BINDING.json'),actions=actions,cell=cell))
        result=dict(cell=cell,actions=actions,local_success=streak>=3,goal_streak=streak,
                    stop_reason='qualified_goal_hold' if streak>=3 else 'step_budget_exhausted',
                    seal_sha256=sha(output/'SEAL.json'),full_task_success_evidence=False,automatic_retry=False)
        dump(output/'RESULT.json',result);return result
    except Exception as error:
        dump(output/'ABORT.json',dict(error=type(error).__name__,reason=str(error),actions=actions,cell=cell,automatic_retry=False));raise
    finally:
        runtime.exit()
        if renderer is not None:renderer.close()
        if env is not None:env.close()
        if vision is not None:vision.close()


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--environment-binding',type=Path,required=True)
    q.add_argument('--entries',type=Path,required=True);q.add_argument('--bc',required=True);q.add_argument('--joint',required=True)
    q.add_argument('--dino',required=True);q.add_argument('--lila',required=True);q.add_argument('--vision-python');q.add_argument('--vision-device',default='cuda')
    q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('run');q.add_argument('--binding',type=Path,required=True);q.add_argument('--index',type=int,required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    if a.command=='prepare':
        if a.output.exists():raise FileExistsError(a.output)
        value=prepare(a.environment_binding,json.loads(a.entries.read_text()),dict(bc=a.bc,fixed=a.joint,online=a.joint),a.dino,a.lila,a.vision_python,a.vision_device)
        a.output.parent.mkdir(parents=True,exist_ok=True);dump(a.output,value)
    else:print(json.dumps(run(a.binding,a.index,a.output)))


if __name__=='__main__':main()
