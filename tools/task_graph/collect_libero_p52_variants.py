#!/usr/bin/env python3
"""Collect 24 predeclared real-predecessor P52 training conditions.

No model training or physics occurs in prepare. collect runs exactly one frozen
entry, initialized once from a real predecessor state; it never retries.
Public feedback and privileged teacher labels are separate data streams.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='1'
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from prove.task_graph.p52_training_teacher import TeacherConfig,transform,box_vertices
from prove.task_graph.p52_training_teacher_v02 import TrainingOnlyBookTeacher,centered_relation
from prove.task_graph.libero_supervisor import contact_facts
from tools.task_graph.collect_libero_demo_gt import prepare as prepare_environment,verify as verify_environment,sha,dump,snapshot


from tools.task_graph.collect_libero_p52_v02 import PublicRenderer,private_measurement,verify as verify_teacher
from tools.task_graph.collect_libero_demo_gt import source_demo
from tools.task_graph.run_libero_p51_v02 import prepare as prepare_prefix, grounding, insert_entry_domain
from prove.task_graph.schema import TaskGraph
from prove.task_graph.executor import GraphExecutor
from prove.task_graph.libero_precision_p51 import MixedController,MixedRouter
from prove.task_graph.libero_supervisor import LiberoSupervisor
from prove.task_graph.p52_training_teacher import norm_limit


def conditions(relation,book_rotation):
    half=np.asarray(relation['book_collision_halfsize']);mouth=np.asarray(relation['back_halfsize'])
    rows=[]
    for sx in (-1,1):
        for sy in (-1,0,1):
            g=len(rows)//4;offset=np.array([sx*.2*half[0],sy*.06*half[1],0.])
            for q,(qx,qy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1))):
                i=len(rows)
                rows.append(dict(index=i,job_id=f'job_{i:02d}',condition_seed=520000+i,
                    split='validation' if q==g%4 else 'train',grasp_index=g,quadrant_index=q,
                    grasp_book_offset=offset.tolist(),grasp_world_offset=(book_rotation@offset).tolist(),
                    mouth_offset=[qx*.1*mouth[0],qy*.1*mouth[1],0.],
                    mouth_yaw=qx*qy*.1*float(np.arctan(half[0]/half[1]))))
    return rows


def prepare(teacher_binding,graph,checkpoint,calibration):
    teacher_binding=Path(teacher_binding).resolve();basis=json.loads(teacher_binding.read_text());verify_teacher(basis)
    environment=basis['environment']
    prefix=prepare_prefix(environment['source'],graph,calibration,environment['libero_root'],environment['robosuite_root'],checkpoint)
    capture=Path(basis['relation']['source_frames'][0]['path']).parent
    frame=capture/'frame_0057.npz';names=json.loads((capture/'SUPERVISOR_NAMES.json').read_text())
    with np.load(frame) as z:R=z['supervisor_body_xmat'][names['body'].index('black_book_1_main')].reshape(3,3)
    provenance={str(teacher_binding):sha(teacher_binding),str(frame):sha(frame)}
    for i in range(4):
        folder=teacher_binding.parent/'teacher_smoke_v02'/f'entry_{i:02d}'
        result=json.loads((folder/'RESULT.json').read_text())
        if not result['teacher_qualified'] or not result['official_success'] or sha(folder/'SEAL.json')!=result['seal_sha256']:raise ValueError('teacher qualification missing')
        for name in ('RESULT.json','SEAL.json'):provenance[str(folder/name)]=sha(folder/name)
    for item in basis['relation']['source_frames']:provenance[item['path']]=item['sha256']
    for item in basis['entries']:
        provenance[item['frame']]=item['frame_sha256']
        for name in ('BINDING.json','FRAMES.jsonl'):
            path=Path(item['capture'])/name;provenance[str(path)]=sha(path)
    implementation=dict(basis['implementation'],**prefix['implementation'])
    implementation[str(Path(__file__).resolve())]=sha(__file__)
    implementation[str(ROOT/'tools/task_graph/collect_libero_p52_v02.py')]=sha(ROOT/'tools/task_graph/collect_libero_p52_v02.py')
    return dict(schema='prove.p52-real-prefix-collection.v1',environment=environment,
        relation=basis['relation'],teacher_config=basis['teacher_config'],public=basis['public'],
        conditions=conditions(basis['relation'],R),prefix=prefix,implementation=implementation,
        provenance=provenance,packages=dict(basis['packages'],**prefix['controller_packages']),
        max_actions_per_episode=180,prefix_max_actions=400,transition_max_actions=24,
        transition_rule='24 real closed-gripper pose-feedback steps to prescribed relative EEF target; attached and no wrong target required at resulting entry; requested offset is not claimed achieved',
        initial_restores_per_episode=1,automatic_retries=0,physics_changes=False,
        training_rule='Only predeclared train and teacher_qualified trajectories may supply successful BC labels; retain all24 outcomes, no replacements')


def verify(binding):
    if binding['schema']!='prove.p52-real-prefix-collection.v1' or len(binding['conditions'])!=24:raise ValueError('wrong collection scope')
    verify_environment(binding['environment'])
    for path,digest in dict(binding['implementation'],**binding['provenance']).items():
        if sha(path)!=digest:raise ValueError(f'frozen input changed: {path}')
    for name,version in binding['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError(f'package changed: {name}')


class VariantController(MixedController):
    def __init__(self,source,graph,checkpoint,offset):
        super().__init__(source,graph,checkpoint)
        self.offset=np.asarray(offset)
        positions=self.code.positions.copy();positions[:57]+=self.offset;positions.setflags(write=False);self.code.positions=positions
        self.paths['s3']['positions']=(np.asarray(self.paths['s3']['positions'])+self.offset).tolist()
    def __call__(self,payload):
        if payload['stage_id']!='s3':return super().__call__(payload)
        original=self.nominal_reference.copy()
        self.nominal_reference[:3,3]+=self.offset
        try:return super().__call__(payload)
        finally:self.nominal_reference=original


def execute_prefix(env,binding,entry,output,names):
    graph=TaskGraph.from_dict(binding['prefix']['runtime_graph'])
    skill=VariantController(binding['environment']['source'],graph,binding['prefix']['checkpoint'],entry['grasp_world_offset'])
    router=MixedRouter()
    for name in {s.skill for s in graph.nodes}:router.register(name,kind=router.required_kind(name),revision=skill.revision,call=skill,compatible=lambda g:g.stage_id in skill.segments)
    executor=GraphExecutor(graph,router,instance_id=entry['job_id'])
    supervisor=LiberoSupervisor(env,json.loads(Path(binding['prefix']['calibration']).read_text()),instance_id=entry['job_id'])
    obs=env._get_observations(force_update=True);active=None;holds=0;actions=0
    def record(phase,action,diagnostics):
        arrays,_=snapshot(env,obs)
        arrays={k:v for k,v in arrays.items() if k.startswith('supervisor_')}
        arrays['action_next']=np.asarray([] if action is None else action)
        path=output/f'{phase}_{actions:04d}.npz';np.savez_compressed(path,**arrays)
        with (output/'TRACE.jsonl').open('a') as f:f.write(json.dumps(dict(t=actions,phase=phase,file=path.name,sha256=sha(path),action=action,diagnostics=diagnostics),allow_nan=False)+'\n')
    while True:
        skill.set_observation(obs);action=None;stop=None;ready=False;events=[]
        for _ in range(8):
            stage=graph.stage(executor.current_stage)
            if active!=stage.id:
                if active is not None:skill.on_exit()
                skill.on_enter(stage.id);active=stage.id
            report=supervisor.observe(obs,stage.id);facts=dict(report.facts)
            facts['public_insert_entry_domain']=insert_entry_domain(skill)[0]
            if actions>=binding['prefix_max_actions']:stop='prefix_budget';break
            result=executor.tick(grounding(graph,stage,entry['job_id'],actions),facts,progress=report.progress)
            event=asdict(result);event['action']=None if result.action is None else np.asarray(result.action).tolist();events.append(event)
            if result.state=='complete':
                if stage.id=='s6':ready=True;break
                continue
            action=event['action']
            if action is None:
                unknown=result.reason in ('invariant_unknown','precondition_unknown')
                if unknown and not any(facts.get(k) is True for k in ('object_lost','wrong_target')) and holds<3:
                    holds+=1;action=[0.,0.,0.,0.,0.,0.,1.]
                else:stop='prefix_'+str(result.reason)
            break
        record('prefix',action,dict(events=events,ready=ready,stop=stop))
        if stop:return dict(qualified=False,stop_reason=stop,prefix_actions=actions)
        if ready:break
        obs,_,_,_=env.step(np.asarray(action));actions+=1
    measured=private_measurement(env,obs,names,binding['relation'])
    initial_eef=measured['world_eef'];C=measured['world_caddy'][:3,:3]
    target=initial_eef.copy();target[:3,3]+=C@np.asarray(entry['mouth_offset'])
    target[:3,:3]=Rotation.from_rotvec(C[:,2]*entry['mouth_yaw']).as_matrix()@target[:3,:3]
    prefix_actions=actions
    for t in range(binding['transition_max_actions']):
        measured=private_measurement(env,obs,names,binding['relation'])
        if measured['finger_contacts'] is None or any(v is None for v in measured['finger_contacts']):
            record('entry_transition',None,dict(stop='transition_insufficient_observation'))
            return dict(qualified=False,stop_reason='transition_insufficient_observation',prefix_actions=prefix_actions,transition_actions=t)
        if measured['wrong_target'] or not any(measured['finger_contacts']):
            record('entry_transition',None,dict(stop='transition_contact_loss'))
            return dict(qualified=False,stop_reason='transition_contact_loss',prefix_actions=prefix_actions,transition_actions=t)
        E=measured['world_eef'];dp=target[:3,3]-E[:3,3];dr=Rotation.from_matrix(target[:3,:3]@E[:3,:3].T).as_rotvec()
        action=np.r_[norm_limit(dp,.01)/.05,norm_limit(dr,.05)/.5,1.].tolist()
        record('entry_transition',action,dict(target=target.tolist(),position_error=dp.tolist(),rotation_error=dr.tolist()))
        obs,_,_,_=env.step(np.asarray(action));actions+=1
    measured=private_measurement(env,obs,names,binding['relation'])
    contacts=measured['finger_contacts']
    qualified=contacts is not None and all(v is True or isinstance(v,np.bool_) and bool(v) for v in contacts) and not measured['wrong_target']
    record('entry_final',None,dict(qualified=qualified,world_book=measured['world_book'].tolist(),world_eef=measured['world_eef'].tolist()))
    return dict(qualified=qualified,stop_reason='entry_ready' if qualified else 'transition_entry_not_bilateral',prefix_actions=prefix_actions,transition_actions=24,
                desired_eef=target.tolist(),actual_eef=measured['world_eef'].tolist(),actual_eef_book=(np.linalg.inv(measured['world_eef'])@measured['world_book']).tolist())


def collect(binding_path,entry_index,output):
    binding=json.loads(binding_path.read_text());verify(binding)
    if entry_index not in range(24):raise ValueError('entry index out of frozen scope')
    entry=binding['conditions'][entry_index];output.mkdir(parents=True,exist_ok=False)
    public=output/'public';private=output/'private';public.mkdir();private.mkdir()
    dump(output/'BINDING.json',binding);dump(output/'ENTRY.json',entry)
    prefix=output/'prefix';prefix.mkdir()
    np.random.seed(entry['condition_seed'])
    envbinding=binding['environment'];config_dir=output/'libero_config';config_dir.mkdir()
    base=Path(envbinding['libero_root'])/'libero/libero'
    dump(config_dir/'config.yaml',dict(benchmark_root=str(base),bddl_files=str(base/'bddl_files'),init_states=str(base/'init_files'),datasets=str(Path(envbinding['source']).parent),assets=str(base/'assets')))
    os.environ['LIBERO_CONFIG_PATH']=str(config_dir.resolve());os.environ.setdefault('MUJOCO_GL','egl');sys.path.insert(0,envbinding['libero_root'])
    from libero.libero.envs import TASK_MAPPING
    env=None;renderer=None;rows=[];actions=0
    try:
        kwargs=dict(envbinding['env_kwargs']);kwargs.update(use_camera_obs=False,has_offscreen_renderer=False)
        env=TASK_MAPPING[envbinding['env_args_original']['problem_name']](**kwargs)
        env.reset();env.reset_from_xml_string(envbinding['relocated_xml']);env.sim.reset()
        env.sim.set_state_from_flattened(source_demo(envbinding['source'])['initial'])
        env.sim.forward();env._post_process();env._update_observables(force=True)
        names={kind:[getattr(env.sim.model,kind+'_id2name')(i) for i in range(getattr(env.sim.model,count))] for kind,count in [('body','nbody'),('site','nsite'),('geom','ngeom')]}
        dump(private/'NAMES.json',names)
        prefix_result=execute_prefix(env,binding,entry,prefix,names)
        dump(output/'PREFIX_RESULT.json',prefix_result)
        if not prefix_result['qualified']:
            result=dict(job_id=entry['job_id'],split=entry['split'],entry=entry,prefix_qualified=False,
                        teacher_qualified=False,official_success=bool(env._check_success()),
                        stop_reason=prefix_result['stop_reason'],actions=0,frames=0,automatic_retry=False)
            verify(binding)
            dump(output/'SEAL.json',dict(binding_sha256=sha(output/'BINDING.json'),files={str(p.relative_to(output)):sha(p) for p in [*sorted(prefix.iterdir()),output/'BINDING.json',output/'PREFIX_RESULT.json',output/'ENTRY.json'] if p.is_file()},initial_restore_only=True))
            result['seal_sha256']=sha(output/'SEAL.json');dump(output/'RESULT.json',result);return result
        renderer=PublicRenderer(env,binding['public']);teacher=TrainingOnlyBookTeacher(binding['relation'],TeacherConfig(**binding['teacher_config']))
        obs=env._get_observations(force_update=True)
        while True:
            public_arrays=renderer.observe(obs)
            public_path=public/f'frame_{actions:04d}.npz';np.savez_compressed(public_path,**public_arrays)
            private_arrays,_=snapshot(env,obs)
            private_arrays={key:value for key,value in private_arrays.items() if key.startswith('supervisor_')}
            measured=private_measurement(env,obs,names,binding['relation'])
            action,label=teacher.command(measured)
            stop=label['stop_reason'] or ('step_budget_exhausted' if actions>=binding['max_actions_per_episode'] else None)
            if measured['wrong_target']:stop='wrong_target'
            private_arrays.update(world_book=measured['world_book'],world_caddy=measured['world_caddy'],world_eef=measured['world_eef'])
            private_path=private/f'frame_{actions:04d}.npz';np.savez_compressed(private_path,**private_arrays)
            label.update(t=actions,contacts=measured['contacts'],finger_contacts=measured['finger_contacts'],rim_height=measured['rim_height'],stop_reason=stop,
                         private_state=private_path.name,private_state_sha256=sha(private_path),public_observation=public_path.name,public_observation_sha256=sha(public_path))
            with (private/'LABELS.jsonl').open('a') as stream:stream.write(json.dumps(label,allow_nan=False)+'\n')
            rows.append(label)
            if actions:
                previous=rows[-2]
                transition=dict(episode_id=entry['job_id'],stage_id='Insert',t=actions-1,
                    observation=f'frame_{actions-1:04d}.npz',observation_sha256=previous['public_observation_sha256'],
                    executed_action=previous['expert_action'],next_observation=public_path.name,next_observation_sha256=sha(public_path))
                with (public/'TRANSITIONS.jsonl').open('a') as stream:stream.write(json.dumps(transition,allow_nan=False)+'\n')
            if stop:break
            obs,_,_,_=env.step(action.copy());actions+=1
        verify(binding)
        seal=dict(schema='prove.p52-training-trajectory-seal.v2',binding_sha256=sha(output/'BINDING.json'),entry_index=entry_index,
                  files={str(p.relative_to(output)):sha(p) for folder in (public,private,prefix) for p in sorted(folder.iterdir()) if p.is_file()},actions=actions,observations=len(rows),
                  public_private_separated=True,initial_restore_only=True,physics_steps_executed=actions>0)
        seal['files'].update({name:sha(output/name) for name in ('BINDING.json','ENTRY.json','PREFIX_RESULT.json')})
        dump(output/'SEAL.json',seal)
        result=dict(schema='prove.p52-training-teacher-result.v2',job_id=entry['job_id'],split=entry['split'],prefix_qualified=True,entry=entry,actions=actions,frames=len(rows),official_success=rows[-1]['official_success'],
                    teacher_qualified=rows[-1]['stop_reason']=='qualified_goal_hold',stop_reason=rows[-1]['stop_reason'],goal_hold_frames=rows[-1]['goal_streak'],
                    seal_sha256=sha(output/'SEAL.json'),scope='local original-LIBERO training teacher only; not learned policy or complete-task success',
                    full_task_success_evidence=False,automatic_retry=False)
        dump(output/'RESULT.json',result);return result
    except Exception as exc:
        dump(output/'ABORT.json',dict(error=type(exc).__name__,reason=str(exc),actions_executed=actions,frames_saved=len(rows),automatic_retry=False))
        raise
    finally:
        if renderer is not None:renderer.close()
        if env is not None:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    for name in ('teacher-binding','graph','checkpoint','calibration','output'):p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('collect');p.add_argument('--binding',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.output.exists():raise FileExistsError(args.output)
        dump(args.output,prepare(args.teacher_binding,args.graph,args.checkpoint,args.calibration))
    else:print(json.dumps(collect(args.binding,args.index,args.output),indent=2))
