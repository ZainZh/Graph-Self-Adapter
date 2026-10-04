#!/usr/bin/env python3
"""Prepare and collect four bounded P52 GT-teacher training qualifications.

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
from prove.task_graph.p52_training_teacher import TrainingOnlyBookTeacher,TeacherConfig,transform,box_vertices
from prove.task_graph.libero_supervisor import contact_facts
from tools.task_graph.collect_libero_demo_gt import prepare as prepare_environment,verify as verify_environment,sha,dump,snapshot


def capture_rows(path):
    return [json.loads(line) for line in (path/'FRAMES.jsonl').read_text().splitlines()]


def prepare(source, teacher_capture, predecessor_capture, calibration, libero_root, robosuite_root):
    source,teacher_capture,predecessor_capture,calibration=map(lambda p:Path(p).resolve(),(source,teacher_capture,predecessor_capture,calibration))
    calibration_data=json.loads(calibration.read_text())
    names=json.loads((teacher_capture/'SUPERVISOR_NAMES.json').read_text())
    book=names['body'].index('black_book_1_main');goal=names['site'].index('desk_caddy_1_back_contain_region')
    relations={}
    teacher_rows=capture_rows(teacher_capture)
    for t in (191,234):
        path=teacher_capture/f'frame_{t:04d}.npz'
        if sha(path)!=teacher_rows[t]['sha256']:raise ValueError('teacher relation frame changed')
        with np.load(path) as z:
            W_B=transform(z['supervisor_body_xpos'][book],z['supervisor_body_xmat'][book].reshape(3,3))
            W_C=transform(z['supervisor_site_xpos'][goal],z['supervisor_site_xmat'][goal].reshape(3,3))
            relations[t]=np.linalg.inv(W_C)@W_B
    points=box_vertices(calibration_data['book_collision_center'],calibration_data['book_collision_halfsize'])
    clearpoints=points@relations[191][:3,:3].T+relations[191][:3,3]
    relation=dict(final_caddy_book=relations[234].tolist(),
                  demonstrated_clearance_bottom=float(np.min(clearpoints@np.array([0.,0.,-1.]))),
                  book_collision_center=calibration_data['book_collision_center'],
                  book_collision_halfsize=calibration_data['book_collision_halfsize'],back_halfsize=calibration_data['back_halfsize'],
                  source_frames=[dict(path=str(teacher_capture/f'frame_{t:04d}.npz'),sha256=sha(teacher_capture/f'frame_{t:04d}.npz')) for t in (191,234)])
    entries=[]
    for kind,capture,steps in [('official_demo_physical_prefix',teacher_capture,(191,200)),('p51_a2_physical_predecessor',predecessor_capture,(333,342))]:
        rows=capture_rows(capture);local_names=json.loads((capture/'SUPERVISOR_NAMES.json').read_text())
        for t in steps:
            frame=capture/f'frame_{t:04d}.npz'
            if sha(frame)!=rows[t]['sha256']:raise ValueError('predecessor frame changed')
            fingers,_=contact_facts(local_names,rows[t]['supervisor']['contacts'])
            if fingers!=(True,True):raise ValueError('qualification entry must have both fingers touching')
            entries.append(dict(index=len(entries),name=f'{kind}_{t}',source_kind=kind,source_frame=t,
                                frame=str(frame),frame_sha256=sha(frame),capture=str(capture),
                                source_binding_sha256=sha(capture/'BINDING.json'),
                                predecessor_trace_sha256=sha(capture/'FRAMES.jsonl'),predecessor_bilateral_contact=True))
    files=[Path(__file__).resolve(),ROOT/'prove/task_graph/p52_training_teacher.py',ROOT/'prove/task_graph/libero_supervisor.py',
           ROOT/'tools/task_graph/collect_libero_demo_gt.py',ROOT/'prove/__init__.py',ROOT/'prove/task_graph/__init__.py',
           ROOT/'tools/__init__.py',ROOT/'tools/task_graph/__init__.py',calibration,teacher_capture/'SUPERVISOR_NAMES.json',predecessor_capture/'SUPERVISOR_NAMES.json']
    return dict(schema='prove.p52-local-teacher-qualification-freeze.v1',phase='development_training_teacher_qualification',
                environment=prepare_environment(source,libero_root,robosuite_root),entries=entries,relation=relation,
                teacher_config=asdict(TeacherConfig()),implementation={str(p):sha(p) for p in files},
                packages={name:importlib.metadata.version(name) for name in ('scipy','mujoco')},
                public=dict(cameras=['agentview','robot0_eye_in_hand'],resolution=448,image_origin='top_left',
                            roles=['actor','object','goal'],goal_mask='actual visible caddy mesh, not projected hidden mouth',
                            inputs=['RGB','visible role masks','camera calibration and own robot FK','EEF pose','gripper qpos','joint state','executed actions']),
                permissions=dict(private='GT teacher pose/relations/actions/contacts/native goal/full physics state: training and evaluation only',
                                 public='candidate runtime and feedback loss may consume public stream only; expert labels are separate'),
                max_episodes=4,max_actions_per_episode=180,initial_restores_per_episode=1,automatic_retries=0,
                initialization='saved physical state and ctrl once; restore gripper command integrator from ctrl range; synchronize OSC goal with current EEF; no further restore',
                entry_scope='real predecessor snapshots restored only at training episode start; not full-task success evidence',
                teacher='raise actual book above mouth/wall clearance, align to final demonstrated book relation, descend along mouth axis; compensate current EEF-to-book transform',
                physics_changes=False,force_learning=False,learned_policy_executed=False)


def verify(binding):
    if binding.get('schema')!='prove.p52-local-teacher-qualification-freeze.v1' or binding.get('max_episodes')!=4 or binding.get('max_actions_per_episode')!=180:
        raise ValueError('unexpected qualification scope')
    verify_environment(binding['environment'])
    for path,digest in binding['implementation'].items():
        if sha(path)!=digest:raise ValueError(f'frozen implementation/input changed: {path}')
    for entry in binding['entries']:
        if sha(entry['frame'])!=entry['frame_sha256'] or sha(Path(entry['capture'])/'BINDING.json')!=entry['source_binding_sha256'] or sha(Path(entry['capture'])/'FRAMES.jsonl')!=entry['predecessor_trace_sha256']:
            raise ValueError('predecessor binding changed')
    for entry in binding['relation']['source_frames']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('teacher geometry source changed')
    for name,version in binding['packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('package changed')
    if binding['teacher_config']!=asdict(TeacherConfig()):raise ValueError('teacher configuration changed')


class PublicRenderer:
    """Actual visible segmentation only, no projected hidden object geometry."""
    def __init__(self,env,config):
        import mujoco
        self.mj=mujoco;self.env=env;self.config=config
        self.model=env.sim.model._model;self.data=env.sim.data._data
        self.renderer=mujoco.Renderer(self.model,height=config['resolution'],width=config['resolution'])
        self.option=mujoco.MjvOption();self.option.geomgroup[0]=0;self.option.geomgroup[1]=1
        self.ids=[]
        for prefix in ('gripper0_','black_book_1_','desk_caddy_1_'):
            self.ids.append([i for i in range(self.model.ngeom) if (self.model.geom(i).name or '').startswith(prefix)])
    def observe(self,obs):
        rgbs=[];masks=[];intrinsics=[];cameras=[];size=self.config['resolution']
        for camera in self.config['cameras']:
            self.renderer.update_scene(self.data,camera=camera,scene_option=self.option)
            rgbs.append(self.renderer.render().copy())
            self.renderer.enable_segmentation_rendering();seg=self.renderer.render().copy();self.renderer.disable_segmentation_rendering()
            physical=seg[:,:,1]==int(self.mj.mjtObj.mjOBJ_GEOM)
            masks.append(np.asarray([physical&np.isin(seg[:,:,0],ids) for ids in self.ids]))
            cid=self.model.camera(camera).id;focal=size/2/np.tan(np.deg2rad(self.model.cam_fovy[cid])/2)
            intrinsics.append([[focal,0,size/2],[0,focal,size/2],[0,0,1]])
            cameras.append(transform(self.data.cam_xpos[cid],self.data.cam_xmat[cid].reshape(3,3)))
        return dict(rgb=np.asarray(rgbs),visible_masks=np.asarray(masks),camera_K=np.asarray(intrinsics),camera_pose=np.asarray(cameras),
                    eef_pos=np.asarray(obs['robot0_eef_pos']).copy(),eef_quat_xyzw=np.asarray(obs['robot0_eef_quat']).copy(),
                    gripper_qpos=np.asarray(obs['robot0_gripper_qpos']).copy(),joint_pos=np.asarray(obs['robot0_joint_pos']).copy(),joint_vel=np.asarray(obs['robot0_joint_vel']).copy())
    def close(self):self.renderer.close()


def private_measurement(env,obs,names,relation):
    data=env.sim.data;model=env.sim.model
    b=model.body_name2id('black_book_1_main');g=model.site_name2id('desk_caddy_1_back_contain_region')
    W_B=transform(data.body_xpos[b],data.body_xmat[b].reshape(3,3));W_C=transform(data.site_xpos[g],data.site_xmat[g].reshape(3,3))
    W_E=transform(obs['robot0_eef_pos'],Rotation.from_quat(obs['robot0_eef_quat']).as_matrix())
    contacts=[dict(geom1=int(data.contact[i].geom1),geom2=int(data.contact[i].geom2),distance=float(data.contact[i].dist)) for i in range(data.ncon)]
    fingers,wrong=contact_facts(names,contacts)
    # Physical collision boxes overlapping the back-mouth footprint determine rim height.
    rim=float(relation['back_halfsize'][2]);half=np.asarray(relation['back_halfsize'])[:2];margin=min(relation['book_collision_halfsize'])
    for i,name in enumerate(names['geom']):
        if not name or not name.startswith('desk_caddy_1_') or int(model.geom_type[i])!=6:continue
        pts=box_vertices(np.zeros(3),model.geom_size[i])@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
        local=(pts-W_C[:3,3])@W_C[:3,:3]
        if np.all(local[:,:2].max(0)>=-half-margin) and np.all(local[:,:2].min(0)<=half+margin):rim=max(rim,float(np.max(-local[:,2])))
    return dict(world_book=W_B,world_caddy=W_C,world_eef=W_E,rim_height=rim,finger_contacts=fingers,
                official_success=bool(env._check_success()),wrong_target=wrong,contacts=contacts)


def collect(binding_path,entry_index,output):
    binding=json.loads(binding_path.read_text());verify(binding)
    if entry_index not in range(4):raise ValueError('entry index out of frozen scope')
    entry=binding['entries'][entry_index];output.mkdir(parents=True,exist_ok=False)
    public=output/'public';private=output/'private';public.mkdir();private.mkdir()
    dump(output/'BINDING.json',binding);dump(output/'ENTRY.json',entry)
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
        with np.load(entry['frame']) as initial:
            env.sim.set_state_from_flattened(initial['supervisor_sim_state'])
            if 'supervisor_ctrl' in initial:env.sim.data.ctrl[:]=initial['supervisor_ctrl']
        env.sim.forward();env._post_process();env._update_observables(force=True)
        # Restore the gripper command integrator from saved actuator controls.
        # Resetting only qpos would otherwise reopen a grasp on its first action.
        for robot in env.robots:
            gripper=robot.gripper
            ids=[env.sim.model.actuator_name2id(name) for name in gripper.actuators]
            ranges=env.sim.model.actuator_ctrlrange[ids]
            gripper.current_action=np.clip((env.sim.data.ctrl[ids]-ranges.mean(axis=1))/(.5*(ranges[:,1]-ranges[:,0])),-1.,1.)
            robot.controller.update(force=True);robot.controller.reset_goal()
        names={kind:[getattr(env.sim.model,kind+'_id2name')(i) for i in range(getattr(env.sim.model,count))] for kind,count in [('body','nbody'),('site','nsite'),('geom','ngeom')]}
        dump(private/'NAMES.json',names)
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
                transition=dict(episode_id=output.name,stage_id='Insert',t=actions-1,
                    observation=f'frame_{actions-1:04d}.npz',observation_sha256=previous['public_observation_sha256'],
                    executed_action=previous['expert_action'],next_observation=public_path.name,next_observation_sha256=sha(public_path))
                with (public/'TRANSITIONS.jsonl').open('a') as stream:stream.write(json.dumps(transition,allow_nan=False)+'\n')
            if stop:break
            obs,_,_,_=env.step(action.copy());actions+=1
        verify(binding)
        seal=dict(schema='prove.p52-training-trajectory-seal.v1',binding_sha256=sha(output/'BINDING.json'),entry_index=entry_index,
                  files={str(p.relative_to(output)):sha(p) for folder in (public,private) for p in sorted(folder.iterdir()) if p.is_file()},actions=actions,observations=len(rows),
                  public_private_separated=True,initial_restore_only=True,physics_steps_executed=actions>0)
        dump(output/'SEAL.json',seal)
        result=dict(schema='prove.p52-training-teacher-result.v1',entry=entry,actions=actions,frames=len(rows),official_success=rows[-1]['official_success'],
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
    for name in ('source','teacher-capture','predecessor-capture','calibration','libero-root','robosuite-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('collect');p.add_argument('--binding',type=Path,required=True);p.add_argument('--entry-index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.output.exists():raise FileExistsError(args.output)
        value=prepare(args.source,args.teacher_capture,args.predecessor_capture,args.calibration,args.libero_root,args.robosuite_root);args.output.parent.mkdir(parents=True,exist_ok=True);dump(args.output,value)
        print(args.output)
    else:print(json.dumps(collect(args.binding,args.entry_index,args.output),indent=2))
