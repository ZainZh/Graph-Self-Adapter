#!/usr/bin/env python3
"""Cached RGB-D + fresh actual feedback + FK-only entry candidate; no motion."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROLES={'white_mug','red_mug','box'}


def validate_boxes(text):
    boxes=json.loads(text)
    if not isinstance(boxes,dict) or set(boxes)!=ROLES:raise ValueError('explicit manual boxes require white_mug, red_mug and box')
    for role,box in boxes.items():
        if not isinstance(box,list) or len(box)!=4 or any(type(v) not in (int,float) or not np.isfinite(v) for v in box):
            raise ValueError('invalid manual box: '+role)
        if not 0<=box[0]<box[2] or not 0<=box[1]<box[3]:raise ValueError('empty or negative box: '+role)
    return boxes


def read_cached_rgbd(path,boxes):
    with np.load(path,allow_pickle=False) as data:
        rgb=data['rgb'].copy();depth=data['depth_m'].copy()
        key='camera_matrix' if 'camera_matrix' in data.files else 'K'
        K=data[key].copy()
        timestamps={name:float(data[name]) for name in ('timestamp_monotonic_sec','timestamp_unix_sec','stamp') if name in data.files}
        serial=int(data['serial']) if 'serial' in data.files else None
    if rgb.ndim!=3 or rgb.shape[2]!=3 or depth.shape!=rgb.shape[:2] or K.shape!=(3,3):raise ValueError('cached RGB-D/intrinsics shape mismatch')
    if serial is not None and serial!=38968158:raise ValueError('wrong cached camera serial')
    h,w=rgb.shape[:2]
    if any(box[2]>w or box[3]>h for box in boxes.values()):raise ValueError('manual box outside cached frame')
    return rgb,depth,K,{'camera_serial':serial,'cached_timestamps':timestamps,'cached_rgb_shape':list(rgb.shape),
                      'observation_cached':True,'live_admission':False,'timestamps_rebased':False}


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--rgbd',type=Path,required=True);p.add_argument('--boxes',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path)
    a=p.parse_args(argv);boxes=validate_boxes(a.boxes)
    if a.output.exists():raise FileExistsError('fresh read-only output required')
    rgb,depth,K,cached=read_cached_rgbd(a.rgbd,boxes)
    package=a.package.resolve();sys.path.insert(0,str(package));here=Path(__file__).resolve().parent
    cli=module(package/'tools/task_graph/xtrainer_portable.py','_readonly_portable_cli')
    framing=module(here/'real_feedback_stream.py','_readonly_framed_feedback')
    planner=module(here/'object_relative_entry.py','_readonly_object_entry')
    from prove.xtrainer.io import load_calibration
    from prove.xtrainer.perception import role_geometry
    cfg_path=a.config or package/'local/CONFIG.json';cfg=json.loads(cfg_path.read_text())
    Tcamera,Tright=load_calibration(cfg['left_calibration'],cfg['right_calibration'])
    bundle=cli.portable_check(package,cfg)[2]
    a.output.mkdir(parents=True,exist_ok=False)
    log={'schema':'graph-self-adapter.readonly-entry-check.v1','NoMotion':True,'hardware_motion_commands':0,
         'camera_opened':False,'cached_observation':cached,'manual_boxes':boxes,'converged':False,
         'fresh_live_object_admission':False,'partial_FK_log':[],
         'bindings':{str(path.resolve()):cli.digest(path) for path in (a.rgbd,cfg_path,Path(__file__),here/'real_feedback_stream.py',here/'object_relative_entry.py')},
         'limitation':'cached object position is historical; candidate cannot authorize live motion'}
    def save():cli.write(a.output/'RESULT.json',log)
    save();streams=[];rpc=None
    try:
        # SAM uses only this explicitly supplied recorded frame. No camera object is constructed.
        config={**cfg['vision'],'role_boxes':boxes};vision=cli.LiveVision(None,Tcamera,config)
        tracker=vision._tracker();masks=tracker.initialize(rgb,boxes)
        roles=role_geometry(depth,masks,K,Tcamera)
        log['cached_roles_shared_m']={k:None if v is None else v.tolist() for k,v in roles.items()};save()
        if roles.get('red_mug') is None:raise ValueError('cached red-cup metric geometry unknown')
        for arm in ('left','right'):
            stream=framing.FramedFeedbackStream(cfg['robot_ips'][arm],cfg['sdk_file']);streams.append(stream);stream.connect()
        samples=[s.snapshot() for s in streams]
        cli.SafetyLayer(cfg['safety']).samples(samples)
        log['actual_q_rad']=[s.q.tolist() for s in samples];log['feedback_local_stamps']=[s.stamp for s in samples]
        log['actual_TCP_native_m']=[s.tcp.tolist() for s in samples]
        log['framing']=[s.framing_diagnostics() for s in streams];save()
        class FKOnlyRPC(cli.NativeRPC):
            def call(self,method,**kwargs):
                if method!='get_fk':raise cli.SafetyError('read-only entry RPC permits get_fk only')
                return super().call(method,**kwargs)
            def command(self,*args,**kwargs):raise cli.SafetyError('motion forbidden in read-only entry check')
        rpc=FKOnlyRPC(timeout_ms=1000);rpc.connect()
        started=time.monotonic();deadline=started+80.
        def fk(q,g,arm):
            if time.monotonic()>=deadline:raise TimeoutError('read-only planner 80-second wall budget exhausted')
            row={'arm':arm,'q_rad':np.asarray(q).tolist(),'started_elapsed_sec':time.monotonic()-started}
            log['partial_FK_log'].append(row);save()
            try:
                xyz,rotation=rpc.fk(q,g,arm)
                row.update(TCP_native_m=xyz.tolist(),rotation_native=rotation.tolist(),completed_elapsed_sec=time.monotonic()-started)
                save()
                if time.monotonic()>=deadline:raise TimeoutError('read-only planner 80-second wall budget exhausted')
                return xyz,rotation
            except Exception as exc:row['error']=str(exc);save();raise
        class FKAdapter:
            def fk(self,*args):return fk(*args)
        g=bundle['initial_g'];log['gripper_command_for_FK_is_expected_not_measurement']=g.tolist()
        comparison=cli.feedback_fk_check(FKAdapter(),samples,g);log['native_feedback_FK_check']=comparison;save()
        if not comparison['passed']:raise cli.SafetyError('actual native TCP/FK comparison failed')
        ref=planner.first_red_grasp_reference(bundle)
        log['reference_entry']={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in ref.items()};save()
        plan=planner.plan_first_red_grasp_entry(actual_q=np.stack([s.q for s in samples]),
            current_object_shared_m=roles['red_mug'],reference_q=ref['reference_q'],
            reference_object_shared_m=ref['reference_object_shared_m'],native_fk=fk,gripper_command=g,
            T_left_from_right=Tright,joint_min_rad=cfg['safety']['joint_min_rad'],joint_max_rad=cfg['safety']['joint_max_rad'],
            workspace_min_m=cfg['safety']['workspace_min_m'],workspace_max_m=cfg['safety']['workspace_max_m'],
            max_iterations=120,max_joint_step_rad=.01,observed_timestamp_sec=None)
        log.update(candidate_plan=plan,converged=plan['converged'],planner_wall_sec=time.monotonic()-started)
        save();return log
    except Exception as exc:
        log.update(error=str(exc),converged=False,partial_log_preserved=True);save();return log
    finally:
        if rpc:rpc.close()
        for stream in streams:stream.close()

if __name__=='__main__':main()
