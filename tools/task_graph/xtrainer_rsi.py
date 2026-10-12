#!/usr/bin/env python3
"""Default file-only check; mock, explicit read-only shadow, or frozen execute."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from prove.xtrainer.io import NativeRPC,FeedbackStream,DashboardStop,load_calibration,SafetyError,SafetyLayer,feedback_fk_check
from prove.xtrainer.runtime import load_bundle,Executor
from prove.xtrainer.agent import RealRootJudge,digest,write
from prove.xtrainer.perception import ZEDCamera,LiveVision
from prove.xtrainer.mock import MockClock,MockRobot,MockVision,MockJudge,MockStop

EXTERNAL=Path('/home/zheng/github/RoboTwin_bin_pick')
SOURCES=('prove/xtrainer/io.py','prove/xtrainer/perception.py','prove/xtrainer/agent.py',
         'prove/xtrainer/runtime.py','prove/xtrainer/mock.py','tools/task_graph/xtrainer_rsi.py')


def nominal_envelopes(bundle,cfg):
    dt=cfg['command_dt_sec'];scale=cfg['hardware_reference_speed_scale']
    if not 0<scale<=1:raise SafetyError('hardware reference scale must be (0,1]')
    rows=[]
    for stage in bundle['graph']['stages']:
        start=stage['reference']['start_time_sec'];end=stage['reference']['end_time_sec']
        variants={}
        for name,factor in (('original',1.),('hardware_scaled',scale)):
            increment=dt*stage.get('reference_speed_scale',1.)*factor
            times=np.r_[np.arange(start,end,increment),end]
            q=bundle['qcurve'](times);peak=float(np.max(abs(np.diff(q,axis=0)))) if len(q)>1 else 0.
            variants[name]={'max_nominal_step_rad':peak,'max_nominal_velocity_rad_sec':peak/dt,
                'passed':peak<=cfg['safety']['max_step_rad']+1e-9 and peak/dt<=cfg['safety']['max_velocity_rad_sec']+1e-9}
        rows.append({'stage_id':stage['stage_id'],**variants})
    return rows


def check(config_path,task):
    cfg=json.loads(config_path.read_text());bundle=load_bundle(task)
    left=Path(cfg['left_calibration']);right=Path(cfg['right_calibration']);Tcamera,Tright=load_calibration(left,right)
    required=[*bundle['paths'].values(),*[REPO/p for p in SOURCES],config_path,left,right,
              Path(cfg['sdk_file']),Path(cfg['vision']['sam_utility'])]
    required += [EXTERNAL/'include/xtrainer_clover/dobot_control/robots/robot_node.py',
                 EXTERNAL/'include/xtrainer_clover/dobot_control/robots/robot.py',
                 EXTERNAL/'include/xtrainer_clover/dobot_control/robots/dobot.py']
    sam_root=Path(cfg['vision']['sam_root'])
    for relative in ('sam2/build_sam.py','sam2/sam2_camera_predictor.py','configs/'+cfg['vision']['sam_config']):
        candidate=sam_root/relative
        if candidate.exists():required.append(candidate)
    optional=Path(cfg['vision']['sam_checkpoint'])
    if optional.exists():required.append(optional)
    report={'schema':'graph-self-adapter.xtrainer-check.v1','task':task,'hardware_connections':0,
            'hardware_motion_commands':0,'graph_stages':len(bundle['graph']['stages']),
            'model_input_dim':bundle['skill'].model.input_dim,'fixed_z':True,
            'dependencies':{},'required_bindings':[{'path':str(p.resolve()),'sha256':digest(p)} for p in required],
            'T_left_from_right':Tright.tolist(),'missing_aperture':'actual aperture unavailable: None; current visual Root judgment required',
            'hardware_ready':False,'limitations':['no measured gripper calibration','no collision planner/scene clearance validation',
              'no live camera/SAM/feedback/driver validation','requires one runtime environment with all dependencies',
              'proposed joint/workspace limits must be reviewed against actual hardware and workspace']}
    for name in ('numpy','scipy','PIL','yaml','zmq','torch','pyzed'):
        report['dependencies'][name]=importlib.util.find_spec(name) is not None
    report['nominal_joint_envelopes']=nominal_envelopes(bundle,cfg)
    report['nominal_plan_within_joint_step_velocity_limits']=all(row['hardware_scaled']['passed'] for row in report['nominal_joint_envelopes'])
    cfg['T_left_from_right']=Tright.tolist();cfg['T_left_from_camera']=Tcamera.tolist()
    return report,cfg,bundle


def validate_authorization(scope_path,freeze_path,report,config_path,*,mode,task,output):
    if scope_path is None or freeze_path is None:raise SafetyError('explicit scope and hash freeze required before any connection')
    scope=json.loads(scope_path.read_text())
    permission='hardware_motion_commands_authorized' if mode=='execute' else 'read_only_connections_authorized'
    if scope.get(permission) is not True:raise SafetyError(permission+' is not authorized; no connection made')
    if mode=='execute' and scope.get('maximum_hardware_motion_episodes',0)<1:raise SafetyError('hardware episode scope is zero')
    freeze=json.loads(freeze_path.read_text())
    if freeze.get('schema')!='graph-self-adapter.xtrainer-freeze.v1' or freeze.get('scope_sha256')!=digest(scope_path):
        raise SafetyError('hardware scope/freeze mismatch')
    bound={str(Path(r['path']).resolve()):r['sha256'] for r in freeze['bindings']}
    for row in report['required_bindings']+[{ 'path':str(scope_path.resolve()),'sha256':digest(scope_path)}]:
        if bound.get(row['path'])!=row['sha256']:raise SafetyError('missing/mismatched hardware binding '+row['path'])
    for path,sha in bound.items():
        if digest(path)!=sha:raise SafetyError('changed hardware source/calibration/model binding '+path)
    if not any(job.get('task')==task and job.get('mode')==mode and Path(job['output']).resolve()==output.resolve() for job in freeze['jobs']):
        raise SafetyError('exact task/mode/output not frozen')
    cfg=json.loads(config_path.read_text())
    if mode=='execute' and not report['nominal_plan_within_joint_step_velocity_limits']:raise SafetyError('nominal hardware plan exceeds step/velocity limits')
    if mode=='execute' and cfg.get('hardware_safety_reviewed') is not True:raise SafetyError('joint/workspace/collision risk review not declared')


class LiveRobot:
    def __init__(self,rpc,streams):self.rpc=rpc;self.streams=streams
    @property
    def commands(self):return self.rpc.commands
    def samples(self):return [stream.snapshot() for stream in self.streams]
    def fk(self,*args):return self.rpc.fk(*args)
    def command(self,*args):return self.rpc.command(*args)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group();group.add_argument('--check',action='store_true');group.add_argument('--mock',action='store_true')
    group.add_argument('--shadow',action='store_true');group.add_argument('--execute',action='store_true')
    p.add_argument('--task',default='put_mug_on_box',choices=('put_mug_on_box','put_knife_board_new'))
    p.add_argument('--config',type=Path,default=REPO/'task_packages/xtrainer_v01/CONFIG_PROPOSED.json')
    p.add_argument('--scope',type=Path);p.add_argument('--execution-freeze',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args(argv);report,cfg,bundle=check(a.config,a.task)
    if not (a.mock or a.shadow or a.execute):
        if a.output:write(a.output/'CHECK.json',report)
        print(json.dumps(report,indent=2));return report
    if a.output is None or a.output.exists():raise SafetyError('fresh explicit output required')
    if a.mock:
        clock=MockClock();robot=MockRobot(bundle,clock);vision=MockVision(robot,clock)
        vision.set_context(bundle['graph']['stages'][0],0.,bundle)
        judge=MockJudge(vision,robot);stops=[MockStop(),MockStop()]
        cfg.update(mode='mock',T_left_from_right=np.eye(4).tolist(),initial_gripper_state_confirmed=True,
                   known_initial_gripper_commands=bundle['initial_g'].tolist())
        # Deliberately permissive fake-cell limits never carry into hardware configuration.
        cfg['safety']={**cfg['safety'],'joint_min_rad':[-8]*6,'joint_max_rad':[8]*6,'workspace_min_m':[-8]*3,
                       'workspace_max_m':[8]*3,'max_step_rad':cfg['safety']['max_step_rad'],'max_velocity_rad_sec':cfg['safety']['max_velocity_rad_sec'],'max_tracking_rad':.03}
        runtime=Executor(bundle,robot,vision,judge,stops,cfg,a.output,clock=clock,sleep=clock.sleep)
        result=runtime.run();write(a.output/'CHECK.json',report)
        print(json.dumps({k:v for k,v in result.items() if k not in ('trace','Root_decisions','bindings')},indent=2));return result
    mode='execute' if a.execute else 'shadow'
    validate_authorization(a.scope,a.execution_freeze,report,a.config,mode=mode,task=a.task,output=a.output)
    if not all(report['dependencies'].values()):raise SafetyError('live dependency gap; no connection made')
    streams=[FeedbackStream(cfg['robot_ips'][arm],cfg['sdk_file']) for arm in ('left','right')]
    rpc=NativeRPC();stops=[DashboardStop(cfg['robot_ips'][arm]) for arm in ('left','right')]
    camera=ZEDCamera();vision=LiveVision(camera,np.asarray(cfg['T_left_from_camera']),cfg['vision'])
    try:
        for stream in streams:stream.connect()
        vision.connect()
        deadline=time.monotonic()+3
        while any(stream.latest is None for stream in streams):
            if time.monotonic()>deadline:raise SafetyError('actual feedback startup timeout')
            time.sleep(.01)
        if a.shadow:
            rpc.connect();obs=vision.capture();samples=[s.snapshot() for s in streams]
            SafetyLayer(cfg['safety']).samples(samples)
            comparison=feedback_fk_check(rpc,samples,cfg['known_initial_gripper_commands'])
            write(a.output/'SHADOW.json',{'schema':'graph-self-adapter.xtrainer-shadow.v1','hardware_motion_commands':0,
                  'roles':{k:None if v is None else v.tolist() for k,v in obs.roles.items()},
                  'actual_q':[s.q.tolist() for s in samples],'actual_aperture_m':[s.aperture for s in samples],
                  'native_feedback_FK_check':comparison})
            if not comparison['passed']:raise SafetyError('native actual TCP/FK mismatch; inspect SHADOW.json')
            return {'shadow':True,'hardware_motion_commands':0}
        for stop in stops:stop.connect()
        rpc.connect();cfg['mode']='execute'
        comparison=feedback_fk_check(rpc,[s.snapshot() for s in streams],cfg['known_initial_gripper_commands'])
        if not comparison['passed']:
            write(a.output/'FK_REJECTED.json',comparison)
            for stop in stops:stop.stop()
            raise SafetyError('actual TCP/FK mismatch before graph; both robot stops sent')
        judge=RealRootJudge(a.output/'REAL_AGENT_HANDOFFS',task=a.task,graph_sha=digest(bundle['paths']['graph']),identity=cfg['root_identity'])
        return Executor(bundle,LiveRobot(rpc,streams),vision,judge,stops,cfg,a.output).run()
    finally:
        rpc.close();vision.close()
        for stream in streams:stream.close()
        for stop in stops:stop.close()

if __name__=='__main__':main()
