#!/usr/bin/env python3
"""Portable check/mock/explicit shadow/execute; never launch a robot service."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
PACKAGE=Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
from prove.xtrainer.portable import portable_check,rebind_config,TASK
from prove.xtrainer.agent import digest,write,RealRootJudge
from prove.xtrainer.runtime import Executor
from prove.xtrainer.io import SafetyError,NativeRPC,FeedbackStream,DashboardStop,SafetyLayer,feedback_fk_check
from prove.xtrainer.perception import ZEDCamera,LiveVision
from prove.xtrainer.mock import MockClock,MockRobot,MockVision,MockJudge,MockStop


def freeze_bindings(report,config_path,scope_path):
    bindings={**report['package_bindings'],**report['external_bindings'],
              str(config_path.resolve()):digest(config_path),str(scope_path.resolve()):digest(scope_path)}
    return [{'path':p,'sha256':sha} for p,sha in sorted(bindings.items())]


def prepare_freeze(report,config_path,scope_path,output,mode,identity):
    # Binding preparation is not permission, safety review or initial-state confirmation.
    if report['missing_external_files']:raise SafetyError('external files missing; cannot prepare exact freeze')
    scope=json.loads(scope_path.read_text())
    return {'schema':'graph-self-adapter.xtrainer-portable-freeze.v1','status':'result_free_prepared',
        'scope_sha256':digest(scope_path),'bindings':freeze_bindings(report,config_path,scope_path),
        'jobs':[{'task':TASK,'mode':mode,'output':str(output.resolve()),'root_identity':identity}],
        'authorization_not_created':True,'scope_declared_permissions':{k:scope.get(k) for k in
            ('read_only_connections_authorized','hardware_motion_commands_authorized','maximum_hardware_motion_episodes')}}


def validate_execute(report,cfg,config_path,scope_path,freeze_path,*,mode,output):
    if not scope_path or not freeze_path:raise SafetyError('explicit portable scope/freeze required before connections')
    scope=json.loads(scope_path.read_text());permission='hardware_motion_commands_authorized' if mode=='execute' else 'read_only_connections_authorized'
    if scope.get(permission) is not True:raise SafetyError('scope does not authorize '+permission)
    if mode=='execute' and (scope.get('maximum_hardware_motion_episodes',0)<1 or cfg['hardware_safety_reviewed'] is not True or cfg['initial_gripper_state_confirmed'] is not True):
        raise SafetyError('motion scope, physical safety or initial grip confirmation absent')
    if report['missing_external_files'] or not all(report['dependencies'].values()) or not report['nominal_plan_within_limits']:
        raise SafetyError('portable external dependency/nominal limit gap')
    freeze=json.loads(freeze_path.read_text())
    if freeze.get('schema')!='graph-self-adapter.xtrainer-portable-freeze.v1' or freeze.get('status')!='result_free_frozen' or freeze.get('scope_sha256')!=digest(scope_path):
        raise SafetyError('portable result-free freeze mismatch; prepared is not frozen')
    expected={row['path']:row['sha256'] for row in freeze_bindings(report,config_path,scope_path)}
    bound={str(Path(row['path']).resolve()):row['sha256'] for row in freeze['bindings']}
    if not expected.items()<=bound.items() or any(digest(p)!=sha for p,sha in bound.items()):raise SafetyError('portable package/config/external source binding mismatch')
    if freeze.get('jobs')!=[{'task':TASK,'mode':mode,'output':str(output.resolve()),'root_identity':cfg['root_identity']}]:
        raise SafetyError('portable task/mode/output/Root identity job mismatch')


class LiveRobot:
    def __init__(self,rpc,streams):self.rpc=rpc;self.streams=streams
    @property
    def commands(self):return self.rpc.commands
    def samples(self):return [s.snapshot() for s in self.streams]
    def fk(self,*a):return self.rpc.fk(*a)
    def command(self,*a):return self.rpc.command(*a)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,default=PACKAGE)
    p.add_argument('--config',type=Path);p.add_argument('--robot-project');p.add_argument('--sam-root');p.add_argument('--sam-checkpoint')
    group=p.add_mutually_exclusive_group();group.add_argument('--check',action='store_true');group.add_argument('--mock',action='store_true')
    group.add_argument('--shadow',action='store_true');group.add_argument('--execute',action='store_true');group.add_argument('--prepare-freeze',action='store_true')
    p.add_argument('--scope',type=Path);p.add_argument('--execution-freeze',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--freeze-mode',choices=('shadow','execute'),default='shadow');a=p.parse_args(argv)
    config_path=a.config or a.package/'CONFIG_PROPOSED.json'
    cfg=rebind_config(json.loads(config_path.read_text()),robot_project=a.robot_project,sam_root=a.sam_root,sam_checkpoint=a.sam_checkpoint)
    report,cfg,bundle=portable_check(a.package,cfg)
    if a.prepare_freeze:
        if not a.scope or not a.output:raise SafetyError('prepare-freeze needs explicit scope and future run output')
        result=prepare_freeze(report,config_path,a.scope,a.output,a.freeze_mode,cfg['root_identity']);print(json.dumps(result,indent=2));return result
    if not (a.mock or a.shadow or a.execute):
        if a.output:write(a.output/'CHECK.json',report)
        print(json.dumps(report,indent=2));return report
    if not a.output or a.output.exists():raise SafetyError('fresh explicit output required')
    if a.mock:
        clock=MockClock();robot=MockRobot(bundle,clock);vision=MockVision(robot,clock);vision.set_context(bundle['graph']['stages'][0],0,bundle)
        cfg.update(mode='mock',T_left_from_right=np.eye(4).tolist(),known_initial_gripper_commands=bundle['initial_g'].tolist(),initial_gripper_state_confirmed=True)
        cfg['safety'].update(joint_min_rad=[-8]*6,joint_max_rad=[8]*6,workspace_min_m=[-8]*3,workspace_max_m=[8]*3,max_tracking_rad=.03)
        result=Executor(bundle,robot,vision,MockJudge(vision,robot),[MockStop(),MockStop()],cfg,a.output,clock=clock,sleep=clock.sleep).run()
        print(json.dumps({'graph_completed':result['graph_completed'],'hardware_motion_commands':0,'mock_commands':result['mock_command_count']}));return result
    mode='execute' if a.execute else 'shadow';validate_execute(report,cfg,config_path,a.scope,a.execution_freeze,mode=mode,output=a.output)
    streams=[FeedbackStream(cfg['robot_ips'][arm],cfg['sdk_file']) for arm in ('left','right')]
    rpc=NativeRPC();stops=[DashboardStop(cfg['robot_ips'][arm]) for arm in ('left','right')]
    vision=LiveVision(ZEDCamera(),np.asarray(cfg['T_left_from_camera']),cfg['vision'])
    try:
        for s in streams:s.connect()
        vision.connect();deadline=time.monotonic()+3
        while any(s.latest is None for s in streams):
            if time.monotonic()>deadline:raise SafetyError('actual feedback startup timeout')
            time.sleep(.01)
        if a.execute:
            for stop in stops:stop.connect()
        rpc.connect();samples=[s.snapshot() for s in streams];SafetyLayer(cfg['safety']).samples(samples)
        comparison=feedback_fk_check(rpc,samples,cfg['known_initial_gripper_commands'])
        if not comparison['passed']:
            write(a.output/'FK_REJECTED.json',comparison)
            if a.execute:
                for stop in stops:stop.stop()
            raise SafetyError('actual native TCP/FK mismatch')
        if a.shadow:
            obs=vision.capture();write(a.output/'SHADOW.json',{'schema':'graph-self-adapter.xtrainer-portable-shadow.v1',
                'hardware_motion_commands':0,'native_feedback_FK_check':comparison,'actual_aperture_m':[s.aperture for s in samples],
                'roles':{k:None if v is None else v.tolist() for k,v in obs.roles.items()}});return {'shadow':True,'hardware_motion_commands':0}
        cfg['mode']='execute';judge=RealRootJudge(a.output/'REAL_AGENT_HANDOFFS',task=TASK,graph_sha=digest(bundle['paths']['graph']),identity=cfg['root_identity'])
        return Executor(bundle,LiveRobot(rpc,streams),vision,judge,stops,cfg,a.output).run()
    finally:
        rpc.close();vision.close()
        for s in streams:s.close()
        for stop in stops:stop.close()

if __name__=='__main__':main()
