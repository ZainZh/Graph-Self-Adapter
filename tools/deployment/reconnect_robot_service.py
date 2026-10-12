#!/usr/bin/env python3
"""Explicit reconnect of existing enabled robots; suppress constructor actuation.

Default is source/INI check only. Does not kill a service, enable, home, move,
set Tool1 or initialize jaw position. Formal clients may command after serving.
"""
import argparse
import configparser
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import socket
import sys

DASHBOARD_SUPPRESS=('EnableRobot','SpeedFactor','AccJ','SpeedJ','SetTool','Tool','StopDrag')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,default=str));temp.replace(path)


@contextmanager
def suppress_constructor_actions(api,gripper_class,records):
    patches=[]
    def suppress(owner,name,result=None):
        original=getattr(owner,name)
        def blocked(self,*args,**kwargs):records.append({'suppressed':owner.__name__+'.'+name,'args':args,'kwargs':kwargs});return result
        setattr(owner,name,blocked);patches.append((owner,name,original))
    try:
        for name in DASHBOARD_SUPPRESS:suppress(api.DobotApiDashboard,name,'0,{},Suppressed();')
        suppress(gripper_class,'move',(True,0));suppress(gripper_class,'set_torque_limit')
        if hasattr(gripper_class,'set_latency_timer'):suppress(gripper_class,'set_latency_timer')
        # Guard unexpected constructor motor methods as well; restore all after construction.
        for name in ('ServoJ','MovJ','MovL','JointMovJ','Home','ServoP'):
            if hasattr(api.DobotApiMove,name):suppress(api.DobotApiMove,name,'0,{},Suppressed();')
        yield
    finally:
        for owner,name,original in reversed(patches):setattr(owner,name,original)


def check(project):
    project=Path(project).resolve()
    paths=[project/'dobot_control/robots/dobot.py',project/'dobot_control/robots/dobot_api.py',
           project/'dobot_control/robots/robot.py',project/'dobot_control/robots/robot_node.py',
           project/'dobot_control/gripper/dobot_gripper.py',project/'scripts/dobot_config/dobot_settings.ini']
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise ValueError('robot-project must be actual xtrainer_clover root; missing: '+str(missing))
    ini=configparser.ConfigParser();ini.read(paths[-1]);serials=[]
    for section in ('GRIPPER_LEFT','GRIPPER_RIGHT'):
        port=ini.get(section,'port');real=Path(port).resolve()
        serials.append({'role':section,'configured_port':port,'resolved_device':str(real),'exists':Path(port).exists(),
                        'servo_id':ini.getint(section,'id'),'position_limits':ini.get(section,'pos')})
    return {'default_check_only':True,'hardware_connections':0,'hardware_motion_commands':0,'source_bindings':[
        {'path':str(p),'sha256':sha(p)} for p in paths],'serial_configuration':serials}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--robot-project',type=Path,required=True)
    p.add_argument('--serve',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--package',type=Path)
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('fresh reconnect report directory required')
    report=check(a.robot_project);a.output.mkdir(parents=True);write(a.output/'BOOT_REPORT.json',report)
    if not a.serve:print(json.dumps(report,indent=2));return report
    # Bind probe only; no packet/robot control. Root must stop old PID separately.
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
        try:probe.bind(('127.0.0.1',6001))
        except OSError as exc:raise RuntimeError('6001 occupied; no automatic PID killing') from exc
    package=(a.package or Path(__file__).resolve().parents[1]).resolve();sys.path.insert(0,str(package))
    spec=importlib.util.spec_from_file_location('_service_framed_feedback',Path(__file__).with_name('real_feedback_stream.py'))
    framed=importlib.util.module_from_spec(spec);spec.loader.exec_module(framed)
    from prove.xtrainer.io import SafetyError
    project=a.robot_project.resolve();sdk=project/'dobot_control/robots/dobot_api.py'
    streams=[];robots=[];server=None;report.update(default_check_only=False,suppressed_calls=[],fresh_preflight=[])
    try:
        for ip in ('192.168.5.1','192.168.5.2'):
            stream=framed.FramedFeedbackStream(ip,sdk);streams.append(stream);stream.connect();sample=stream.snapshot()
            if (not sample.enabled or sample.mode!=5 or sample.error or sample.running or sample.user!=0 or sample.tool!=1):
                raise SafetyError('existing robot must already be enabled stationary mode5 User0 Tool1')
            import numpy as np
            if not np.allclose(sample.user_offset,np.zeros(6),atol=1e-5) or not np.allclose(sample.tool_offset,[0,0,197,0,0,0],atol=.01):
                raise SafetyError('existing User/Tool offsets differ; reconnect will not change them')
            report['fresh_preflight'].append({'ip':ip,'mode':sample.mode,'enabled':sample.enabled,'error':sample.error,'q_rad':sample.q.tolist(),
                                              'TCP_native_m':sample.tcp.tolist(),'framing':stream.framing_diagnostics()})
        write(a.output/'BOOT_REPORT.json',report)
        sys.path.insert(0,str(project))
        api=importlib.import_module('dobot_control.robots.dobot_api')
        driver=importlib.import_module('dobot_control.robots.dobot');robot_module=importlib.import_module('dobot_control.robots.robot')
        node=importlib.import_module('dobot_control.robots.robot_node');gripper=importlib.import_module('dobot_control.gripper.dobot_gripper')
        with suppress_constructor_actions(api,gripper.DobotGripper,report['suppressed_calls']):
            for number,ip in enumerate(('192.168.5.1','192.168.5.2'),1):robots.append(driver.DobotRobot(robot_ip=ip,robot_number=number))
        report['gripper_ping']=[]
        for robot in robots:
            result=robot.gripper.ping()
            if result!=0:raise RuntimeError('actual reconnected gripper ping failed: '+str(result))
            port=robot.gripper.port
            report['gripper_ping'].append({'ip':robot.robot_ip,'configured_port':port,'resolved_device':str(Path(port).resolve()),
                                           'servo_id':robot.gripper._servo_id,'communication_result':result,'ping_passed':True})
        for stream in streams:
            sample=stream.snapshot()
            if not sample.enabled or sample.mode!=5 or sample.error:raise SafetyError('robot readiness changed during reconnect')
        server=node.ZMQServerRobot(robot=robot_module.BimanualRobot(robots[0],robots[1]),host='127.0.0.1',port=6001)
        report.update(server_bound=True,constructor_motion_suppressed=True,client_commands_restored=True,
                      boot_motion_commands=0,scope='reconnect/ping only; subsequent existing client commands are normal')
        write(a.output/'BOOT_REPORT.json',report);server.serve()
    except Exception as exc:
        report['error']=str(exc);write(a.output/'BOOT_REPORT.json',report);raise
    finally:
        for stream in streams:stream.close()
        if server:server.stop()
        for robot in robots:
            if hasattr(robot,'_stop_thread'):robot._stop_thread.set()
            if hasattr(robot,'gripper'):robot.gripper.disconnect()

if __name__=='__main__':main()
