import importlib.util
from pathlib import Path
from types import SimpleNamespace

path=Path(__file__).resolve().parents[1]/'tools/deployment/reconnect_robot_service.py'
spec=importlib.util.spec_from_file_location('_reconnect_no_devices',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_constructor_actuation_suppressed_then_original_methods_restored():
    calls=[]
    class Dashboard:pass
    for name in m.DASHBOARD_SUPPRESS:
        setattr(Dashboard,name,lambda self,*a,_name=name,**k:calls.append(_name))
    class Move:
        def ServoJ(self,*a):calls.append('ServoJ')
    class Gripper:
        def move(self,*a):calls.append('gripper.move')
        def set_torque_limit(self,*a):calls.append('torque')
        def set_latency_timer(self):calls.append('latency')
    api=SimpleNamespace(DobotApiDashboard=Dashboard,DobotApiMove=Move);records=[]
    with m.suppress_constructor_actions(api,Gripper,records):
        Dashboard().EnableRobot();Dashboard().SetTool(1,0,0,197,0,0,0);Move().ServoJ(0,0,0,0,0,0)
        Gripper().move(0,100,1);Gripper().move(255,100,1);Gripper().set_torque_limit(300)
    assert not calls and len(records)==6
    Move().ServoJ(1,2,3,4,5,6);Gripper().move(100,100,1)
    assert calls==['ServoJ','gripper.move']
