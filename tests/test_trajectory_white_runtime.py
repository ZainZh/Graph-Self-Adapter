import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/deployment'))
import trajectory_white_runtime as module
from test_stage_trajectory import Clock,Robot,fk

def executor():
    x=module.TrajectoryWhiteExecutor.__new__(module.TrajectoryWhiteExecutor)
    x.stage={'cycle_id':'white_direct','phase':'Approach','stage_id':'white.Approach'}
    x.direct_plan={'pre':(np.array([.02,0,.03]),np.eye(3)),'entry':(np.array([.03,0,.02]),np.eye(3)),'pre_reached':False}
    x.grasp_tolerance=.002;x.phase_goals={};x.stage_timing={};x.segment_reports=[]
    return x

def test_approach_orders_clearance_before_entry_and_grasp_only_tolerance():
    x=executor();goals=x.segment_goals(x.stage,(np.zeros(3),np.eye(3)))
    assert goals==[x.direct_plan['pre'],x.direct_plan['entry']]
    x.stage['phase']='Grasp';assert x._reached((np.zeros(3),np.eye(3)),(np.array([.0015,0,0]),np.eye(3)))
    x.stage['phase']='Lower';assert not x._reached((np.zeros(3),np.eye(3)),(np.array([.0015,0,0]),np.eye(3)))

def test_segment_stream_preserves_intent_budget_passive_and_fresh_boundaries(tmp_path):
    x=executor();clock=Clock();start=np.zeros((2,6));start[1,:3]=.1
    robot=Robot(start,clock);robot.fk=fk;x.robot=robot;x.command_g=np.array([1.,1.]);x.clock=clock;x.sleep=clock.sleep;x.T=np.eye(4)
    x.b={'initial_g':[1.,1.]};x.macro_grasp=None;x.output=tmp_path;x._hardware_attempts=0;x._hardware_acknowledged=0;x.command_index=0
    x.safety=SimpleNamespace(last_target=None,samples=lambda samples:None);x.stops=[robot];x.guard_limits=lambda:None
    captures=[]
    def capture():captures.append(clock());return robot.samples(),SimpleNamespace(stamp=clock())
    x.boundary_capture=capture;x._shared_pose=lambda q,g,arm:fk(q,g,arm)
    x.trajectory_limits=module.Limits([-1]*6,[1]*6,[-1]*3,[1]*3,rate_hz=25,velocity=.3,acceleration=.6,position_tolerance=.001)
    target,progress,record=x.command(x.stage,.1,robot.samples(),None,.1)
    assert progress==1 and x.direct_plan['pre_reached'] and len(captures)==3
    assert x.command_index==len(robot.commands)==x._hardware_attempts==x._hardware_acknowledged
    assert np.array_equal(robot.q[1],start[1]) and (tmp_path/'COMMAND_INTENT.json').exists()
    assert record['segment']['FK_calls_during_streaming']==0

def test_actual_owner_config_mapping_without_device_initialization(monkeypatch):
    import json
    cfg=json.loads((ROOT/'evaluations/stage_trajectory_v16/CONFIG_PROPOSED.json').read_text())['runtime']
    monkeypatch.setattr(module.DirectExecutor,'__init__',lambda self,*a,**k:setattr(self,'cfg',cfg))
    x=module.TrajectoryWhiteExecutor()
    assert x.trajectory_limits.rate_hz==25 and x.trajectory_limits.acceleration==.6
    assert x.trajectory_limits.tracking==cfg['safety']['max_tracking_rad']
    assert x.trajectory_limits.feedback_velocity==cfg['safety']['max_velocity_rad_sec']
    assert x.grasp_tolerance==.002

def test_entry_loader_selects_new_scheduler_and_rejects_invalid_settings(monkeypatch):
    import trajectory_entry as entry
    monkeypatch.setattr(entry,'_original_load',lambda path,name:Path(path).name)
    assert entry.load(Path('direct_white_runtime.py'),'runtime')=='trajectory_white_runtime.py'
    assert entry.load(Path('sim_robot_io.py'),'io')=='sim_robot_io.py'
    with pytest.raises(module.SafetyError,match='positive finite'):
        entry.bindings(ROOT/'unused',{'runtime':{'trajectory':{'rate_hz':0}}},'sim')
