import importlib.util
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from prove.xtrainer.io import RobotSample,SafetyError
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_stage_trajectory_test',ROOT/'tools/deployment/stage_trajectory.py');module=importlib.util.module_from_spec(spec)
import sys;sys.modules[spec.name]=module;spec.loader.exec_module(module)


class Clock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t
    def sleep(self,t):self.t+=t


def fk(q,g,arm):
    row=q[0 if arm=='left' else 1];return row[:3].copy(),Rotation.from_rotvec(row[3:]).as_matrix()


def setup():
    limits=module.Limits([-1]*6,[1]*6,[-1]*3,[1]*3);start=np.zeros((2,6));start[1]=[.1,.1,.1,0,0,0]
    goals=[(np.array([.04,0,.04]),np.eye(3)),(np.array([.08,.02,.06]),Rotation.from_rotvec([0,.1,0]).as_matrix())]
    return limits,start,module.plan_stage(fk,start,goals,[0.,1.],'left',limits)


class Robot:
    def __init__(self,start,clock):self.q=start.copy();self.clock=clock;self.commands=[];self.stopped=False;self.delay=0.;self.fault=False;self.stale=False
    def samples(self):
        self.clock.sleep(self.delay)
        return [RobotSample(self.clock()-(1. if self.stale else 0.),q.copy(),np.zeros(6),q[:3].copy(),Rotation.from_rotvec(q[3:]).as_matrix(),None,'mock_actual_own_feedback',error=self.fault) for q in self.q]
    def command(self,q,g,flags):self.commands.append((q.copy(),flags));self.q[0]=q[0]
    def stop(self):self.stopped=True
    def fk(self,*args):pytest.fail('stream must not call FK')


def test_complete_trajectory_endpoint_velocity_acceleration_and_passive_hold():
    limits,start,plan=setup()
    assert np.linalg.norm(plan.tcp[-1]-plan.goal[0])<limits.position_tolerance
    assert np.max(abs(plan.qdot))<=limits.velocity and np.max(abs(plan.qddot))<=limits.acceleration
    assert np.max(abs(np.diff(plan.q[:,0],axis=0)))<=limits.step
    assert np.array_equal(plan.q[:,1],np.repeat(start[None,1],len(plan.times),axis=0))
    clock=Clock();robot=Robot(start,clock);report=module.stream_stage(robot,plan,limits,robot.stop,clock=clock,sleep=clock.sleep)
    assert report['completed'] and report['FK_calls_during_streaming']==0 and all(flags==[True,False] for _,flags in robot.commands)
    assert np.array_equal(robot.q[1],start[1]) and len(robot.commands)==len(plan.times)


def test_feedback_blocking_past_deadline_stops_without_stale_command():
    limits,start,plan=setup();clock=Clock();robot=Robot(start,clock);robot.delay=.06
    with pytest.raises(SafetyError,match='deadline'):module.stream_stage(robot,plan,limits,robot.stop,clock=clock,sleep=clock.sleep)
    assert robot.stopped and not robot.commands # second deadline check, no catchup burst


@pytest.mark.parametrize('failure',['stale','fault','emergency','passive_workspace'])
def test_real_feedback_and_emergency_fail_closed(failure):
    limits,start,plan=setup();clock=Clock();robot=Robot(start,clock)
    if failure=='stale':robot.stale=True
    if failure=='fault':robot.fault=True
    if failure=='passive_workspace':
        original=robot.samples
        def samples():
            result=original();result[1].tcp[0]=2.;return result
        robot.samples=samples
    with pytest.raises(SafetyError,match='workspace' if failure=='passive_workspace' else None):module.stream_stage(robot,plan,limits,robot.stop,clock=clock,sleep=clock.sleep,emergency=lambda:failure=='emergency')
    assert robot.stopped and not robot.commands


def test_actual_endpoint_must_be_measured_not_assumed_from_sent_trajectory():
    limits,start,plan=setup();limits.tracking=.5;clock=Clock();robot=Robot(start,clock)
    robot.command=lambda q,g,flags:robot.commands.append((q.copy(),flags))
    with pytest.raises(SafetyError,match='endpoint'):module.stream_stage(robot,plan,limits,robot.stop,clock=clock,sleep=clock.sleep)
    assert robot.stopped and len(robot.commands)==len(plan.times) # streamed without per-step reached wait


def test_transport_failure_mid_stream_stops_and_does_not_continue():
    limits,start,plan=setup();clock=Clock();robot=Robot(start,clock);original=robot.command
    def command(q,g,flags):
        if len(robot.commands)==2:raise SafetyError('mock partial transport failure')
        original(q,g,flags)
    robot.command=command
    with pytest.raises(SafetyError,match='transport failure'):module.stream_stage(robot,plan,limits,robot.stop,clock=clock,sleep=clock.sleep)
    assert robot.stopped and len(robot.commands)==2

def test_subthreshold_feedback_delay_cannot_trigger_catchup_burst():
    limits,start,plan=setup();clock=Clock();robot=Robot(start,clock);robot.delay=.025;issued=[];original=robot.command
    def command(q,g,flags):issued.append(clock());original(q,g,flags)
    robot.command=command
    with pytest.raises(SafetyError,match='deadline'):module.stream_stage(robot,plan,limits,robot.stop,clock=clock,sleep=clock.sleep)
    assert robot.stopped and len(issued)>0
    assert len(issued)<2 or min(np.diff(issued))>=1/limits.rate_hz-1e-10

def test_fast_feedback_does_not_accumulate_scheduler_lateness():
    limits,start,plan=setup();clock=Clock();robot=Robot(start,clock);robot.delay=.001;issued=[];original=robot.command
    def command(q,g,flags):issued.append(clock());original(q,g,flags)
    robot.command=command
    report=module.stream_stage(robot,plan,limits,robot.stop,clock=clock,sleep=clock.sleep)
    assert report['completed'] and len(issued)==len(plan.times)>100
    assert min(np.diff(issued))>=1/limits.rate_hz-1e-10
    assert report['maximum_dispatch_lateness']<.002
