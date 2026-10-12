"""V17 tests use device-neutral frozen V16 fixtures; no hardware or simulator."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/deployment'))
import stage_trajectory_v17 as module
from test_stage_trajectory import Clock,Robot,setup

@pytest.mark.parametrize('delay',[.04,.07,.1])
def test_moderate_feedback_io_slows_stage_without_burst_or_accumulated_failure(delay):
    limits,start,plan=setup();clock=Clock();robot=Robot(start,clock);robot.delay=delay;issued=[];original=robot.command
    def command(q,g,flags):issued.append(clock());original(q,g,flags)
    robot.command=command
    result=module.stream_stage(robot,plan,limits,robot.stop,clock=clock,sleep=clock.sleep)
    assert result['completed'] and result['rebase_count']>0 and len(issued)==len(plan.times)>100
    assert min(np.diff(issued))>=1/limits.rate_hz-1e-10
    assert np.array_equal(robot.q[1],start[1]) and not robot.stopped
    assert result['FK_calls_during_streaming']==0 and result['actual_feedback_log']

@pytest.mark.parametrize('failure',['feedback_stall','dispatch_stall','combined_stall','partial_transport','stale','emergency'])
def test_stalls_and_true_faults_stop_and_preserve_attempt_feedback_metrics(failure):
    limits,start,plan=setup();clock=Clock();robot=Robot(start,clock);original=robot.command
    if failure=='feedback_stall':robot.delay=.26
    if failure=='combined_stall':robot.delay=.14
    if failure=='stale':robot.stale=True
    def command(q,g,flags):
        if failure=='partial_transport':raise module.SafetyError('partial transport motion possible')
        original(q,g,flags)
        if failure in ('dispatch_stall','combined_stall'):clock.sleep(.26 if failure=='dispatch_stall' else .14)
    robot.command=command
    with pytest.raises(module.SafetyError) as error:
        module.stream_stage(robot,plan,limits,robot.stop,clock=clock,sleep=clock.sleep,emergency=lambda:failure=='emergency')
    metrics=error.value.stream_metrics
    assert robot.stopped and not metrics['completed'] and metrics['actual_feedback_log']
    if failure=='partial_transport':assert metrics['command_attempts']==1 and metrics['command_acknowledgments']==0 and metrics['actual_dispatch_log'][0]['acknowledged'] is False
    if failure=='combined_stall':assert metrics['command_acknowledgments']==1 # aggregate blocked interval is bounded, too


def test_v17_wrapper_reads_new_stall_parameter_and_preserves_safety(monkeypatch):
    import json
    import trajectory_white_runtime_v17 as runtime
    cfg=json.loads((ROOT/'evaluations/stage_trajectory_v17/CONFIG_PROPOSED.json').read_text())['runtime']
    monkeypatch.setattr(runtime.DirectExecutor,'__init__',lambda self,*a,**k:setattr(self,'cfg',cfg))
    x=runtime.TrajectoryWhiteExecutor()
    assert x.maximum_stall_sec==.25 and x.trajectory_limits.rate_hz==25
    assert x.trajectory_limits.step==.009 and x.trajectory_limits.actual_step_limit==.04
    assert x.trajectory_limits.feedback_velocity==.5 and x.grasp_tolerance==.002
