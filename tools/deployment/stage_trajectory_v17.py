"""V17 elastic host stream: slow on IO latency; never burst missed setpoints.
Frozen V16 planner is reused. This is not a hard-real-time/native queue.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from stage_trajectory import Limits,StageTiming,Trajectory,plan_stage,_pose,_joint,_workspace
from prove.xtrainer.io import SafetyError,array

def stream_stage(robot,trajectory,limits,stop,*,emergency=lambda:False,clock=time.monotonic,sleep=time.sleep,context_timing=None,maximum_stall_sec=.25):
    """Synchronous host streamer; fixed deadlines, fresh feedback, zero FK/SAM/judge calls.

    No per-setpoint pose settling. Bounded lateness shifts the following timebase rather than bursting
    old setpoints. Caller must explicitly own command transport and stop hook.
    """
    timing=StageTiming();index=0 if trajectory.arm=='left' else 1;last=None;attempts=0;acks=0;lags=[];started=clock();last_dispatch=None;rebases=[];dispatch_log=[];feedback_log=[]
    if not 0<maximum_stall_sec<=.25:raise SafetyError('stall budget must be positive and <=.25s')
    def result(completed,failure=None):
        return {'completed':completed,'failure':failure,'command_attempts':attempts,'command_acknowledgments':acks,'planning':trajectory.planning,'streaming':timing.report(),'elapsed_seconds':clock()-started,'maximum_dispatch_lateness':max(lags,default=0.),'rebase_count':len(rebases),'timebase_rebases':rebases,'actual_dispatch_log':dispatch_log,'actual_feedback_log':feedback_log,'maximum_stall_sec':maximum_stall_sec,'vision_judge_context':context_timing or {},'FK_calls_during_streaming':0,'SAM_calls_during_streaming':0,'native_SDK_queue_used':False,'hard_realtime_claim':False,'collision_planning_claim':False}
    def feedback():
        timing.count('feedback_calls')
        feedback_started=clock()
        with timing.measure('feedback',clock):samples=robot.samples()
        feedback_log.append({'start':feedback_started,'end':clock(),'seconds':clock()-feedback_started})
        if clock()-feedback_started>maximum_stall_sec:raise SafetyError('continuous feedback stall exceeded .25s budget')
        if len(samples)!=2:raise SafetyError('two actual arm samples required')
        for arm_index,sample in enumerate(samples):
            if not 0<=clock()-sample.stamp<=limits.feedback_age:raise SafetyError('stale actual robot feedback')
            array(sample.q,(6,));array(sample.qdot,(6,));_pose(sample.tcp,sample.rotation)
            if sample.error or not sample.enabled or sample.mode not in (5,7):raise SafetyError('robot feedback mode/error fault')
            if sample.user!=0 or sample.tool!=1 or not np.allclose(sample.user_offset,np.zeros(6),atol=1e-5) or not np.allclose(sample.tool_offset,[0,0,197,0,0,0],atol=.01):raise SafetyError('User0/Tool1 calibration mismatch')
            if np.max(abs(sample.qdot))>(limits.feedback_velocity if limits.feedback_velocity is not None else limits.velocity):raise SafetyError('actual joint velocity limit')
            own_tcp=sample.tcp if arm_index==0 else (trajectory.T_left_from_right@np.r_[sample.tcp,1])[:3]
            _workspace(own_tcp,limits)
        actual=np.stack([s.q for s in samples]);_joint(actual,limits)
        p,R=samples[index].tcp,samples[index].rotation
        if index:p=(trajectory.T_left_from_right@np.r_[p,1])[:3];R=trajectory.T_left_from_right[:3,:3]@R
        _workspace(p,limits)
        if last is not None and np.max(abs(actual[index]-last[index]))>limits.tracking:raise SafetyError('actual trajectory tracking limit')
        if emergency():raise SafetyError('emergency stop requested')
        return samples,actual,(p,R)
    try:
        samples,actual,_=feedback()
        if np.max(abs(actual-trajectory.start_q))>limits.start_drift:raise SafetyError('actual start changed during offline planning')
        if np.max(abs(samples[index].qdot))>.01:raise SafetyError('actual trajectory entry is not stationary')
        origin=clock()
        for i,t in enumerate(trajectory.times):
            delay=origin+float(t)-clock()
            if delay>0:sleep(delay)
            lag=clock()-(origin+float(t));lags.append(lag)
            if lag>maximum_stall_sec:raise SafetyError('continuous scheduler stall exceeded budget')
            samples,actual,_=feedback()
            if last_dispatch is not None and clock()<last_dispatch+1/limits.rate_hz:sleep(last_dispatch+1/limits.rate_hz-clock())
            if any(not 0<=clock()-sample.stamp<=limits.feedback_age for sample in samples):raise SafetyError('actual feedback expired before dispatch')
            target=trajectory.q[i].copy();target[1-index]=actual[1-index]
            if last is not None and np.max(abs(target[index]-last[index]))>limits.step+1e-9:raise SafetyError('planned joint step exceeded')
            if np.max(abs(target[index]-actual[index]))>limits.tracking:raise SafetyError('pre-dispatch tracking limit')
            if limits.actual_step_limit is not None and np.max(abs(target[index]-actual[index]))>limits.actual_step_limit:raise SafetyError('pre-dispatch actual joint step limit')
            lag=clock()-(origin+float(t));lags.append(lag)
            if lag>maximum_stall_sec or (last_dispatch is not None and clock()-last_dispatch>maximum_stall_sec):raise SafetyError('continuous pre-dispatch stall exceeded budget')
            if lag>1e-9:
                rebases.append({'sample_index':i,'seconds':lag,'wall_time':clock()});origin+=lag
            attempts+=1;timing.count('command_attempts')
            last_dispatch=clock()
            dispatch_log.append({'sample_index':i,'start':last_dispatch,'acknowledged':False})
            with timing.measure('dispatch',clock):robot.command(target,trajectory.gripper,[index==0,index==1])
            acks+=1;last=target.copy();dispatch_log[-1].update(end=clock(),seconds=clock()-last_dispatch,acknowledged=True)
            if clock()-last_dispatch>maximum_stall_sec:raise SafetyError('continuous transport stall exceeded budget')
        settle_start=clock();accepted=False
        while clock()-settle_start<=limits.settle_seconds:
            samples,actual,pose=feedback();position=np.linalg.norm(pose[0]-trajectory.goal[0]);rotation=Rotation.from_matrix(pose[1]@trajectory.goal[1].T).magnitude()
            if position<=limits.position_tolerance and rotation<=limits.rotation_tolerance and np.max(abs(samples[index].qdot))<=.01:accepted=True;break
            sleep(min(.02,1/limits.rate_hz))
        timing.durations['end_settle']=clock()-settle_start
        if not accepted:raise SafetyError('actual stage endpoint not reached in bounded settle')
        return result(True)
    except Exception as error:
        metrics=result(False,str(error));error.stream_metrics=metrics
        if dispatch_log and 'end' not in dispatch_log[-1]:dispatch_log[-1].update(end=clock(),seconds=clock()-dispatch_log[-1]['start'])
        try:stop()
        except Exception as stop_error:raise SafetyError(str(error)+'; stop failed: '+str(stop_error)) from error
        raise
