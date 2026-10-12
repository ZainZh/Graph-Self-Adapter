"""Device-neutral stage planning then fixed-rate streaming; no SAM/Agent in servo loop.

NativeRPC/SimRobot satisfy fk(q2x6,g2,arm), samples(), command(q,g,flags).
This is host scheduling, not a native SDK queue or collision planner.
"""
from dataclasses import dataclass,field
from contextlib import contextmanager
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp
from prove.xtrainer.io import SafetyError,array


@dataclass
class Limits:
    joint_min: object
    joint_max: object
    workspace_min: object
    workspace_max: object
    velocity: float=.5
    acceleration: float=1.
    step: float=.01
    rate_hz: float=50.
    feedback_age: float=.25
    tracking: float=.05
    start_drift: float=.01
    position_tolerance: float=.002
    rotation_tolerance: float=.02
    settle_seconds: float=2.
    maximum_lateness: float=.04
    feedback_velocity: float | None=None
    actual_step_limit: float | None=None
    planning_seconds: float=80.


@dataclass
class StageTiming:
    durations: dict=field(default_factory=dict)
    counts: dict=field(default_factory=dict)
    @contextmanager
    def measure(self,name,clock=time.monotonic):
        start=clock()
        try:yield
        finally:self.durations[name]=self.durations.get(name,0.)+clock()-start
    def count(self,name,n=1):self.counts[name]=self.counts.get(name,0)+n
    def report(self):return {'seconds':dict(self.durations),'counts':dict(self.counts)}


@dataclass
class Trajectory:
    times: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray
    tcp: np.ndarray
    gripper: np.ndarray
    arm: str
    goal: tuple
    start_q: np.ndarray
    T_left_from_right: np.ndarray
    planning: dict


def _pose(position,rotation):
    p=array(position,(3,));R=array(rotation,(3,3))
    if not np.allclose(R.T@R,np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(R),1.,atol=1e-5):raise SafetyError('invalid rotation')
    return p,R


def _shared_pose(fk,q,g,arm,T):
    p,R=_pose(*fk(q,g,arm))
    return ((T@np.r_[p,1])[:3],T[:3,:3]@R) if arm=='right' else (p,R)


def _workspace(p,limits):
    if np.any(p<np.asarray(limits.workspace_min)) or np.any(p>np.asarray(limits.workspace_max)):raise SafetyError('TCP workspace limit')


def _joint(q,limits):
    if np.any(q<np.asarray(limits.joint_min)) or np.any(q>np.asarray(limits.joint_max)):raise SafetyError('joint position limit')


def plan_stage(fk,start_q,goals,gripper,arm,limits,*,T_left_from_right=None,cartesian_step=.01,rotation_step=.05,ik_iterations=120,clock=time.monotonic):
    """Actual-q seed, ordered caller-specified clearance poses; all FK before dispatch."""
    if arm not in ('left','right') or not goals:raise SafetyError('actor and at least one declared goal required')
    if min(limits.velocity,limits.acceleration,limits.step,limits.rate_hz,limits.planning_seconds)<=0:raise SafetyError('positive trajectory limits required')
    started=clock();timing=StageTiming();q=array(start_q,(2,6)).copy();g=array(gripper,(2,)).copy();_joint(q,limits)
    if np.any(g<0) or np.any(g>1):raise SafetyError('gripper command range')
    T=np.eye(4) if T_left_from_right is None else array(T_left_from_right,(4,4));index=0 if arm=='left' else 1
    def query(q):
        if clock()-started>limits.planning_seconds:raise SafetyError('offline planning wall budget exhausted')
        timing.count('fk_calls')
        with timing.measure('fk',clock):return _shared_pose(fk,q,g,arm,T)
    timing.count('fk_calls')
    with timing.measure('fk',clock):passive=_shared_pose(fk,q,g,'right' if arm=='left' else 'left',T)
    _workspace(passive[0],limits)
    path=[q.copy()];pose=query(q);_workspace(pose[0],limits)
    for declared in goals:
        goal=_pose(*declared);_workspace(goal[0],limits)
        length=np.linalg.norm(goal[0]-pose[0]);angle=Rotation.from_matrix(goal[1]@pose[1].T).magnitude()
        n=max(1,int(np.ceil(max(length/cartesian_step,angle/rotation_step))))
        rotations=Slerp([0,1],Rotation.from_matrix(np.stack([pose[1],goal[1]])));origin=pose
        for u in np.linspace(0,1,n+1)[1:]:
            target=(origin[0]+u*(goal[0]-origin[0]),rotations(u).as_matrix());converged=False
            for _ in range(ik_iterations):
                timing.count('ik_iterations');current=query(q);error=np.r_[target[0]-current[0],Rotation.from_matrix(target[1]@current[1].T).as_rotvec()]
                if np.linalg.norm(error[:3])<=limits.position_tolerance*.25 and np.linalg.norm(error[3:])<=limits.rotation_tolerance*.25:converged=True;break
                J=np.zeros((6,6));eps=1e-4
                for j in range(6):
                    probe=q.copy();probe[index,j]+=eps;pp,RR=query(probe);J[:3,j]=(pp-current[0])/eps;J[3:,j]=Rotation.from_matrix(RR@current[1].T).as_rotvec()/eps
                delta=J.T@np.linalg.solve(J@J.T+.015**2*np.eye(6),error);delta*=min(1.,.08/max(np.max(abs(delta)),1e-12))
                q[index]+=delta;_joint(q,limits)
            if not converged:raise SafetyError('offline bounded IK did not converge')
            _workspace(current[0],limits);path.append(q.copy())
        pose=query(q)
    path=np.asarray(path);distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path[:,index],axis=0),axis=1))];keep=np.r_[True,np.diff(distance)>1e-10];path=path[keep];distance=distance[keep]
    if len(path)==1:path=np.concatenate([path,path]);distance=np.array([0.,1.])
    u=distance/distance[-1];spline=CubicSpline(u,path,axis=0,bc_type=((1,np.zeros((2,6))),(1,np.zeros((2,6)))))
    # Analytic derivative/position extrema of each cubic interval, not just endpoints.
    extrema=list(u)
    for k,width in enumerate(np.diff(u)):
        for joint in range(6):
            a,b,c=spline.c[:3,k,index,joint]
            if abs(a)>1e-12:
                point=-b/(3*a)
                if 0<point<width:extrema.append(u[k]+point)
            for root in np.roots([3*a,2*b,c]) if abs(a)+abs(b)>1e-12 else []:
                if abs(np.imag(root))<1e-10 and 0<float(np.real(root))<width:extrema.append(u[k]+float(np.real(root)))
    extrema=np.asarray(extrema);_joint(spline(extrema),limits)
    vmax=float(np.max(abs(spline(extrema,1)[:,index])));amax=float(np.max(abs(spline(u,2)[:,index])))
    dt=1/limits.rate_hz;duration=max(dt,vmax/limits.velocity,np.sqrt(amax/limits.acceleration),vmax*dt/limits.step)*1.02
    duration=np.ceil(duration/dt)*dt
    times=np.r_[np.arange(0,duration,dt),duration];parameter=times/duration;targets=spline(parameter);vel=spline(parameter,1)/duration;acc=spline(parameter,2)/duration**2
    _joint(targets,limits);tcp=[]
    with timing.measure('workspace_validation',clock):
        for target in targets:
            p,R=query(target);_workspace(p,limits);tcp.append(p)
    timing.durations['planning_total']=clock()-started
    return Trajectory(times,targets,vel,acc,np.asarray(tcp),g,arm,goal,array(start_q,(2,6)).copy(),T,timing.report())


def stream_stage(robot,trajectory,limits,stop,*,emergency=lambda:False,clock=time.monotonic,sleep=time.sleep,context_timing=None):
    """Synchronous host streamer; fixed deadlines, fresh feedback, zero FK/SAM/judge calls.

    No per-setpoint pose settling. A missed deadline fails rather than bursting
    old setpoints. Caller must explicitly own command transport and stop hook.
    """
    timing=StageTiming();index=0 if trajectory.arm=='left' else 1;last=None;attempts=0;acks=0;lags=[];started=clock();last_dispatch=None
    def feedback():
        timing.count('feedback_calls')
        with timing.measure('feedback',clock):samples=robot.samples()
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
            if lag>limits.maximum_lateness:raise SafetyError('fixed-rate dispatch deadline missed')
            samples,actual,_=feedback()
            if last_dispatch is not None and clock()<last_dispatch+1/limits.rate_hz:sleep(last_dispatch+1/limits.rate_hz-clock())
            if any(not 0<=clock()-sample.stamp<=limits.feedback_age for sample in samples):raise SafetyError('actual feedback expired before dispatch')
            target=trajectory.q[i].copy();target[1-index]=actual[1-index]
            if last is not None and np.max(abs(target[index]-last[index]))>limits.step+1e-9:raise SafetyError('planned joint step exceeded')
            if np.max(abs(target[index]-actual[index]))>limits.tracking:raise SafetyError('pre-dispatch tracking limit')
            if limits.actual_step_limit is not None and np.max(abs(target[index]-actual[index]))>limits.actual_step_limit:raise SafetyError('pre-dispatch actual joint step limit')
            if clock()-(origin+float(t))>limits.maximum_lateness:raise SafetyError('dispatch deadline expired during actual feedback')
            attempts+=1;timing.count('command_attempts')
            last_dispatch=clock()
            with timing.measure('dispatch',clock):robot.command(target,trajectory.gripper,[index==0,index==1])
            acks+=1;last=target.copy()
        settle_start=clock();accepted=False
        while clock()-settle_start<=limits.settle_seconds:
            samples,actual,pose=feedback();position=np.linalg.norm(pose[0]-trajectory.goal[0]);rotation=Rotation.from_matrix(pose[1]@trajectory.goal[1].T).magnitude()
            if position<=limits.position_tolerance and rotation<=limits.rotation_tolerance and np.max(abs(samples[index].qdot))<=.01:accepted=True;break
            sleep(min(.02,1/limits.rate_hz))
        timing.durations['end_settle']=clock()-settle_start
        if not accepted:raise SafetyError('actual stage endpoint not reached in bounded settle')
        return {'completed':True,'command_attempts':attempts,'command_acknowledgments':acks,'planning':trajectory.planning,'streaming':timing.report(),
                'elapsed_seconds':clock()-started,'maximum_dispatch_lateness':max(lags,default=0.),'vision_judge_context':context_timing or {},
                'FK_calls_during_streaming':0,'SAM_calls_during_streaming':0,'native_SDK_queue_used':False,'collision_planning_claim':False}
    except Exception as error:
        try:stop()
        except Exception as stop_error:raise SafetyError(str(error)+'; stop failed: '+str(stop_error)) from error
        raise
