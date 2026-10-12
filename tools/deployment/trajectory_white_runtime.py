"""V16 engineering scheduler; frozen V15 graph, skill and controller stay unchanged."""
import numpy as np
from direct_white_runtime import DirectExecutor,SharedSAMVision
from direct_white_geometry import tcp_at_handle
from stage_trajectory import Limits,StageTiming,plan_stage,stream_stage
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError

SEGMENTS=('Approach','Lift','Raise','Move','Lower','Retreat')
VARIANT='white_stage_trajectory_v16'

class TrajectoryWhiteExecutor(DirectExecutor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.segment_reports=[];self.stage_timing={}
        config=self.cfg['trajectory'];s=self.cfg['safety']
        self.trajectory_limits=Limits(s['joint_min_rad'],s['joint_max_rad'],s['workspace_min_m'],s['workspace_max_m'],
            velocity=config['velocity_rad_sec'],acceleration=config['acceleration_rad_sec2'],step=.009,rate_hz=config['rate_hz'],
            feedback_age=s['max_age_sec'],tracking=s['max_tracking_rad'],position_tolerance=.001,rotation_tolerance=.02,
            settle_seconds=config['settle_seconds'],maximum_lateness=config['maximum_lateness_sec'],feedback_velocity=s['max_velocity_rad_sec'],actual_step_limit=s['max_step_rad'])
        self.grasp_tolerance=float(config['grasp_position_tolerance_m'])
        if not 0<self.grasp_tolerance<=.002:raise SafetyError('white Grasp tolerance must be positive and <=2mm')
    def _reached(self,goal,pose):
        if self.stage and self.stage['cycle_id']=='white_direct' and self.stage['phase']=='Grasp':
            error=self._error(goal,pose)
            return bool(np.linalg.norm(error[:3])<=self.grasp_tolerance and np.linalg.norm(error[3:])<=.02)
        return super()._reached(goal,pose)
    def timing(self):return self.stage_timing.setdefault(self.stage['stage_id'],StageTiming())
    def boundary_capture(self):
        self.vision.invalidate();previous=self.vision.memo_enabled;self.vision.memo_enabled=False
        try:
            with self.timing().measure('vision_and_feedback_boundary',self.clock):return self.capture()
        finally:self.vision.invalidate();self.vision.memo_enabled=previous
    def semantics(self,*args,**kwargs):
        with self.timing().measure('judge',self.clock):return super().semantics(*args,**kwargs)
    def segment_goals(self,stage,pose):
        plan=self.direct_plan;phase=stage['phase']
        if phase=='Approach':return [plan['pre'],plan['entry']]
        if phase not in self.phase_goals:
            point=pose[0]+pose[1]@plan['pad'];R=plan['entry'][1]
            if phase=='Lift':point[2]=max(point[2],plan['prior']['initial_table_z_m']+plan['prior']['visible_extent_radius_m']+.25*plan['prior']['observed_visible_height_m']);goal=tcp_at_handle(point,R,plan['pad'])
            elif phase=='Raise':point[2]=max(point[2],plan['targets']['transport_handle_height_m']);goal=tcp_at_handle(point,R,plan['pad'])
            elif phase=='Move':point[:2]=plan['targets']['handle_support_goal_m'][:2];point[2]=plan['targets']['transport_handle_height_m'];goal=tcp_at_handle(point,R,plan['pad'])
            elif phase=='Lower':goal=tcp_at_handle(plan['targets']['handle_support_goal_m'],R,plan['pad'])
            elif phase=='Retreat':p=pose[0].copy();p[2]+=plan['prior']['observed_visible_height_m'];goal=(p,pose[1].copy())
            else:raise SafetyError('undeclared trajectory phase')
            self.phase_goals[phase]=goal
        return [self.phase_goals[phase]]
    def command(self,stage,elapsed,samples,obs,dt):
        if stage['cycle_id']!='white_direct' or stage['phase'] not in SEGMENTS:return super().command(stage,elapsed,samples,obs,dt)
        if self.direct_plan is None:self.prepare_white(stage,samples,obs)
        samples,obs=self.boundary_capture();actual=np.stack([s.q for s in samples]);pose=self._shared_pose(actual,self.command_g,'left')
        goals=self.segment_goals(stage,pose);g=self.command_g.copy();phase=stage['phase']
        if phase=='Approach':g[0]=max(g[0],self.b['initial_g'][0])
        elif phase=='Retreat':g[0]=1.
        else:
            if self.macro_grasp is None:raise SafetyError('trajectory certified grasp command missing')
            g[0]=self.macro_grasp['g_command']
        with self.timing().measure('planning',self.clock):
            trajectory=plan_stage(self.robot.fk,actual,goals,g,'left',self.trajectory_limits,T_left_from_right=self.T,clock=self.clock)
        # Planning never renews stamps. Reacquire RGBD and own feedback before dispatch.
        self.boundary_capture();owner=self
        class Dispatch:
            def samples(self):
                owner.guard_limits();samples=owner.robot.samples();owner.safety.samples(samples);return samples
            def command(self,target,gripper,flags):
                owner.guard_limits();owner._hardware_attempts+=1
                write(owner.output/'COMMAND_INTENT.json',{'attempt_count':owner._hardware_attempts,'acknowledged_count':owner._hardware_acknowledged,'stage_id':stage['stage_id'],'q_target_rad':target.tolist(),'gripper_command':gripper.tolist(),'flags':flags,'partial_motion_possible_without_ack':True})
                owner.robot.command(target,gripper,flags);owner._hardware_acknowledged+=1;owner.command_index+=1
                owner.safety.last_target=target.copy();owner.command_g=gripper.copy()
        def stop():
            for hook in self.stops:hook.stop()
        report=stream_stage(Dispatch(),trajectory,self.trajectory_limits,stop,clock=self.clock,sleep=self.sleep)
        samples,obs=self.boundary_capture();pose=self._shared_pose(np.stack([s.q for s in samples]),self.command_g,'left')
        self._last_cartesian_goal=goals[-1];self._goal_reached=self._reached(goals[-1],pose)
        if not self._goal_reached:raise SafetyError('fresh segment-end pose outside original tolerance')
        if phase=='Approach':self.direct_plan['pre_reached']=True
        report.update(stage_id=stage['stage_id'],phase=phase,stage_timing=self.timing().report(),actual_transport_measured=False)
        self.segment_reports.append(report);write(self.output/'STAGE_TRAJECTORIES.json',self.segment_reports)
        return trajectory.q[-1],1.,{'controller':VARIANT,'phase':phase,'segment':report,'goal_TCP_m':goals[-1][0].tolist(),'no_per_setpoint_pose_wait':True}
    def run(self):
        result=super().run();result.update(execution_variant=VARIANT,stage_trajectories=self.segment_reports,stage_timings={key:value.report() for key,value in self.stage_timing.items()},whole_task_speedup_verified=False)
        write(self.output/'RESULT.json',result);return result

SharedCupExecutor=TrajectoryWhiteExecutor
