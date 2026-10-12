"""V25 ordinary task own graphs, precise frozen NN and object-frame Code streams."""
from dataclasses import replace
import numpy as np
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError
from multitask_ordinary_runtime_v21 import RelativeLiftExecutor,relative_lift_goal
from multitask_ordinary_runtime import OrdinaryExecutor
from all_code_stage_runtime import AllCodeStageExecutor,ExactFKCacheRobot,PRECISION_PHASES
from trajectory_white_runtime_v17 import TrajectoryWhiteExecutor
from all_code_stage_trajectory import Limits,StageTiming,plan_stage,stream_stage
from shared_cup_vision import SharedSAMVision
VARIANT='ordinary_task_stage_trajectory_v25'
class OrdinaryV25Executor(RelativeLiftExecutor):
    def __init__(self,bundle,robot,*args,**kwargs):
        super().__init__(bundle,ExactFKCacheRobot(robot),*args,**kwargs)
        self.all_code_stage_reports=[];self.stage_timing={}
        cfg=self.cfg['trajectory'];s=self.cfg['safety'];self.maximum_stall_sec=cfg['maximum_stall_sec']
        self.trajectory_limits=Limits(s['joint_min_rad'],s['joint_max_rad'],s['workspace_min_m'],s['workspace_max_m'],velocity=cfg['velocity_rad_sec'],acceleration=cfg['acceleration_rad_sec2'],step=.009,rate_hz=cfg['rate_hz'],feedback_age=s['max_age_sec'],tracking=s['max_tracking_rad'],position_tolerance=.003,rotation_tolerance=.02,settle_seconds=cfg['settle_seconds'],maximum_lateness=cfg['maximum_lateness_sec'],feedback_velocity=s['max_velocity_rad_sec'],actual_step_limit=s['max_step_rad'])
    timing=TrajectoryWhiteExecutor.timing
    def boundary_capture(self):
        with self.timing().measure('vision_and_feedback_boundary',self.clock):return self.capture()
    _ground_red_cycle=AllCodeStageExecutor._ground_red_cycle
    _reference_times=AllCodeStageExecutor._reference_times
    _apply_gripper_profile=AllCodeStageExecutor._apply_gripper_profile
    _fresh_pre_dispatch=AllCodeStageExecutor._fresh_pre_dispatch
    def _step_ik(self,*args,**kwargs):
        target,pose,error=super()._step_ik(*args,**kwargs);self.robot.prefetch_target_fk(target,self.command_g);return target,pose,error
    def _code_goals_and_gripper(self,stage,pose,obs):
        self._ground_red_cycle(stage,obs);times=self._reference_times(stage)
        if stage['phase']=='Lift':
            anchor=self.accepted_grasp_pose.get(stage['cycle_id'])
            if anchor is None:raise SafetyError('accepted actual Grasp TCP missing')
            start_pose=self._shared_pose(np.asarray(self.b['qcurve'](times[0])),self.command_g,stage['arm'])
            goals=[relative_lift_goal(anchor,start_pose,self._shared_pose(np.asarray(self.b['qcurve'](t)),self.command_g,stage['arm'])) for t in times]
        else:goals=[AllCodeStageExecutor._red_goal(self,stage,float(t),pose) for t in times]
        g=self.command_g.copy();actor=0 if stage['arm']=='left' else 1
        if stage['phase']=='Approach':g[actor]=max(float(g[actor]),float(self.b['initial_g'][actor]))
        elif stage['phase'] in ('Lift','Move'):
            if self.accepted_grasp is None:raise SafetyError('accepted Grasp missing for transfer')
            g[actor]=self.accepted_grasp['g_command']
        elif stage['phase']=='Wait':g[actor]=1.
        return goals,g
    def command(self, stage, elapsed, samples, obs, dt):
        self.robot.clear_fk_cache()
        if stage["controller"] != "code_reference_retarget":
            return super().command(stage, elapsed, samples, obs, dt)
        if stage["phase"] in PRECISION_PHASES:
            raise SafetyError("precision node routed into Code trajectory")
        samples, obs = self.boundary_capture()
        actual = np.stack([sample.q for sample in samples])
        pose = self._shared_pose(actual, self.command_g, stage["arm"])
        goals, g = self._code_goals_and_gripper(stage, pose, obs)
        limits = self.trajectory_limits
        if stage["cycle_id"] in ("c01", "c02"):
            limits = replace(limits, position_tolerance=.003)
        with self.timing().measure("planning", self.clock):
            trajectory = plan_stage(self.robot.fk, actual, goals, g, stage["arm"], limits,
                                    T_left_from_right=self.T, clock=self.clock)
            self._apply_gripper_profile(trajectory, stage, g)
        role_check = self._fresh_pre_dispatch(stage, trajectory)
        owner = self
        class Dispatch:
            def samples(self):
                owner.guard_limits(); fresh = owner.robot.samples(); owner.safety.samples(fresh); return fresh
            def command(self, target, gripper, flags):
                owner.guard_limits(); owner._hardware_attempts += 1
                write(owner.output / "COMMAND_INTENT.json", {
                    "attempt_count": owner._hardware_attempts,
                    "acknowledged_count": owner._hardware_acknowledged,
                    "stage_id": stage["stage_id"], "q_target_rad": target.tolist(),
                    "gripper_command": gripper.tolist(), "flags": flags,
                    "partial_motion_possible_without_ack": True})
                owner.robot.command(target, gripper, flags)
                owner._hardware_acknowledged += 1; owner.command_index += 1
                owner.safety.last_target = target.copy(); owner.command_g = gripper.copy()
        def stop():
            for hook in self.stops:
                hook.stop()
        try:
            report = stream_stage(Dispatch(), trajectory, limits, stop, clock=self.clock,
                                  sleep=self.sleep, maximum_stall_sec=self.maximum_stall_sec)
        except Exception as error:
            report = getattr(error, "stream_metrics", {"completed": False, "failure": str(error)})
            report.update(stage_id=stage["stage_id"], phase=stage["phase"], pre_dispatch=role_check)
            self.all_code_stage_reports.append(report)
            write(self.output / "ALL_CODE_STAGE_TRAJECTORIES.json", self.all_code_stage_reports)
            raise
        samples, obs = self.boundary_capture()
        actual_pose = self._shared_pose(np.stack([sample.q for sample in samples]), self.command_g, stage["arm"])
        self._phase = stage["stage_id"]
        self._phase_elapsed = float(stage["reference"]["end_time_sec"] - stage["reference"]["start_time_sec"])
        self._last_cartesian_goal = goals[-1]
        self._goal_reached = self._reached(goals[-1], actual_pose)
        if not self._goal_reached:
            raise SafetyError("fresh segment-end pose outside original tolerance")
        report.update(stage_id=stage["stage_id"], phase=stage["phase"], pre_dispatch=role_check,
                      stage_timing=self.timing().report(), complete_reference_start_to_end=True,
                      semantic_boundary_calls_during_streaming=0)
        self.all_code_stage_reports.append(report)
        write(self.output / "ALL_CODE_STAGE_TRAJECTORIES.json", self.all_code_stage_reports)
        return trajectory.q[-1], 1., {
            "controller": VARIANT, "phase": stage["phase"], "segment": report,
            "goal_TCP_m": goals[-1][0].tolist(), "goal_count": len(goals),
            "no_per_setpoint_SAM_Root_or_IK": True,
        }

    def run(self):
        result=super().run();result.update(execution_variant=VARIANT,all_code_stage_trajectories=self.all_code_stage_reports,stage_timings={k:v.report() for k,v in self.stage_timing.items()},precision_nodes_unchanged=True)
        write(self.output/'RESULT.json',result);return result
SharedCupExecutor=OrdinaryV25Executor
