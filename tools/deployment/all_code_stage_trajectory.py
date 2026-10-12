"""Trajectory streaming for the ALL_CODE_STAGE candidate.

Planning is delegated to the accepted V16 planner.  This module only extends
the V17 elastic streamer with a per-sample gripper profile so a complete
Release node can be planned once and streamed without re-entering IK/SAM.
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from stage_trajectory_v17 import Limits, StageTiming, Trajectory, plan_stage
from stage_trajectory import _joint, _pose, _workspace
from prove.xtrainer.io import SafetyError, array


def _gripper_at(trajectory, index):
    profile = np.asarray(trajectory.gripper, dtype=float)
    if profile.shape == (2,):
        return profile.copy()
    if profile.shape != (len(trajectory.times), 2):
        raise SafetyError("trajectory gripper profile shape mismatch")
    return profile[index].copy()


def stream_stage(robot, trajectory, limits, stop, *, emergency=lambda: False,
                 clock=time.monotonic, sleep=time.sleep, context_timing=None,
                 maximum_stall_sec=.25):
    """Stream one preplanned node; only own feedback, limits and stop run here."""
    timing = StageTiming()
    actor = 0 if trajectory.arm == "left" else 1
    last = None
    attempts = acknowledgments = 0
    lags, rebases, dispatch_log, feedback_log = [], [], [], []
    started = clock()
    last_dispatch = None
    if not 0 < maximum_stall_sec <= .25:
        raise SafetyError("stall budget must be positive and <=.25s")

    def result(completed, failure=None):
        return {
            "completed": completed,
            "failure": failure,
            "command_attempts": attempts,
            "command_acknowledgments": acknowledgments,
            "planning": trajectory.planning,
            "streaming": timing.report(),
            "elapsed_seconds": clock() - started,
            "maximum_dispatch_lateness": max(lags, default=0.),
            "rebase_count": len(rebases),
            "timebase_rebases": rebases,
            "actual_dispatch_log": dispatch_log,
            "actual_feedback_log": feedback_log,
            "maximum_stall_sec": maximum_stall_sec,
            "vision_judge_context": context_timing or {},
            "FK_calls_during_streaming": 0,
            "IK_calls_during_streaming": 0,
            "SAM_calls_during_streaming": 0,
            "judge_calls_during_streaming": 0,
            "native_SDK_queue_used": False,
            "hard_realtime_claim": False,
            "collision_planning_claim": False,
        }

    def feedback():
        timing.count("feedback_calls")
        feedback_started = clock()
        with timing.measure("feedback", clock):
            samples = robot.samples()
        feedback_log.append({"start": feedback_started, "end": clock(),
                             "seconds": clock() - feedback_started})
        if clock() - feedback_started > maximum_stall_sec:
            raise SafetyError("continuous feedback stall exceeded .25s budget")
        if len(samples) != 2:
            raise SafetyError("two actual arm samples required")
        for arm_index, sample in enumerate(samples):
            if not 0 <= clock() - sample.stamp <= limits.feedback_age:
                raise SafetyError("stale actual robot feedback")
            array(sample.q, (6,)); array(sample.qdot, (6,)); _pose(sample.tcp, sample.rotation)
            if sample.error or not sample.enabled or sample.mode not in (5, 7):
                raise SafetyError("robot feedback mode/error fault")
            if sample.user != 0 or sample.tool != 1 or not np.allclose(sample.user_offset, np.zeros(6), atol=1e-5) or not np.allclose(sample.tool_offset, [0, 0, 197, 0, 0, 0], atol=.01):
                raise SafetyError("User0/Tool1 calibration mismatch")
            speed_limit = limits.feedback_velocity if limits.feedback_velocity is not None else limits.velocity
            if np.max(abs(sample.qdot)) > speed_limit:
                raise SafetyError("actual joint velocity limit")
            own_tcp = sample.tcp if arm_index == 0 else (trajectory.T_left_from_right @ np.r_[sample.tcp, 1])[:3]
            _workspace(own_tcp, limits)
        actual = np.stack([sample.q for sample in samples]); _joint(actual, limits)
        p, R = samples[actor].tcp, samples[actor].rotation
        if actor:
            p = (trajectory.T_left_from_right @ np.r_[p, 1])[:3]
            R = trajectory.T_left_from_right[:3, :3] @ R
        _workspace(p, limits)
        if last is not None and np.max(abs(actual[actor] - last[actor])) > limits.tracking:
            raise SafetyError("actual trajectory tracking limit")
        if emergency():
            raise SafetyError("emergency stop requested")
        return samples, actual, (p, R)

    try:
        samples, actual, _ = feedback()
        if np.max(abs(actual - trajectory.start_q)) > limits.start_drift:
            raise SafetyError("actual start changed during offline planning")
        if np.max(abs(samples[actor].qdot)) > .01:
            raise SafetyError("actual trajectory entry is not stationary")
        origin = clock()
        for i, offset in enumerate(trajectory.times):
            delay = origin + float(offset) - clock()
            if delay > 0:
                sleep(delay)
            lag = clock() - (origin + float(offset)); lags.append(lag)
            if lag > maximum_stall_sec:
                raise SafetyError("continuous scheduler stall exceeded budget")
            samples, actual, _ = feedback()
            if last_dispatch is not None and clock() < last_dispatch + 1 / limits.rate_hz:
                sleep(last_dispatch + 1 / limits.rate_hz - clock())
            if any(not 0 <= clock() - sample.stamp <= limits.feedback_age for sample in samples):
                raise SafetyError("actual feedback expired before dispatch")
            target = trajectory.q[i].copy(); target[1 - actor] = actual[1 - actor]
            if last is not None and np.max(abs(target[actor] - last[actor])) > limits.step + 1e-9:
                raise SafetyError("planned joint step exceeded")
            if np.max(abs(target[actor] - actual[actor])) > limits.tracking:
                raise SafetyError("pre-dispatch tracking limit")
            if limits.actual_step_limit is not None and np.max(abs(target[actor] - actual[actor])) > limits.actual_step_limit:
                raise SafetyError("pre-dispatch actual joint step limit")
            lag = clock() - (origin + float(offset)); lags.append(lag)
            if lag > maximum_stall_sec or (last_dispatch is not None and clock() - last_dispatch > maximum_stall_sec):
                raise SafetyError("continuous pre-dispatch stall exceeded budget")
            if lag > 1e-9:
                rebases.append({"sample_index": i, "seconds": lag, "wall_time": clock()})
                origin += lag
            attempts += 1; timing.count("command_attempts"); last_dispatch = clock()
            dispatch_log.append({"sample_index": i, "start": last_dispatch, "acknowledged": False})
            with timing.measure("dispatch", clock):
                robot.command(target, _gripper_at(trajectory, i), [actor == 0, actor == 1])
            acknowledgments += 1; last = target.copy()
            dispatch_log[-1].update(end=clock(), seconds=clock() - last_dispatch, acknowledged=True)
            if clock() - last_dispatch > maximum_stall_sec:
                raise SafetyError("continuous transport stall exceeded budget")
        settle_started = clock(); accepted = False
        while clock() - settle_started <= limits.settle_seconds:
            samples, actual, pose = feedback()
            position = np.linalg.norm(pose[0] - trajectory.goal[0])
            rotation = Rotation.from_matrix(pose[1] @ trajectory.goal[1].T).magnitude()
            if position <= limits.position_tolerance and rotation <= limits.rotation_tolerance and np.max(abs(samples[actor].qdot)) <= .01:
                accepted = True; break
            sleep(min(.02, 1 / limits.rate_hz))
        timing.durations["end_settle"] = clock() - settle_started
        if not accepted:
            raise SafetyError("actual stage endpoint not reached in bounded settle")
        return result(True)
    except Exception as error:
        metrics = result(False, str(error)); error.stream_metrics = metrics
        if dispatch_log and "end" not in dispatch_log[-1]:
            dispatch_log[-1].update(end=clock(), seconds=clock() - dispatch_log[-1]["start"])
        try:
            stop()
        except Exception as stop_error:
            raise SafetyError(str(error) + "; stop failed: " + str(stop_error)) from error
        raise


__all__ = ["Limits", "StageTiming", "Trajectory", "plan_stage", "stream_stage"]
