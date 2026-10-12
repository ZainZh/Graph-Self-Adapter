"""ALL_CODE_STAGE candidate over the frozen V19 graph/model semantics.

Every simple Code node is planned once from current feedback and streamed as a
continuous host trajectory.  Precision Grasp/Align nodes remain on the
existing fixed-z closed loop.  SAM, IK and semantic judges are absent from the
trajectory setpoint loop.
"""
from dataclasses import replace

import numpy as np

from trajectory_white_runtime_v18 import TrajectoryWhiteExecutor as V18Executor
from trajectory_white_runtime_v18 import SharedSAMVision
from all_code_stage_trajectory import plan_stage, stream_stage
from all_code_stage_route import CODE_STAGE_IDS
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError


VARIANT = "all_code_stage_trajectory_v01"
RED_CODE_PHASES = frozenset(("Approach", "Lift", "Move", "Release", "Wait"))
WHITE_CODE_PHASES = frozenset(("Approach", "Lift", "Raise", "Move", "Lower", "Release", "Retreat"))
PRECISION_PHASES = frozenset(("Grasp", "Align"))


def is_code_stage(stage):
    return stage.get("stage_id") in CODE_STAGE_IDS if stage.get("stage_id") else (
        stage["cycle_id"] in ("c01", "c02") and stage["controller"] == "code_reference_retarget" and stage["phase"] in RED_CODE_PHASES
        or stage["cycle_id"] == "white_direct" and stage["phase"] in WHITE_CODE_PHASES)


class ExactFKCacheRobot:
    """One-control-update cache populated only by an explicit target prefetch."""
    def __init__(self, robot):
        self.robot = robot
        self._entries = {}

    @property
    def commands(self):
        return self.robot.commands

    def clear_fk_cache(self):
        self._entries.clear()

    @staticmethod
    def _key(q, g, arm):
        return (str(arm), np.asarray(q, dtype=float).shape,
                np.asarray(q, dtype=float).tobytes(),
                np.asarray(g, dtype=float).shape,
                np.asarray(g, dtype=float).tobytes())

    def prefetch_target_fk(self, q, g):
        self.clear_fk_cache()
        for arm in ("left", "right"):
            xyz, rotation = self.robot.fk(q, g, arm)
            self._entries[self._key(q, g, arm)] = (
                np.asarray(xyz, dtype=float).copy(),
                np.asarray(rotation, dtype=float).copy(),
            )

    def fk(self, q, g, arm):
        cached = self._entries.get(self._key(q, g, arm))
        if cached is None:
            return self.robot.fk(q, g, arm)
        return cached[0].copy(), cached[1].copy()

    def samples(self):
        return self.robot.samples()

    def command(self, *args, **kwargs):
        return self.robot.command(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.robot, name)


class AllCodeStageExecutor(V18Executor):
    def __init__(self, bundle, robot, *args, **kwargs):
        self.raw_robot = robot
        super().__init__(bundle, ExactFKCacheRobot(robot), *args, **kwargs)
        self.all_code_stage_reports = []

    def _step_ik(self, actual_q, g, arm, goal):
        target, pose, error = super()._step_ik(actual_q, g, arm, goal)
        # Deterministic native FK is deliberately paid before the new RGB-D
        # sample. SafetyLayer.validate will reuse it only for byte-identical
        # q/g plus the same arm during this one control update.
        self.robot.prefetch_target_fk(target, g)
        return target, pose, error

    def _ground_red_cycle(self, stage, obs):
        if self._cycle != stage["cycle_id"]:
            if stage["phase"] != "Approach":
                raise SafetyError("new manipulation cycle must begin with object Approach")
            self._cycle = stage["cycle_id"]
            self.coarse_source = self.coarse_target = None
            self._coarse_source_finalized = False
            self.attachment = self._empty_hand_anchor = self._release_pose_anchor = None
            self.static_source_env = self.static_target_env = None
            self.code_static_source_handled = self.code_static_target_handled = None
        if self.coarse_source is not None:
            return
        start = float(stage["reference"]["start_time_sec"])
        start_ref = self.nearest_ref(start)
        initial_ref = getattr(self, "static_reference_points", None) or self.b["frames"][0]["surface_centroids_world_m"]
        initial = getattr(self, "initial_roles", {})
        object_now = obs.roles.get(stage["object"])
        object_ref = start_ref["surface_centroids_world_m"].get(stage["object"])
        self.source_prior = "fresh cycle Approach START paired static environment cue"
        if stage["object"] not in self._manipulated and initial.get(stage["object"]) is not None and initial_ref.get(stage["object"]) is not None:
            object_now = initial[stage["object"]]; object_ref = initial_ref[stage["object"]]
            self.source_prior = "scene initial visible paired prior; assumed unmanipulated static role, not fresh measurement"
        if object_now is None or object_ref is None:
            raise SafetyError("static source environment grounding unknown")
        target = self.target_point(obs, stage)
        target_ref = start_ref["surface_centroids_world_m"].get(stage["target"])
        self.target_prior = "cycle static visible/fixed-handover target prior"
        if stage["target"] == "box" and initial.get("box") is not None and initial_ref.get("box") is not None:
            target = initial["box"]; target_ref = initial_ref["box"]
            self.target_prior = "scene initial SAM/reference paired box prior; static assumption, not fresh measurement"
        if target_ref is None and stage["target"] == "table_handover_region":
            target_ref = self.b["refs"]["handover_anchors"][stage["cycle_id"]]["centroid_world_m"]
        if target is None or target_ref is None:
            raise SafetyError("static target environment grounding unknown")
        self.source_anchor = np.asarray(object_now).copy()
        self.target_anchor = np.asarray(target).copy()
        self.coarse_source = self.source_anchor - np.asarray(object_ref)
        self.coarse_target = self.target_anchor - np.asarray(target_ref)
        self.static_source_env = self.coarse_source.copy()
        self.static_target_env = self.coarse_target.copy()
        self.code_static_source_handled = self.coarse_source.copy()
        self.code_static_target_handled = self.coarse_target.copy()

    def _reference_times(self, stage):
        start = float(stage["reference"]["start_time_sec"])
        end = float(stage["reference"]["end_time_sec"])
        values = np.asarray(self.b["times"], dtype=float)
        within = values[(values > start) & (values < end)]
        # A bounded geometric representation of the demonstrated stage. This
        # includes both endpoints; Approach therefore cannot finish at START.
        if len(within) > 10:
            within = within[np.linspace(0, len(within) - 1, 10).round().astype(int)]
        return np.unique(np.r_[start, within, end])

    def _red_goal(self, stage, t, actual_pose):
        start = float(stage["reference"]["start_time_sec"])
        end = float(stage["reference"]["end_time_sec"])
        progress = (t - start) / max(end - start, 1e-9)
        index = 0 if stage["arm"] == "left" else 1
        q_ref = np.asarray(self.b["qcurve"](t))
        g = np.array([np.interp(t, self.b["times"], self.b["gripper"][:, i]) for i in range(2)])
        g[1 - index] = self.command_g[1 - index]
        native = self._shared_pose(q_ref, g, stage["arm"])
        phase = stage["phase"]
        if phase in ("Approach", "Lift"):
            coarse = self.coarse_source.copy()
        elif phase == "Move":
            coarse = (1 - progress) * self.coarse_source + progress * self.coarse_target
        elif phase in ("Release", "Wait"):
            coarse = self.coarse_target.copy()
        else:
            raise SafetyError("undeclared red Code stage")
        if phase == "Release":
            if self._release_pose_anchor is None:
                self._release_pose_anchor = (actual_pose[0].copy(), actual_pose[1].copy())
            return self._release_pose_anchor
        if phase == "Wait":
            if self._empty_hand_anchor is None:
                raise SafetyError("actual Release-end empty-hand anchor missing")
            reference_start = self._shared_pose(np.asarray(self.b["qcurve"](start)), g, stage["arm"])
            return (self._empty_hand_anchor[0] + native[0] - reference_start[0],
                    native[1] @ reference_start[1].T @ self._empty_hand_anchor[1])
        if phase in ("Lift", "Move") and self.attachment is not None:
            return self.attached_goal(native, coarse)[0]
        return native[0] + coarse, native[1]

    def _code_goals_and_gripper(self, stage, pose, obs):
        if stage["cycle_id"] == "white_direct":
            if self.direct_plan is None:
                self.prepare_white(stage, self.robot.samples(), obs)
            if stage["phase"] == "Release":
                self.phase_goals.setdefault("Release", (pose[0].copy(), pose[1].copy()))
                goals = [self.phase_goals["Release"]]
            else:
                goals = self.segment_goals(stage, pose)
            g = self.command_g.copy()
            if stage["phase"] == "Approach":
                g[0] = max(g[0], self.b["initial_g"][0])
            elif stage["phase"] in ("Release", "Retreat"):
                g[0] = 1.
            else:
                if self.macro_grasp is None:
                    raise SafetyError("trajectory certified grasp command missing")
                g[0] = self.macro_grasp["g_command"]
            return goals, g
        self._ground_red_cycle(stage, obs)
        times = self._reference_times(stage)
        goals = [self._red_goal(stage, float(t), pose) for t in times]
        index = 0 if stage["arm"] == "left" else 1
        g = self.command_g.copy()
        if stage["phase"] == "Approach":
            g[index] = max(float(g[index]), float(np.interp(times[0], self.b["times"], self.b["gripper"][:, index])))
        elif stage["phase"] in ("Lift", "Move"):
            if self.macro_grasp is None:
                raise SafetyError("confirmed macro gripper command missing")
            g[index] = self.macro_grasp["g_command"]
        elif stage["phase"] == "Wait":
            g[index] = 1.
        return goals, g

    def _apply_gripper_profile(self, trajectory, stage, start_g):
        count = len(trajectory.times)
        profile = np.repeat(np.asarray(start_g)[None, :], count, axis=0)
        actor = 0 if stage["arm"] == "left" else 1
        if stage["phase"] == "Release":
            duration = float(stage["reference"]["end_time_sec"] - stage["reference"]["start_time_sec"])
            minimum = duration / max(float(self.cfg["hardware_reference_speed_scale"]), 1e-9)
            if trajectory.times[-1] < minimum:
                dt = 1 / self.trajectory_limits.rate_hz
                new_times = np.r_[np.arange(0., minimum, dt), minimum]
                trajectory.q = np.repeat(trajectory.q[-1][None, :], len(new_times), axis=0)
                trajectory.qdot = np.zeros_like(trajectory.q)
                trajectory.qddot = np.zeros_like(trajectory.q)
                trajectory.tcp = np.repeat(trajectory.tcp[-1][None, :], len(new_times), axis=0)
                trajectory.times = new_times; count = len(new_times)
                profile = np.repeat(np.asarray(start_g)[None, :], count, axis=0)
            if stage["cycle_id"] == "white_direct":
                profile[:, actor] = np.linspace(self.command_g[actor], 1., count)
            else:
                start = float(stage["reference"]["start_time_sec"])
                end = float(stage["reference"]["end_time_sec"])
                ref_t = start + (end - start) * trajectory.times / trajectory.times[-1]
                profile[:, actor] = np.interp(ref_t, self.b["times"], self.b["gripper"][:, actor])
        trajectory.gripper = profile

    def _fresh_pre_dispatch(self, stage, trajectory):
        samples, obs = self.boundary_capture()
        age = self.clock() - obs.stamp
        if not 0 <= age <= self.cfg["safety"]["max_age_sec"]:
            raise SafetyError("fresh post-plan roles stale before trajectory")
        actual = np.stack([sample.q for sample in samples])
        if np.max(abs(actual - trajectory.start_q)) > self.trajectory_limits.start_drift:
            raise SafetyError("actual pose changed during trajectory planning")
        object_visible_or_held = obs.roles.get(stage["object"]) is not None or self.holding(samples, stage) is True
        target_known = stage["target"] == "table_handover_region" or obs.roles.get(stage["target"]) is not None
        if not object_visible_or_held or not target_known:
            raise SafetyError("fresh post-plan role check unknown")
        return {"stamp": obs.stamp, "age_sec": age, "object_visible_or_held": object_visible_or_held,
                "target_known": target_known, "q_drift_rad": float(np.max(abs(actual - trajectory.start_q)))}

    def command(self, stage, elapsed, samples, obs, dt):
        self.robot.clear_fk_cache()
        if not is_code_stage(stage):
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
        if stage["cycle_id"] == "white_direct" and stage["phase"] == "Lower":
            limits = replace(limits, position_tolerance=self.lower_contact_tolerance)
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
        if stage["cycle_id"] == "white_direct" and stage["phase"] == "Approach":
            self.direct_plan["pre_reached"] = True
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
        result = super().run()
        result.update(execution_variant=VARIANT,
                      all_code_stage_trajectories=self.all_code_stage_reports,
                      code_route={"red": sorted(RED_CODE_PHASES), "white_direct": sorted(WHITE_CODE_PHASES)},
                      precision_route={"red": ["Grasp", "Align"], "white_direct": ["Grasp"]},
                      fixed_z=True, latent_updates=0, same_frozen_graph_and_weights=True)
        write(self.output / "RESULT.json", result)
        return result


SharedCupExecutor = AllCodeStageExecutor
