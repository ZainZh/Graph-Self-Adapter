"""P70 object-centric book->caddy controller primitives.

New P70 code only.  Runtime actions are computed from public masks/cameras and
robot proprioception.  Simulator/private object poses may judge stages outside
this module, but they are not accepted as controller inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import importlib
import json
from pathlib import Path
from typing import Mapping, Any

import numpy as np
from scipy.spatial.transform import Rotation

from prove.task_graph.p69_public_grounding import RoleDelta, estimate_role_delta, load_reference
from prove.task_graph.p57_public_geometry import eef_transform

PUBLIC_KEYS = frozenset((
    "rgb", "visible_masks", "camera_K", "camera_pose", "eef_pos", "eef_quat_xyzw",
    "gripper_qpos", "joint_pos", "joint_vel",
))
STAGES = (
    "approach_book", "short_grasp", "lift_clear", "orient_clear",
    "transport_above_caddy", "near_entry", "p68_insert",
)


def _array(value, shape, label):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{label} must be finite shape {shape}")
    return arr


def _unit_quat_xyzw(value):
    q = _array(value, (4,), "quaternion_xyzw")
    n = np.linalg.norm(q)
    if n < 1e-9:
        raise ValueError("zero quaternion")
    return q / n


def _yaw_matrix(yaw: float) -> np.ndarray:
    return Rotation.from_euler("z", float(yaw)).as_matrix()


def _pose_from_xy_yaw_height(xy, yaw: float, z: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _yaw_matrix(yaw)
    T[:2, 3] = np.asarray(xy, dtype=np.float64)
    T[2, 3] = float(z)
    return T


def _norm_limit(vec, limit: float) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v if n <= limit or n < 1e-12 else v * (limit / n)


def _clip_action(delta_pos, delta_rot, grip, *, translation_scale=0.05, rotation_scale=0.5, max_action=1.0):
    # P70 object-code stages use the same conservative per-command physical
    # bounds as the P52/P53 teachers: 1cm translation and 0.05rad rotation,
    # then convert to LIBERO OSC scale.  This avoids 5cm/0.5rad jumps from a
    # mere component clip.
    dp = _norm_limit(delta_pos, 0.01) / translation_scale
    dr = _norm_limit(delta_rot, 0.05) / rotation_scale
    action = np.r_[dp, dr, float(grip)]
    return np.clip(action, -max_action, max_action)


@dataclass(frozen=True)
class StageTarget:
    stage: str
    target_T_world_eef: list[list[float]]
    grip: float
    source: str


@dataclass
class P70ControllerState:
    stage: str = "approach_book"
    t: int = 0
    stage_t: int = 0
    held_T_eef_book: list[list[float]] | None = None
    lift_goal_T_world_book: list[list[float]] | None = None
    orient_goal_T_world_book: list[list[float]] | None = None
    approach_substage: str = "above_current_book"
    grasp_substage: str = "direct_close"
    latched_T_world_eef_grasp: list[list[float]] | None = None
    close_latched_once: bool = False
    held_pose_calibrated: bool = False
    lift_refined_once: bool = False
    frozen_book_delta: dict | None = None
    frozen_caddy_delta: dict | None = None
    stage_entries: list[dict] | None = None

    def __post_init__(self):
        if self.stage_entries is None:
            self.stage_entries = []

    def to_dict(self):
        return asdict(self)


class P70ObjectController:
    """Deterministic object-centric controller for P70 development.

    It uses a fixed public grounding reference from P69/P70 calibration.  Book is
    located by the accepted silhouette helper; caddy remains view0 planar.  Once
    a grasp is judged externally and the controller advances to held stages, the
    book pose is propagated from proprioception through the frozen
    ``held_T_eef_book`` transform instead of reading current private pose.
    """

    revision = "p70-object-centric-public-controller-v0"

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.reference = load_reference(self.config["grounding_reference"]["path"])
        self.relation_targets = self._load_bound_json(self.config.get("object_relation_targets"), "object_relation_targets")
        self.grasp_relations = self._load_bound_json(self.config.get("grasp_policy", {}).get("demo_relations"), "grasp_policy.demo_relations")
        self.held_pose_calibration = self._load_bound_json(self.config.get("held_pose_calibration"), "held_pose_calibration")
        self.grasp_policy = None
        self.state = P70ControllerState()
        self.records: list[dict] = []
        self._last_public_hash = None
        self._load_grasp_policy()
        self._validate_config()

    def _sha256(self, path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def _load_bound_json(self, binding, label: str) -> dict | None:
        if binding is None:
            return None
        if not isinstance(binding, Mapping) or "path" not in binding or "sha256" not in binding:
            raise ValueError(f"{label} requires path and sha256")
        path = Path(binding["path"])
        if self._sha256(path) != binding["sha256"]:
            raise ValueError(f"{label} hash mismatch")
        return json.loads(path.read_text())

    def _validate_bound_file(self, binding, label: str) -> None:
        if not isinstance(binding, Mapping) or "path" not in binding or "sha256" not in binding:
            raise ValueError(f"{label} requires path and sha256")
        if self._sha256(binding["path"]) != binding["sha256"]:
            raise ValueError(f"{label} hash mismatch")

    def _load_grasp_policy(self) -> None:
        gp = self.config.get("grasp_policy")
        if gp is None:
            return
        self._validate_bound_file(gp.get("checkpoint"), "grasp_policy.checkpoint")
        module = gp.get("module", "prove.task_graph.p70_grasp_policy_v02")
        cls = getattr(importlib.import_module(module), "GraspPolicy")
        self.grasp_policy = cls.load(gp["checkpoint"]["path"], gp.get("device", "cpu"))

    def _validate_config(self):
        required = {"grounding_reference", "heights", "offsets", "stage_budgets", "gains", "object_relation_targets", "grasp_policy"}
        if not required <= set(self.config):
            raise ValueError(f"P70 controller config missing {sorted(required - set(self.config))}")
        if self.config.get("runtime_inputs") != ["public_masks", "camera", "proprio"]:
            raise ValueError("P70 runtime inputs must be public masks/camera/proprio")
        for stage in STAGES[:-1]:
            if stage not in self.config["stage_budgets"]:
                raise ValueError(f"missing stage budget {stage}")
        if self.relation_targets is None or self.relation_targets.get("schema") != "prove.p70-object-relation-targets.v1":
            raise ValueError("P70 object relation targets must be bound")
        if self.grasp_relations is None or self.grasp_relations.get("schema") != "prove.p70-demo-grasp-relations.v1":
            raise ValueError("P70 grasp demo relations must be bound")
        if self.grasp_policy is None:
            raise ValueError("P70 grasp policy checkpoint must be bound and loadable")
        if self.held_pose_calibration is not None:
            if self.held_pose_calibration.get("schema") != "prove.p70-held-relation-calibration.v1":
                raise ValueError("wrong P70 held relation calibration schema")
            self._held_pose_correction_matrix()

    def reset(self):
        self.state = P70ControllerState()
        self.records.clear()

    def observe_grounding(self, public: Mapping[str, Any]) -> tuple[RoleDelta, RoleDelta]:
        self._validate_public(public)
        book = estimate_role_delta(public, self.reference, "book")
        caddy = estimate_role_delta(public, self.reference, "caddy")
        return book, caddy

    def _validate_public(self, public: Mapping[str, Any]) -> None:
        if set(public) != PUBLIC_KEYS:
            raise ValueError("P70 controller accepts only public frame keys")
        rgb = np.asarray(public["rgb"])
        masks = np.asarray(public["visible_masks"])
        if rgb.shape != (2, 448, 448, 3) or masks.shape != (2, 3, 448, 448):
            raise ValueError("P70 requires native dual-view 448 public frame")
        _array(public["camera_K"], (2, 3, 3), "camera_K")
        _array(public["camera_pose"], (2, 4, 4), "camera_pose")
        _array(public["eef_pos"], (3,), "eef_pos")
        _unit_quat_xyzw(public["eef_quat_xyzw"])
        _array(public["gripper_qpos"], (2,), "gripper_qpos")

    def _eef(self, public) -> np.ndarray:
        return eef_transform(public["eef_pos"], public["eef_quat_xyzw"])

    def _book_pose_from_delta(self, book: RoleDelta) -> np.ndarray:
        h = self.config["heights"]
        return _pose_from_xy_yaw_height(book.current_xy, book.current_yaw, h["book_body_z"])

    def _caddy_pose_from_delta(self, caddy: RoleDelta) -> np.ndarray:
        # P69 caddy grounding reports a public mask-PCA pose, not the body frame.
        # Use only visual delta_xy/delta_yaw to retarget the frozen demo body pose.
        # This preserves the real caddy body yaw/offset when the visual PCA yaw is
        # offset by ~90deg from the MuJoCo body frame.
        T_demo = self._relation_matrix("T_world_caddy_demo")
        T = T_demo.copy()
        T[:2, 3] = T_demo[:2, 3] + np.asarray(caddy.delta_xy, dtype=np.float64)
        T[2, 3] = float(T_demo[2, 3])
        T[:3, :3] = _yaw_matrix(float(caddy.delta_yaw)) @ T_demo[:3, :3]
        return T

    def _delta_from_cache(self, role: str, current: RoleDelta) -> RoleDelta:
        # P70 locks the initial settled public book/caddy estimates.  Caddy is
        # static; book is used only before attachment and then replaced by the
        # held EEF-book transform.  This avoids target drift from gripper
        # occlusion or fixed-Z silhouette fits after lift.
        cached = self.state.frozen_book_delta if role == "book" else self.state.frozen_caddy_delta
        if cached:
            return RoleDelta(role, tuple(cached["demo_xy"]), tuple(cached["current_xy"]), tuple(cached["delta_xy"]),
                             float(cached["delta_yaw"]), float(cached["current_yaw"]), True, "locked_initial_settled_public_grounding")
        return current

    def _current_book_pose(self, public, book: RoleDelta) -> np.ndarray | None:
        if self.state.held_T_eef_book is not None:
            return self._eef(public) @ np.asarray(self.state.held_T_eef_book, dtype=np.float64)
        book = self._delta_from_cache("book", book)
        if book.valid:
            return self._book_pose_from_delta(book)
        return None

    def enter_stage(self, stage: str, public: Mapping[str, Any], *, reason: str):
        if stage not in STAGES:
            raise ValueError("unknown P70 stage")
        self.state.stage = stage
        self.state.stage_t = 0
        self.state.stage_entries.append(dict(t=self.state.t, stage=stage, reason=reason))
        if stage in ("lift_clear", "orient_clear", "transport_above_caddy", "near_entry") and self.state.held_T_eef_book is None:
            book, _ = self.observe_grounding(public)
            book = self._delta_from_cache("book", book)
            if not book.valid:
                raise ValueError("cannot enter held stage without public book estimate")
            held = np.linalg.inv(self._eef(public)) @ self._book_pose_from_delta(book)
            self.state.held_T_eef_book = held.tolist()
            if stage == "lift_clear":
                B = self._book_pose_from_delta(book)
                self.state.lift_goal_T_world_book = self._lifted_book_goal(self._book_pose_from_delta(book)).tolist()

    def command(self, public: Mapping[str, Any], facts: Mapping[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        """Return one OSC action and telemetry.

        ``facts`` may contain boolean stage-judge facts such as ``attached`` or
        ``near_entry``.  These facts only advance stages; they never provide pose
        values or continuous geometry to the controller.
        """
        facts = dict(facts or {})
        self._validate_public(public)
        book, caddy = self.observe_grounding(public)
        if self.state.frozen_book_delta is None and book.valid:
            self.state.frozen_book_delta = book.to_dict()
        if self.state.frozen_caddy_delta is None and caddy.valid:
            self.state.frozen_caddy_delta = caddy.to_dict()
        transition = self._maybe_advance(public, facts)
        self._current_facts = facts
        self._current_transition = transition
        target, action, action_diagnostics = self._target_and_action(public, book, caddy)
        record = dict(t=self.state.t, stage=self.state.stage, stage_t=self.state.stage_t,
                      transition=transition, target=asdict(target), action=action.tolist(), action_diagnostics=action_diagnostics,
                      grounding=dict(book=book.to_dict(), caddy=caddy.to_dict()),
                      facts={k: bool(v) for k, v in facts.items() if isinstance(v, (bool, np.bool_))},
                      controller_state=self.state.to_dict(), runtime_private_pose_used=False)
        self.records.append(record)
        self.state.t += 1
        self.state.stage_t += 1
        return action, record

    def _maybe_advance(self, public, facts: Mapping[str, Any]) -> dict | None:
        s = self.state.stage
        next_stage = None
        reason = None
        if s == "approach_book" and facts.get("grasp_ready") is True:
            next_stage, reason = "short_grasp", "grasp_ready"
        elif s == "short_grasp" and facts.get("attached") is True:
            next_stage, reason = "lift_clear", "attached"
        elif s == "lift_clear" and facts.get("lift_clear") is True:
            next_stage, reason = "orient_clear", "lift_clear"
        elif s == "orient_clear" and facts.get("orient_clear") is True:
            next_stage, reason = "transport_above_caddy", "orient_clear"
        elif s == "transport_above_caddy" and facts.get("above_destination_vicinity") is True:
            next_stage, reason = "near_entry", "above_destination_vicinity"
        elif s == "near_entry" and facts.get("near_entry") is True:
            next_stage, reason = "p68_insert", "near_entry"
        budget = self.config["stage_budgets"].get(s)
        if budget is not None and self.state.stage_t >= int(budget):
            return dict(stop=f"{s}_budget", stage=s)
        if next_stage is not None:
            self.enter_stage(next_stage, public, reason=reason)
            return dict(from_stage=s, to_stage=next_stage, reason=reason)
        return None

    def _book_box_bottom(self, world_book: np.ndarray) -> float:
        geom = self.relation_targets["book_geometry"]
        center = np.asarray(geom["center"], dtype=np.float64)
        half = np.asarray(geom["halfsize"], dtype=np.float64)
        vertices = np.asarray([[sx, sy, sz] for sx in (-half[0], half[0]) for sy in (-half[1], half[1]) for sz in (-half[2], half[2])], dtype=np.float64) + center
        world = vertices @ world_book[:3, :3].T + world_book[:3, 3]
        return float(world[:, 2].min())

    def _lifted_book_goal(self, world_book: np.ndarray) -> np.ndarray:
        goal = world_book.copy()
        bottom = self._book_box_bottom(goal)
        half = np.asarray(self.relation_targets["book_geometry"]["halfsize"], dtype=np.float64)
        # Aim slightly above the Boolean judge floor; the judge threshold itself
        # remains unchanged.  The margin is the original geometry tolerance
        # scale used for local gates, not a result-tuned centimeter target.
        floor = float(self.relation_targets["safety_heights"]["book_bottom_min_world_z"]) + self._lift_refinement_amount_m()
        goal[2, 3] += max(0.0, floor - bottom)
        return goal

    def _lift_refinement_amount_m(self) -> float:
        half = np.asarray(self.relation_targets["book_geometry"]["halfsize"], dtype=np.float64)
        return float(np.min(half) / 4.0)

    def _pose_error(self, current: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        pos_err = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
        rot_err = float(Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).magnitude())
        return pos_err, rot_err

    def _maybe_refine_lift_goal(self, public, facts: Mapping[str, Any], B_goal: np.ndarray) -> dict:
        if self.config.get("lift_clearance_refinement") is not True:
            return {"applied": False, "reason": "disabled"}
        if self.state.lift_refined_once:
            return {"applied": False, "reason": "already_applied"}
        if isinstance(getattr(self, "_current_transition", None), dict) and self._current_transition.get("stop"):
            return {"applied": False, "reason": "transition_stop", "stop": self._current_transition.get("stop")}
        if self.state.stage != "lift_clear":
            return {"applied": False, "reason": "not_lift_clear"}
        if facts.get("attached") is not True or facts.get("bilateral_contact") is not True or facts.get("lift_clear") is not False:
            return {"applied": False, "reason": "stage_facts_not_eligible",
                    "attached": bool(facts.get("attached")), "bilateral_contact": bool(facts.get("bilateral_contact")),
                    "lift_clear": bool(facts.get("lift_clear"))}
        original_goal = np.asarray(B_goal, dtype=np.float64).copy()
        target_eef = self._held_compensated_eef(original_goal)
        current_eef = self._eef(public)
        pos_err, rot_err = self._pose_error(current_eef, target_eef)
        amount = self._lift_refinement_amount_m()
        pos_tol = amount / 2.0
        rot_tol = float(self.config.get("lift_refinement_rotation_tolerance_rad", np.deg2rad(10.0)))
        if pos_err > pos_tol or rot_err > rot_tol:
            return {"applied": False, "reason": "public_goal_error_too_large",
                    "position_error_m": pos_err, "position_tolerance_m": pos_tol,
                    "rotation_error_rad": rot_err, "rotation_tolerance_rad": rot_tol}
        refined = original_goal.copy()
        refined[2, 3] += amount
        self.state.lift_goal_T_world_book = refined.tolist()
        self.state.lift_refined_once = True
        return {"applied": True, "reason": "attached_bilateral_public_goal_reached_but_lift_not_clear",
                "position_error_m": pos_err, "position_tolerance_m": pos_tol,
                "rotation_error_rad": rot_err, "rotation_tolerance_rad": rot_tol,
                "z_increment_m": amount,
                "original_lift_goal_T_world_book": original_goal.tolist(),
                "refined_lift_goal_T_world_book": refined.tolist()}

    def _relation_matrix(self, key: str) -> np.ndarray:
        T = np.asarray(self.relation_targets[key], dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError(f"relation target {key} must be finite 4x4")
        return T

    def _world_caddy_site(self, C_body: np.ndarray) -> np.ndarray:
        return C_body @ self._relation_matrix("T_caddy_body_back_site")

    def _held_compensated_eef(self, world_book_goal: np.ndarray) -> np.ndarray:
        if self.state.held_T_eef_book is None:
            raise ValueError("held_T_eef_book unavailable")
        return world_book_goal @ np.linalg.inv(np.asarray(self.state.held_T_eef_book, dtype=np.float64))

    def _grasp_goal(self, name: str) -> np.ndarray:
        T = np.asarray(self.grasp_relations["relations"][name]["T_book_eef"], dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError(f"bad grasp relation {name}")
        return T

    def _grasp_gate(self, B: np.ndarray, E: np.ndarray) -> dict:
        goal = B @ self._grasp_goal("grasp")
        geom = self.relation_targets["book_geometry"]
        half = np.asarray(geom["halfsize"], dtype=np.float64)
        gate = self.config.get("grasp_gate", {})
        pos_tol = float(gate.get("position_m", np.min(half) / 4.0))
        rot_tol = float(gate.get("rotation_rad", 0.15))
        pos_err = float(np.linalg.norm(goal[:3, 3] - E[:3, 3]))
        rot_err = float(Rotation.from_matrix(goal[:3, :3] @ E[:3, :3].T).magnitude())
        return dict(goal=goal, position_error_m=pos_err, rotation_error_rad=rot_err,
                    position_tolerance_m=pos_tol, rotation_tolerance_rad=rot_tol,
                    passed=bool(pos_err <= pos_tol and rot_err <= rot_tol))

    def _short_grasp_action(self, public, B: np.ndarray, E: np.ndarray) -> tuple[StageTarget, np.ndarray, dict]:
        sequence = self.config.get("grasp_sequence", "direct_close")
        if sequence != "position_then_close":
            action, diag = self.grasp_policy.command(B, E, public["gripper_qpos"], self._grasp_goal("grasp"), close=True)
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            target = StageTarget("short_grasp", (B @ self._grasp_goal("grasp")).tolist(), float(action[6]), "v02_grasp_policy_grasp")
            return target, action, dict(diag, grasp_sequence="direct_close")
        gate = self._grasp_gate(B, E)
        if self.state.grasp_substage == "direct_close":
            self.state.grasp_substage = "position_open"
        if self.state.grasp_substage == "position_open" and gate["passed"]:
            self.state.grasp_substage = "close_hold"
            self.state.latched_T_world_eef_grasp = gate["goal"].tolist()
            self.state.close_latched_once = True
        if self.state.grasp_substage == "position_open":
            action, diag = self.grasp_policy.command(B, E, public["gripper_qpos"], self._grasp_goal("grasp"), close=False)
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            target = StageTarget("short_grasp", gate["goal"].tolist(), float(action[6]), "v02_grasp_policy_open_to_grasp_gate")
            return target, action, dict(diag, grasp_sequence=sequence, grasp_substage="position_open", gate={k: v for k, v in gate.items() if k != "goal"})
        latched = np.asarray(self.state.latched_T_world_eef_grasp, dtype=np.float64) if self.state.latched_T_world_eef_grasp is not None else gate["goal"]
        target = StageTarget("short_grasp", latched.tolist(), 1.0, "latched_grasp_pose_close_hold")
        action = self._action_to_target(public, target)
        return target, action, dict(controller="latched_pose_close_hold", grasp_sequence=sequence, grasp_substage="close_hold", gate={k: v for k, v in gate.items() if k != "goal"})

    def _held_pose_correction_matrix(self) -> np.ndarray:
        if self.held_pose_calibration is None:
            raise ValueError("held pose calibration is not configured")
        if "T_predicted_book_actual_book" not in self.held_pose_calibration:
            raise ValueError("held relation calibration missing T_predicted_book_actual_book")
        value = self.held_pose_calibration["T_predicted_book_actual_book"]
        D = np.asarray(value, dtype=np.float64)
        if D.shape != (4, 4) or not np.isfinite(D).all() or not np.allclose(D[3], [0, 0, 0, 1]):
            raise ValueError("held pose calibration correction must be finite 4x4")
        return D

    def _apply_held_pose_calibration_once(self) -> dict:
        if self.held_pose_calibration is None:
            return {"applied": False, "reason": "no_held_pose_calibration"}
        if self.state.held_pose_calibrated:
            return {"applied": False, "reason": "already_applied"}
        if self.state.held_T_eef_book is None:
            return {"applied": False, "reason": "no_held_T_eef_book"}
        D = self._held_pose_correction_matrix()
        H = np.asarray(self.state.held_T_eef_book, dtype=np.float64)
        self.state.held_T_eef_book = (H @ D).tolist()
        self.state.held_pose_calibrated = True
        return {"applied": True, "source": self.config["held_pose_calibration"]["path"]}

    def _orient_book_goal(self, current_book: np.ndarray, entry_book: np.ndarray) -> np.ndarray:
        if self.config.get("freeze_orient_goal") is True and self.state.orient_goal_T_world_book is not None:
            return np.asarray(self.state.orient_goal_T_world_book, dtype=np.float64)
        goal = entry_book.copy()
        goal[:2, 3] = current_book[:2, 3]
        goal = self._lifted_book_goal(goal)
        if self.config.get("freeze_orient_goal") is True:
            self.state.orient_goal_T_world_book = goal.tolist()
        return goal

    def _target_and_action(self, public, book: RoleDelta, caddy: RoleDelta) -> tuple[StageTarget, np.ndarray, dict]:
        stage = self.state.stage
        E = self._eef(public)
        B = self._current_book_pose(public, book)
        caddy = self._delta_from_cache("caddy", caddy)
        C_body = self._caddy_pose_from_delta(caddy) if caddy.valid else None
        if stage == "approach_book":
            if B is None:
                target = StageTarget(stage, E.tolist(), -1.0, "hold_missing_book")
                return target, self._action_to_target(public, target), {"controller": "hold_missing_book"}
            goal = B @ self._grasp_goal("pregrasp")
            gate = self.config.get("approach_gate", {"position_m": 0.025, "rotation_rad": 0.25})
            if self.state.approach_substage == "above_current_book":
                above = goal.copy()
                above[2, 3] = max(float(E[2, 3]), float(goal[2, 3]) + float(self.config["offsets"].get("approach_clearance_z", 0.08)))
                pos_err = float(np.linalg.norm(above[:3, 3] - E[:3, 3]))
                rot_err = float(Rotation.from_matrix(above[:3, :3] @ E[:3, :3].T).magnitude())
                if pos_err <= float(gate["position_m"]) and rot_err <= float(gate["rotation_rad"]):
                    self.state.approach_substage = "pregrasp"
                else:
                    target = StageTarget(stage, above.tolist(), -1.0, "public_book_above_pregrasp")
                    return target, self._action_to_target(public, target), {"controller": "above_current_book", "position_error_m": pos_err, "rotation_error_rad": rot_err}
            action, diag = self.grasp_policy.command(B, E, public["gripper_qpos"], self._grasp_goal("pregrasp"), close=False)
            action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
            target = StageTarget(stage, goal.tolist(), float(action[6]), "v02_grasp_policy_pregrasp")
            return target, action, dict(diag, approach_substage=self.state.approach_substage)
        if stage == "short_grasp":
            if B is None:
                target = StageTarget(stage, E.tolist(), -1.0, "hold_missing_book")
                return target, self._action_to_target(public, target), {"controller": "hold_missing_book"}
            return self._short_grasp_action(public, B, E)
        if B is None:
            target = StageTarget(stage, E.tolist(), 1.0, "hold_missing_held_book")
            return target, self._action_to_target(public, target), {"controller": "hold_missing_held_book"}
        if stage == "lift_clear":
            B_goal = np.asarray(self.state.lift_goal_T_world_book, dtype=np.float64) if self.state.lift_goal_T_world_book is not None else B.copy()
            refinement = self._maybe_refine_lift_goal(public, self._current_facts, B_goal)
            if refinement.get("applied"):
                B_goal = np.asarray(self.state.lift_goal_T_world_book, dtype=np.float64)
            target = StageTarget(stage, self._held_compensated_eef(B_goal).tolist(), 1.0, "entry_frozen_book_lift_goal")
            return target, self._action_to_target(public, target), {"controller": "held_compensated_lift", "lift_clearance_refinement": refinement}
        if C_body is None:
            target = StageTarget(stage, E.tolist(), 1.0, "hold_missing_caddy")
            return target, self._action_to_target(public, target), {"controller": "hold_missing_caddy"}
        C_site = self._world_caddy_site(C_body)
        entry_site_book = self._relation_matrix("T_back_site_book_entry")
        B_entry = C_site @ entry_site_book
        if stage == "orient_clear":
            # Rotate the held upright book toward the entry relation.  Optional
            # held-pose calibration is a fixed training prior and is applied
            # exactly once before the first orient goal is constructed.
            calibration = self._apply_held_pose_calibration_once()
            B = self._current_book_pose(public, book)
            B_goal = self._orient_book_goal(B, B_entry)
            target = StageTarget(stage, self._held_compensated_eef(B_goal).tolist(), 1.0, "held_book_entry_orientation")
            return target, self._action_to_target(public, target), {"controller": "held_compensated_orient", "freeze_orient_goal": bool(self.config.get("freeze_orient_goal")), "orient_goal_frozen": self.state.orient_goal_T_world_book is not None, "held_pose_calibration": calibration}
        if stage == "transport_above_caddy":
            B_goal = B_entry.copy()
            B_goal = self._lifted_book_goal(B_goal)
            target = StageTarget(stage, self._held_compensated_eef(B_goal).tolist(), 1.0, "held_book_to_caddy_entry_clearance")
            return target, self._action_to_target(public, target), {"controller": "held_compensated_transport"}
        if stage in ("near_entry", "p68_insert"):
            target = StageTarget(stage, self._held_compensated_eef(B_entry).tolist(), 1.0, "held_book_caddy_entry_relation")
            return target, self._action_to_target(public, target), {"controller": "held_compensated_near_entry"}
        raise ValueError("unknown stage")

    def _action_to_target(self, public, target: StageTarget) -> np.ndarray:
        E = self._eef(public)
        T = np.asarray(target.target_T_world_eef, dtype=np.float64)
        dp = T[:3, 3] - E[:3, 3]
        dr = Rotation.from_matrix(T[:3, :3] @ E[:3, :3].T).as_rotvec()
        gains = self.config["gains"]
        return _clip_action(dp * float(gains.get("position_gain", 1.0)),
                            dr * float(gains.get("rotation_gain", 1.0)), target.grip,
                            translation_scale=float(gains.get("translation_scale", 0.05)),
                            rotation_scale=float(gains.get("rotation_scale", 0.5)),
                            max_action=float(gains.get("max_action", 1.0)))


def load_controller_config(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "prove.p70-object-controller-config.v1":
        raise ValueError("wrong P70 controller config schema")
    return value
