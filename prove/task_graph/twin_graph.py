"""Compiler and runtime primitives for root-designed digital-twin task graphs.

The compiler derives *reference windows* from reviewed gripper events.  Those
windows select priors and training samples; runtime transitions still require
explicit Boolean facts and never succeed from elapsed reference time alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import numpy as np


GRAPH_SCHEMA = "graph-self-adapter.twin-task-graph.v1"
ROOT_PLAN_SCHEMA = "graph-self-adapter.twin-root-plan.v1"
REVISION_SCHEMA = "graph-self-adapter.twin-root-overrides.v1"
SHORT_PHASES = frozenset(("Grasp", "Align", "Insert", "Push", "Pull"))
HOLDING_CONTRACTS = frozenset(("bilateral_contact", "loaded_contact", "held_or_supported", "none"))
SOURCE_GROUNDING_CONTRACTS = frozenset(("legacy", "initial_visible_until_first_manipulation"))
SOURCE_GROUNDING_PLANES = frozenset(("none", "fixed_support_xy_z0"))
PREDICATES = frozenset((
    "reference_progress", "holding", "functional_grasp_offset", "object_target_relation",
    "visible_displacement", "released", "stabilized", "supported_contact", "goal_relation",
    "stage_goal", "actor_command_reached", "finite_simulation", "holding_confirmed",
))
PUBLIC_OBSERVATION_KEYS = frozenset((
    "time_sec", "qpos", "qvel", "motor_position", "rgb", "depth_m",
    "role_masks", "camera_K", "camera_to_world", "eef_pose", "eef_jacobian",
))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _nearest_time(times: np.ndarray, value: float, tolerance: float = 0.15) -> tuple[int, float]:
    i = int(np.argmin(np.abs(times - value)))
    if abs(float(times[i]) - value) > tolerance:
        raise ValueError(f"reviewed event {value:.3f}s has no reference sample within {tolerance:.3f}s")
    return i, float(times[i])


def _phase_done(phase: str, cycle: Mapping[str, Any]) -> list[str]:
    relation = str(cycle["goal_relation"])
    if phase == "Approach":
        return ["near_object"]
    if phase == "Grasp":
        return ["bilateral_contact"]
    if phase == "Lift":
        return ["bilateral_contact", "lift_clear"]
    if phase == "Move":
        return ["bilateral_contact", "near_target"]
    if phase == "Align":
        return ["bilateral_contact", "near_target"]
    if phase in ("Insert", "Push", "Pull"):
        return ["bilateral_contact", relation]
    if phase == "Release":
        return ["released", "object_stabilized", relation]
    if phase == "Hold":
        return ["bilateral_contact", relation]
    if phase in ("Retreat", "Wait"):
        # These post-release reference segments preserve continuous robot
        # motion between root-authored cycles; task semantics were already
        # completed by the Boolean-gated Release.
        return []
    raise ValueError(f"unsupported root semantic phase {phase!r}")


def _stage_windows(cycle: Mapping[str, Any], start: float, end: float) -> list[tuple[float, float]]:
    phases = list(cycle["semantic_plan"])
    close_t = float(cycle["close_time_sec"])
    open_t = None if cycle.get("open_time_sec") is None else float(cycle["open_time_sec"])
    if not start <= close_t <= end + 0.2 or (open_t is not None and not close_t < open_t <= end + 0.2):
        raise ValueError(f"invalid cycle event ordering for {cycle['cycle_id']}")
    n = len(phases)
    grasp_i = phases.index("Grasp") if "Grasp" in phases else -1
    release_i = phases.index("Release") if "Release" in phases else n
    cuts = np.zeros(n + 1, dtype=np.float64)
    cuts[0], cuts[-1] = start, end
    before_close = max(start, close_t - 0.30)
    after_close = min(end, close_t + 0.30)
    if grasp_i >= 0:
        cuts[:grasp_i + 1] = np.linspace(start, before_close, grasp_i + 1)
        cuts[grasp_i], cuts[grasp_i + 1] = before_close, after_close
    middle_start = grasp_i + 1
    if release_i < n:
        assert open_t is not None
        before_open = max(after_close, open_t - 0.40)
        count = release_i - middle_start
        cuts[middle_start:release_i + 1] = np.linspace(after_close, before_open, count + 1)
        cuts[release_i], cuts[release_i + 1] = before_open, min(end, open_t + 0.60)
        if release_i + 1 < n:
            cuts[release_i + 1:] = np.linspace(cuts[release_i + 1], end, n - release_i)
    else:
        count = n - middle_start
        cuts[middle_start:] = np.linspace(after_close, end, count + 1)
    cuts = np.maximum.accumulate(cuts)
    return [(float(cuts[i]), float(cuts[i + 1])) for i in range(n)]


def _predicate_rows(values: Iterable[Any], *, label: str) -> list[dict[str, Any]]:
    rows = []
    for i, value in enumerate(values):
        row = {"predicate": value} if isinstance(value, str) else dict(value)
        kind = row.get("predicate")
        if kind not in PREDICATES:
            raise ValueError(f"{label}[{i}] unsupported predicate: {kind!r}")
        if kind == "stage_goal":
            key = row.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label}[{i}] stage_goal requires a non-empty Boolean key")
            for optional in ("object_role", "target_role", "goal_relation"):
                if optional in row and (not isinstance(row[optional], str) or not row[optional]):
                    raise ValueError(f"{label}[{i}] stage_goal {optional} must be a non-empty string")
            debounce = float(row.get("debounce_sec", 0.0))
            if not 0.0 <= debounce <= 0.15:
                raise ValueError(f"{label}[{i}] stage_goal debounce_sec must be within [0,.15]")
            row["debounce_sec"] = debounce
        if kind == "actor_command_reached":
            tolerance = float(row.get("tolerance_rad", 0.025))
            if not 0.0 < tolerance <= 0.10:
                raise ValueError(
                    f"{label}[{i}] actor_command_reached tolerance_rad must be within (0,.10]")
            row["tolerance_rad"] = tolerance
        row.setdefault("id", f"{label}.{i:02d}.{kind}")
        rows.append(row)
    return rows


def _task_override(revision: Mapping[str, Any] | None, task_name: str) -> dict[str, Any] | None:
    if revision is None:
        return None
    tasks = revision.get("tasks", {})
    if isinstance(tasks, list):
        matches = [x for x in tasks if x.get("task") == task_name]
        return None if not matches else dict(matches[0])
    value = tasks.get(task_name)
    return None if value is None else dict(value)


def _cycle_override(task_revision: Mapping[str, Any] | None, cycle_id: str) -> dict[str, Any] | None:
    if task_revision is None:
        return None
    cycles = task_revision.get("cycles", {})
    if isinstance(cycles, list):
        matches = [x for x in cycles if x.get("cycle_id") == cycle_id]
        return None if not matches else dict(matches[0])
    value = cycles.get(cycle_id)
    return None if value is None else dict(value)


def compile_task_graph(task: Mapping[str, Any], *, root_plan_sha256: str,
                       reference_times: Iterable[float], revision: Mapping[str, Any] | None = None,
                       revision_sha256: str | None = None) -> dict[str, Any]:
    """Compile one root task without inferring or replacing semantic cycles."""
    times = np.asarray(tuple(reference_times), dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("reference_times must be finite and strictly increasing")
    cycles = list(task.get("cycles", ()))
    if not cycles:
        raise ValueError("root-authored cycles are required")
    stages: list[dict[str, Any]] = []
    task_revision = _task_override(revision, str(task["task"]))
    revision_source_grounding = ("legacy" if revision is None else
                                 revision.get("source_grounding", "legacy"))
    if revision_source_grounding not in SOURCE_GROUNDING_CONTRACTS:
        raise ValueError(f"unsupported root source grounding {revision_source_grounding!r}")
    previous_end = float(times[0])
    for ci, cycle in enumerate(cycles):
        for key in ("cycle_id", "actor", "arm", "object", "object_part", "target",
                    "close_time_sec", "open_time_sec", "goal_relation", "semantic_plan"):
            if key not in cycle:
                raise ValueError(f"cycle missing {key}")
        _nearest_time(times, float(cycle["close_time_sec"]))
        if cycle.get("open_time_sec") is not None:
            _nearest_time(times, float(cycle["open_time_sec"]))
        cycle_end = float(times[-1]) if ci + 1 == len(cycles) else float(cycles[ci + 1]["close_time_sec"]) - 0.45
        override = _cycle_override(task_revision, str(cycle["cycle_id"]))
        source_grounding_plane = ("none" if override is None else
                                  override.get("source_grounding_plane", "none"))
        if source_grounding_plane not in SOURCE_GROUNDING_PLANES:
            raise ValueError(
                f"unsupported source grounding plane {source_grounding_plane!r} for {cycle['cycle_id']}")
        target_grounding = ("current_visible_pair" if override is None else
                            override.get("target_grounding", "current_visible_pair"))
        if target_grounding not in ("current_visible_pair", "scene_entry_stationary_target"):
            raise ValueError(f"unsupported target grounding {target_grounding!r}")
        holding_default = "bilateral_contact" if override is None else override.get("holding_contract", "bilateral_contact")
        if holding_default not in HOLDING_CONTRACTS:
            raise ValueError(f"unsupported holding contract {holding_default!r}")
        contracts = None if override is None else override.get("stage_contracts")
        if contracts is not None:
            contracts = [dict(x) for x in contracts]
            phases = [str(x["phase"]) for x in contracts]
            windows = [(float(x["start_time_sec"]), float(x["end_time_sec"])) for x in contracts]
            if any(not previous_end - .2 <= a < b <= cycle_end + .2 for a, b in windows):
                raise ValueError(f"stage override outside cycle bounds: {task['task']} {cycle['cycle_id']}")
        else:
            compiled_cycle = dict(cycle)
            phases = list(cycle["semantic_plan"])
            open_time = cycle.get("open_time_sec")
            if open_time is not None and "Release" in phases and float(open_time) + 0.65 < cycle_end:
                phases.append("Wait")
            compiled_cycle["semantic_plan"] = phases
            windows = _stage_windows(compiled_cycle, previous_end, cycle_end)
        occurrences: dict[str, int] = {}
        for pi, (phase, window) in enumerate(zip(phases, windows)):
            occurrences[phase] = occurrences.get(phase, 0) + 1
            contract = None if contracts is None else contracts[pi]
            if contract is not None and int(contract.get("occurrence", occurrences[phase])) != occurrences[phase]:
                raise ValueError(f"stage occurrence mismatch for {task['task']} {cycle['cycle_id']} {phase}")
            a, b = window
            ia, snapped_a = _nearest_time(times, a, max(0.15, float(np.max(np.diff(times)))))
            ib, snapped_b = _nearest_time(times, b, max(0.15, float(np.max(np.diff(times)))))
            stage_id = f"{cycle['cycle_id']}.{pi + 1:02d}.{phase.lower()}"
            hold = phase not in ("Approach", "Grasp", "Release", "Retreat", "Wait")
            stage_holding = holding_default if contract is None else contract.get("holding_contract", holding_default)
            if stage_holding not in HOLDING_CONTRACTS:
                raise ValueError(f"unsupported holding contract {stage_holding!r}")
            if stage_holding == "held_or_supported" and phase not in ("Align", "Insert"):
                raise ValueError(f"{stage_id} held_or_supported is restricted to Align/Insert")
            source_grounding = (revision_source_grounding if override is None else
                                override.get("source_grounding", revision_source_grounding))
            if source_grounding not in SOURCE_GROUNDING_CONTRACTS:
                raise ValueError(f"{stage_id} unsupported source grounding {source_grounding!r}")
            if contract is None:
                done = _phase_done(str(phase), cycle)
                done.append("reference_progress_reached")
                completion_predicates = []
                invariant_predicates = []
            else:
                completion_predicates = _predicate_rows(contract.get("completion", ()), label="completion")
                invariant_predicates = _predicate_rows(contract.get("invariants", ()), label="invariant")
                if phase not in ("Wait", "Retreat") and not any(x["predicate"] != "reference_progress" for x in completion_predicates):
                    raise ValueError(f"{stage_id} cannot complete from reference progress alone")
                done = [x["id"] for x in completion_predicates]
            speed_by_phase = {} if override is None else override.get("reference_speed_scale_by_phase", {})
            speed = float(({} if contract is None else contract).get(
                "reference_speed_scale", speed_by_phase.get(phase, 1.0)))
            if not np.isfinite(speed) or not 0.0 < speed <= 1.0:
                raise ValueError(f"{stage_id} reference_speed_scale must be within (0,1]")
            confirm_sec = (float(({} if override is None else override).get(
                "grasp_stable_contact_sec", 0.0)) if phase == "Grasp" else 0.0)
            if not np.isfinite(confirm_sec) or not 0.0 <= confirm_sec <= 1.0:
                raise ValueError(f"{stage_id} grasp_stable_contact_sec must be within [0,1]")
            if confirm_sec:
                if contract is None or not any(row["predicate"] == "holding" for row in completion_predicates):
                    raise ValueError("grasp confirmation requires a contracted completion holding predicate")
                confirmation = {"id": "completion.holding_confirmed", "predicate": "holding_confirmed",
                                "minimum_sec": confirm_sec}
                completion_predicates.append(confirmation)
                done.append(confirmation["id"])
            reference_commands = max(2, int(np.ceil(max(0.0, snapped_b - snapped_a) / speed * 10.0)))
            entry = {
                "previous_stage_complete": bool(stages),
                "role_visible_or_held": True,
                "cache_max_age_sec": .5,
            }
            if contract is not None and contract.get("entry") is not None:
                override_entry = contract["entry"]
                if not isinstance(override_entry, Mapping):
                    raise ValueError(f"{stage_id} entry override must be an object")
                unknown_entry = set(override_entry) - set(entry)
                if unknown_entry:
                    raise ValueError(f"{stage_id} unsupported entry fields: {sorted(unknown_entry)}")
                entry.update(override_entry)
                if not isinstance(entry["role_visible_or_held"], bool):
                    raise ValueError(f"{stage_id} role_visible_or_held must be Boolean")
            stages.append({
                "stage_id": stage_id,
                "cycle_id": cycle["cycle_id"],
                "phase": phase,
                "controller": "neural_short" if phase in SHORT_PHASES else "code_reference_retarget",
                "actor": cycle["actor"], "arm": cycle["arm"],
                "object": cycle["object"], "object_part": cycle["object_part"],
                "source_anchor_role": (cycle["object"] if override is None else
                                       override.get("source_anchor_role", cycle["object"])),
                "source_grounding": source_grounding,
                "source_grounding_plane": source_grounding_plane,
                "target": cycle["target"], "goal_relation": cycle["goal_relation"],
                "handover_region": bool(cycle.get("handover_region", False)),
                "handover_anchor": None if override is None else override.get("handover_anchor"),
                "role_semantics": None if override is None else override.get("role_semantics"),
                "entry": entry,
                "holding_contract": stage_holding,
                "done": done,
                "done_mode": "all",
                "completion_predicates": completion_predicates,
                "invariant_predicates": invariant_predicates,
                "invariants": (([stage_holding] if hold and stage_holding != "none" else []) + ["finite_simulation"])
                              if contract is None else [x["id"] for x in invariant_predicates],
                "reference_commands": reference_commands,
                "max_budget_commands": reference_commands + (20 if phase == "Release" else 4) + int(np.ceil(confirm_sec * 10.0)),
                "reference": {
                    "start_index": ia, "end_index": ib,
                    "start_time_sec": snapped_a, "end_time_sec": snapped_b,
                    "progress_only": True,
                    "boundary_evidence": "root reviewed gripper events; nearest demonstration FK/contact samples",
                },
                "next": None,
            })
            if target_grounding != "current_visible_pair":
                stages[-1]["target_grounding"] = target_grounding
            if speed != 1.0 or confirm_sec:
                stages[-1].update(reference_speed_scale=speed, grasp_stable_contact_sec=confirm_sec)
        previous_end = cycle_end
    for current, nxt in zip(stages, stages[1:]):
        current["next"] = nxt["stage_id"]
    feedback_contract = None if revision is None else revision.get("local_feedback_contract")
    if feedback_contract is not None:
        if (not isinstance(feedback_contract, Mapping) or
                feedback_contract.get("schema") != "graph-self-adapter.twin-feedback-contract.v2"):
            raise ValueError("local_feedback_contract must be the root-authored v2 contract")
        feedback_contract = json.loads(json.dumps(feedback_contract))
    return {
        "schema": GRAPH_SCHEMA,
        "task": task["task"],
        "root_plan_sha256": root_plan_sha256,
        "revision_override_sha256": revision_sha256,
        "root_designed_cycles": True,
        "runtime_inputs": ["visible_role_masks", "rgbd", "camera_calibration", "proprioception", "fixed_training_reference", "boolean_supervisor"],
        "forbidden_runtime_inputs": ["continuous_object_body_pose", "future_recorded_object_pose", "success_label_for_adaptation"],
        "single_initial_restore": True,
        "local_feedback_contract": feedback_contract,
        "source_grounding": revision_source_grounding,
        "stages": stages,
    }


def compile_root_plan(path: str | Path, revision_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    path = Path(path)
    plan = json.loads(path.read_text())
    if plan.get("schema") != ROOT_PLAN_SCHEMA or plan.get("root_designed") is not True:
        raise ValueError("a root-designed v1 plan is required")
    digest = sha256(path)
    revision = None
    revision_digest = None
    if revision_path is not None:
        revision_path = Path(revision_path)
        revision = json.loads(revision_path.read_text())
        if revision.get("schema") != REVISION_SCHEMA or revision.get("root_designed") is not True:
            raise ValueError("a root-designed R1 override is required")
        revision_digest = sha256(revision_path)
    out = {}
    for task in plan["tasks"]:
        with np.load(task["sources"]["motor_targets.npz"]["path"], allow_pickle=False) as data:
            times = data["timestamps"]
        out[task["task"]] = compile_task_graph(task, root_plan_sha256=digest, reference_times=times,
                                                revision=revision, revision_sha256=revision_digest)
    return out


def surface_centroid(depth_m: Any, mask: Any, camera_K: Any,
                     camera_to_world: Any, *, min_points: int = 12) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Robust visible-surface centroid; this never accepts a simulator body pose."""
    depth = np.asarray(depth_m, dtype=np.float64)
    visible = np.asarray(mask, dtype=bool)
    K = np.asarray(camera_K, dtype=np.float64).reshape(3, 3)
    T = np.asarray(camera_to_world, dtype=np.float64).reshape(4, 4)
    valid = visible & np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(valid)
    if len(xx) < min_points:
        return None, {"known": False, "visible_points": int(len(xx)), "reason": "insufficient_visible_depth"}
    z = depth[yy, xx]
    xyz_camera = np.stack(((xx - K[0, 2]) * z / K[0, 0],
                           (yy - K[1, 2]) * z / K[1, 1], z), axis=1)
    xyz_world = xyz_camera @ T[:3, :3].T + T[:3, 3]
    center = np.median(xyz_world, axis=0)
    return center, {"known": True, "visible_points": int(len(xx)), "estimator": "median_calibrated_visible_surface"}


class SurfaceMemory:
    """Hold the previous valid public estimate while retaining unknown status."""

    def __init__(self) -> None:
        self.last: dict[str, np.ndarray] = {}

    def estimate(self, role: str, observation: Mapping[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
        masks = observation["role_masks"]
        if role not in masks:
            value = self.last.get(role)
            return value, {"known": False, "held_previous": value is not None, "reason": "role_mask_missing"}
        estimates, records = [], []
        for i, mask in enumerate(masks[role]):
            value, record = surface_centroid(observation["depth_m"][i], mask,
                                             observation["camera_K"][i], observation["camera_to_world"][i])
            records.append(record)
            if value is not None:
                estimates.append(value)
        if estimates:
            value = np.median(np.stack(estimates), axis=0)
            self.last[role] = value
            return value, {"known": True, "held_previous": False, "views": records}
        value = self.last.get(role)
        return value, {"known": False, "held_previous": value is not None, "reason": "role_not_visible", "views": records}


def validate_public_observation(observation: Mapping[str, Any]) -> None:
    extra = set(observation) - PUBLIC_OBSERVATION_KEYS
    missing = PUBLIC_OBSERVATION_KEYS - set(observation)
    if extra or missing:
        raise ValueError(f"public observation boundary mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for key in ("qpos", "qvel", "motor_position", "depth_m", "camera_K", "camera_to_world", "eef_pose", "eef_jacobian"):
        if not np.isfinite(np.asarray(observation[key], dtype=np.float64)).all():
            raise ValueError(f"non-finite public observation: {key}")


@dataclass
class RuntimeState:
    stage_index: int = 0
    stage_commands: int = 0
    total_commands: int = 0
    failure: str | None = None
    completed: bool = False
    entry_object_delta: list[float] | None = None
    entry_target_delta: list[float] | None = None
    archived_z: list[dict[str, Any]] = field(default_factory=list)
    entry_rejections: list[dict[str, Any]] = field(default_factory=list)


class TwinWorldProtocol(Protocol):
    def reset(self, condition: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def public_observation(self) -> Mapping[str, Any]: ...
    def boolean_facts(self, stage: Mapping[str, Any]) -> Mapping[str, bool | None]: ...
    def step_motor(self, motor_position: np.ndarray) -> Mapping[str, Any]: ...
    def task_outcome(self) -> Mapping[str, Any]: ...


class TwinGraphRuntime:
    """Stage machine enforcing Boolean transitions and a single world reset."""

    def __init__(self, graph: Mapping[str, Any]) -> None:
        if graph.get("schema") != GRAPH_SCHEMA:
            raise ValueError("unsupported twin graph")
        self.graph = dict(graph)
        self.state = RuntimeState()
        self.surface_memory = SurfaceMemory()
        self.trace: list[dict[str, Any]] = []

    @property
    def stage(self) -> Mapping[str, Any]:
        return self.graph["stages"][self.state.stage_index]

    def reset(self) -> None:
        self.state = RuntimeState()
        self.surface_memory = SurfaceMemory()
        self.trace.clear()

    def observe_roles(self, public: Mapping[str, Any]) -> dict[str, Any]:
        validate_public_observation(public)
        stage = self.stage
        out = {}
        for role in (stage["object"], stage["target"]):
            value, record = self.surface_memory.estimate(role, public)
            out[role] = {"surface_centroid_world": None if value is None else value.tolist(), **record}
        return out

    def transition(self, facts: Mapping[str, bool | None]) -> bool:
        stage = self.stage
        invariant_values = [facts.get(name) for name in stage["invariants"]]
        if any(value is False for value in invariant_values):
            self.state.failure = "invariant_failed"
            return False
        if any(value is None for value in invariant_values):
            if self.state.stage_commands >= int(stage["max_budget_commands"]):
                self.state.failure = "stage_invariant_unknown"
            return False
        done = all(facts.get(name) is True for name in stage["done"])
        if not done:
            if self.state.stage_commands >= int(stage["max_budget_commands"]):
                values = [facts.get(name) for name in stage["done"]]
                self.state.failure = ("stage_completion_unknown" if any(value is None for value in values)
                                      else "stage_budget_without_boolean_completion")
            return False
        if stage["next"] is None:
            self.state.completed = True
            return True
        self.state.stage_index += 1
        self.state.stage_commands = 0
        self.state.entry_object_delta = None
        self.state.entry_target_delta = None
        return True

    def record_step(self, public: Mapping[str, Any], roles: Mapping[str, Any],
                    facts: Mapping[str, Any], motor_target: Any,
                    controller_record: Mapping[str, Any]) -> None:
        self.trace.append({
            "command": self.state.total_commands,
            "time_sec": float(public["time_sec"]),
            "stage_id": self.stage["stage_id"],
            "qpos": np.asarray(public["qpos"]).tolist(),
            "public_roles": dict(roles),
            "boolean_facts": dict(facts),
            "motor_target": np.asarray(motor_target).tolist(),
            "controller": dict(controller_record),
        })
        self.state.stage_commands += 1
        self.state.total_commands += 1
