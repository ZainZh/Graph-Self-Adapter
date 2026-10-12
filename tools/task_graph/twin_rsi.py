#!/usr/bin/env python3
"""Prepare, train, execute, and summarize root-designed twin task packages."""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SHORT_PHASE_ORDER = ("Grasp", "Align", "Insert", "Push", "Pull")
DEFAULT_PLAN = REPO / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json"
DEFAULT_ASSETS = REPO / "evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json"
DEFAULT_PACKAGES = REPO / "task_packages"
TWIN_PYTHON = Path("/home/zheng/github/Mujoco-RSI/.venv/bin/python")
TRAIN_PYTHON = Path("/home/zheng/github/Interaction-RSI/data/runtime/libero-pi05-venv-clean/bin/python")
V02_RUNTIME_SCHEMA = "graph-self-adapter.twin-runtime-contract.v2"


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GRAPH = _load_file("_twin_graph_runtime", REPO / "prove/task_graph/twin_graph.py")


def _physics():
    return _load_file("_twin_physics_runtime", REPO / "prove/task_graph/twin_physics.py")


def _skill():
    return _load_file("_twin_local_skill_runtime", REPO / "prove/fast_adapter/twin_local_skill.py")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def task_from_plan(plan: Mapping[str, Any], task: str) -> dict[str, Any]:
    matches = [x for x in plan["tasks"] if x["task"] == task]
    if len(matches) != 1:
        raise ValueError(f"task must occur exactly once in root plan: {task}")
    return matches[0]


def verify_source_bindings(task: Mapping[str, Any], asset_catalog: Mapping[str, Any]) -> dict[str, Any]:
    checked = []
    for label, item in task["sources"].items():
        path = Path(item["path"])
        actual = digest(path)
        if actual != item["sha256"]:
            raise ValueError(f"frozen source changed: {label}: {path}")
        checked.append({"label": label, "path": str(path), "sha256": actual})
    binding = asset_catalog["task_assets"][task["task"]]
    if binding["scene_sha256"] != task["sources"]["scene.xml"]["sha256"]:
        raise ValueError("scene hash disagrees with root external asset catalog")
    for path_string, expected in binding["external_asset_hashes"].items():
        path = Path(path_string)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"external scene asset changed or missing: {path}")
    return {"sources": checked, "external_assets": len(binding["external_asset_hashes"])}


def package_dir(task: str, root: Path = DEFAULT_PACKAGES) -> Path:
    return root / f"twin_{task}" / "solution"


def _obj(value: Any) -> Any:
    return dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value


def _field(value: Any, name: str, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def visible_centroid(frame: Any, role: str) -> list[float] | None:
    roles = _field(frame, "roles", {})
    item = roles.get(role) if isinstance(roles, Mapping) else None
    value = _field(item, "centroid_world_m") if item is not None else None
    return None if value is None else np.asarray(value, dtype=float).reshape(3).tolist()


class VisibleCentroidCache:
    """Bounded public last-valid cache; cached values are never new z feedback."""

    def __init__(self, max_age_sec: float = .5) -> None:
        self.max_age_sec = float(max_age_sec)
        self.rows: dict[str, tuple[np.ndarray, float]] = {}

    def update(self, frame: Any) -> None:
        now = float(_field(frame, "physical_time_sec"))
        for role in _field(frame, "roles", {}):
            value = visible_centroid(frame, role)
            if value is not None:
                self.rows[role] = (np.asarray(value, dtype=np.float64), now)

    def get(self, role: str, frame: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
        now = float(_field(frame, "physical_time_sec"))
        fresh = visible_centroid(frame, role)
        if fresh is not None:
            value = np.asarray(fresh, dtype=np.float64)
            self.rows[role] = (value, now)
            return value, {"known": True, "fresh": True, "held_previous": False, "age_sec": 0.0}
        previous = self.rows.get(role)
        if previous is None:
            return None, {"known": False, "fresh": False, "held_previous": False, "age_sec": None}
        value, timestamp = previous
        age = now - timestamp
        if age <= self.max_age_sec:
            return value.copy(), {"known": True, "fresh": False, "held_previous": True, "age_sec": age}
        return None, {"known": False, "fresh": False, "held_previous": False, "age_sec": age,
                      "reason": "last_valid_expired"}


class ReferenceCentroidCache:
    """Bounded past-only cache for offline visible reference centroids."""

    def __init__(self, max_age_sec: float = .5) -> None:
        self.max_age_sec = float(max_age_sec)
        self.rows: dict[str, tuple[np.ndarray, float]] = {}

    def get(self, role: str, centroids: Mapping[str, Any],
            reference_time_sec: float) -> tuple[np.ndarray | None, dict[str, Any]]:
        now = float(reference_time_sec)
        current = centroids.get(role)
        if current is not None:
            value = np.asarray(current, dtype=np.float64).reshape(3)
            self.rows[role] = (value, now)
            return value.copy(), {"known": True, "fresh": True, "held_previous": False,
                                  "age_sec": 0.0, "reference_time_sec": now}
        previous = self.rows.get(role)
        if previous is None:
            return None, {"known": False, "fresh": False, "held_previous": False,
                          "age_sec": None, "reference_time_sec": now}
        value, timestamp = previous
        age = now - timestamp
        if 0.0 <= age <= self.max_age_sec:
            return value.copy(), {"known": True, "fresh": False, "held_previous": True,
                                  "age_sec": age, "reference_time_sec": timestamp}
        return None, {"known": False, "fresh": False, "held_previous": False,
                      "age_sec": age, "reference_time_sec": timestamp,
                      "reason": "past_reference_expired" if age >= 0 else "future_reference_forbidden"}


class PublicVisualStability:
    """Fresh visible-centroid stability; used for contract-aware Release."""

    def __init__(self, motion_tolerance_m: float = .005) -> None:
        self.motion_tolerance_m = float(motion_tolerance_m)
        self.rows: dict[str, tuple[np.ndarray, float]] = {}

    def update(self, role: str, frame: Any) -> bool:
        value = visible_centroid(frame, role)
        if value is None:
            return False
        now = float(_field(frame, "physical_time_sec"))
        value = np.asarray(value, dtype=np.float64)
        previous = self.rows.get(role)
        stable_since = now if previous is None else previous[1]
        if previous is not None and np.linalg.norm(value - previous[0]) > self.motion_tolerance_m:
            stable_since = now
        self.rows[role] = (value, stable_since)
        return now - stable_since >= .20


def scene_entry_grounding_priors(frame: Any, reference_frame: Mapping[str, Any],
                                 roles: list[str]) -> dict[str, dict[str, Any]]:
    """Pair initial rendered surfaces only; never reads simulator object truth."""
    reference_centroids = reference_frame.get("surface_centroids_world_m", {})
    priors: dict[str, dict[str, Any]] = {}
    for role in roles:
        current = visible_centroid(frame, role)
        reference = reference_centroids.get(role)
        if current is None or reference is None:
            continue
        current_array = np.asarray(current, dtype=np.float64)
        reference_array = np.asarray(reference, dtype=np.float64)
        priors[role] = {
            "delta_m": (current_array - reference_array).tolist(),
            "scene_entry_time_sec": float(_field(frame, "physical_time_sec")),
            "reference_time_sec": float(reference_frame["time_sec"]),
            "assumption": "initial_visible_surface_pair_until_first_manipulation",
            "fresh_feedback": False,
        }
    return priors


def use_scene_entry_grounding(stage: Mapping[str, Any], manipulated_roles: set[str]) -> bool:
    return bool(
        stage.get("source_grounding") == "initial_visible_until_first_manipulation" and
        stage["object"] not in manipulated_roles)


def stationary_target_prior(stage: Mapping[str, Any], *, priors: Mapping[str, Any],
                            planned_objects: set[str], manipulated_roles: set[str],
                            physical_time: float) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Select a labeled initial visible fixture prior; no world/condition input."""
    requested = stage.get("target_grounding", "current_visible_pair")
    role = stage["target"]
    record = {"mode": "current_visible_pair", "requested_mode": requested,
              "target_role": role, "fresh_feedback": False, "visible_delta_m": None,
              "raw_visible_delta_m": None, "scene_entry_time_sec": None,
              "reference_time_sec": None, "age_sec": None,
              "fallback_reason": None, "assumption": "fixed_visible_pair_until_cycle_exit",
              "prior_assumption": "fixed_visible_pair_until_cycle_exit"}
    if requested == "current_visible_pair":
        return None, record
    if requested != "scene_entry_stationary_target":
        raise ValueError(f"unsupported target grounding {requested!r}")
    if role in manipulated_roles:
        record["fallback_reason"] = "target_already_manipulated"
    elif role in planned_objects:
        record["fallback_reason"] = "target_planned_for_manipulation"
    elif role not in priors:
        record["fallback_reason"] = "initial_visible_pair_unavailable"
    else:
        prior = priors[role]
        raw = np.asarray(prior["delta_m"], dtype=np.float64).reshape(3).copy()
        if not np.isfinite(raw).all():
            record["fallback_reason"] = "initial_visible_pair_nonfinite"
            return None, record
        record.update(mode=requested, visible_delta_m=raw.tolist(), raw_visible_delta_m=raw.tolist(),
                      scene_entry_time_sec=prior["scene_entry_time_sec"],
                      reference_time_sec=prior["reference_time_sec"],
                      age_sec=max(0.0, physical_time-float(prior["scene_entry_time_sec"])),
                      assumption="target_is_unmanipulated_stationary_fixture_for_this_task",
                      prior_assumption="target_is_unmanipulated_stationary_fixture_for_this_task")
        return raw, record
    return None, record


def augment_reference_contracts(graph: Mapping[str, Any], refs: Mapping[str, Any]) -> dict[str, Any]:
    """Derive public object-relative reference relations from existing rendered refs."""
    out = json.loads(json.dumps(refs))
    frames = out["frames"]
    relations: dict[str, Any] = {}
    handovers: dict[str, Any] = {}
    for stage in graph["stages"]:
        start = frames[str(stage["reference"]["start_index"])]
        end = frames[str(stage["reference"]["end_index"])]
        arm_index = 0 if stage["arm"] == "left" else 1
        os = start["surface_centroids_world_m"].get(stage["object"])
        oe = end["surface_centroids_world_m"].get(stage["object"])
        ts = start["surface_centroids_world_m"].get(stage["target"])
        te = end["surface_centroids_world_m"].get(stage["target"])
        tcp_end = np.asarray(end["tcp_position"])[arm_index]
        relations[stage["stage_id"]] = {
            "object_tcp_end": None if oe is None else (np.asarray(oe) - tcp_end).tolist(),
            "object_target_start": None if os is None or ts is None else (np.asarray(os) - np.asarray(ts)).tolist(),
            "object_target_end": None if oe is None or te is None else (np.asarray(oe) - np.asarray(te)).tolist(),
            "object_start": os, "object_end": oe, "target_start": ts, "target_end": te,
            "authority": "offline_visible_rgbd_reference_only",
        }
        if stage.get("handover_anchor") is not None and stage["phase"] == "Release" and oe is not None:
            handovers[stage["cycle_id"]] = {
                "role": stage["target"], "centroid_world_m": oe,
                "source_stage_id": stage["stage_id"], "source": "release_endpoint_after_demonstrated_deposit",
            }
    # A symbolic handover target is cycle-local.  Replace only frames belonging
    # to that cycle, never all occurrences of the shared role string.
    for stage in graph["stages"]:
        anchor = handovers.get(stage["cycle_id"])
        if anchor is None:
            continue
        for index in range(stage["reference"]["start_index"], stage["reference"]["end_index"] + 1):
            frames[str(index)]["surface_centroids_world_m"][stage["target"]] = anchor["centroid_world_m"]
        row = relations[stage["stage_id"]]
        if row["object_start"] is not None:
            row["object_target_start"] = (np.asarray(row["object_start"]) - np.asarray(anchor["centroid_world_m"])).tolist()
        if row["object_end"] is not None:
            row["object_target_end"] = (np.asarray(row["object_end"]) - np.asarray(anchor["centroid_world_m"])).tolist()
        row["target_start"] = anchor["centroid_world_m"]
        row["target_end"] = anchor["centroid_world_m"]
    out["stage_relations"] = relations
    out["handover_anchors"] = handovers
    out["schema"] = "graph-self-adapter.twin-visible-reference.v2"
    return out


def collect_reference_and_dataset(task: Mapping[str, Any], graph: Mapping[str, Any], *, samples_per_stage: int,
                                  seed: int, qpos_jitter_rad: float = 0.0,
                                  runtime_contract: Mapping[str, Any] | None = None
                                  ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run one engineering reference trace; labels use actual twin Jacobians only."""
    if samples_per_stage <= 0 or not 0.0 <= qpos_jitter_rad <= .05:
        raise ValueError("samples_per_stage must be positive and qpos_jitter_rad within [0,.05]")
    physics = _physics()
    run = Path(task["source_run"])
    world = physics.TwinWorld.from_run(
        run, width=640, height=360, control_hz=10, render=True,
        cup_release_control=("legacy_object_distance" if runtime_contract is None else
                             runtime_contract["cup_release_control"]))
    motor = np.load(task["sources"]["motor_targets.npz"]["path"], allow_pickle=False)
    times, joints, gripper = motor["timestamps"], motor["joints"], motor["gripper"]
    from scipy.interpolate import PchipInterpolator
    joint_curve = PchipInterpolator(times, joints, axis=0, extrapolate=False)
    joint_velocity = joint_curve.derivative()
    def grip_curve(value: float) -> np.ndarray:
        return np.asarray([np.interp(value, times, gripper[:, arm]) for arm in range(2)])
    semantic_roles = sorted({s["object"] for s in graph["stages"]} |
                            {s["target"] for s in graph["stages"]} |
                            {s.get("source_anchor_role", s["object"]) for s in graph["stages"]})
    roles = sorted(world.role_geom_ids)
    world.initialize(joints[0], gripper[0])
    frames: dict[int, Any] = {0: world.observe_visible(roles=roles)}
    proprio: dict[int, Any] = {0: world.proprioception()}
    wanted = sorted({i for s in graph["stages"] for i in range(
        int(s["reference"]["start_index"]), int(s["reference"]["end_index"]) + 1)})
    for i in range(1, len(times)):
        remaining = float(times[i] - world.data.time)
        if remaining <= 0:
            raise RuntimeError(f"reference replay advanced past sample {i}: {world.data.time} > {times[i]}")
        result = world.step_command_callback(
            lambda absolute_time: (
                joint_curve(np.clip(absolute_time, times[0], times[-1])),
                grip_curve(float(np.clip(absolute_time, times[0], times[-1]))),
                joint_velocity(np.clip(absolute_time, times[0], times[-1]))),
            remaining, observe=i in wanted, roles=roles, include_endpoint=False)
        if runtime_contract is not None and not world.numerical_safety()["simulation_finite"]:
            safety = world.numerical_safety()
            world.close()
            raise RuntimeError(f"portable reference trace numerical failure at sample {i}: {safety}")
        if i in wanted:
            frame = _field(result, "visible", None) or _field(result, "observation", None) or world.observe_visible(roles=roles)
            frames[i], proprio[i] = frame, world.proprioception()
    synthetic_anchors: dict[str, list[float] | None] = {}
    for stage in graph["stages"]:
        if stage["target"] in roles or stage["target"] in synthetic_anchors:
            continue
        endpoint = int(stage["reference"]["end_index"])
        synthetic_anchors[stage["target"]] = visible_centroid(frames[endpoint], stage["object"])
    refs = {"schema": "graph-self-adapter.twin-visible-reference.v1", "task": task["task"], "frames": {},
            "synthetic_semantic_anchors": synthetic_anchors,
            "runtime_materialization": {
                "cup_release_control": world.cup_release_control,
                "runtime_contract_canonical_sha256": (None if runtime_contract is None else
                    canonical_json_sha256(runtime_contract)),
            }}
    for i in wanted:
        if i not in frames:
            raise RuntimeError(f"reference frame {i} was not captured at its physical sample time")
        refs["frames"][str(i)] = {
            "time_sec": float(_field(frames[i], "physical_time_sec")),
            "surface_centroids_world_m": {
                role: (visible_centroid(frames[i], role) if role in roles else synthetic_anchors.get(role))
                for role in semantic_roles
            },
            "tcp_position": np.asarray(_field(proprio[i], "tcp_position")).tolist(),
            "tcp_rotation": np.asarray(_field(proprio[i], "tcp_rotation")).tolist(),
            "joint_pos": np.asarray(_field(proprio[i], "joint_pos")).tolist(),
        }
    rng = np.random.default_rng(seed)
    features, labels, stage_ids = [], [], []
    for stage in graph["stages"]:
        if stage["controller"] != "neural_short":
            continue
        arm_index = 0 if stage["arm"] == "left" else 1
        a, b = int(stage["reference"]["start_index"]), int(stage["reference"]["end_index"])
        candidate_indices = np.arange(max(0, a), min(len(joints), b + 1))
        if not len(candidate_indices):
            candidate_indices = np.asarray([min(a, len(joints) - 1)])
        for _ in range(samples_per_stage):
            i = int(rng.choice(candidate_indices))
            q = (joints[i, arm_index].astype(np.float64) +
                 rng.normal(0.0, qpos_jitter_rad, 6))
            qvel = np.asarray(joint_velocity(float(times[i]))[arm_index], dtype=np.float64) + rng.normal(0.0, 0.015, 6)
            object_delta = rng.uniform(-0.008, 0.008, 3)
            target_delta = rng.uniform(-0.008, 0.008, 3)
            z = rng.uniform(-0.006, 0.006, 3)
            progress = rng.uniform(0.0, 1.0)
            arm_onehot = np.eye(2)[arm_index]
            phase_onehot = np.eye(len(SHORT_PHASE_ORDER))[SHORT_PHASE_ORDER.index(stage["phase"])]
            correction = object_delta if stage["phase"] == "Grasp" else target_delta
            # The feature and teacher share reference-current correction sign.
            desired_pose_delta = np.r_[correction + z, np.zeros(3)]
            dq = np.asarray(world.solve_ik_delta(stage["arm"], desired_pose_delta, joints=q), dtype=np.float64).reshape(6) - q
            feature = np.r_[q, qvel, object_delta, target_delta, progress, arm_onehot, phase_onehot, z]
            features.append(feature); labels.append(dq); stage_ids.append(stage["stage_id"])
    metadata = {
        "schema": "graph-self-adapter.twin-skill-dataset.v1",
        "task": task["task"], "teacher": "actual_twin_jacobian_damped_least_squares",
        "teacher_runtime": "TwinWorld.solve_ik_delta with scratch/current MuJoCo kinematics",
        "teacher_max_joint_step_rad": 0.08,
        "feature_names": ["joint_pos[6]", "joint_vel[6]", "object_current_minus_reference[3]",
                          "target_current_minus_reference[3]", "reference_progress[1]", "arm_onehot[2]",
                          "short_phase_onehot_grasp_align_insert_push_pull[5]", "z_reference_pose_correction[3]"],
        "z_feature_slice": [26, 29], "correction_feature_slices": [[12, 18], [26, 29]],
        "samples": len(features), "samples_per_stage": samples_per_stage,
        "qpos_jitter_rad_std": qpos_jitter_rad,
        "joint_velocity_noise_rad_per_sec_std": 0.015,
        "geometry_perturbation_range_m": [-0.008, 0.008],
        "z_training_range_m": [-0.006, 0.006],
        "stage_counts": {s: stage_ids.count(s) for s in sorted(set(stage_ids))}, "seed": seed,
        "reference_runtime_materialization": refs["runtime_materialization"],
    }
    world.close()
    return refs, {"features": np.asarray(features, np.float32), "labels": np.asarray(labels, np.float32),
                  "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    plan, assets = load_json(args.plan), load_json(args.assets)
    if plan.get("schema") != GRAPH.ROOT_PLAN_SCHEMA:
        raise ValueError("wrong root plan schema")
    selected = [args.task] if args.task else [x["task"] for x in plan["tasks"]]
    if args.overrides is not None:
        overrides = load_json(args.overrides)
        if overrides.get("candidate_id") != args.candidate_id or overrides.get("root_designed") is not True:
            raise ValueError("candidate id/root authority does not match preparation override")
    runtime_contract = load_bound_runtime_contract(getattr(args, "runtime_contract", None), overrides if args.overrides else None)
    compiled = GRAPH.compile_root_plan(args.plan, args.overrides)
    report = {"schema": "graph-self-adapter.twin-prepare-report.v1", "tasks": {}}
    for task_name in selected:
        task, solution = task_from_plan(plan, task_name), package_dir(task_name, args.packages)
        verified = verify_source_bindings(task, assets)
        solution.mkdir(parents=True, exist_ok=True)
        dump_json(solution / "graph/task_graph.json", compiled[task_name])
        bindings = {"schema": "graph-self-adapter.twin-external-bindings.v1", "task": task_name,
                    "root_plan": {"path": str(args.plan.resolve()), "sha256": digest(args.plan)},
                    "external_assets": {"path": str(args.assets.resolve()), "sha256": digest(args.assets)},
                    "sources": task["sources"]}
        if args.overrides is not None:
            bindings["revision_override"] = {
                "path": str(args.overrides.resolve()), "sha256": digest(args.overrides)}
        dump_json(solution / "EXTERNAL_BINDINGS.json", bindings)
        if args.no_render:
            refs = {"schema": "graph-self-adapter.twin-visible-reference.v1", "task": task_name,
                    "frames": {}, "status": "render_pending"}
            dataset = None
        else:
            refs, dataset = collect_reference_and_dataset(task, compiled[task_name],
                                                          samples_per_stage=args.samples_per_stage,
                                                          seed=args.seed,
                                                          qpos_jitter_rad=args.qpos_jitter_rad,
                                                          runtime_contract=runtime_contract)
            if args.overrides is not None:
                refs = augment_reference_contracts(compiled[task_name], refs)
        dump_json(solution / "refs/visible_centroids.json", refs)
        if dataset is not None:
            (solution / "training").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(solution / "training/local_skill_dataset.npz", **dataset)
        manifest = {"schema": "graph-self-adapter.twin-solution.v1", "task": task_name,
                    "candidate_id": args.candidate_id, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "graph_sha256": digest(solution / "graph/task_graph.json"),
                    "references_sha256": digest(solution / "refs/visible_centroids.json"),
                    "dataset_sha256": None if dataset is None else digest(solution / "training/local_skill_dataset.npz"),
                    "verified_bindings": verified, "model": None, "candidate_freeze": None}
        if args.runtime_contract is not None:
            manifest["runtime_contract"] = {
                "path": str(args.runtime_contract.resolve()), "sha256": digest(args.runtime_contract)}
        dump_json(solution / "MANIFEST.json", manifest)
        report["tasks"][task_name] = manifest
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0 or args.batch_size <= 0 or args.action_limit_rad <= 0:
        raise ValueError("steps, batch_size and action_limit_rad must be positive")
    selected = [args.task] if args.task else [p.parent.parent.name.removeprefix("twin_") for p in args.packages.glob("twin_*/solution/MANIFEST.json")]
    report = {"schema": "graph-self-adapter.twin-train-report.v1", "tasks": {}}
    for task in selected:
        solution = package_dir(task, args.packages)
        dataset, model = solution / "training/local_skill_dataset.npz", solution / "models/local_skill.npz"
        if not dataset.is_file():
            raise FileNotFoundError(f"prepared training dataset missing for {task}")
        model.parent.mkdir(parents=True, exist_ok=True)
        command = [str(args.train_python), str(REPO / "tools/task_graph/twin_train_skills.py"),
                   "--dataset", str(dataset), "--output", str(model), "--steps", str(args.steps),
                   "--batch-size", str(args.batch_size), "--seed", str(args.seed),
                   "--action-limit", str(args.action_limit_rad)]
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        stats = json.loads(completed.stdout)
        manifest = load_json(solution / "MANIFEST.json")
        manifest["model"] = {"path": "models/local_skill.npz", "sha256": digest(model), "training": stats}
        dump_json(solution / "MANIFEST.json", manifest)
        freeze = {
            "schema": "graph-self-adapter.twin-candidate-freeze.v1", "task": task, "candidate_id": args.candidate_id,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "executed_rollouts": 0,
            "graph_sha256": digest(solution / "graph/task_graph.json"),
            "model_sha256": digest(model), "references_sha256": digest(solution / "refs/visible_centroids.json"),
            "dataset_sha256": digest(dataset),
            "runtime_inputs": load_json(solution / "graph/task_graph.json")["runtime_inputs"],
            "stage_budgets": {s["stage_id"]: s["max_budget_commands"] for s in load_json(solution / "graph/task_graph.json")["stages"]},
            "condition_design": {"development_seed": 51005, "test_seed": 71005, "translation_m": [0.003, 0.008], "orientation_shift": False, "force_shift": False},
            "code": {str(path.relative_to(REPO)): digest(path) for path in (
                REPO / "prove/task_graph/twin_graph.py", REPO / "prove/task_graph/twin_physics.py",
                REPO / "prove/fast_adapter/twin_local_skill.py", REPO / "tools/task_graph/twin_rsi.py")},
        }
        dump_json(solution / "CANDIDATE_FREEZE.json", freeze)
        manifest["candidate_freeze"] = {"path": "CANDIDATE_FREEZE.json", "sha256": digest(solution / "CANDIDATE_FREEZE.json")}
        dump_json(solution / "MANIFEST.json", manifest)
        report["tasks"][task] = {"model": manifest["model"], "candidate_freeze": manifest["candidate_freeze"]}
    return report


def observer_reference_refresh(*, parent_observer_sha256: str | None,
                               current_observer_sha256: str,
                               requested: bool | None) -> bool:
    """Resolve explicit refresh choice, otherwise refresh on encoder change."""
    if requested is not None:
        return bool(requested)
    return parent_observer_sha256 != current_observer_sha256


def repackage(args: argparse.Namespace) -> dict[str, Any]:
    """Create a revised graph/reference freeze while reusing R0 model bytes."""
    plan, assets, overrides = load_json(args.plan), load_json(args.assets), load_json(args.overrides)
    if overrides.get("candidate_id") != args.candidate_id or overrides.get("root_designed") is not True:
        raise ValueError("candidate id/root authority does not match revision override")
    runtime_contract = load_bound_runtime_contract(getattr(args, "runtime_contract", None), overrides)
    compiled = GRAPH.compile_root_plan(args.plan, args.overrides)
    selected = [args.task] if args.task else [x["task"] for x in plan["tasks"]]
    report = {"schema": "graph-self-adapter.twin-repackage-report.v1", "candidate_id": args.candidate_id,
              "parent_candidate": overrides.get("parent_candidate"), "tasks": {}}
    for task_name in selected:
        task = task_from_plan(plan, task_name)
        verify_source_bindings(task, assets)
        source = package_dir(task_name, args.source_packages)
        target = package_dir(task_name, args.packages)
        if target.exists():
            raise FileExistsError(f"revised package already exists: {target}")
        source_freeze = load_json(source / "CANDIDATE_FREEZE.json")
        source_checks = {
            "graph": digest(source / "graph/task_graph.json") == source_freeze["graph_sha256"],
            "model": digest(source / "models/local_skill.npz") == source_freeze["model_sha256"],
            "references": digest(source / "refs/visible_centroids.json") == source_freeze["references_sha256"],
        }
        if not all(source_checks.values()):
            raise ValueError(f"parent package artifact mismatch for {task_name}: {source_checks}")
        observer_rel = "prove/task_graph/twin_physics.py"
        # The relevant parent hash is the observer that materialized its
        # references, not merely the observer bundled with its runtime code.
        # Legacy packages lack this provenance and therefore refresh safely.
        parent_materialization = source_freeze.get("observer_reference_materialization", {})
        parent_observer_sha256 = parent_materialization.get("current_observer_sha256")
        parent_candidate_observer_sha256 = source_freeze.get("code", {}).get(observer_rel)
        current_observer_sha256 = digest(REPO / observer_rel)
        refresh_references = observer_reference_refresh(
            parent_observer_sha256=parent_observer_sha256,
            current_observer_sha256=current_observer_sha256,
            requested=args.refresh_observer_reference)
        reference_packages = getattr(args, "reference_packages", None)
        if reference_packages is not None:
            if runtime_contract is None:
                raise ValueError("shared v02 references require --runtime-contract")
            reference_solution = package_dir(task_name, reference_packages)
            reference_freeze = load_json(reference_solution / "CANDIDATE_FREEZE.json")
            reference_path = reference_solution / "refs/visible_centroids.json"
            if digest(reference_path) != reference_freeze["references_sha256"]:
                raise ValueError("shared reference freeze mismatch")
            parent_refs = load_json(reference_path)
            refresh_references = False
        elif refresh_references:
            # Re-render through the current observer on the exact native motor
            # trace.  The one-sample dataset is deliberately discarded: R1
            # keeps the original training file and exact frozen model bytes.
            parent_refs, _discarded_dataset = collect_reference_and_dataset(
                task, compiled[task_name], samples_per_stage=1, seed=51005,
                runtime_contract=runtime_contract)
        else:
            parent_refs = load_json(source / "refs/visible_centroids.json")
        if runtime_contract is not None:
            materialization = parent_refs.get("runtime_materialization", {})
            if (materialization.get("cup_release_control") != runtime_contract["cup_release_control"] or
                    materialization.get("runtime_contract_canonical_sha256") !=
                    canonical_json_sha256(runtime_contract)):
                raise ValueError("v02 visible references must be materialized under the bound portable contract")
        revised_refs = augment_reference_contracts(compiled[task_name], parent_refs)
        (target / "graph").mkdir(parents=True)
        (target / "refs").mkdir(parents=True)
        (target / "models").mkdir(parents=True)
        dump_json(target / "graph/task_graph.json", compiled[task_name])
        dump_json(target / "refs/visible_centroids.json", revised_refs)
        shutil.copy2(source / "models/local_skill.npz", target / "models/local_skill.npz")
        if (source / "training/local_skill_dataset.npz").is_file():
            (target / "training").mkdir(parents=True)
            shutil.copy2(source / "training/local_skill_dataset.npz", target / "training/local_skill_dataset.npz")
        bindings = load_json(source / "EXTERNAL_BINDINGS.json")
        bindings["revision_override"] = {"path": str(args.overrides.resolve()), "sha256": digest(args.overrides)}
        if args.runtime_contract is not None:
            bindings["runtime_contract"] = {
                "path": str(args.runtime_contract.resolve()), "sha256": digest(args.runtime_contract)}
        bindings["parent_candidate_freeze"] = {
            "path": str((source / "CANDIDATE_FREEZE.json").resolve()),
            "sha256": digest(source / "CANDIDATE_FREEZE.json"),
        }
        bindings["observer_reference_materialization"] = {
            "refreshed": refresh_references,
            "shared_reference_packages": (None if reference_packages is None else str(reference_packages.resolve())),
            "visible_frames_sha256": canonical_json_sha256(parent_refs["frames"]),
            "parent_observer_sha256": parent_observer_sha256,
            "parent_candidate_observer_sha256": parent_candidate_observer_sha256,
            "current_observer_sha256": current_observer_sha256,
            "method": ("native_motor_trace_current_visible_observer"
                       if refresh_references else "reused_parent_visible_references"),
        }
        dump_json(target / "EXTERNAL_BINDINGS.json", bindings)
        graph = compiled[task_name]
        freeze = {
            "schema": "graph-self-adapter.twin-candidate-freeze.v1", "task": task_name,
            "candidate_id": args.candidate_id, "parent_candidate": overrides.get("parent_candidate"),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "executed_rollouts": 0,
            "graph_sha256": digest(target / "graph/task_graph.json"),
            "model_sha256": digest(target / "models/local_skill.npz"),
            "references_sha256": digest(target / "refs/visible_centroids.json"),
            "dataset_sha256": (digest(target / "training/local_skill_dataset.npz")
                               if (target / "training/local_skill_dataset.npz").is_file() else None),
            "revision_override_sha256": digest(args.overrides),
            "parent_candidate_freeze_sha256": digest(source / "CANDIDATE_FREEZE.json"),
            "model_reused_without_retraining": True,
            "observer_reference_materialization": bindings["observer_reference_materialization"],
            "adaptation_scope": "visible_geometry_baseline; unchanged R0 weights; no P68 or meta-learned core claim",
            "runtime_inputs": graph["runtime_inputs"],
            "stage_budgets": {s["stage_id"]: s["max_budget_commands"] for s in graph["stages"]},
            "condition_design": source_freeze["condition_design"],
            "code": {str(path.relative_to(REPO)): digest(path) for path in (
                REPO / "prove/task_graph/twin_graph.py", REPO / "prove/task_graph/twin_physics.py",
                REPO / "prove/fast_adapter/twin_local_skill.py", REPO / "tools/task_graph/twin_rsi.py")},
        }
        if args.runtime_contract is not None:
            freeze["runtime_contract_sha256"] = digest(args.runtime_contract)
            freeze["condition_design"] = runtime_contract["condition_design"]
        dump_json(target / "CANDIDATE_FREEZE.json", freeze)
        source_manifest = load_json(source / "MANIFEST.json")
        manifest = {
            "schema": "graph-self-adapter.twin-solution.v1", "task": task_name,
            "candidate_id": args.candidate_id, "parent_candidate": overrides.get("parent_candidate"),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "graph_sha256": freeze["graph_sha256"], "references_sha256": freeze["references_sha256"],
            "dataset_sha256": freeze["dataset_sha256"],
            "model": {**source_manifest["model"], "sha256": freeze["model_sha256"],
                      "reused_from_candidate": overrides.get("parent_candidate"), "retrained": False},
            "candidate_freeze": {"path": "CANDIDATE_FREEZE.json", "sha256": digest(target / "CANDIDATE_FREEZE.json")},
            "revision_override": {"path": str(args.overrides.resolve()), "sha256": digest(args.overrides)},
            "observer_reference_materialization": bindings["observer_reference_materialization"],
        }
        if args.runtime_contract is not None:
            manifest["runtime_contract"] = bindings["runtime_contract"]
        dump_json(target / "MANIFEST.json", manifest)
        report["tasks"][task_name] = manifest
    return report


def validate_runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(value)
    if contract.get("schema") != V02_RUNTIME_SCHEMA:
        raise ValueError(f"runtime contract must use {V02_RUNTIME_SCHEMA}")
    if contract.get("cup_release_control") != "portable_tcp_displacement_v1":
        raise ValueError("v02 runtime contract requires portable_tcp_displacement_v1")
    if contract.get("numerical_integrity_fail_closed") is not True:
        raise ValueError("v02 runtime contract must fail closed on numerical integrity")
    design = contract.get("condition_design")
    if not isinstance(design, Mapping):
        raise ValueError("v02 runtime contract requires a condition_design object")
    grouping = design.get("knife_fixture_group_translation")
    if not isinstance(grouping, bool):
        raise ValueError("condition_design must declare knife_fixture_group_translation Boolean")
    return contract


def load_bound_runtime_contract(path: Path | None,
                                overrides: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    binding = None if overrides is None else overrides.get("runtime_contract_binding")
    if path is None:
        if binding is not None:
            raise ValueError("root override binds a runtime contract; pass --runtime-contract explicitly")
        return None
    contract = validate_runtime_contract(load_json(path))
    if binding is not None:
        if not isinstance(binding, Mapping):
            raise ValueError("runtime_contract_binding must be an object")
        bound_path = Path(str(binding.get("path", "")))
        if not bound_path.is_absolute():
            bound_path = REPO / bound_path
        if bound_path.resolve() != path.resolve() or binding.get("sha256") != digest(path):
            raise ValueError("runtime contract does not match root override binding")
    return contract


def condition_manifest(task: Mapping[str, Any], *, seed: int, trial: int, nominal: bool,
                       condition_design: Mapping[str, Any] | None = None) -> dict[str, Any]:
    reconstruction = load_json(Path(task["sources"]["reconstruction.json"]["path"]))
    roles = sorted(row["name"] for row in reconstruction["objects"])
    if nominal:
        transforms = {role: [0.0, 0.0, 0.0] for role in roles}
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, trial, sum(map(ord, task["task"]))]))
        transforms = {}
        for role in roles:
            radius = float(rng.uniform(0.003, 0.008)); angle = float(rng.uniform(-np.pi, np.pi))
            transforms[role] = [radius * np.cos(angle), radius * np.sin(angle), 0.0]
        if "drawer_shell" in transforms and "drawer" in transforms:
            transforms["drawer"] = list(transforms["drawer_shell"])
        if (task["task"] == "put_knife_board_new" and condition_design is not None and
                condition_design.get("knife_fixture_group_translation") is True):
            fixture_roles = ("knife_holder", "red_knife", "blue_knife")
            if not all(role in transforms for role in fixture_roles):
                raise ValueError("knife fixture grouping requires holder, red knife, and blue knife")
            fixture_delta = list(transforms["knife_holder"])
            for role in fixture_roles:
                transforms[role] = list(fixture_delta)
    grouped = bool(condition_design is not None and
                   condition_design.get("knife_fixture_group_translation") is True and
                   task["task"] == "put_knife_board_new")
    result = {"schema": "graph-self-adapter.twin-condition.v1", "task": task["task"], "seed": seed,
            "trial": trial, "family": ("nominal" if nominal else
                ("coherent_knife_fixture_table_plane_position_3_to_8mm" if grouped else
                 ("table_plane_position_shift_3_to_8mm" if condition_design is not None else
                  "table_plane_position_ood_3_to_8mm"))),
            "initial_translation": transforms, "actual_initial_translation_m": transforms,
            "orientation_shift": False, "force_shift": False}
    if condition_design is not None:
        result["condition_design"] = {
                "knife_fixture_group_translation": grouped,
                "knife_fixture_group_roles": (["knife_holder", "red_knife", "blue_knife"]
                                               if grouped else []),
            }
    return result


def project_source_grounding(delta: Any, plane: str) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(delta, dtype=np.float64).reshape(3)
    if plane == "none":
        return raw.copy(), {"plane": "none", "projected": False,
                            "raw_delta_m": raw.tolist(), "delta_m": raw.tolist()}
    if plane != "fixed_support_xy_z0":
        raise ValueError(f"unsupported source grounding plane {plane!r}")
    projected = raw.copy(); projected[2] = 0.0
    return projected, {
        "plane": plane, "projected": True,
        "assumption": "root_declared_source_on_fixed_height_support",
        "raw_delta_m": raw.tolist(), "delta_m": projected.tolist(),
        "uses_continuous_object_truth": False,
    }


def verify_freeze(solution: Path) -> dict[str, Any]:
    freeze = load_json(solution / "CANDIDATE_FREEZE.json")
    checks = {
        "graph": digest(solution / "graph/task_graph.json") == freeze["graph_sha256"],
        "model": digest(solution / "models/local_skill.npz") == freeze["model_sha256"],
        "references": digest(solution / "refs/visible_centroids.json") == freeze["references_sha256"],
    }
    for rel, expected in freeze["code"].items():
        checks[rel] = digest(REPO / rel) == expected
    if not all(checks.values()):
        raise ValueError(f"candidate freeze mismatch: {checks}")
    return checks


def validate_test_selection(selection: Mapping[str, Any], *, task: str,
                            candidate_id: str, solution: Path,
                            selection_path: Path) -> dict[str, Any]:
    """Bind one independent-test task to its exact frozen package artifacts."""
    if selection.get("status") != "frozen":
        raise ValueError("selection freeze status must be frozen")
    rows = selection.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("selection freeze requires a per-task tasks list")
    matches = [row for row in rows if isinstance(row, Mapping) and
               row.get("task") == task and row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"selection must contain exactly one {task}/{candidate_id} row")
    row = dict(matches[0])
    expected = {
        "graph_sha256": digest(solution / "graph/task_graph.json"),
        "model_sha256": digest(solution / "models/local_skill.npz"),
        "reference_sha256": digest(solution / "refs/visible_centroids.json"),
        "candidate_freeze_sha256": digest(solution / "CANDIDATE_FREEZE.json"),
    }
    mismatch = {key: {"selected": row.get(key), "actual": value}
                for key, value in expected.items() if row.get(key) != value}
    if mismatch:
        raise ValueError(f"selection artifact binding mismatch: {mismatch}")
    selected_packages = row.get("packages")
    if selected_packages is not None:
        if not isinstance(selected_packages, str) or not selected_packages:
            raise ValueError("selection row packages must be a non-empty path")
        selected_solution = Path(selected_packages)
        if not selected_solution.is_absolute():
            selected_solution = selection_path.resolve().parent / selected_solution
        if selected_solution.resolve() != solution.resolve():
            raise ValueError("selection row packages path does not match active solution")
    snapshot = row.get("source_snapshot")
    snapshot_path_value = snapshot.get("path") if isinstance(snapshot, Mapping) else snapshot
    if not isinstance(snapshot_path_value, str) or not snapshot_path_value:
        raise ValueError("selection row requires source_snapshot path")
    snapshot_path = Path(snapshot_path_value)
    if not snapshot_path.is_absolute():
        snapshot_path = selection_path.resolve().parent / snapshot_path
    if not snapshot_path.exists():
        raise ValueError(f"selected source_snapshot does not exist: {snapshot_path}")
    if isinstance(snapshot, Mapping) and snapshot.get("sha256") is not None:
        if not snapshot_path.is_file() or digest(snapshot_path) != snapshot["sha256"]:
            raise ValueError("selected source_snapshot hash mismatch")
    return row


def validate_v02_test_condition(selection: Mapping[str, Any], *, row: Mapping[str, Any],
                                selection_path: Path, runtime_contract_path: Path,
                                seed: int, trial: int, condition: Mapping[str, Any]) -> None:
    if selection.get("condition_seed") != seed or seed != 92005:
        raise ValueError("v02 test selection condition seed mismatch")
    actual_runtime = digest(runtime_contract_path)
    if selection.get("runtime_contract_sha256") != actual_runtime:
        raise ValueError("v02 test selection runtime contract mismatch")
    scope = selection.get("scope_binding")
    if not isinstance(scope, Mapping):
        raise ValueError("v02 test selection requires scope_binding")
    scope_path = Path(str(scope.get("path", "")))
    if not scope_path.is_absolute():
        scope_path = selection_path.resolve().parent / scope_path
    if not scope_path.is_file() or digest(scope_path) != scope.get("sha256"):
        raise ValueError("v02 test selection scope binding mismatch")
    hashes = row.get("condition_sha256_by_trial")
    if not isinstance(hashes, Mapping) or hashes.get(str(trial)) != canonical_json_sha256(condition):
        raise ValueError("v02 test selection condition hash mismatch")


@dataclasses.dataclass(frozen=True)
class StageReferenceClock:
    start: float
    end: float
    speed: float = 1.0

    def at(self, physical_elapsed: float) -> float:
        return min(self.end, self.start + max(0.0, physical_elapsed) * self.speed)

    def duration(self, physical_elapsed: float, nominal_duration: float) -> float:
        remaining = max(0.0, self.end - self.at(physical_elapsed)) / self.speed
        return min(nominal_duration, remaining) if remaining > 1e-6 else nominal_duration

    def progress(self, physical_elapsed: float) -> float:
        return 1.0 if self.end <= self.start else (self.at(physical_elapsed) - self.start) / (self.end - self.start)


def reference_motion_command(clock: StageReferenceClock, *, stage_elapsed: float,
                             command_elapsed: float, command_duration: float,
                             previous_correction: np.ndarray, correction_delta: np.ndarray,
                             joint_curve: Any, joint_velocity: Any, grip_curve: Any,
                             trajectory_start: float, trajectory_end: float):
    """Scale demonstration time; interpolate correction in physical time."""
    local = np.clip(command_elapsed / command_duration, 0.0, 1.0)
    query = float(np.clip(clock.at(stage_elapsed + max(0.0, command_elapsed)),
                          trajectory_start, trajectory_end))
    correction = previous_correction + local * correction_delta
    q_command = np.asarray(joint_curve(query)) + correction
    base_velocity = (np.zeros_like(q_command) if query >= clock.end - 1e-9 else
                     clock.speed * np.asarray(joint_velocity(query)))
    return q_command, grip_curve(query), base_velocity + correction_delta / command_duration


@dataclasses.dataclass
class HoldingConfirmation:
    """Debounce only completed-command contracted Boolean holding observations."""
    since: float | None = None
    elapsed: float = 0.0

    def reset(self) -> None:
        self.since = None
        self.elapsed = 0.0

    def update(self, holding: bool | None, physical_time: float, required: float) -> bool | None:
        if holding is not True:
            self.reset()
            return None if holding is None else False
        if self.since is None or physical_time < self.since:
            self.since = physical_time
        self.elapsed = max(0.0, physical_time - self.since)
        return self.elapsed + 1e-12 >= required



def holding_fact(contract: str, raw: Mapping[str, Any], *, target_role: str | None = None,
                 goal_relation: str | None = None) -> bool | None:
    if contract == "none":
        return True
    if contract == "bilateral_contact":
        return raw.get("bilateral_contact")
    if contract == "loaded_contact":
        return raw.get("single_finger_loaded_contact", raw.get("loaded_contact"))
    if contract == "held_or_supported":
        held = raw.get("bilateral_contact")
        if target_role == "table_handover_region":
            supported = (raw.get("object_table_support_contact")
                         if goal_relation == "accessible_to_right_arm" else None)
        else:
            supported = raw.get("object_support_contact", raw.get("supported_contact"))
        return None if held is None and supported is None else bool(held or supported)
    raise ValueError(f"unknown holding contract {contract!r}")


def attachment_feedback_holding(raw: Mapping[str, Any]) -> bool | None:
    """R2 feedback admits only a current bilateral object/TCP grip."""
    return raw.get("bilateral_contact")


def evaluate_contract_predicate(predicate: Mapping[str, Any], *, stage: Mapping[str, Any],
                                progress: float, raw: Mapping[str, Any],
                                object_value: np.ndarray | None, target_value: np.ndarray | None,
                                tcp: np.ndarray, relation_reference: Mapping[str, Any],
                                entry_geometry: Mapping[str, np.ndarray | None],
                                stage_goal_raw: Mapping[str, bool | None] | None = None,
                                actor_joint_error_max_rad: float | None = None) -> tuple[bool | None, dict[str, Any]]:
    kind = predicate["predicate"]
    result: bool | None
    diagnostic: dict[str, Any] = {"predicate": kind}
    if kind == "reference_progress":
        result = progress >= 1.0
    elif kind == "holding_confirmed":
        result = raw.get("holding_confirmed")
        diagnostic["continuous_holding_sec"] = raw.get("continuous_holding_sec", 0.0)
        diagnostic["minimum_sec"] = predicate["minimum_sec"]
    elif kind == "holding":
        result = holding_fact(stage["holding_contract"], raw, target_role=stage["target"],
                              goal_relation=stage["goal_relation"])
    elif kind == "functional_grasp_offset":
        reference = relation_reference.get("object_tcp_end")
        if object_value is None or reference is None:
            result = None
        else:
            residual = (object_value - tcp) - np.asarray(reference)
            error = float(np.linalg.norm(residual)); tolerance = float(predicate["tolerance_m"])
            result = error <= tolerance; diagnostic.update(error_m=error, tolerance_m=tolerance)
    elif kind == "object_target_relation":
        reference = relation_reference.get("object_target_end")
        if object_value is None or target_value is None or reference is None:
            result = None
        else:
            residual = (object_value - target_value) - np.asarray(reference)
            metric = predicate.get("metric", "xyz")
            selected = residual[:2] if metric == "xy" else residual
            error = float(np.linalg.norm(selected)); tolerance = float(predicate["tolerance_m"])
            result = error <= tolerance; diagnostic.update(error_m=error, tolerance_m=tolerance, metric=metric)
    elif kind == "visible_displacement":
        role = predicate.get("role", stage["object"])
        current_role = object_value if role == stage["object"] else target_value
        start_role = entry_geometry.get(role)
        support = predicate.get("support")
        current_support = None if support is None else (target_value if support == stage["target"] else None)
        start_support = None if support is None else entry_geometry.get(support)
        if current_role is None or start_role is None or (support is not None and (current_support is None or start_support is None)):
            result = None
        else:
            delta = current_role - start_role
            if support is not None:
                delta -= current_support - start_support
            axis = predicate.get("axis")
            value = float(delta[{"x": 0, "y": 1, "z": 2}[axis]]) if axis else float(np.linalg.norm(delta))
            if predicate.get("absolute", False):
                value = abs(value)
            minimum = float(predicate["min_m"])
            result = value >= minimum; diagnostic.update(displacement_m=value, min_m=minimum, axis=axis)
    elif kind == "released":
        result = (raw.get("loaded_released") if stage["holding_contract"] == "loaded_contact"
                  else raw.get("released"))
    elif kind == "stabilized":
        result = raw.get("object_stabilized", raw.get("stabilized"))
    elif kind == "supported_contact":
        result = raw.get("object_support_contact", raw.get("supported_contact"))
    elif kind == "goal_relation":
        result = raw.get(stage["goal_relation"])
    elif kind == "stage_goal":
        key = str(predicate["key"])
        result = None if stage_goal_raw is None else stage_goal_raw.get(predicate["id"])
        diagnostic.update(key=key, authority="private_boolean_stage_supervisor",
                          debounce_sec=float(predicate.get("debounce_sec", 0.0)),
                          object_role=predicate.get("object_role", stage["object"]),
                          target_role=predicate.get("target_role", stage["target"]),
                          goal_relation=predicate.get("goal_relation", stage["goal_relation"]))
    elif kind == "actor_command_reached":
        tolerance = float(predicate.get("tolerance_rad", 0.025))
        result = (None if actor_joint_error_max_rad is None
                  else actor_joint_error_max_rad <= tolerance)
        diagnostic.update(error_max_rad=actor_joint_error_max_rad,
                          tolerance_rad=tolerance,
                          authority="active_arm_post_step_proprioception_vs_issued_joint_target")
    elif kind == "finite_simulation":
        result = raw.get("finite_simulation")
    else:
        raise ValueError(f"unsupported predicate {kind!r}")
    diagnostic["value"] = result
    return result, diagnostic


def evaluate_stage_contract(stage: Mapping[str, Any], *, progress: float, raw: Mapping[str, Any],
                            object_value: np.ndarray | None, target_value: np.ndarray | None,
                            tcp: np.ndarray, relation_reference: Mapping[str, Any],
                            entry_geometry: Mapping[str, np.ndarray | None],
                            stage_goal_raw: Mapping[str, bool | None] | None = None,
                            actor_joint_error_max_rad: float | None = None) -> tuple[dict[str, bool | None], list[dict[str, Any]]]:
    """Evaluate every root-authored predicate without coercing UNKNOWN to false."""
    facts: dict[str, bool | None] = {}
    diagnostics: list[dict[str, Any]] = []
    for row in [*stage.get("completion_predicates", ()), *stage.get("invariant_predicates", ())]:
        value, diagnostic = evaluate_contract_predicate(
            row, stage=stage, progress=progress, raw=raw, object_value=object_value,
            target_value=target_value, tcp=tcp, relation_reference=relation_reference,
            entry_geometry=entry_geometry, stage_goal_raw=stage_goal_raw,
            actor_joint_error_max_rad=actor_joint_error_max_rad)
        facts[row["id"]] = value
        diagnostics.append({"id": row["id"], **diagnostic})
    return facts, diagnostics


def bilateral_holding_annotation(stage: Mapping[str, Any], controller: Mapping[str, Any]) -> bool:
    """Recover bilateral holding from the contract-bound predicate record."""
    if stage.get("holding_contract") != "bilateral_contact":
        return False
    return any(row.get("predicate") == "holding" and row.get("value") is True
               for row in controller.get("predicate_diagnostics", ()))


def query_stage_goal_predicates(world: Any, stage: Mapping[str, Any],
                                rows: list[Mapping[str, Any]]) -> dict[str, bool | None]:
    """Call the Boolean-only supervisor independently for each bound row."""
    if rows and not hasattr(world, "stage_goal_facts"):
        raise RuntimeError("stage_goal predicate requires TwinWorld.stage_goal_facts")
    values: dict[str, bool | None] = {}
    for row in rows:
        supervisor = _obj(world.stage_goal_facts(
            row.get("object_role", stage["object"]),
            row.get("target_role", stage["target"]), arm=stage["arm"],
            goal_relation=row.get("goal_relation", stage["goal_relation"]),
            debounce_sec=float(row.get("debounce_sec", 0.0))))
        values[row["id"]] = supervisor.get(row["key"])
    return values


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan, assets = load_json(args.plan), load_json(args.assets)
    task = task_from_plan(plan, args.task); solution = package_dir(args.task, args.packages)
    verify_source_bindings(task, assets); verify_freeze(solution)
    candidate_freeze = load_json(solution / "CANDIDATE_FREEZE.json")
    if candidate_freeze.get("candidate_id") != args.candidate_id:
        raise ValueError("candidate id does not match package freeze")
    selection_row = None
    selection = None
    if args.condition_set in ("test", "test_v02"):
        if args.selection_freeze is None:
            raise ValueError("independent test conditions require a reviewed --selection-freeze JSON")
        selection = load_json(args.selection_freeze)
        selection_row = validate_test_selection(
            selection, task=args.task,
            candidate_id=args.candidate_id, solution=solution,
            selection_path=args.selection_freeze)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.task}_{args.condition_set}_{args.trial:02d}_{'online' if args.online_z else 'fixed'}"
    visual_path = args.output / f"{stem}_visual_trace.npz"
    path = args.output / f"{stem}.json"
    if path.exists() or visual_path.exists():
        raise FileExistsError(f"run output already exists; choose a new candidate/output: {path}")
    physics, skill_module = _physics(), _skill()
    graph, refs = load_json(solution / "graph/task_graph.json"), load_json(solution / "refs/visible_centroids.json")
    runtime_contract = (None if args.runtime_contract is None else
                        validate_runtime_contract(load_json(args.runtime_contract)))
    if args.condition_set.endswith("_v02") and runtime_contract is None:
        raise ValueError("v02 condition sets require an explicit --runtime-contract")
    if args.condition_set.endswith("_v02"):
        if candidate_freeze.get("runtime_contract_sha256") != digest(args.runtime_contract):
            raise ValueError("v02 runtime contract differs from candidate freeze")
        materialization = refs.get("runtime_materialization", {})
        if (materialization.get("cup_release_control") != runtime_contract["cup_release_control"] or
                materialization.get("runtime_contract_canonical_sha256") != canonical_json_sha256(runtime_contract)):
            raise ValueError("v02 candidate references lack matching portable provenance")
        if args.trial not in range(10):
            raise ValueError("v02 trial must be within 0..9")
        expected_seed = 51005 if args.condition_set == "development_v02" else 92005
        if args.condition_seed != expected_seed:
            raise ValueError("v02 condition seed violates the frozen condition set")
    feedback_contract = graph.get("local_feedback_contract")
    local_skill = skill_module.TwinLocalSkill(
        solution / "models/local_skill.npz", online_z=args.online_z,
        feedback_contract=feedback_contract)
    if args.condition_seed is None:
        if args.condition_set.endswith("_v02"):
            raise ValueError("v02 condition sets require an explicit --condition-seed")
        seed = 71005 if args.condition_set == "test" else 51005
    else:
        seed = int(args.condition_seed)
    condition = condition_manifest(task, seed=seed, trial=args.trial,
                                   nominal=args.condition_set in ("development", "development_v02") and args.trial == 0,
                                   condition_design=(None if runtime_contract is None else
                                                     runtime_contract["condition_design"]))
    if args.condition_set == "test_v02":
        assert selection is not None and selection_row is not None and args.runtime_contract is not None
        validate_v02_test_condition(
            selection, row=selection_row, selection_path=args.selection_freeze,
            runtime_contract_path=args.runtime_contract, seed=seed, trial=args.trial,
            condition=condition)
    translations = condition["actual_initial_translation_m"]
    world = physics.TwinWorld.from_run(Path(task["source_run"]), width=640, height=360, control_hz=10,
                                       render=True, initial_translation=translations,
                                       cup_release_control=("legacy_object_distance" if runtime_contract is None else
                                                            runtime_contract["cup_release_control"]))
    motor = np.load(task["sources"]["motor_targets.npz"]["path"], allow_pickle=False)
    times, joints, gripper = motor["timestamps"], motor["joints"], motor["gripper"]
    from scipy.interpolate import PchipInterpolator
    joint_curve = PchipInterpolator(times, joints, axis=0, extrapolate=False)
    joint_velocity = joint_curve.derivative()
    def grip_curve(value: float) -> np.ndarray:
        return np.asarray([np.interp(value, times, gripper[:, arm]) for arm in range(2)])
    world.initialize(joints[0], gripper[0])
    roles = sorted(world.role_geom_ids)
    initial_frame = world.observe_visible(roles=roles)
    centroid_cache = VisibleCentroidCache(max_age_sec=.5)
    centroid_cache.update(initial_frame)
    reference_cache = ReferenceCentroidCache(max_age_sec=.5)
    uses_r3_grounding = any(
        stage.get("source_grounding") == "initial_visible_until_first_manipulation"
        for stage in graph["stages"])
    initial_reference = refs["frames"].get("0")
    if uses_r3_grounding and initial_reference is None:
        raise ValueError("R3 initial source grounding requires visible reference frame 0")
    scene_entry_priors = ({} if initial_reference is None else
                          scene_entry_grounding_priors(initial_frame, initial_reference, roles))
    if initial_reference is not None:
        for role in initial_reference["surface_centroids_world_m"]:
            reference_cache.get(role, initial_reference["surface_centroids_world_m"],
                                float(initial_reference["time_sec"]))
    visual_stability = PublicVisualStability()
    for role in roles:
        world.visible_stage_facts(role, initial_frame, arm=graph["stages"][0]["arm"])
    runtime = GRAPH.TwinGraphRuntime(graph)
    trace = []
    previous_stage = None
    previous_cycle = None
    entry_short_delta = None
    source_delta_latch = None
    target_delta_latch = None
    manipulated_roles: set[str] = set()
    planned_objects = {stage["object"] for stage in graph["stages"]}
    target_grounding_record: dict[str, Any] = {}
    cycle_uses_initial_grounding = False
    cycle_grounding_record: dict[str, Any] = {"mode": "legacy"}
    entry_geometry: dict[str, np.ndarray | None] = {}
    entry_records: list[dict[str, Any]] = []
    stage_world_start = float(world.data.time)
    previous_correction = np.zeros((2, 6), dtype=np.float64)
    holding_confirmation = HoldingConfirmation()
    while not runtime.state.completed and runtime.state.failure is None:
        stage = runtime.stage
        if stage["cycle_id"] != previous_cycle:
            previous_cycle = stage["cycle_id"]
            source_delta_latch = None
            target_delta_latch, target_grounding_record = stationary_target_prior(
                stage, priors=scene_entry_priors, planned_objects=planned_objects,
                manipulated_roles=manipulated_roles, physical_time=float(world.data.time))
            source_anchor_role = stage.get("source_anchor_role", stage["object"])
            cycle_uses_initial_grounding = use_scene_entry_grounding(stage, manipulated_roles)
            if cycle_uses_initial_grounding and source_anchor_role in scene_entry_priors:
                prior = scene_entry_priors[source_anchor_role]
                source_delta_latch, plane_record = project_source_grounding(
                    prior["delta_m"], stage.get("source_grounding_plane", "none"))
                cycle_grounding_record = {
                    "mode": "scene_entry_visible_pair",
                    "source_anchor_role": source_anchor_role,
                    **prior, **plane_record,
                }
            else:
                cycle_grounding_record = {
                    "mode": ("recent_visible_pair" if stage.get("source_grounding") ==
                             "initial_visible_until_first_manipulation" else "legacy"),
                    "source_anchor_role": source_anchor_role,
                }
            manipulated_roles.add(stage["object"])
        if stage["stage_id"] != previous_stage:
            local_skill.reset_phase(completed_stage_id=previous_stage)
            previous_stage = stage["stage_id"]
            entry_short_delta = None
            entry_geometry = {}
            stage_world_start = float(world.data.time)
            holding_confirmation.reset()
        # Whole-scene public observations seed the bounded cache during Wait
        # and before a later role becomes occluded.  Only fresh pixels are
        # admitted to the online latent update below.
        frame = world.observe_visible(roles=roles)
        centroid_cache.update(frame)
        prop = world.proprioception()
        fresh_object = visible_centroid(frame, stage["object"])
        fresh_target = visible_centroid(frame, stage["target"])
        current_object, object_visibility = centroid_cache.get(stage["object"], frame)
        current_target, target_visibility = centroid_cache.get(stage["target"], frame)
        source_anchor_role = stage.get("source_anchor_role", stage["object"])
        current_source_anchor, source_anchor_visibility = centroid_cache.get(source_anchor_role, frame)
        ref_start = float(stage["reference"]["start_time_sec"])
        ref_end = float(stage["reference"]["end_time_sec"])
        nominal_command_duration = 1.0 / world.control_hz
        stage_elapsed = float(world.data.time) - stage_world_start
        reference_clock = StageReferenceClock(ref_start, ref_end, float(stage.get("reference_speed_scale", 1.0)))
        command_duration = reference_clock.duration(stage_elapsed, nominal_command_duration)
        ref_time = reference_clock.at(stage_elapsed + command_duration)
        progress = reference_clock.progress(stage_elapsed + command_duration)
        ri = int(np.argmin(np.abs(times - ref_time)))
        ref = refs["frames"].get(str(ri))
        if ref is None:
            raise ValueError(f"visible reference missing for stage {stage['stage_id']}")
        ref_object = ref["surface_centroids_world_m"].get(stage["object"])
        ref_target = ref["surface_centroids_world_m"].get(stage["target"])
        if stage.get("source_grounding") == "initial_visible_until_first_manipulation":
            grounding_reference_time = reference_clock.at(stage_elapsed)
            grounding_ri = int(np.argmin(np.abs(times - grounding_reference_time)))
            grounding_ref = refs["frames"].get(str(grounding_ri))
            if grounding_ref is None:
                raise ValueError(
                    f"source grounding reference missing for stage {stage['stage_id']} at {grounding_ri}")
            for reference_role in grounding_ref["surface_centroids_world_m"]:
                if reference_role != source_anchor_role:
                    reference_cache.get(reference_role, grounding_ref["surface_centroids_world_m"],
                                        float(grounding_ref["time_sec"]))
            ref_source_anchor, source_reference_visibility = reference_cache.get(
                source_anchor_role, grounding_ref["surface_centroids_world_m"],
                float(grounding_ref["time_sec"]))
            source_reference_visibility["query_time_sec"] = float(grounding_reference_time)
            source_reference_visibility["sample_index"] = grounding_ri
        else:
            ref_source_anchor = ref["surface_centroids_world_m"].get(source_anchor_role)
            source_reference_visibility = {
                "known": ref_source_anchor is not None, "fresh": ref_source_anchor is not None,
                "held_previous": False, "age_sec": 0.0 if ref_source_anchor is not None else None,
                "reference_time_sec": float(ref["time_sec"]), "mode": "legacy_current_reference",
            }
        anchor = refs.get("handover_anchors", {}).get(stage["cycle_id"])
        if current_target is None and anchor is not None and stage["target"] == anchor["role"]:
            current_target = np.asarray(anchor["centroid_world_m"], dtype=np.float64)
            target_visibility = {"known": True, "fresh": False, "held_previous": False,
                                 "age_sec": None, "source": "fixed_offline_handover_anchor"}
        support = stage["target"] if stage["target"] in world.role_geom_ids else None
        entry_raw = _obj(world.visible_stage_facts(
            stage["object"], frame, arm=stage["arm"], support=support,
            goal_relation=stage["goal_relation"]))
        if not entry_geometry:
            held = holding_fact(stage["holding_contract"], entry_raw, target_role=stage["target"],
                                goal_relation=stage["goal_relation"])
            initial_object_prior = (scene_entry_priors.get(stage["object"])
                                    if cycle_uses_initial_grounding else None)
            object_known = current_object is not None
            object_entry_assumed = initial_object_prior is not None and not object_known
            role_required = bool(stage.get("entry", {}).get("role_visible_or_held", True))
            role_accepted = bool(not role_required or object_known or object_entry_assumed or held is True)
            source_anchor_known = (source_delta_latch is not None or
                                   (current_source_anchor is not None and ref_source_anchor is not None))
            source_anchor_required = source_delta_latch is None and role_required
            accepted = bool(role_accepted and (not source_anchor_required or source_anchor_known))
            entry_geometry = {stage["object"]: None if current_object is None else current_object.copy(),
                              stage["target"]: None if current_target is None else current_target.copy()}
            entry_record = {"stage_id": stage["stage_id"], "accepted": accepted,
                            "role_visible_or_held_required": role_required,
                            "object_visible_or_cached": object_known, "holding": held,
                            "object_entry_initial_prior": object_entry_assumed,
                            "initial_prior_age_sec": (None if initial_object_prior is None else
                                                      float(world.data.time) -
                                                      float(initial_object_prior["scene_entry_time_sec"])),
                            "initial_prior_assumption": (None if initial_object_prior is None else
                                                         initial_object_prior["assumption"]),
                            "source_anchor_role": source_anchor_role,
                            "source_anchor_known": source_anchor_known,
                            "source_anchor_visibility": source_anchor_visibility,
                            "source_reference_visibility": source_reference_visibility,
                            "object_visibility": object_visibility, "target_visibility": target_visibility}
            entry_records.append(entry_record)
            if not accepted:
                runtime.state.entry_rejections.append(entry_record)
                runtime.state.failure = ("entry_rejected_missing_visible_or_held" if not role_accepted
                                         else "entry_rejected_source_anchor_unknown")
                break
        observed_object_delta = (None if current_source_anchor is None or ref_source_anchor is None else
                                 np.asarray(current_source_anchor) - ref_source_anchor)
        observed_target_delta = None if current_target is None or ref_target is None else np.asarray(current_target) - ref_target
        if source_delta_latch is None and observed_object_delta is not None:
            source_delta_latch, plane_record = project_source_grounding(
                observed_object_delta, stage.get("source_grounding_plane", "none"))
            cycle_grounding_record.update({
                "current_visibility": source_anchor_visibility,
                "reference_visibility": source_reference_visibility,
                "delta_m": source_delta_latch.tolist(),
                "fresh_feedback": False,
                "support_plane_projection": plane_record,
            })
        if target_delta_latch is None and observed_target_delta is not None:
            target_delta_latch = observed_target_delta.copy()
            target_grounding_record.update(
                visible_delta_m=target_delta_latch.tolist(), raw_visible_delta_m=target_delta_latch.tolist(),
                scene_entry_time_sec=None, observation_time_sec=float(world.data.time),
                reference_time_sec=float(ref["time_sec"]), age_sec=0.0)
        object_delta = np.zeros(3) if source_delta_latch is None else source_delta_latch
        target_delta = np.zeros(3) if target_delta_latch is None else target_delta_latch
        grounding_for_record = json.loads(json.dumps(cycle_grounding_record))
        if grounding_for_record.get("mode") == "scene_entry_visible_pair":
            grounding_for_record["age_sec"] = (
                float(world.data.time) - float(grounding_for_record["scene_entry_time_sec"]))
        target_grounding_for_record = json.loads(json.dumps(target_grounding_record))
        prior_time = target_grounding_for_record.get("scene_entry_time_sec")
        observed_time = target_grounding_for_record.get("observation_time_sec")
        if prior_time is not None or observed_time is not None:
            target_grounding_for_record["age_sec"] = max(0.0, float(world.data.time) - float(
                prior_time if prior_time is not None else observed_time))
        arm_index = 0 if stage["arm"] == "left" else 1
        q = np.asarray(_field(prop, "joint_pos"))[arm_index]; qvel = np.asarray(_field(prop, "joint_velocity"))[arm_index]
        nominal_joints = np.asarray(joint_curve(ref_time), dtype=np.float64)
        target_correction = np.zeros((2, 6), dtype=np.float64)
        target_gripper = grip_curve(ref_time)
        if stage["controller"] == "neural_short":
            phase_onehot = np.eye(len(SHORT_PHASE_ORDER))[SHORT_PHASE_ORDER.index(stage["phase"])]
            feature = np.r_[q, qvel, object_delta, target_delta, progress, np.eye(2)[arm_index], phase_onehot, np.zeros(3)]
            # Legacy R0/R1 feedback used the selected object/target surface.
            # R2 always observes the held object itself with fresh pixels.
            if feedback_contract is None:
                selected_current = fresh_object if stage["phase"] == "Grasp" else fresh_target
                selected_ref = ref_object if stage["phase"] == "Grasp" else ref_target
            else:
                selected_current = fresh_object
                matched_time = reference_clock.at(stage_elapsed)
                matched_ri = int(np.argmin(np.abs(times - matched_time)))
                matched_ref = refs["frames"].get(str(matched_ri))
                if matched_ref is None:
                    raise ValueError(f"feedback reference missing for stage {stage['stage_id']} at {matched_ri}")
                selected_ref = matched_ref["surface_centroids_world_m"].get(stage["object"])
            if entry_short_delta is None and selected_current is not None and selected_ref is not None:
                entry_short_delta = np.asarray(selected_current) - np.asarray(selected_ref)
            adapted_reference = None if selected_ref is None or entry_short_delta is None else np.asarray(selected_ref) + entry_short_delta
            command_kwargs: dict[str, Any] = {}
            if feedback_contract is not None:
                adapted_reference = selected_ref
                command_kwargs = {
                    "phase": stage["phase"],
                    # Attachment feedback requires an actual current bilateral
                    # TCP grip, independent of held_or_supported stage gates.
                    "holding": attachment_feedback_holding(entry_raw),
                    "matched_reference_time_sec": float(matched_ref["time_sec"]),
                    "current_tcp_xyz": np.asarray(_field(prop, "tcp_position"))[arm_index],
                    "current_tcp_rotation": np.asarray(_field(prop, "tcp_rotation"))[arm_index],
                    "reference_tcp_xyz": np.asarray(matched_ref["tcp_position"])[arm_index],
                    "reference_tcp_rotation": np.asarray(matched_ref["tcp_rotation"])[arm_index],
                }
            step = local_skill.command(
                feature, current_surface_xyz=selected_current,
                reference_surface_xyz=adapted_reference, **command_kwargs)
            nominal_joint_prior = nominal_joints[arm_index].copy()
            target_correction[arm_index] = step.action
            controller = step.record
            controller["nominal_reference_joint_prior"] = nominal_joint_prior.tolist()
            controller["latched_source_delta_m"] = object_delta.tolist()
            controller["latched_target_delta_m"] = target_delta.tolist()
            controller["source_anchor_role"] = source_anchor_role
            controller["source_anchor_visibility"] = source_anchor_visibility
            controller["source_grounding"] = grounding_for_record
            if feedback_contract is not None:
                controller["feedback_reference_query_time_sec"] = float(matched_time)
                controller["feedback_reference_sample_index"] = matched_ri
        else:
            blend = object_delta if stage["phase"] in ("Approach", "Lift") else (1-progress)*object_delta + progress*target_delta
            scratch_target = np.asarray(world.solve_ik_delta(
                stage["arm"], np.r_[blend, np.zeros(3)], joints=nominal_joints[arm_index]))
            target_correction[arm_index] = scratch_target - nominal_joints[arm_index]
            controller = {"controller": "code_reference_retarget", "blend_delta_m": blend.tolist(),
                          "reference_progress": progress, "orientation_prior": "continuous_demo_fk",
                          "source_anchor_role": source_anchor_role,
                          "source_anchor_visibility": source_anchor_visibility,
                          "source_grounding": grounding_for_record}
        command_world_start = float(world.data.time)
        correction_delta = target_correction - previous_correction
        command_ref_start = reference_clock.at(stage_elapsed)
        def continuous_command(absolute_time: float):
            return reference_motion_command(
                reference_clock, stage_elapsed=stage_elapsed,
                command_elapsed=absolute_time - command_world_start, command_duration=command_duration,
                previous_correction=previous_correction, correction_delta=correction_delta,
                joint_curve=joint_curve, joint_velocity=joint_velocity, grip_curve=grip_curve,
                trajectory_start=float(times[0]), trajectory_end=float(times[-1]))
        controller["target_grounding"] = target_grounding_for_record
        controller["reference_clock"] = {
            "speed_scale": reference_clock.speed, "physical_stage_elapsed_sec": stage_elapsed,
            "current_reference_time_sec": command_ref_start, "command_endpoint_reference_time_sec": ref_time,
            "physical_command_duration_sec": command_duration,
        }
        result = world.step_command_callback(continuous_command, command_duration, observe=False, include_endpoint=True)
        target_joints = nominal_joints + target_correction
        previous_correction = target_correction.copy()
        post_frame = world.observe_visible(roles=roles)
        centroid_cache.update(post_frame)
        post_prop = world.proprioception()
        post_actor_joints = np.asarray(_field(post_prop, "joint_pos"), dtype=np.float64)[arm_index]
        actor_joint_error_max_rad = float(np.max(np.abs(
            post_actor_joints - np.asarray(target_joints[arm_index], dtype=np.float64))))
        relation_distance = .22 if stage["phase"] == "Move" else .10
        facts_raw = world.visible_stage_facts(stage["object"], post_frame, arm=stage["arm"], support=support,
                                              goal_relation=stage["goal_relation"], relation_distance_m=relation_distance)
        facts_raw = _obj(facts_raw)
        facts_raw["finite_simulation"] = bool(
            _obj(world.numerical_safety()).get("simulation_finite", False))
        confirmation_sec = float(stage.get("grasp_stable_contact_sec", 0.0))
        if stage["phase"] == "Grasp" and confirmation_sec:
            contracted_holding = holding_fact(stage["holding_contract"], facts_raw,
                                             target_role=stage["target"], goal_relation=stage["goal_relation"])
            facts_raw["holding_confirmed"] = holding_confirmation.update(
                contracted_holding, float(_field(result, "physical_time_sec")), confirmation_sec)
            facts_raw["continuous_holding_sec"] = holding_confirmation.elapsed
        visibly_stable = visual_stability.update(stage["object"], post_frame)
        if stage["holding_contract"] == "loaded_contact":
            facts_raw["object_stabilized"] = bool(facts_raw.get("loaded_released") and visibly_stable)
            facts_raw["stabilized"] = facts_raw["object_stabilized"]
        post_object, post_object_visibility = centroid_cache.get(stage["object"], post_frame)
        post_target, post_target_visibility = centroid_cache.get(stage["target"], post_frame)
        if post_target is None and anchor is not None and stage["target"] == anchor["role"]:
            post_target = np.asarray(anchor["centroid_world_m"], dtype=np.float64)
            post_target_visibility = {"known": True, "fresh": False, "held_previous": False,
                                      "age_sec": None, "source": "fixed_offline_handover_anchor"}
        is_revised = bool(stage.get("completion_predicates"))
        if not is_revised and stage["phase"] in ("Move", "Align") and post_object is not None and post_target is not None:
            # Move/Align establish planar above-target vicinity.  Vertical
            # placement and the goal relation remain Release/Insert gates.
            facts_raw["near_target"] = bool(np.linalg.norm(
                np.asarray(post_object)[:2] - np.asarray(post_target)[:2]) <= (.18 if stage["phase"] == "Move" else .14))
        if not is_revised and support is None and post_object is not None and ref_target is not None:
            synthetic_relation = bool(np.linalg.norm(np.asarray(post_object) - np.asarray(ref_target)) <= .10)
            facts_raw[stage["goal_relation"]] = synthetic_relation
            facts_raw["near_target"] = synthetic_relation
        if is_revised:
            goal_rows = [row for row in (*stage.get("completion_predicates", ()),
                                         *stage.get("invariant_predicates", ()))
                         if row.get("predicate") == "stage_goal"]
            stage_goal_raw = None
            if goal_rows:
                stage_goal_raw = query_stage_goal_predicates(world, stage, goal_rows)
            relation_reference = refs.get("stage_relations", {}).get(stage["stage_id"], {})
            facts, predicate_diagnostics = evaluate_stage_contract(
                stage, progress=progress, raw=facts_raw,
                object_value=post_object, target_value=post_target,
                tcp=np.asarray(_field(post_prop, "tcp_position"))[arm_index],
                relation_reference=relation_reference, entry_geometry=entry_geometry,
                stage_goal_raw=stage_goal_raw,
                actor_joint_error_max_rad=actor_joint_error_max_rad)
        else:
            facts = {name: bool(facts_raw.get(name, False)) for name in set(stage["done"] + stage["invariants"])}
            facts["reference_progress_reached"] = progress >= 1.0
            predicate_diagnostics = []
        # The stage invariant is exactly finiteness.  Accumulated clearance and
        # joint-limit safety remain strict final-evaluator gates and are never
        # relabeled as an early non-finite transition failure.
        if not is_revised:
            facts["finite_simulation"] = facts_raw["finite_simulation"]
        runtime.record_step({"time_sec": _field(result, "physical_time_sec", 0.0),
                             "qpos": np.asarray(_field(post_prop, "joint_pos")),},
                            {stage["object"]: {"surface_centroid_world": None if post_object is None else post_object.tolist(),
                                               "visibility": post_object_visibility},
                             stage["target"]: {"surface_centroid_world": None if post_target is None else post_target.tolist(),
                                               "visibility": post_target_visibility}},
                            facts, target_joints, {**controller, "predicate_diagnostics": predicate_diagnostics})
        runtime.transition(facts)
    # Archive an active short phase on terminal failure/completion.  Completed
    # short phases were already archived on entry to their successor.
    local_skill.reset_phase(
        completed_stage_id=(runtime.stage["stage_id"] if local_skill.command_index else None))
    external_root = Path("/home/zheng/github/Mujoco-RSI")
    if str(external_root) not in sys.path:
        sys.path.insert(0, str(external_root))
    from twin.physics_evaluation import task_outcome
    outcome = _obj(world.finalize_evaluator(task_outcome))
    strict_success = bool(runtime.state.completed and runtime.state.failure is None and
                          outcome.get("numerical_safety", {}).get("passed", True) and
                          outcome.get("task_outcome", outcome).get("passed", False))
    evaluator_rows = (list(world.evaluator_trace())[-len(runtime.trace):] if runtime.trace else [])
    stage_lookup = {s["stage_id"]: (i, s) for i, s in enumerate(graph["stages"])}
    annotations = []
    for row in runtime.trace:
        index, stage = stage_lookup[row["stage_id"]]
        latent = row["controller"].get("latent", {})
        z = latent.get("z_after_m", [0.0, 0.0, 0.0])
        annotations.append((index, stage, z, bilateral_holding_annotation(stage, row["controller"])))
    if annotations and runtime.state.failure:
        final_failure = runtime.state.failure
    else:
        final_failure = ""
    np.savez_compressed(
        visual_path,
        timestamps=np.asarray([row["physical_time_sec"] for row in evaluator_rows], dtype=np.float64),
        qpos=np.asarray([row["qpos"] for row in evaluator_rows], dtype=np.float64),
        ctrl=np.asarray([row["ctrl"] for row in evaluator_rows], dtype=np.float64),
        stage_id=np.asarray([row["stage_id"] for row in runtime.trace]),
        stage_index=np.asarray([x[0] for x in annotations], dtype=np.int32),
        arm=np.asarray([x[1]["arm"] for x in annotations]),
        object_role=np.asarray([x[1]["object"] for x in annotations]),
        target_role=np.asarray([x[1]["target"] for x in annotations]),
        controller_kind=np.asarray([row["controller"]["controller"] for row in runtime.trace]),
        z=np.asarray([x[2] for x in annotations], dtype=np.float64),
        bilateral_holding=np.asarray([x[3] for x in annotations], dtype=bool),
        failure=np.asarray(["" for _ in annotations[:-1]] + ([final_failure] if annotations else [])),
    )
    report = {"schema": "graph-self-adapter.twin-run.v1", "task": args.task,
              "candidate_id": args.candidate_id, "condition_set": args.condition_set, "condition": condition,
              "controller": "online_z" if args.online_z else "fixed_z", "graph_completed": runtime.state.completed,
              "graph_failure": runtime.state.failure, "original_task_outcome": outcome,
              "entry_records": entry_records,
              "adaptation_scope": ("visible_geometry_baseline; R1 reuses R0 weights and is not a P68 or meta-learned core claim"
                                   if feedback_contract is None else feedback_contract["claim_boundary"]),
              "local_feedback_contract": feedback_contract,
              "runtime_contract": runtime_contract,
              "local_feedback_phase_archive": local_skill.phase_archive,
              "strict_success": strict_success,
              "bindings": {"scene_sha256": task["sources"]["scene.xml"]["sha256"],
                           "graph_sha256": digest(solution / "graph/task_graph.json"),
                           "model_sha256": digest(solution / "models/local_skill.npz"),
                           "condition_sha256": canonical_json_sha256(condition),
                           "runtime_contract_sha256": (None if args.runtime_contract is None else
                                                       digest(args.runtime_contract)),
                           "candidate_freeze_sha256": digest(solution / "CANDIDATE_FREEZE.json"),
                           "selection_freeze_sha256": (None if args.selection_freeze is None else
                                                       digest(args.selection_freeze)),
                           "visual_trace": {"path": str(visual_path), "sha256": digest(visual_path)}},
              "selection_row": selection_row,
              "trace": runtime.trace}
    world.close()
    dump_json(path, report)
    return {"output": str(path), "strict_success": strict_success, "failure": runtime.state.failure}


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    rows = [load_json(p) for p in sorted(args.input.glob("*.json")) if load_json(p).get("schema") == "graph-self-adapter.twin-run.v1"]
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['task']}:{row.get('candidate_id','legacy')}:{row.get('condition_set','unknown')}:{row['controller']}"
        g = groups.setdefault(key, {"task": row["task"], "candidate_id": row.get("candidate_id", "legacy"),
                                    "condition_set": row.get("condition_set", "unknown"),
                                    "controller": row["controller"], "trials": 0, "successes": 0, "failures": {}})
        g["trials"] += 1; g["successes"] += int(row["strict_success"])
        reason = row.get("graph_failure") or ("success" if row["strict_success"] else "original_task_outcome_failed")
        g["failures"][reason] = g["failures"].get(reason, 0) + 1
    for g in groups.values():
        g["success_rate"] = g["successes"] / g["trials"] if g["trials"] else None
    report = {"schema": "graph-self-adapter.twin-summary.v1", "runs": len(rows), "groups": list(groups.values()),
              "scientific_results": groups, "physics_engineering_replay": "reported separately; never included in algorithm denominator"}
    if args.output:
        dump_json(args.output, report)
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, default=DEFAULT_PLAN); p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--packages", type=Path, default=DEFAULT_PACKAGES)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--task"); q.add_argument("--candidate-id", required=True); q.add_argument("--no-render", action="store_true")
    q.add_argument("--overrides", type=Path)
    q.add_argument("--runtime-contract", type=Path)
    q.add_argument("--samples-per-stage", type=int, default=384)
    q.add_argument("--qpos-jitter-rad", type=float, default=0.0,
                   help="offline teacher joint perturbation std; R3 planned value is 0.015")
    q.add_argument("--seed", type=int, default=51005); q.set_defaults(func=prepare)
    q = sub.add_parser("train"); q.add_argument("--task"); q.add_argument("--candidate-id", required=True); q.add_argument("--steps", type=int, default=2500)
    q.add_argument("--batch-size", type=int, default=256); q.add_argument("--seed", type=int, default=51005)
    q.add_argument("--action-limit-rad", type=float, default=.04)
    q.add_argument("--train-python", type=Path, default=TRAIN_PYTHON); q.set_defaults(func=train)
    q = sub.add_parser("repackage"); q.add_argument("--task"); q.add_argument("--candidate-id", required=True)
    q.add_argument("--source-packages", type=Path, required=True); q.add_argument("--overrides", type=Path, required=True)
    q.add_argument("--refresh-observer-reference", action=argparse.BooleanOptionalAction, default=None,
                   help="re-render visible references; default: auto when observer hash differs from parent")
    q.add_argument("--runtime-contract", type=Path)
    q.add_argument("--reference-packages", type=Path, help="reuse bound portable visible frames from this package root")
    q.set_defaults(func=repackage)
    q = sub.add_parser("run"); q.add_argument("--task", required=True); q.add_argument(
        "--condition-set", choices=("development", "test", "development_v02", "test_v02"),
        default="development")
    q.add_argument("--trial", type=int, default=0); q.add_argument("--online-z", action="store_true"); q.add_argument("--candidate-id", required=True)
    q.add_argument("--condition-seed", type=int,
                   help="explicit v02 condition seed; legacy sets retain their historical defaults")
    q.add_argument("--runtime-contract", type=Path,
                   help="explicit v02 portable backend and condition-design contract")
    q.add_argument("--selection-freeze", type=Path); q.add_argument("--output", type=Path, default=REPO / "runs/twin_rsi_v01"); q.set_defaults(func=run)
    q = sub.add_parser("summarize"); q.add_argument("--input", type=Path, default=REPO / "runs/twin_rsi_v01")
    q.add_argument("--output", type=Path); q.set_defaults(func=summarize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = args.func(args)
    json.dump(result, sys.stdout, indent=2, sort_keys=True); print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
