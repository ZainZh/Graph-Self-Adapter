#!/usr/bin/env python3
"""P69 minimal full-task runner for outer RSI book->caddy rounds.

New file only.  It composes existing P55 full-task prefix/Code APIs with the
frozen P68 reference-continuous book inner adapter at s9.  It restores exactly
one explicit 45-vector LIBERO initial state at episode start, never restores a
near-entry state, and never uses continuous evaluator geometry for actions.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.task_graph.collect_libero_demo_gt import dump, sha, snapshot
from tools.task_graph.collect_libero_p52_v02 import private_measurement
from tools.task_graph.p52_full_task import InsertJudge, structural_measurement
from tools.task_graph.p54_full_task import near_gate, run_prefix, _names
from tools.task_graph.p55_relation_approach import run_code as run_p55_code
from tools.task_graph.p64_book_evaluate import PUBLIC_KEYS, _append, _seal

SCHEMA = "prove.p69-full-rsi-freeze.v1"
APPROVAL_SCHEMA = "prove.p69-root-review.v1"
RESULT_SCHEMA = "prove.p69-full-rsi-result.v1"
ARMS = ("continuous",)
STAGES = ("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s8a", "s9")


def read(path: Path):
    return json.loads(Path(path).read_text())


def binding(path: Path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def file_sha256(path: Path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_bindings():
    files = [
        "tools/task_graph/p69_full_rsi_runner.py",
        "tools/task_graph/p55_relation_approach.py",
        "tools/task_graph/p54_full_task.py",
        "tools/task_graph/p53_full_task.py",
        "tools/task_graph/collect_libero_p53_v03.py",
        "tools/task_graph/collect_libero_p52_v02.py",
        "tools/task_graph/p52_full_task.py",
        "tools/task_graph/run_libero_p51_v02.py",
        "tools/task_graph/p64_book_evaluate.py",
        "prove/fast_adapter/p68_book_runtime.py",
        "prove/fast_adapter/p64_book_runtime.py",
        "prove/fast_adapter/p64_book_bridge.py",
        "prove/fast_adapter/p64_common.py",
        "prove/fast_adapter/p61_encoder.py",
        "prove/fast_adapter/visual_alignment.py",
        "prove/task_graph/p55_relation_approach.py",
        "prove/task_graph/p53_micro_insert.py",
        "prove/task_graph/p52_stage_policy.py",
        "prove/task_graph/libero_demo_skill.py",
        "prove/task_graph/libero_precision_p51.py",
        "prove/task_graph/libero_supervisor.py",
        "prove/task_graph/executor.py",
        "prove/task_graph/schema.py",
        "prove/task_graph/grounding.py",
        "prove/task_graph/p57_geometry_policy.py",
        "prove/task_graph/p57_public_geometry.py",
        "prove/task_graph/p69_public_grounding.py",
        "prove/task_graph/p69_book_silhouette.py",
        "prove/task_graph/precision_action_policy.py",
    ]
    return {str((ROOT / f).resolve()): sha(ROOT / f) for f in files if (ROOT / f).exists()}


def scope_entries(scope: dict) -> list[dict]:
    rows = scope.get("entries", scope.get("initial_states"))
    if not isinstance(rows, list):
        raise ValueError("P69 INIT_SCOPE requires entries")
    return rows




def binding_chain(path: Path, *, max_depth: int = 4) -> list[dict]:
    chain = []
    current = Path(path).resolve()
    for _ in range(max_depth):
        if not current.exists():
            raise FileNotFoundError(f"P69 parent binding missing: {current}")
        value = read(current)
        chain.append({"path": str(current), "sha256": sha(current), "schema": value.get("schema")})
        parent = value.get("parent")
        if not isinstance(parent, dict) or "path" not in parent:
            break
        if sha(parent["path"]) != parent.get("sha256"):
            raise ValueError("P69 recursive parent binding hash mismatch: " + parent["path"])
        current = Path(parent["path"]).resolve()
    return chain


def load_p53_full_binding(path: Path) -> dict:
    current = Path(path).resolve()
    for _ in range(6):
        value = read(current)
        if value.get("schema") == "prove.p53-full-original-task.v1":
            required = {"environment", "prefix", "public", "reference_poses", "relation", "task_graph"}
            if not required <= set(value):
                raise ValueError("P53 full binding lacks required runtime keys")
            return value
        parent = value.get("parent")
        if not isinstance(parent, dict) or "path" not in parent:
            raise ValueError("P69 parent chain ended before prove.p53-full-original-task.v1")
        if sha(parent["path"]) != parent.get("sha256"):
            raise ValueError("P69 recursive parent binding hash mismatch: " + parent["path"])
        current = Path(parent["path"]).resolve()
    raise ValueError("P69 parent chain too deep before P53 full binding")

def validate_init_scope(scope: dict) -> dict:
    if scope.get("schema") != "prove.p69-init-scope.v1":
        raise ValueError("P69 INIT_SCOPE schema required")
    rows = scope_entries(scope)
    if len(rows) != 50:
        raise ValueError("P69 requires exactly 50 official pruned initial states")
    counts = {"train": 0, "explore": 0, "test": 0}
    seen_id, seen_hash = set(), set()
    for row in rows:
        vec = np.asarray(row.get("state_flattened", row.get("qpos45")), dtype=np.float64)
        if vec.shape != (45,) or not np.isfinite(vec).all():
            raise ValueError("each P69 initial state must be a finite complete state_flattened45")
        split = row.get("split")
        if split not in counts:
            raise ValueError("P69 initial state split must be train/explore/test")
        counts[split] += 1
        if row["id"] in seen_id:
            raise ValueError("duplicate initial state id")
        seen_id.add(row["id"])
        digest = row.get("state_sha256")
        if digest is not None:
            if digest in seen_hash:
                raise ValueError("duplicate initial state hash")
            seen_hash.add(digest)
            if hashlib.sha256(vec.tobytes()).hexdigest() != digest:
                raise ValueError("state_flattened bytes do not match state_sha256")
    if counts != {"train": 30, "explore": 10, "test": 10}:
        raise ValueError(f"P69 split counts must be 30/10/10, got {counts}")
    return scope


def validate_graph_config(graph: dict) -> dict:
    if graph.get("schema") != "prove.p69-graph-config.v1":
        raise ValueError("P69 graph config schema required")
    if graph.get("demo_segments") != {
        "Move0": [0, 30], "Approach": [30, 57], "Grasp": [57, 68], "Lift": [68, 85],
        "Move1": [85, 130], "Align": [130, 200], "Insert": [200, 234],
    }:
        raise ValueError("P69 graph must bind the reviewed seven demo segments; no Release")
    task_graph = graph.get("task_graph")
    if not task_graph or [n["id"] for n in task_graph.get("nodes", [])] != ["s7", "s8", "s8a", "s9"]:
        raise ValueError("P69 first runner expects P55-style s7/s8/s8a/s9 graph")
    prefix_mode = graph.get("prefix_runtime_graph_mode", "inherit_parent_original")
    if prefix_mode not in ("inherit_parent_original", "override"):
        raise ValueError("prefix_runtime_graph_mode must be inherit_parent_original or override")
    if prefix_mode == "override":
        prefix_graph = graph.get("prefix_runtime_graph")
        node_ids = [] if not prefix_graph else [n.get("id") for n in prefix_graph.get("nodes", [])]
        handoff_after = graph.get("prefix_handoff_after", "s6")
        if handoff_after == "s5":
            expected = ["s1", "s2", "s3", "s4", "s5", "s7"]
        elif handoff_after == "s6":
            expected = [f"s{i}" for i in range(1, 8)]
        else:
            raise ValueError("prefix_handoff_after must be s5 or s6")
        if node_ids != expected:
            raise ValueError(f"prefix_runtime_graph override must define executable nodes {expected}")
    if graph.get("public_observation") != {"rgb": [2, 448, 448, 3], "visible_masks": [2, 3, 448, 448]}:
        raise ValueError("P69/P68 book runtime requires explicit native dual-view 448 public config")
    grounding_cfg = graph.get("r2_public_grounding")
    if grounding_cfg is not None:
        if grounding_cfg.get("schema") != "prove.p69-r2-public-grounding-config.v1":
            raise ValueError("R2 grounding config schema required")
        if grounding_cfg.get("runtime_inputs") != ["visible_masks", "camera_K", "camera_pose", "robot_proprio"]:
            raise ValueError("R2 runtime inputs must remain public mask/camera/proprio only")
        settling = grounding_cfg.get("settling")
        if not isinstance(settling, dict) or settling.get("max_open_steps") != 12 or settling.get("stable_frames") != 3:
            raise ValueError("R2 settling must be the reviewed public-pixel stable protocol")
        if settling.get("min_open_steps_before_lock") != 5 or float(settling.get("stable_pixel_max", -1)) != 1.0:
            raise ValueError("R2 settling must use min_open_steps=5 and <=1px public book centroid stability")
        ref = grounding_cfg.get("grounding_reference")
        if not isinstance(ref, dict) or "path" not in ref or "sha256" not in ref:
            raise ValueError("R2 grounding reference binding required")
        from prove.task_graph.p69_public_grounding import load_reference
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("R2 grounding reference hash mismatch")
        load_reference(ref["path"])
    return graph


def round_index(version: str) -> int:
    if not version.startswith("R") or not version[1:].isdigit():
        raise ValueError("P69 version must be R0..R3")
    value = int(version[1:])
    if not 0 <= value <= 3:
        raise ValueError("P69 version must be R0..R3")
    return value


def cells_from_scope(init_scope: dict, split: str, version: str, arms=("continuous",)):
    if tuple(arms) != ("continuous",):
        raise ValueError("P69 uses only the main reference-continuous arm")
    if split not in ("explore", "test"):
        raise ValueError("P69 physical runs are only explore or test; train is for offline data")
    rows = [r for r in scope_entries(init_scope) if r["split"] == split]
    if len(rows) != 10:
        raise ValueError(f"P69 split {split!r} needs exactly ten initial states")
    cells = []
    r = round_index(version)
    for i, row in enumerate(rows):
        stream_seed = 69020 + r if split == "explore" else 69030 + i
        cells.append(dict(index=len(cells), version=version, split=split, init_id=row["id"],
                          init_index=i, arm="continuous", state_sha256=row.get("state_sha256"),
                          execution_seed=stream_seed))
    return cells


def prepare(*, output: Path, p55_binding: Path, init_scope: Path, graph_config: Path,
            precision_head: Path, encoder_checkpoint: Path, action_checkpoint: Path, a2_head: Path,
            priors: Path, dino_binding: Path, split: str, version: str, arms: list[str], version_source_root: Path | None = None):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    p55 = read(p55_binding)
    init = validate_init_scope(read(init_scope))
    graph = validate_graph_config(read(graph_config))
    cells = cells_from_scope(init, split, version, tuple(arms))
    frozen = dict(
        schema=SCHEMA,
        status="frozen_before_results",
        version=version,
        split=split,
        cells=cells,
        initial_restores=1,
        later_restores=0,
        automatic_retries=0,
        budgets=dict(prefix_max_actions=400, code_max_actions=120, micro_max_actions=24, total_max_actions=544),
        success="graph_complete and original official goal plus held-book finger contact hold3; official instant success is diagnostic only",
        p55_binding=binding(p55_binding),
        parent_binding_chain=binding_chain(p55_binding),
        init_scope=binding(init_scope),
        graph_config=binding(graph_config),
        graph=graph,
        init_scope_materialized=init,
        p55_parent_summary=dict(schema=p55.get("schema"), seed=p55.get("seed"), source=p55.get("parent")),
        version_source_root=None if version_source_root is None else str(Path(version_source_root).resolve()),
        files=dict(
            precision_head=binding(precision_head),
            encoder_checkpoint=binding(encoder_checkpoint),
            action_checkpoint=binding(action_checkpoint),
            a2_head=binding(a2_head),
            priors=binding(priors),
            dino_binding=binding(dino_binding),
        ),
        sources=source_bindings(),
        rng_streams=dict(training="69010+r", explore="69020+r", test="69030", shuffle=69000),
        runtime_contract=dict(
            full_initial_state="one explicit complete 45D MuJoCo flattened state restored at episode start",
            no_mid_restore=True,
            prefix="existing P55/P53 full-task GraphExecutor closed-loop API",
            code="existing P55 public EEF path + relation approach using the P69 precision_head checkpoint; GT only as boolean supervisor gates",
            s9="P68 ReferenceBookRuntime with frozen P64/P68 inner adapter core from R0 onward",
            action_inputs="public images/masks/proprio and frozen config/checkpoints only; no continuous evaluator GT",
            prefix_runtime_graph_mode=graph.get("prefix_runtime_graph_mode", "inherit_parent_original"),
        ),
        interpretation="outer RSI development/evaluation runner; no physics until exact Root review",
    )
    dump(output / "PHYSICAL_FREEZE.json", frozen)
    return frozen


def validate_freeze(path: Path, approval: Path | None = None, check_files: bool = True) -> dict:
    frozen = read(path)
    if frozen.get("schema") != SCHEMA or frozen.get("status") != "frozen_before_results":
        raise ValueError("wrong P69 freeze")
    validate_init_scope(frozen["init_scope_materialized"])
    validate_graph_config(frozen["graph"])
    if frozen.get("initial_restores") != 1 or frozen.get("later_restores") != 0 or frozen.get("automatic_retries") != 0:
        raise ValueError("P69 restore/retry contract drift")
    if check_files and frozen.get("parent_binding_chain") != binding_chain(frozen["p55_binding"]["path"]):
        raise ValueError("P69 recursive parent binding chain changed")
    if approval is not None:
        review = read(approval)
        if (review.get("schema") != APPROVAL_SCHEMA or review.get("freeze_sha256") != sha(path)
                or review.get("root_reviewed") is not True or review.get("physical_execution_authorized") is not True):
            raise ValueError("missing P69 exact Root review")
    if check_files:
        for item in [frozen["p55_binding"], frozen["init_scope"], frozen["graph_config"], *frozen["files"].values()]:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("P69 frozen file changed: " + item["path"])
        for path, digest in frozen["sources"].items():
            if sha(path) != digest:
                raise ValueError("P69 source changed: " + path)
    return frozen


def init_row(frozen: dict, cell: dict) -> dict:
    row = next((r for r in scope_entries(frozen["init_scope_materialized"]) if r["id"] == cell["init_id"]), None)
    if row is None:
        raise ValueError("cell init id absent from frozen scope")
    return row


def initialize_full_env(parent: dict, output: Path, state_flattened: np.ndarray):
    # Reuse the original LIBERO scene setup from P52/P53/P55, but restore the
    # complete explicit P69 flattened MuJoCo state.  This function intentionally
    # has no near-entry frame argument and never overlays a prefix of an old state.
    environment = parent["environment"]
    config = output / "libero_config"
    config.mkdir()
    base = Path(environment["libero_root"]) / "libero/libero"
    dump(config / "config.yaml", dict(benchmark_root=str(base), bddl_files=str(base / "bddl_files"),
                                      init_states=str(base / "init_files"), datasets=str(Path(environment["source"]).parent),
                                      assets=str(base / "assets")))
    import os
    os.environ["LIBERO_CONFIG_PATH"] = str(config.resolve())
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, environment["libero_root"])
    from libero.libero.envs import TASK_MAPPING
    kwargs = dict(environment["env_kwargs"])
    kwargs.update(use_camera_obs=False, has_offscreen_renderer=False)
    env = TASK_MAPPING[environment["env_args_original"]["problem_name"]](**kwargs)
    try:
        env.reset()
        env.reset_from_xml_string(environment["relocated_xml"])
        env.sim.reset()
        state_flattened = np.asarray(state_flattened, dtype=float)
        if state_flattened.shape != (45,) or not np.isfinite(state_flattened).all():
            raise ValueError("P69 requires one finite complete state_flattened45")
        env.sim.set_state_from_flattened(state_flattened.copy())
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        return env
    except Exception:
        env.close()
        raise


def run_s9_book(env, frozen: dict, cell: dict, child: dict, output: Path, names: dict):
    from tools.task_graph import p56_collection as collection
    from tools.task_graph.collect_libero_demo_gt import snapshot
    from prove.fast_adapter.p64_common import load_encoder, _state_hash
    from prove.fast_adapter.p68_book_runtime import ReferenceBookRuntime
    from prove.fast_adapter.p64_book_bridge import BookPriors
    from prove.task_graph.p57_geometry_policy import load_geometry_policy
    from prove.fast_adapter.visual_alignment import FrozenDinoROI

    device = read(frozen["files"]["dino_binding"]["path"]).get("device", "cuda")
    encoder = load_encoder(frozen["files"]["encoder_checkpoint"]["path"], device=device,
                           expected_sha256=frozen["files"]["encoder_checkpoint"]["sha256"])
    policy, _, _ = load_geometry_policy(frozen["files"]["action_checkpoint"]["path"],
                                        frozen["files"]["a2_head"]["path"], device=device)
    priors = BookPriors(**read(frozen["files"]["priors"]["path"])["values"])
    dino = read(frozen["files"]["dino_binding"]["path"])
    vision = FrozenDinoROI(dino["model_dir"], device=device, microbatch=2)
    runtime = ReferenceBookRuntime(encoder, policy, priors, cell["arm"], vision=vision, device=device)
    before = (_state_hash(encoder), _state_hash(policy))
    renderer = collection.PublicRenderer(env, child["public"])
    judge, hold, steps = InsertJudge(), 0, 0
    update_events = 0
    reason = "micro_not_started"
    strict = native_ever = native_instant = False
    obs = env._get_observations(force_update=True)
    try:
        while True:
            public = renderer.observe(obs)
            public = {key: public[key] for key in PUBLIC_KEYS}
            if np.asarray(public["rgb"]).shape != (2, 448, 448, 3) or np.asarray(public["visible_masks"]).shape != (2, 3, 448, 448):
                raise ValueError("P69/P68 s9 requires native dual-view 448 public frames")
            frame = output / "public" / f"frame_{steps:04d}.npz"
            np.savez_compressed(frame, **public)
            action, telemetry = runtime.step(public)
            update_events += int(bool(telemetry.get("event"))) if isinstance(telemetry, dict) else 0
            action = np.asarray(action, float)
            if action.shape != (7,) or not np.isfinite(action).all() or action[6] != 1:
                raise ValueError("Invalid P69 frozen neural OSC command")
            _append(output / "public/PREDICTIONS.jsonl", dict(t=steps, observation_sha256=sha(frame),
                                                              action=action, telemetry=telemetry))
            official, fingers, wrong = structural_measurement(env, names)
            judgment = judge.observe(official, fingers, wrong)
            hold = hold + 1 if official else 0
            native_ever = native_ever or hold >= 3
            native_instant = native_instant or official
            reason = judgment["stop"] or ("micro_budget24" if steps >= frozen["budgets"]["micro_max_actions"] else None)
            _append(output / "private/JUDGMENTS.jsonl", dict(t=steps, official=official, fingers=fingers,
                                                             wrong_target=wrong, **judgment))
            arrays, _ = snapshot(env, obs)
            np.savez_compressed(output / "private" / f"frame_{steps:04d}.npz",
                                **{k: v for k, v in arrays.items() if k.startswith("supervisor_")})
            _append(output / "public/EXECUTED.jsonl", dict(t=steps, executed=reason is None,
                                                           action=action if reason is None else None))
            if reason:
                strict = bool(judgment["complete"])
                break
            obs, _, _, _ = env.step(action.copy())
            steps += 1
        after = (_state_hash(encoder), _state_hash(policy))
        dump(output / "public/WEIGHT_AUDIT.json", dict(schema="prove.p69-book-weight-audit.v1",
                                                       encoder_before=before[0], encoder_after=after[0],
                                                       policy_before=before[1], policy_after=after[1], unchanged=before == after))
        if before != after:
            raise RuntimeError("Frozen P69/P68 s9 weights changed")
        runtime.exit()
        dump(output / "public/LIFECYCLE.json", runtime.lifecycle)
        return dict(actions=steps, graph_complete=strict, official_success=native_ever,
                    native_instant_ever=native_instant, stop_reason=reason,
                    updates=update_events)
    finally:
        if runtime.active:
            runtime.exit()
        renderer.close()
        del vision





def retarget_reference_poses_with_caddy(reference_poses: list[dict], caddy_delta: dict) -> list[dict]:
    """Apply the public caddy SE2 delta to P55 Code reference poses only."""
    from scipy.spatial.transform import Rotation
    demo_xy = np.asarray(caddy_delta["demo_xy"], dtype=np.float64)
    current_xy = np.asarray(caddy_delta["current_xy"], dtype=np.float64)
    delta_yaw = float(caddy_delta["delta_yaw"])
    if demo_xy.shape != (2,) or current_xy.shape != (2,) or not np.isfinite(demo_xy).all() or not np.isfinite(current_xy).all() or not np.isfinite(delta_yaw):
        raise ValueError("finite caddy SE2 delta required for Code reference retarget")
    Rz2 = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)], [np.sin(delta_yaw), np.cos(delta_yaw)]], dtype=np.float64)
    Rz3 = Rotation.from_euler("z", delta_yaw).as_matrix()
    out = []
    for item in reference_poses:
        row = copy.deepcopy(item)
        p = np.asarray(row["position"], dtype=np.float64)
        q = np.asarray(row["quaternion_xyzw"], dtype=np.float64)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("malformed reference pose")
        p[:2] = current_xy + Rz2 @ (p[:2] - demo_xy)
        row["position"] = p.tolist()
        row["quaternion_xyzw"] = (Rotation.from_matrix(Rz3 @ Rotation.from_quat(q).as_matrix()).as_quat()).tolist()
        out.append(row)
    return out

def run_prefix_public_grounded(env, binding, frozen: dict, output: Path, instance: str, entry: dict):
    """R2 prefix: settle book from public masks, retarget frozen demo trajectory, then run graph.

    Runtime grounding reads only PublicRenderer outputs (visible_masks/camera_K/camera_pose)
    plus robot proprio through the existing controller.  Private supervisor facts are still
    used only by LiberoSupervisor/GraphExecutor for the unchanged gates and progress judge.
    """
    from prove.task_graph.schema import TaskGraph
    from prove.task_graph.executor import GraphExecutor
    from prove.task_graph.libero_precision_p51 import MixedRouter
    from prove.task_graph.libero_supervisor import LiberoSupervisor
    from prove.task_graph.p69_public_grounding import (
        PublicRetargetVariantController, estimate_role_delta, load_reference)
    from tools.task_graph.collect_libero_p52_v02 import PublicRenderer
    from tools.task_graph.run_libero_p51_v02 import grounding, insert_entry_domain

    prefix = binding["prefix"]
    graph = TaskGraph.from_dict(prefix["runtime_graph"])
    handoff_after = frozen["graph"].get("prefix_handoff_after", "s6")
    if handoff_after not in ("s5", "s6"):
        raise ValueError("prefix_handoff_after must be s5 or s6")
    cfg = frozen["graph"]["r2_public_grounding"]
    reference = load_reference(cfg["grounding_reference"]["path"])
    renderer = PublicRenderer(env, binding["public"])
    supervisor = LiberoSupervisor(env, json.loads(Path(prefix["calibration"]).read_text()), instance_id=instance)
    obs = env._get_observations(force_update=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "public_grounding").mkdir(exist_ok=True)
    actions = 0
    stable = []
    public = None
    def book_centroid_px(frame):
        mask = np.asarray(frame["visible_masks"])[0, 1].astype(bool)
        yy, xx = np.nonzero(mask)
        if len(xx) == 0:
            return None
        return np.array([float(xx.mean() + 0.5), float(yy.mean() + 0.5)], dtype=float)
    def write_terminal(reason, grounding=None):
        arrays, _ = snapshot(env, obs)
        arrays = {k: v for k, v in arrays.items() if k.startswith("supervisor_")}
        arrays["action_next"] = np.asarray([])
        private_path = output / f"frame_{actions:04d}.npz"
        np.savez_compressed(private_path, **arrays)
        _append(output / "TRACE.jsonl", dict(t=actions, phase="r2_terminal", file=private_path.name,
                                             sha256=sha(private_path), action=None, ready=False, stop=reason,
                                             grounding=grounding))

    try:
        settling = cfg["settling"]
        for settle_t in range(int(settling["max_open_steps"]) + 1):
            public = renderer.observe(obs)
            frame = output / "public_grounding" / f"settle_{settle_t:04d}.npz"
            np.savez_compressed(frame, **public)
            centroid = book_centroid_px(public)
            if centroid is not None:
                stable.append(centroid)
                stable = stable[-int(settling["stable_frames"]):]
            else:
                stable = []
            stable_ok = (settle_t >= int(settling["min_open_steps_before_lock"])
                         and len(stable) == int(settling["stable_frames"])
                         and max(np.linalg.norm(x - stable[-1]) for x in stable) <= float(settling["stable_pixel_max"]))
            _append(output / "PUBLIC_GROUNDING_TRACE.jsonl", dict(t=actions, settle_index=settle_t, file=str(frame.relative_to(output)),
                                                                  sha256=sha(frame), book_centroid_px=None if centroid is None else centroid.tolist(),
                                                                  recent_book_centroids_px=[x.tolist() for x in stable],
                                                                  stable_ok=bool(stable_ok)))
            if stable_ok:
                break
            if settle_t >= int(settling["max_open_steps"]):
                write_terminal("r2_grounding_unstable_book")
                return dict(qualified=False, stop_reason="r2_grounding_unstable_book", actions=actions,
                            completed_s6=False, public_entry_qualified=False)
            open_action = np.asarray([0., 0., 0., 0., 0., 0., -1.], dtype=float)
            arrays, _ = snapshot(env, obs)
            arrays = {k: v for k, v in arrays.items() if k.startswith("supervisor_")}
            arrays["action_next"] = open_action.copy()
            private_path = output / f"frame_{actions:04d}.npz"
            np.savez_compressed(private_path, **arrays)
            _append(output / "TRACE.jsonl", dict(t=actions, phase="r2_settle_open", file=private_path.name,
                                                 sha256=sha(private_path), action=open_action.tolist(), ready=False, stop=None))
            obs, _, _, _ = env.step(open_action)
            actions += 1
        assert public is not None
        book = estimate_role_delta(public, reference, "book")
        caddy = estimate_role_delta(public, reference, "caddy")
        if not book.valid or not caddy.valid:
            grounding = dict(book=book.to_dict(), caddy=caddy.to_dict())
            write_terminal("r2_grounding_missing_role", grounding)
            return dict(qualified=False, stop_reason="r2_grounding_missing_role", actions=actions,
                        completed_s6=False, public_entry_qualified=False,
                        grounding=grounding)
        skill = PublicRetargetVariantController(binding["environment"]["source"], graph, prefix["checkpoint"],
                                                entry.get("grasp_world_offset", [0.0, 0.0, 0.0]),
                                                book_delta=book, caddy_delta=caddy)
        router = MixedRouter()
        for name in {s.skill for s in graph.nodes}:
            router.register(name, kind=router.required_kind(name), revision=skill.revision,
                            call=skill, compatible=lambda g: g.stage_id in skill.segments)
        executor = GraphExecutor(graph, router, instance_id=instance)
        active = None
        holds = 0
        while True:
            skill.set_observation(obs)
            action = None
            stop = None
            ready = False
            events = []
            for _ in range(8):
                stage = graph.stage(executor.current_stage)
                if stage.id == "s7":
                    raise RuntimeError("Old s7 controller must never execute")
                if active != stage.id:
                    if active is not None:
                        skill.on_exit()
                    skill.on_enter(stage.id)
                    active = stage.id
                report = supervisor.observe(obs, stage.id)
                facts = dict(report.facts)
                entry_ok, entry_info = insert_entry_domain(skill)
                facts["public_insert_entry_domain"] = entry_ok
                if actions >= binding["prefix_max_actions"]:
                    stop = "prefix_budget"
                    break
                result = executor.tick(grounding(graph, stage, instance, actions), facts, progress=report.progress)
                event = asdict(result)
                event["action"] = None if result.action is None else np.asarray(result.action).tolist()
                events.append(dict(judge=event, supervisor=asdict(report), public_entry=dict(qualified=entry_ok, **entry_info),
                                   controller=dict(skill.diagnostics), r2_grounding=dict(book=book.to_dict(), caddy=caddy.to_dict())))
                if result.state == "complete":
                    if stage.id == handoff_after:
                        ready = True
                        break
                    continue
                action = event["action"]
                if action is None:
                    unknown = result.reason in ("invariant_unknown", "precondition_unknown")
                    if unknown and result.state != "structural_failure" and not any(facts.get(k) is True for k in ("object_lost", "wrong_target")) and holds < 3:
                        holds += 1
                        action = [0., 0., 0., 0., 0., 0., 1.]
                    else:
                        stop = "prefix_" + str(result.reason)
                break
            else:
                raise RuntimeError("Unbounded same-observation prefix transition")
            arrays, _ = snapshot(env, obs)
            arrays = {k: v for k, v in arrays.items() if k.startswith("supervisor_")}
            arrays["action_next"] = np.asarray([] if action is None else action)
            path = output / f"frame_{actions:04d}.npz"
            np.savez_compressed(path, **arrays)
            _append(output / "TRACE.jsonl", dict(t=actions, file=path.name, sha256=sha(path), events=events,
                                                 action=action, ready=ready, stop=stop))
            if stop or ready:
                return dict(qualified=ready, stop_reason=stop, actions=actions, completed_s6=bool(ready and handoff_after == "s6"),
                            handoff_stage=handoff_after if ready else None,
                            public_entry_qualified=bool(entry_ok), grounding=dict(book=book.to_dict(), caddy=caddy.to_dict()))
            obs, _, _, _ = env.step(np.asarray(action))
            actions += 1
    finally:
        renderer.close()
        if "skill" in locals() and getattr(skill, "stage_id", None) is not None:
            skill.on_exit()

def infer_failure_stage(output: Path, prefix: dict, code: dict, result: dict) -> str | None:
    if result.get("graph_complete"):
        return None
    if code.get("qualified"):
        return "s9"
    if prefix.get("qualified"):
        trace = output / "code/TRACE.jsonl"
        if trace.exists() and trace.read_text().splitlines():
            return json.loads(trace.read_text().splitlines()[-1]).get("stage_id", "code")
        return "code"
    trace = output / "prefix/TRACE.jsonl"
    if trace.exists() and trace.read_text().splitlines():
        last = json.loads(trace.read_text().splitlines()[-1])
        events = last.get("events") or []
        if events:
            return events[-1].get("judge", {}).get("stage_id", "prefix")
    return "prefix"

def run_cell(freeze_path: Path, approval_path: Path, index: int, output: Path):
    frozen = validate_freeze(freeze_path, approval_path)
    if not 0 <= index < len(frozen["cells"]):
        raise ValueError("P69 cell outside frozen scope")
    cell = frozen["cells"][index]
    np.random.seed(int(cell["execution_seed"]))
    torch.manual_seed(int(cell["execution_seed"]))
    output.mkdir(parents=True, exist_ok=False)
    for name in ("prefix", "code", "public", "private"):
        (output / name).mkdir()
    dump(output / "BINDING.json", frozen)
    dump(output / "CELL.json", cell)
    dump(output / "TASK_GRAPH.json", frozen["graph"]["task_graph"])
    dump(output / "EXECUTION.json", dict(started_unix=time.time(), automatic_retry=False))
    original = load_p53_full_binding(frozen["p55_binding"]["path"])
    # R0/P69 graph config owns the Code graph, but the existing P55 code expects
    # a child binding with P53/P55 fields. Reuse the bound full-task scene/references
    # and replace only the precision MLP checkpoint with the frozen P69 head.
    child = copy.deepcopy(original)
    child["task_graph"] = copy.deepcopy(frozen["graph"]["task_graph"])
    child.setdefault("public", {})["resolution"] = 448
    if frozen["graph"].get("prefix_runtime_graph_mode", "inherit_parent_original") == "override":
        child.setdefault("prefix", {})["runtime_graph"] = copy.deepcopy(frozen["graph"]["prefix_runtime_graph"])
    child.setdefault("prefix", {})["checkpoint"] = frozen["files"]["precision_head"]["path"]
    row = init_row(frozen, cell)
    env = None
    try:
        env = initialize_full_env(original, output, np.asarray(row.get("state_flattened", row.get("qpos45")), dtype=float))
        names = _names(env)
        dump(output / "private/NAMES.json", names)
        if "r2_public_grounding" in frozen["graph"]:
            prefix = run_prefix_public_grounded(env, child, frozen, output / "prefix", output.name,
                                               dict(grasp_world_offset=row.get("grasp_world_offset", [0.0, 0.0, 0.0])))
        else:
            prefix = run_prefix(env, child, output / "prefix", output.name,
                                dict(grasp_world_offset=row.get("grasp_world_offset", [0.0, 0.0, 0.0])))
        if prefix.get("qualified") and "r2_public_grounding" in frozen["graph"]:
            caddy = prefix.get("grounding", {}).get("caddy")
            if not isinstance(caddy, dict) or caddy.get("valid") is not True:
                raise ValueError("R2 qualified prefix requires valid public caddy grounding for Code retarget")
            child["reference_poses"] = retarget_reference_poses_with_caddy(child["reference_poses"], caddy)
        dump(output / "PREFIX_RESULT.json", prefix)
        code = run_p55_code(env, child, dict(near_position_tolerance=.003585, near_gap_max=.00717), output / "code", names) if prefix["qualified"] else dict(qualified=False, actions=0, near_entry_qualified=False, stop_reason="prefix_failed")
        dump(output / "CODE_RESULT.json", code)
        transitions = []
        if prefix["qualified"]:
            handoff_stage = prefix.get("handoff_stage") or ("s6" if prefix.get("completed_s6") else "prefix")
            transitions.append(dict(from_stage=handoff_stage, to_stage="s7", after_actions=prefix["actions"]))
        if (output / "code/TRACE.jsonl").exists():
            for line in (output / "code/TRACE.jsonl").read_text().splitlines():
                record = json.loads(line)
                for transition in record.get("transitions", []):
                    transitions.append(dict(**transition, after_actions=prefix["actions"] + record["t"]))
        if code["qualified"]:
            transitions.append(dict(from_stage="s8a", to_stage="s9", after_actions=prefix["actions"] + code["actions"]))
            result = run_s9_book(env, frozen, cell, child, output, names)
        else:
            result = dict(actions=0, graph_complete=False, official_success=bool(env._check_success()),
                          stop_reason=code["stop_reason"], updates=0)
        if result["graph_complete"]:
            transitions.append(dict(from_stage="s9", to_stage=None,
                                    after_actions=prefix["actions"] + code["actions"] + result["actions"]))
        dump(output / "STAGE_TRANSITIONS.json", transitions)
        _seal(output, freeze_path, cell, prefix["actions"] + code["actions"] + result["actions"])
        result.update(schema=RESULT_SCHEMA, cell=cell, init_state=row, prefix_actions=prefix["actions"],
                      code_actions=code["actions"], near_entry_qualified=code["qualified"],
                      total_actions=prefix["actions"] + code["actions"] + result["actions"],
                      original_initial_restores=1, later_state_restores=0, automatic_retry=False,
                      failure_stage=infer_failure_stage(output, prefix, code, result),
                      seal_sha256=sha(output / "SEAL.json"), full_task_success_evidence=True)
        dump(output / "RESULT.json", result)
        return result
    except Exception as exc:
        dump(output / "ABORT.json", dict(cell=cell, error=type(exc).__name__, reason=str(exc), automatic_retry=False))
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("output", "p55-binding", "init-scope", "graph-config", "precision-head", "encoder-checkpoint", "action-checkpoint", "a2-head", "priors", "dino-binding"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", choices=["explore", "test"], required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--version-source-root", type=Path)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=["continuous"])
    p = sub.add_parser("run-cell")
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(output=args.output, p55_binding=args.p55_binding, init_scope=args.init_scope,
                        graph_config=args.graph_config, precision_head=args.precision_head, encoder_checkpoint=args.encoder_checkpoint,
                        action_checkpoint=args.action_checkpoint, a2_head=args.a2_head, priors=args.priors,
                        dino_binding=args.dino_binding, split=args.split, version=args.version, arms=args.arms,
                        version_source_root=args.version_source_root)
        summary = {"schema": value["schema"], "freeze": str((args.output / "PHYSICAL_FREEZE.json").resolve()),
                   "freeze_sha256": sha(args.output / "PHYSICAL_FREEZE.json"),
                   "version": value["version"], "split": value["split"], "cells": len(value["cells"])}
    else:
        value = run_cell(args.freeze, args.approval, args.index, args.output)
        summary = {"schema": value["schema"], "cell_index": value["cell"]["index"],
                   "success": bool(value.get("graph_complete")), "failure_stage": value.get("failure_stage"),
                   "total_actions": value.get("total_actions"), "seal_sha256": value.get("seal_sha256")}
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
