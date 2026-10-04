#!/usr/bin/env python3
"""P70 object-centric full-task runner skeleton.

This is a new P70 runner.  It preserves P69 artifacts and imports only stable
helpers for environment setup, public rendering, sealing and the frozen P68 s9
runtime.  Formal physics requires a Root review approval file; prepare itself
starts no simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from prove.task_graph.p70_object_controller import P70ObjectController, STAGES, load_controller_config
from tools.task_graph.collect_libero_demo_gt import dump, sha, snapshot
from tools.task_graph.collect_libero_p52_v02 import PublicRenderer
from tools.task_graph.p54_full_task import _names
from tools.task_graph.p52_full_task import InsertJudge, structural_measurement
from tools.task_graph.p64_book_evaluate import PUBLIC_KEYS, _append, _seal
from tools.task_graph.p69_full_rsi_runner import (
    binding, binding_chain, validate_init_scope, scope_entries, init_row,
    initialize_full_env, load_p53_full_binding, run_s9_book,
)

SCHEMA = "prove.p70-object-centric-freeze.v1"
APPROVAL_SCHEMA = "prove.p70-root-review.v1"
RESULT_SCHEMA = "prove.p70-object-centric-result.v1"


def read(path: Path):
    return json.loads(Path(path).read_text())


def controller_file_bindings(cfg: dict) -> dict:
    bindings = {}
    def add(label: str, item):
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            bindings[label] = dict(path=item["path"], sha256=item["sha256"])
    add("grounding_reference", cfg.get("grounding_reference"))
    add("object_relation_targets", cfg.get("object_relation_targets"))
    add("held_pose_calibration", cfg.get("held_pose_calibration"))
    gp = cfg.get("grasp_policy", {})
    for key in ("checkpoint", "demo_relations", "training_freeze", "training_receipt"):
        add("grasp_policy." + key, gp.get(key))
    try:
        ref = read(Path(cfg["grounding_reference"]["path"]))
        br = ref.get("role_calibration", {}).get("book", {}).get("book_reference")
        add("grounding_reference.book_reference", br)
        pg = ref.get("source_demo", {}).get("private_geometry")
        pf = ref.get("source_demo", {}).get("public_frame")
        add("grounding_reference.source_demo.private_geometry", pg)
        add("grounding_reference.source_demo.public_frame", pf)
    except Exception:
        pass
    return bindings


def verify_bound_files(bindings: dict):
    for label, item in bindings.items():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("P70 bound file changed: " + label + " " + item["path"])


def flatten_state(env):
    state = env.sim.get_state()
    return np.asarray(state.flatten(), dtype=np.float64)


def mask_centroid(mask: np.ndarray):
    yy, xx = np.nonzero(np.asarray(mask).astype(bool))
    if len(xx) == 0:
        return None
    return np.array([float(xx.mean()), float(yy.mean())], dtype=np.float64)




def trace_record(global_t: int, telemetry: dict, **prefix) -> dict:
    record = dict(telemetry)
    controller_t = record.pop("t", None)
    record.update(prefix)  # explicit runtime/log fields override controller record keys
    record["t"] = int(global_t)
    if controller_t is not None:
        record["controller_t"] = controller_t
    return record


def placement_verification_limit(frozen: dict, prefix_actions: int, s9_actions: int) -> int:
    requested = int(frozen.get("controller", {}).get("placement_verification_max_actions", 0) or 0)
    if requested <= 0:
        return 0
    remaining = int(frozen["budgets"]["total_max_actions"]) - int(prefix_actions) - int(s9_actions)
    return max(0, min(requested, remaining))


def placement_verification_precondition(s9: dict, official: bool, fingers, wrong: bool) -> bool:
    return (s9.get("stop_reason") == "micro_budget24"
            and bool(official)
            and fingers is not None
            and all(bool(x) for x in fingers)
            and not bool(wrong))


def stationary_closed_action() -> np.ndarray:
    return np.asarray([0., 0., 0., 0., 0., 0., 1.], dtype=np.float64)


def reset_osc_goal_to_current(env) -> dict:
    audits = []
    for idx, robot in enumerate(getattr(env, "robots", [])):
        controller = getattr(robot, "controller", None)
        if controller is not None:
            if hasattr(controller, "update"):
                controller.update(force=True)
            if hasattr(controller, "reset_goal"):
                controller.reset_goal()
        gripper = getattr(robot, "gripper", None)
        if gripper is not None and hasattr(gripper, "current_action"):
            try:
                gripper.current_action = np.ones_like(np.asarray(gripper.current_action, dtype=np.float64))
            except Exception:
                gripper.current_action = np.ones(len(getattr(gripper, "actuators", [0, 1])), dtype=np.float64)
        audits.append({"robot_index": idx, "controller_goal_reset_to_current_proprio": controller is not None,
                       "gripper_commanded_closed": gripper is not None})
    return {"object_gt_used_for_action": False, "robots": audits}

def source_bindings():
    files = [
        "tools/task_graph/p70_stage_runner.py",
        "prove/task_graph/p70_object_controller.py",
        "prove/task_graph/p69_public_grounding.py",
        "prove/task_graph/p69_book_silhouette.py",
        "prove/task_graph/p57_public_geometry.py",
        "prove/task_graph/p70_grasp_policy_v02.py",
        "prove/task_graph/p70_stage_judge.py",
        "tools/task_graph/collect_libero_p52_v02.py",
        "tools/task_graph/collect_libero_demo_gt.py",
        "tools/task_graph/p52_full_task.py",
        "tools/task_graph/p69_full_rsi_runner.py",
        "prove/fast_adapter/p68_book_runtime.py",
        "prove/fast_adapter/p64_book_bridge.py",
        "prove/fast_adapter/p64_common.py",
        "prove/fast_adapter/p61_encoder.py",
        "prove/fast_adapter/visual_alignment.py",
    ]
    return {str((ROOT / f).resolve()): sha(ROOT / f) for f in files if (ROOT / f).exists()}


def split_cells(init_scope: dict, split: str, version: str) -> list[dict]:
    if split not in ("qualification", "explore", "test"):
        raise ValueError("P70 physical splits are qualification/explore/test only")
    if split == "qualification":
        wanted = [20, 2, 12, 26]
        by_id = {int(r["id"]): r for r in scope_entries(init_scope) if r["split"] == "train"}
        rows = [by_id[i] for i in wanted]
        base_seed = 70010
    else:
        rows = [r for r in scope_entries(init_scope) if r["split"] == split]
        if len(rows) != 10:
            raise ValueError(f"P70 split {split} must contain 10 initial states")
        base_seed = 70020 if split == "explore" else 70030
    return [dict(index=i, version=version, split=split, init_id=row["id"], init_index=i,
                 execution_seed=base_seed + i, state_sha256=row.get("state_sha256")) for i, row in enumerate(rows)]


def prepare(*, output: Path, p69_freeze: Path, init_scope: Path, controller_config: Path,
            split: str, version: str, stop_after: str | None = None):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    p69 = read(p69_freeze)
    init = validate_init_scope(read(init_scope))
    cfg = load_controller_config(controller_config)
    P70ObjectController(cfg)  # result-free CPU load validates nested bindings/checkpoint schemas
    nested_bindings = controller_file_bindings(cfg)
    verify_bound_files(nested_bindings)
    cells = split_cells(init, split, version)
    frozen = dict(schema=SCHEMA, status="frozen_before_results", version=version, split=split,
                  cells=cells, p69_freeze=binding(p69_freeze), p69_binding_schema=p69.get("schema"),
                  p55_binding=p69["p55_binding"], parent_binding_chain=binding_chain(p69["p55_binding"]["path"]),
                  init_scope=binding(init_scope), init_scope_materialized=init,
                  controller_config=binding(controller_config), controller_file_bindings=nested_bindings, controller=cfg,
                  files=p69.get("files", {}), budgets=dict(total_max_actions=720, p68_micro_max_actions=24, micro_max_actions=24, stop_after=stop_after, placement_verification_max_actions=int(cfg.get("placement_verification_max_actions", 0) or 0)),
                  stages=list(STAGES), initial_restores=1, later_restores=0, automatic_retries=0,
                  runtime_contract=dict(
                      complete_initial_state="one full 45D MuJoCo state restore at episode start only",
                      controller="object-centric public mask/camera/proprio closed loop",
                      private_pose_for_actions=False,
                      boolean_supervisor="stage judge only; no continuous geometry to action",
                      held_book="rigid T_eef_book captured at grasp and propagated by proprio",
                      s9="unchanged frozen P68 ReferenceBookRuntime after near_entry", placement_verification="optional post-P68 stationary Code hold; no adapter updates; default disabled; frames are post-action with global_t=prefix+s9+t+1", qualification_stop_after=stop_after),
                  sources=source_bindings(), rng_streams=dict(explore="70020+i", test="70030+i"))
    dump(output / "PHYSICAL_FREEZE.json", frozen)
    print(json.dumps({"schema": SCHEMA, "freeze": str((output / "PHYSICAL_FREEZE.json").resolve()),
                      "freeze_sha256": sha(output / "PHYSICAL_FREEZE.json"), "cells": len(cells)}, indent=2))
    return frozen


def validate_freeze(path: Path, approval: Path | None = None, check_files: bool = True) -> dict:
    frozen = read(path)
    if frozen.get("schema") != SCHEMA or frozen.get("status") != "frozen_before_results":
        raise ValueError("wrong P70 freeze")
    validate_init_scope(frozen["init_scope_materialized"])
    load_controller_config(frozen["controller_config"]["path"])
    if approval is not None:
        review = read(approval)
        if (review.get("schema") != APPROVAL_SCHEMA or review.get("freeze_sha256") != sha(path)
                or review.get("root_reviewed") is not True or review.get("physical_execution_authorized") is not True):
            raise ValueError("missing exact P70 Root review")
    if check_files:
        verify_bound_files(frozen.get("controller_file_bindings", {}))
        for item in [frozen["p69_freeze"], frozen["init_scope"], frozen["controller_config"], *frozen["files"].values()]:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("P70 frozen file changed: " + item["path"])
        for path, digest in frozen["sources"].items():
            if sha(path) != digest:
                raise ValueError("P70 source changed: " + path)
    return frozen


def build_stage_judge(env, names: dict, child: dict, controller_config: dict):
    try:
        from prove.task_graph.p70_stage_judge import StageJudge
    except ImportError as exc:
        raise RuntimeError("P70 StageJudge missing; Root assigned prove/task_graph/p70_stage_judge.py to p64_cup") from exc
    return StageJudge(env, names, child, controller_config)



def public_settling(env, obs, renderer: PublicRenderer, output: Path, *, start_t: int, max_steps: int = 12, min_steps: int = 5) -> tuple[object, int, dict]:
    trace = output / "PUBLIC_SETTLING_TRACE.jsonl"
    centroids = []
    actions = 0
    stable = False
    action = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float64)
    while actions < max_steps:
        public_all = renderer.observe(obs)
        public = {key: public_all[key] for key in PUBLIC_KEYS}
        public_path = output / "public" / f"frame_{start_t + actions:04d}.npz"
        np.savez_compressed(public_path, **public)
        arrays, _ = snapshot(env, obs)
        arrays = {k: v for k, v in arrays.items() if k.startswith("supervisor_")}
        arrays["action_next"] = action.copy()
        arrays["sim_state_flattened"] = flatten_state(env)
        private_path = output / "private" / f"frame_{start_t + actions:04d}.npz"
        np.savez_compressed(private_path, **arrays)
        c = mask_centroid(public["visible_masks"][0, 1])
        centroids.append(None if c is None else c.tolist())
        if actions + 1 >= min_steps and len(centroids) >= 3 and all(x is not None for x in centroids[-3:]):
            pts = np.asarray(centroids[-3:], dtype=np.float64)
            stable = bool(np.max(np.linalg.norm(pts - pts[-1], axis=1)) <= 1.0)
        _append(trace, dict(t=start_t + actions, phase="public_settling", public=f"public/{public_path.name}",
                            public_sha256=sha(public_path), private=f"private/{private_path.name}", private_sha256=sha(private_path),
                            action=action.tolist(), book_centroid_px=centroids[-1], stable=stable, min_steps=min_steps, max_steps=max_steps))
        obs, _, _, _ = env.step(action.copy())
        actions += 1
        if stable:
            return obs, actions, dict(stable=True, actions=actions, last_centroid=centroids[-1])
    return obs, actions, dict(stable=False, actions=actions, last_centroid=centroids[-1] if centroids else None, stop="public_settling_unstable")

def run_object_prefix(env, frozen: dict, output: Path, names: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "public").mkdir(exist_ok=True)
    (output / "private").mkdir(exist_ok=True)
    (output / "controller").mkdir(exist_ok=True)
    child = load_p53_full_binding(frozen["p55_binding"]["path"])
    child = json.loads(json.dumps(child))
    child.setdefault("public", {})["resolution"] = 448
    renderer = PublicRenderer(env, child["public"])
    controller = P70ObjectController(frozen["controller"])
    judge = build_stage_judge(env, names, child, frozen["controller"])
    obs = env._get_observations(force_update=True)
    actions = 0
    stop = None
    stage_entries = []
    try:
        obs, used, settling = public_settling(env, obs, renderer, output, start_t=actions)
        actions += used
        if not settling.get("stable"):
            return dict(qualified=False, stop_reason="public_settling_unstable", actions=actions, stage="public_settling", stage_entries=[], settling=settling)
        while actions < int(frozen["budgets"]["total_max_actions"] - frozen["budgets"]["p68_micro_max_actions"]):
            observed_public = renderer.observe(obs)
            public = {key: observed_public[key] for key in PUBLIC_KEYS}
            public_path = output / "public" / f"frame_{actions:04d}.npz"
            np.savez_compressed(public_path, **public)
            arrays, _ = snapshot(env, obs)
            arrays = {k: v for k, v in arrays.items() if k.startswith("supervisor_")}
            arrays["sim_state_flattened"] = flatten_state(env)
            facts, fact_diagnostics = judge.observe(obs, controller.state.stage, actions)
            action, telemetry = controller.command(public, facts)
            transition = telemetry.get("transition") or {}
            terminal_stop = None
            if isinstance(transition, dict) and transition.get("stop"):
                terminal_stop = str(transition["stop"])
            if facts.get("object_lost"):
                terminal_stop = terminal_stop or "object_lost"
            if facts.get("wrong_target"):
                terminal_stop = terminal_stop or "wrong_target"
            executed = terminal_stop is None
            arrays["action_next"] = action.copy() if executed else np.asarray([])
            private_path = output / "private" / f"frame_{actions:04d}.npz"
            np.savez_compressed(private_path, **arrays)
            _append(output / "controller/TRACE.jsonl", trace_record(actions, telemetry, public=f"public/{public_path.name}",
                                                            public_sha256=sha(public_path), private=f"private/{private_path.name}",
                                                            private_sha256=sha(private_path), fact_diagnostics=fact_diagnostics,
                                                            executed=executed, terminal_stop=terminal_stop))
            if controller.state.stage_entries and len(controller.state.stage_entries) > len(stage_entries):
                for row in controller.state.stage_entries[len(stage_entries):]:
                    snap_path = output / "private" / f"stage_entry_{len(stage_entries):03d}_{row['stage']}.npz"
                    np.savez_compressed(snap_path, sim_state_flattened=flatten_state(env), controller_state=np.asarray(json.dumps(controller.state.to_dict()), dtype=object))
                    _append(output / "STAGE_SNAPSHOTS.jsonl", dict(**row, private=f"private/{private_path.name}", private_sha256=sha(private_path),
                                                                    stage_snapshot=f"private/{snap_path.name}", stage_snapshot_sha256=sha(snap_path),
                                                                    controller_state=controller.state.to_dict()))
                stage_entries = list(controller.state.stage_entries)
            completed_stage = transition.get("from_stage") if isinstance(transition, dict) and "to_stage" in transition else None
            if frozen["budgets"].get("stop_after") and completed_stage == frozen["budgets"]["stop_after"]:
                return dict(qualified=True, stop_reason="stop_after_completed_" + completed_stage, actions=actions,
                            stage=controller.state.stage, completed_stage=completed_stage, stage_entries=stage_entries, settling=settling)
            if controller.state.stage == "p68_insert":
                return dict(qualified=True, stop_reason="near_entry_ready", actions=actions,
                            stage="p68_insert", stage_entries=stage_entries, settling=settling)
            if terminal_stop:
                stop = terminal_stop
                break
            obs, _, _, _ = env.step(action.copy())
            actions += 1
        stop = stop or "p70_prefix_budget"
        return dict(qualified=False, stop_reason=stop, actions=actions, stage=controller.state.stage,
                    stage_entries=stage_entries, settling=settling)
    finally:
        renderer.close()


def run_placement_verification(env, frozen: dict, output: Path, names: dict, child: dict, *, max_actions: int, global_offset: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "public").mkdir(exist_ok=True)
    (output / "private").mkdir(exist_ok=True)
    (output / "actions").mkdir(exist_ok=True)
    if max_actions <= 0:
        return {"enabled": False, "actions": 0, "graph_complete": False, "stop_reason": "verification_disabled", "updates": 0}
    child = json.loads(json.dumps(child))
    child.setdefault("public", {})["resolution"] = 448
    renderer = PublicRenderer(env, child["public"])
    judge = InsertJudge()
    action = stationary_closed_action()
    reset_audit = reset_osc_goal_to_current(env)
    dump(output / "RESET_AUDIT.json", reset_audit)
    obs = env._get_observations(force_update=True)
    actions = 0
    native_instant_ever = False
    try:
        while actions < max_actions:
            obs, _, _, _ = env.step(action.copy())
            global_t = int(global_offset) + actions + 1  # post-action observation index for video alignment
            public_all = renderer.observe(obs)
            public = {key: public_all[key] for key in PUBLIC_KEYS}
            public_path = output / "public" / f"frame_{actions:04d}.npz"
            np.savez_compressed(public_path, **public)
            arrays, _ = snapshot(env, obs)
            arrays = {k: v for k, v in arrays.items() if k.startswith("supervisor_")}
            arrays["sim_state_flattened"] = flatten_state(env)
            arrays["action_executed"] = action.copy()
            private_path = output / "private" / f"frame_{actions:04d}.npz"
            np.savez_compressed(private_path, **arrays)
            official, fingers, wrong = structural_measurement(env, names)
            native_instant_ever = native_instant_ever or bool(official)
            judgment = judge.observe(official, fingers, wrong)
            strict_hold = bool(judgment["complete"])
            row = dict(t=actions, global_t=global_t, phase="verify_placement", action=action.tolist(), executed=True,
                       public=f"public/{public_path.name}", public_sha256=sha(public_path),
                       private=f"private/{private_path.name}", private_sha256=sha(private_path),
                       official_success=bool(official), native_instant=bool(official), native_hold3=strict_hold,
                       finger_contacts=fingers, wrong_target=bool(wrong), **judgment)
            _append(output / "actions/VERIFY_ACTIONS.jsonl", row)
            _append(output / "private/JUDGMENTS.jsonl", row)
            actions += 1
            if strict_hold:
                return {"enabled": True, "actions": actions, "graph_complete": True, "official_success": True,
                        "native_instant_ever": native_instant_ever, "stop_reason": "verify_placement_complete",
                        "final_judgment": judgment, "updates": 0, "reset_audit_sha256": sha(output / "RESET_AUDIT.json")}
            if wrong or fingers is None or not all(bool(x) for x in fingers) or not official:
                return {"enabled": True, "actions": actions, "graph_complete": False, "official_success": False,
                        "native_instant_ever": native_instant_ever,
                        "stop_reason": judgment.get("stop") or "verify_placement_goal_or_contact_lost",
                        "final_judgment": judgment, "updates": 0, "reset_audit_sha256": sha(output / "RESET_AUDIT.json")}
        return {"enabled": True, "actions": actions, "graph_complete": False, "official_success": False,
                "native_instant_ever": native_instant_ever, "stop_reason": "verify_placement_budget",
                "updates": 0, "reset_audit_sha256": sha(output / "RESET_AUDIT.json")}
    finally:
        renderer.close()


def run_cell(freeze_path: Path, approval_path: Path, index: int, output: Path):
    frozen = validate_freeze(freeze_path, approval_path)
    if not 0 <= index < len(frozen["cells"]):
        raise ValueError("P70 cell outside frozen scope")
    cell = frozen["cells"][index]
    np.random.seed(int(cell["execution_seed"]))
    torch.manual_seed(int(cell["execution_seed"]))
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "BINDING.json", frozen)
    dump(output / "CELL.json", cell)
    original = load_p53_full_binding(frozen["p55_binding"]["path"])
    row = init_row(frozen, cell)
    env = None
    try:
        env = initialize_full_env(original, output, np.asarray(row.get("state_flattened", row.get("qpos45")), dtype=float))
        names = _names(env)
        dump(output / "private_NAMES.json", names)
        prefix = run_object_prefix(env, frozen, output / "prefix", names)
        dump(output / "PREFIX_RESULT.json", prefix)
        placement_verification = {"enabled": False, "actions": 0, "graph_complete": False, "stop_reason": "not_attempted", "updates": 0}
        if prefix["qualified"] and not frozen["budgets"].get("stop_after"):
            (output / "s9" / "public").mkdir(parents=True, exist_ok=True)
            (output / "s9" / "private").mkdir(parents=True, exist_ok=True)
            s9 = run_s9_book(env, frozen, cell | {"arm": "continuous"}, original, output / "s9", names)
            official, fingers, wrong = structural_measurement(env, names)
            limit = placement_verification_limit(frozen, prefix["actions"], s9["actions"])
            if limit > 0 and placement_verification_precondition(s9, official, fingers, wrong):
                placement_verification = run_placement_verification(env, frozen, output / "placement_verification", names, original, max_actions=limit, global_offset=prefix["actions"] + s9["actions"])
            elif int(frozen.get("controller", {}).get("placement_verification_max_actions", 0) or 0) > 0:
                placement_verification = {"enabled": False, "actions": 0, "graph_complete": False,
                                          "stop_reason": "verification_precondition_failed", "updates": 0,
                                          "precondition": {"s9_stop_reason": s9.get("stop_reason"), "official_success": bool(official),
                                                           "finger_contacts": fingers, "wrong_target": bool(wrong), "budget_limit": limit}}
        else:
            s9 = dict(actions=0, graph_complete=False, official_success=bool(env._check_success()), stop_reason=prefix["stop_reason"], updates=0)
        total = prefix["actions"] + s9["actions"] + placement_verification.get("actions", 0)
        if total > int(frozen["budgets"]["total_max_actions"]):
            raise RuntimeError("P70 total action budget exceeded")
        strict_from_verification = bool(placement_verification.get("graph_complete"))
        final_graph_complete = bool(s9.get("graph_complete")) or strict_from_verification
        _seal(output, freeze_path, cell, total)
        result = dict(schema=RESULT_SCHEMA, cell=cell, init_state=row, prefix=prefix, s9=s9, placement_verification=placement_verification,
                      total_actions=total, graph_complete=final_graph_complete, strict_from_verification=strict_from_verification,
                      official_success=bool(s9.get("official_success") or placement_verification.get("official_success")), original_initial_restores=1,
                      later_state_restores=0, automatic_retry=False, seal_sha256=sha(output / "SEAL.json"),
                      full_task_success_evidence=bool(not frozen["budgets"].get("stop_after")), stage_qualification_only=bool(frozen["budgets"].get("stop_after")))
        dump(output / "RESULT.json", result)
        print(json.dumps({"schema": RESULT_SCHEMA, "cell_index": index, "success": result["graph_complete"],
                          "total_actions": total, "seal_sha256": result["seal_sha256"]}, indent=2))
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--p69-freeze", type=Path, required=True)
    p.add_argument("--init-scope", type=Path, required=True)
    p.add_argument("--controller-config", type=Path, required=True)
    p.add_argument("--split", choices=["qualification", "explore", "test"], required=True)
    p.add_argument("--stop-after", choices=list(STAGES), default=None)
    p.add_argument("--version", required=True)
    p = sub.add_parser("run-cell")
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(output=args.output, p69_freeze=args.p69_freeze, init_scope=args.init_scope,
                controller_config=args.controller_config, split=args.split, version=args.version, stop_after=args.stop_after)
    else:
        run_cell(args.freeze, args.approval, args.index, args.output)


if __name__ == "__main__":
    main()
