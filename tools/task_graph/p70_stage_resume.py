#!/usr/bin/env python3
"""P70 single-stage resume runner from sealed Q0 stage snapshots.

This tool is independent of the Q0 full/qualification runner.  It never edits
Q0 artifacts.  A resume episode restores one sealed 45D MuJoCo state from a real
upstream stage-entry snapshot, preserves the saved P70 controller state, warms
up the simulator/controller with three closed hold actions, then runs only the
requested stage until the stage completes or its budget expires.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from prove.task_graph.p70_object_controller import P70ObjectController, P70ControllerState, STAGES
from tools.task_graph.collect_libero_demo_gt import dump, sha, snapshot
from tools.task_graph.collect_libero_p52_v02 import PublicRenderer
from tools.task_graph.p54_full_task import _names
from tools.task_graph.p64_book_evaluate import PUBLIC_KEYS, _append, _seal
from tools.task_graph.p69_full_rsi_runner import load_p53_full_binding, initialize_full_env
from tools.task_graph.p70_stage_runner import (
    RESULT_SCHEMA as Q0_RESULT_SCHEMA,
    validate_freeze as validate_q0_freeze,
    read,
    flatten_state,
    build_stage_judge,
    binding,
    controller_file_bindings,
    verify_bound_files,
)

SCHEMA = "prove.p70-stage-resume-freeze.v1"
APPROVAL_SCHEMA = "prove.p70-stage-resume-root-review.v1"
RESULT_SCHEMA = "prove.p70-stage-resume-result.v1"
RESUMABLE_STAGES = ("lift_clear", "orient_clear", "transport_above_caddy", "near_entry")
SUFFIX_STAGES = ("orient_clear", "transport_above_caddy", "near_entry")


def _json_controller_state(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    raise ValueError("controller_state is not JSON-decodable")


def load_snapshot(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as z:
        state = np.asarray(z["sim_state_flattened"], dtype=np.float64)
        controller = _json_controller_state(z["controller_state"])
    if state.ndim != 1 or state.size != 45 or not np.isfinite(state).all():
        raise ValueError("stage snapshot must contain finite flattened 45D MuJoCo state")
    if not isinstance(controller, dict) or controller.get("stage") not in STAGES:
        raise ValueError("stage snapshot controller state is invalid")
    return state, controller



def load_snapshot_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k].copy() for k in z.files}


def restore_non45d_control_state(env, snapshot_arrays: dict) -> dict:
    if "supervisor_ctrl" not in snapshot_arrays:
        raise ValueError("stage resume snapshot must include supervisor_ctrl for OSC/gripper restore")
    ctrl = np.asarray(snapshot_arrays["supervisor_ctrl"], dtype=np.float64)
    if ctrl.shape != env.sim.data.ctrl.shape:
        raise ValueError(f"snapshot ctrl shape {ctrl.shape} does not match env ctrl {env.sim.data.ctrl.shape}")
    env.sim.data.ctrl[:] = ctrl
    restored = []
    for index, robot in enumerate(env.robots):
        if not hasattr(robot, "gripper") or not hasattr(robot.gripper, "actuators"):
            raise ValueError("robot gripper actuator metadata unavailable for resume")
        ids = [env.sim.model.actuator_name2id(name) for name in robot.gripper.actuators]
        ranges = env.sim.model.actuator_ctrlrange[ids]
        if ranges.shape[0] != len(ids) or ranges.shape[1] != 2:
            raise ValueError("invalid gripper actuator ctrl range")
        denom = 0.5 * (ranges[:, 1] - ranges[:, 0])
        if np.any(np.abs(denom) < 1e-12):
            raise ValueError("zero gripper actuator ctrl range")
        current = np.clip((env.sim.data.ctrl[ids] - ranges.mean(axis=1)) / denom, -1.0, 1.0)
        robot.gripper.current_action = current
        if not hasattr(robot, "controller") or not hasattr(robot.controller, "update") or not hasattr(robot.controller, "reset_goal"):
            raise ValueError("robot OSC controller reset API unavailable for resume")
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        restored.append(dict(robot_index=index, gripper_actuators=list(robot.gripper.actuators), actuator_ids=[int(x) for x in ids],
                             restored_current_action=current.tolist(), ctrl_values=env.sim.data.ctrl[ids].tolist()))
    return dict(authority="restored_from_sealed_stage_snapshot_supervisor_ctrl_and_public_proprio_controller_reset",
                object_gt_used_for_action=False, ctrl_shape=list(ctrl.shape), robots=restored)


def control_arrays_for_prepare(stage_snapshot: Path, control_snapshot: Path | None) -> tuple[dict, Path]:
    stage_arrays = load_snapshot_arrays(stage_snapshot)
    if "supervisor_ctrl" in stage_arrays:
        return stage_arrays, stage_snapshot
    if control_snapshot is None:
        raise ValueError("stage snapshot lacks supervisor_ctrl; provide --control-snapshot from the sealed paired private frame")
    control_arrays = load_snapshot_arrays(control_snapshot)
    if "supervisor_ctrl" not in control_arrays:
        raise ValueError("control snapshot must include supervisor_ctrl")
    if "sim_state_flattened" not in control_arrays:
        raise ValueError("control snapshot must include sim_state_flattened for pairing check")
    if not np.array_equal(np.asarray(stage_arrays["sim_state_flattened"]), np.asarray(control_arrays["sim_state_flattened"])):
        raise ValueError("control snapshot sim_state_flattened does not exactly match stage snapshot")
    return control_arrays, control_snapshot


def restore_snapshot(env, flattened: np.ndarray):
    env.sim.reset()
    env.sim.set_state_from_flattened(np.asarray(flattened, dtype=np.float64))
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations(force_update=True)


def set_controller_state(controller: P70ObjectController, state: dict, stage: str | None = None):
    value = dict(state)
    if stage is not None:
        value["stage"] = stage
    allowed = P70ControllerState.__dataclass_fields__.keys()
    controller.state = P70ControllerState(**{k: value.get(k) for k in allowed if k in value})
    return controller


def closed_hold_action() -> np.ndarray:
    return np.array([0., 0., 0., 0., 0., 0., 1.], dtype=np.float64)




def trace_record(global_t: int, telemetry: dict, **prefix) -> dict:
    record = dict(telemetry)
    controller_t = record.pop("t", None)
    record.update(prefix)  # explicit runtime/log fields override controller record keys
    record["t"] = int(global_t)
    if controller_t is not None:
        record["controller_t"] = controller_t
    return record

def source_bindings():
    files = [
        "tools/task_graph/p70_stage_resume.py",
        "tools/task_graph/p70_stage_runner.py",
        "prove/task_graph/p70_object_controller.py",
        "prove/task_graph/p70_stage_judge.py",
        "prove/task_graph/p70_grasp_policy_v02.py",
        "prove/task_graph/p69_public_grounding.py",
        "prove/task_graph/p69_book_silhouette.py",
        "tools/task_graph/collect_libero_p52_v02.py",
        "tools/task_graph/collect_libero_demo_gt.py",
        "tools/task_graph/p69_full_rsi_runner.py",
    ]
    return {str((ROOT / f).resolve()): sha(ROOT / f) for f in files if (ROOT / f).exists()}




def validate_scope(mode: str, stage: str, budget: int, stop_after: str | None) -> str:
    if mode not in ("single", "suffix"):
        raise ValueError("resume mode must be single or suffix")
    if mode == "single":
        if stage not in RESUMABLE_STAGES:
            raise ValueError("P70 stage resume is only for held stages before P68 insert")
        if stop_after is not None and stop_after != stage:
            raise ValueError("single-stage stop_after must be omitted or equal to stage")
        if not (0 < int(budget) <= 240):
            raise ValueError("single-stage resume budget must be in 1..240 and includes 3 warmup actions")
        return stage
    if stage != "orient_clear":
        raise ValueError("suffix mode starts only from a real orient_clear snapshot")
    target = stop_after or "near_entry"
    if target not in SUFFIX_STAGES:
        raise ValueError("suffix stop_after must be one of orient_clear/transport_above_caddy/near_entry")
    if SUFFIX_STAGES.index(target) < SUFFIX_STAGES.index(stage):
        raise ValueError("suffix stop_after cannot precede the start stage")
    if not (0 < int(budget) <= 392):
        raise ValueError("suffix resume budget must be in 1..392 and includes 3 warmup actions")
    return target

def prepare(*, output: Path, q0_result: Path, stage_snapshot: Path, stage: str, budget: int, controller_config: Path | None = None, mode: str = "single", stop_after: str | None = None, control_snapshot: Path | None = None):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    stop_after = validate_scope(mode, stage, int(budget), stop_after)
    result = read(q0_result)
    if result.get("schema") != Q0_RESULT_SCHEMA or not result.get("stage_qualification_only"):
        raise ValueError("resume source must be a P70 Q0 stage qualification result")
    cell_root = q0_result.parent
    q0_freeze = cell_root / "BINDING.json"
    q0_seal = cell_root / "SEAL.json"
    if not q0_freeze.exists() or not q0_seal.exists():
        raise ValueError("source Q0 cell must contain BINDING.json and SEAL.json")
    if result.get("seal_sha256") != sha(q0_seal):
        raise ValueError("RESULT.seal_sha256 does not match source SEAL.json")
    seal = read(q0_seal)
    rel_snapshot = stage_snapshot.resolve().relative_to(cell_root.resolve()).as_posix()
    seal_files = seal.get("files", {})
    if rel_snapshot not in seal_files or sha(stage_snapshot) != seal_files[rel_snapshot]:
        raise ValueError("stage snapshot is not sealed by source Q0 SEAL.files")
    if control_snapshot is not None:
        rel_control = control_snapshot.resolve().relative_to(cell_root.resolve()).as_posix()
        if rel_control not in seal_files or sha(control_snapshot) != seal_files[rel_control]:
            raise ValueError("control snapshot is not sealed by source Q0 SEAL.files")
    q0 = validate_q0_freeze(q0_freeze, approval=None, check_files=True)
    if controller_config is None:
        candidate_controller = q0["controller"]
        candidate_binding = q0.get("controller_config")
        candidate_nested = q0.get("controller_file_bindings", {})
        candidate_source = "source_q0_controller"
    else:
        from prove.task_graph.p70_object_controller import load_controller_config
        candidate_controller = load_controller_config(controller_config)
        P70ObjectController(candidate_controller)  # CPU load validates model/config schema
        candidate_nested = controller_file_bindings(candidate_controller)
        verify_bound_files(candidate_nested)
        candidate_binding = binding(controller_config)
        candidate_source = "override_controller_config"
    state, controller_state = load_snapshot(stage_snapshot)
    control_arrays, resolved_control_snapshot = control_arrays_for_prepare(stage_snapshot, control_snapshot)
    if controller_state.get("stage") != stage:
        raise ValueError(f"snapshot stage {controller_state.get('stage')} does not match requested {stage}")
    frozen = dict(schema=SCHEMA, status="frozen_before_results", mode=mode, stage=stage, stop_after=stop_after, budget=int(budget),
                  source=dict(result={"path": str(q0_result.resolve()), "sha256": sha(q0_result)},
                              seal={"path": str(q0_seal.resolve()), "sha256": sha(q0_seal)},
                              q0_freeze={"path": str(q0_freeze.resolve()), "sha256": sha(q0_freeze)},
                              stage_snapshot={"path": str(stage_snapshot.resolve()), "sha256": sha(stage_snapshot)},
                              control_snapshot={"path": str(resolved_control_snapshot.resolve()), "sha256": sha(resolved_control_snapshot)}),
                  q0_freeze=q0, q0_controller=q0["controller"], p55_binding=q0["p55_binding"],
                  candidate_controller_source=candidate_source, candidate_controller_config=candidate_binding,
                  candidate_controller_file_bindings=candidate_nested, candidate_controller=candidate_controller,
                  initial_state_sha256=sha(stage_snapshot), initial_controller_state=controller_state,
                  restore_contract=dict(full_45d_state=True, single_restore=True, no_public_settling=True,
                                        closed_hold_warmup_steps=3, source="real upstream Q0 STAGE_SNAPSHOTS artifact",
                                        non45d_control_state="restore supervisor_ctrl, gripper.current_action, and OSC reset_goal from source snapshot"),
                  runtime_contract=dict(stage_qualification_only=True, suffix_qualification_only=mode == "suffix", full_task_success_evidence=False,
                                        no_demo_ideal_state=True, no_mid_episode_retry=True, p68_not_executed=True),
                  sources=source_bindings())
    dump(output / "STAGE_RESUME_FREEZE.json", frozen)
    print(json.dumps({"schema": SCHEMA, "freeze": str((output / "STAGE_RESUME_FREEZE.json").resolve()),
                      "freeze_sha256": sha(output / "STAGE_RESUME_FREEZE.json"), "stage": stage}, indent=2))
    return frozen


def validate_freeze(path: Path, approval: Path | None = None, check_files: bool = True) -> dict:
    frozen = read(path)
    if frozen.get("schema") != SCHEMA or frozen.get("status") != "frozen_before_results":
        raise ValueError("wrong P70 stage resume freeze")
    validate_scope(frozen.get("mode", "single"), frozen.get("stage"), int(frozen.get("budget", 0)), frozen.get("stop_after"))
    validate_q0_freeze(Path(frozen["source"]["q0_freeze"]["path"]), approval=None, check_files=check_files)
    load_snapshot(Path(frozen["source"]["stage_snapshot"]["path"]))
    if approval is not None:
        review = read(approval)
        if (review.get("schema") != APPROVAL_SCHEMA or review.get("freeze_sha256") != sha(path)
                or review.get("root_reviewed") is not True or review.get("physical_execution_authorized") is not True):
            raise ValueError("missing exact P70 stage resume Root review")
    if check_files:
        for group in frozen["source"].values():
            if sha(group["path"]) != group["sha256"]:
                raise ValueError("resume source binding changed: " + group["path"])
        cfg = frozen.get("candidate_controller_config")
        if cfg and sha(cfg["path"]) != cfg["sha256"]:
            raise ValueError("candidate controller config changed: " + cfg["path"])
        verify_bound_files(frozen.get("candidate_controller_file_bindings", {}))
        for p, digest in frozen["sources"].items():
            if sha(p) != digest:
                raise ValueError("resume source code changed: " + p)
    return frozen


def run_stage(freeze_path: Path, approval_path: Path, output: Path):
    frozen = validate_freeze(freeze_path, approval_path)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "public").mkdir(); (output / "private").mkdir(); (output / "controller").mkdir()
    dump(output / "BINDING.json", frozen)
    q0 = frozen["q0_freeze"]
    original = load_p53_full_binding(q0["p55_binding"]["path"])
    snapshot_path = Path(frozen["source"]["stage_snapshot"]["path"])
    control_snapshot_path = Path(frozen["source"].get("control_snapshot", frozen["source"]["stage_snapshot"])["path"])
    state, controller_state = load_snapshot(snapshot_path)
    snapshot_arrays = load_snapshot_arrays(control_snapshot_path)
    if not np.array_equal(np.asarray(load_snapshot_arrays(snapshot_path)["sim_state_flattened"]), np.asarray(snapshot_arrays["sim_state_flattened"])):
        raise ValueError("bound control snapshot no longer matches stage snapshot state")
    env = None; renderer = None
    actions = 0
    try:
        env = initialize_full_env(original, output, state)
        # initialize_full_env performs the single declared full-state restore.
        # Restore the sealed non-45D control state (OSC goal/gripper integrator)
        # from the same upstream snapshot, then read observation.  No object GT
        # is used to choose actions.
        restore_audit = restore_non45d_control_state(env, snapshot_arrays)
        obs = env._get_observations(force_update=True)
        names = _names(env)
        dump(output / "private_NAMES.json", names)
        dump(output / "RESTORE_AUDIT.json", restore_audit)
        child = load_p53_full_binding(q0["p55_binding"]["path"])
        child = json.loads(json.dumps(child)); child.setdefault("public", {})["resolution"] = 448
        renderer = PublicRenderer(env, child["public"])
        controller = set_controller_state(P70ObjectController(frozen["candidate_controller"]), controller_state, frozen["stage"])
        judge = build_stage_judge(env, names, child, frozen["candidate_controller"])
        hold = closed_hold_action()
        for warm in range(min(3, int(frozen["budget"]))):
            facts, diag = judge.observe(obs, controller.state.stage, actions)
            arrays, _ = snapshot(env, obs); arrays = {k: v for k, v in arrays.items() if k.startswith("supervisor_")}
            arrays["sim_state_flattened"] = flatten_state(env); arrays["action_next"] = hold.copy()
            private_path = output / "private" / f"warmup_{warm:04d}.npz"; np.savez_compressed(private_path, **arrays)
            _append(output / "controller/TRACE.jsonl", dict(t=actions, phase="closed_hold_warmup", private=f"private/{private_path.name}", private_sha256=sha(private_path), action=hold.tolist(), facts=facts, fact_diagnostics=diag, executed=True))
            obs, _, _, _ = env.step(hold.copy()); actions += 1
        stop = None; completed = False
        remaining = max(0, int(frozen["budget"]) - actions)
        for local_t in range(remaining):
            public_all = renderer.observe(obs); public = {k: public_all[k] for k in PUBLIC_KEYS}
            public_path = output / "public" / f"frame_{local_t:04d}.npz"; np.savez_compressed(public_path, **public)
            arrays, _ = snapshot(env, obs); arrays = {k: v for k, v in arrays.items() if k.startswith("supervisor_")}
            arrays["sim_state_flattened"] = flatten_state(env)
            facts, diag = judge.observe(obs, controller.state.stage, actions)
            action, telemetry = controller.command(public, facts)
            transition = telemetry.get("transition") or {}
            terminal = None
            if isinstance(transition, dict) and transition.get("stop"):
                terminal = str(transition["stop"])
            if facts.get("object_lost"):
                terminal = terminal or "object_lost"
            if facts.get("wrong_target"):
                terminal = terminal or "wrong_target"
            completed_stage = transition.get("from_stage") if isinstance(transition, dict) and transition.get("to_stage") else None
            completed = completed_stage == frozen["stop_after"]
            if completed:
                terminal = "completed_" + completed_stage
            executed = terminal is None
            arrays["action_next"] = action.copy() if executed else np.asarray([])
            private_path = output / "private" / f"frame_{local_t:04d}.npz"; np.savez_compressed(private_path, **arrays)
            _append(output / "controller/TRACE.jsonl", trace_record(actions, telemetry, phase="stage", local_t=local_t, public=f"public/{public_path.name}", public_sha256=sha(public_path), private=f"private/{private_path.name}", private_sha256=sha(private_path), facts=facts, fact_diagnostics=diag, executed=executed, terminal_stop=terminal))
            if terminal:
                stop = terminal; break
            obs, _, _, _ = env.step(action.copy()); actions += 1
        stop = stop or "stage_budget"
        arrays, _ = snapshot(env, obs); arrays = {k: v for k, v in arrays.items() if k.startswith("supervisor_")}
        arrays["sim_state_flattened"] = flatten_state(env)
        arrays["controller_state"] = np.asarray(json.dumps(controller.state.to_dict()), dtype=object)
        terminal_snapshot = output / "private" / "terminal_stage_snapshot.npz"
        np.savez_compressed(terminal_snapshot, **arrays)
        _seal(output, freeze_path, frozen["source"], actions)
        result = dict(schema=RESULT_SCHEMA, mode=frozen.get("mode", "single"), stage=frozen["stage"], stop_after=frozen["stop_after"], source=frozen["source"], actions=actions,
                      completed_stage=bool(completed), stop_reason=stop, stage_qualification_only=True,
                      suffix_qualification_only=bool(frozen.get("mode") == "suffix"), full_task_success_evidence=False, original_initial_restores=1, later_state_restores=0,
                      warmup_closed_hold_steps=3, restore_audit_sha256=sha(output / "RESTORE_AUDIT.json"), candidate_controller_source=frozen["candidate_controller_source"],
                      terminal_stage_snapshot={"path": "private/terminal_stage_snapshot.npz", "sha256": sha(terminal_snapshot)},
                      seal_sha256=sha(output / "SEAL.json"))
        dump(output / "RESULT.json", result)
        print(json.dumps({"schema": RESULT_SCHEMA, "stage": frozen["stage"], "completed_stage": completed, "stop_reason": stop, "seal_sha256": result["seal_sha256"]}, indent=2))
        return result
    except Exception as exc:
        dump(output / "ABORT.json", dict(error=type(exc).__name__, reason=str(exc), automatic_retry=False))
        raise
    finally:
        if renderer is not None: renderer.close()
        if env is not None: env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--q0-result", type=Path, required=True)
    p.add_argument("--stage-snapshot", type=Path, required=True)
    p.add_argument("--stage", choices=list(RESUMABLE_STAGES), required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--controller-config", type=Path, default=None)
    p.add_argument("--control-snapshot", type=Path, default=None)
    p.add_argument("--mode", choices=["single", "suffix"], default="single")
    p.add_argument("--stop-after", choices=list(SUFFIX_STAGES), default=None)
    p = sub.add_parser("run-stage")
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(output=args.output, q0_result=args.q0_result, stage_snapshot=args.stage_snapshot, stage=args.stage, budget=args.budget, controller_config=args.controller_config, mode=args.mode, stop_after=args.stop_after, control_snapshot=args.control_snapshot)
    else:
        run_stage(args.freeze, args.approval, args.output)


if __name__ == "__main__":
    main()
