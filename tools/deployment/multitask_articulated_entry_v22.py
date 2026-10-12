#!/usr/bin/env python3
"""Check, freeze, or run one articulated V22 task from its own leaf."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prove.xtrainer.agent import digest, write
from prove.xtrainer.io import SafetyError, load_calibration
from prove.xtrainer.multitask_bundle import load_bundle
from multitask_articulated_v22 import ArticulatedExecutorV22, compile_route, load_leaf
from multitask_ordinary_route import role_contract
import direct_white_entry as backend
from macro_one_command import reserve


TASKS = ("open_drawer_and_place_object", "hanging_mug", "stack_cups")
SOURCES = (
    Path(__file__),
    HERE / "multitask_articulated_v22.py",
    HERE / "multitask_special_runtime.py",
    HERE / "multitask_special_judge.py",
    HERE / "compact_special_judge.py",
    HERE / "multitask_ordinary_runtime.py",
    HERE / "object_centered_runtime.py",
    REPO / "prove/xtrainer/multitask_bundle.py",
    REPO / "prove/xtrainer/runtime.py",
    REPO / "tests/test_multitask_articulated_v22.py",
)


def paths_for(task: str):
    if task not in TASKS:
        raise SafetyError("unsupported articulated V22 task")
    package = REPO / "task_packages/multitask_parallel_v22" / task
    owner = REPO / "evaluations/multitask_parallel_v22/articulated" / task
    return package, owner, package / "TASK_LEAF.json"


def runtime_hash():
    return hashlib.sha256("".join(digest(path) for path in SOURCES).encode()).hexdigest()


def build(task: str):
    package, owner, leaf_path = paths_for(task)
    bundle = load_bundle(task)
    leaf = load_leaf(leaf_path)
    graph = compile_route(bundle["graph"], leaf)
    roles = role_contract(graph, bundle["refs"])
    return package, owner, leaf_path, bundle, leaf, graph, roles


def check(task: str):
    package, owner, leaf_path, bundle, leaf, graph, roles = build(task)
    phases = [s["phase"] for s in graph["stages"]]
    return {
        "task": task,
        "status": "freeze_ready_not_executed",
        "leaf": str(leaf_path),
        "nodes": [{"node_id": s["stage_id"], "node_type": s["node_type"], "phase": s["phase"], "goal_relation": s["goal_relation"]} for s in graph["stages"]],
        "frozen_model_input_dim": bundle["skill"].model.input_dim,
        "neural_phases": [s["phase"] for s in graph["stages"] if s["controller"] == "neural_short"],
        "pull_insert_not_aliased_to_align": all(s["phase"] in ("Grasp", "Align", "Insert", "Pull") for s in graph["stages"] if s["controller"] == "neural_short"),
        "roles": roles,
        "semantic_calls": sum(s["phase"] in ("Grasp", "Complete") for s in graph["stages"]),
        "runtime_sha256": runtime_hash(),
        "device_connections": 0,
        "api_calls": 0,
        "episode_started": False,
        "scope": str(owner / "repair_v22_1/TRIAL_SCOPE.json"),
        "config_source": str(REPO / "evaluations/multitask_runtime_v20" / task / "CONFIG_PROPOSED.json"),
    }


def bindings(config_path, config, mode):
    task = config["task"]
    _, _, leaf_path, bundle, _, expected, _ = build(task)
    route = Path(config["special_route_graph"])
    actual = json.loads(route.read_text())
    if actual != expected:
        raise SafetyError("compiled task leaf route differs")
    bundle["graph"] = actual
    bundle["paths"]["graph"] = route
    runtime = copy.deepcopy(config["runtime"])
    if runtime.get("hardware_reference_speed_scale") != 1.0 or runtime.get("vision", {}).get("role_boxes"):
        raise SafetyError("fresh initial grounding and reference clock 1.0 required")
    from multitask_special_judge import validate_config
    validate_config(runtime["stage_judge"])
    required = sum(s["phase"] in ("Grasp", "Complete") for s in actual["stages"])
    if runtime["stage_judge"]["maximum_calls"] != required:
        raise SafetyError("exact task semantic call budget required")
    camera, right = load_calibration(runtime["left_calibration"], runtime["right_calibration"])
    runtime.update(T_left_from_camera=camera.tolist(), T_left_from_right=right.tolist())
    paths = list(SOURCES) + [leaf_path, config_path, route, Path(config["formal_run_reservation"]), Path(config["sim"]["physics_source"]), Path(runtime["sdk_file"]), Path(runtime["left_calibration"]), Path(runtime["right_calibration"]), Path(runtime["vision"]["sam_utility"]), Path(runtime["vision"]["sam_checkpoint"])]
    scene = Path(config["sim"]["scene_run"])
    for name in ("scene.xml", "reconstruction.json", "motor_targets.npz", "reference.npz", "dynamics.npz", "physics_validation.json"):
        paths.append(scene / name)
    catalog = REPO / "evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json"
    paths.append(catalog)
    assets = json.loads(catalog.read_text())
    for evaluator in assets.get("external_evaluator_sources", []):
        if digest(evaluator["path"]) != evaluator["sha256"]:
            raise SafetyError("postseal evaluator source changed")
        paths.append(Path(evaluator["path"]))
    for path, sha in assets["task_assets"][task]["external_asset_hashes"].items():
        if digest(path) != sha:
            raise SafetyError("task scene asset changed")
        paths.append(Path(path))
    bound = dict(bundle["bindings"])
    bound.update({str(Path(path).resolve()): digest(path) for path in paths})
    return {"dependencies": {}}, runtime, bundle, bound


def execute(task, config, scope, output, freeze):
    cfg = json.loads(config.read_text())
    if cfg.get("task") != task:
        raise SafetyError("configuration belongs to another task")
    original_load = backend.load
    def load(path, name):
        if Path(path).name == "direct_white_runtime.py":
            from multitask_ordinary_runtime import SharedSAMVision
            return SimpleNamespace(SharedCupExecutor=ArticulatedExecutorV22, SharedSAMVision=SharedSAMVision)
        return original_load(path, name)
    from multitask_special_judge import SpecialSemanticJudge
    def judge(*args, **kwargs):
        spec = cfg["runtime"]["stage_judge"]
        return SpecialSemanticJudge(*args, **kwargs, key_file=spec["key_file"], maximum_calls=spec["maximum_calls"])
    backend.TASK = task
    backend.VARIANT = "multitask_articulated_object_centered_v22"
    backend.bindings = bindings
    backend.runtime_hash = runtime_hash
    backend.load = load
    backend.RealRootJudge = judge
    return backend.main(["sim", "--config", str(config), "--scope", str(scope), "--execution-freeze", str(freeze), "--output", str(output)])


def real_interface(task, mode, config_path, scope_path, output):
    """Bind the reviewed camera/feedback/RPC interfaces; shadow never commands."""
    if mode not in ("shadow", "real"):
        raise SafetyError("real interface mode required")
    authority = json.loads(scope_path.read_text())
    if authority.get("task") != task:
        raise SafetyError("real scope belongs to another task")
    if mode == "shadow" and authority.get("read_only_shadow_authorized") is not True:
        raise SafetyError("read-only shadow is not authorized")
    if mode == "real":
        if authority.get("hardware_motion_commands_authorized") is not True:
            raise SafetyError("explicit task hardware motion authorization missing")
        receipt_binding = authority.get("passed_same_runtime_simulation")
        if not isinstance(receipt_binding, dict) or not receipt_binding.get("path") or not receipt_binding.get("sha256"):
            raise SafetyError("same-runtime full SAM simulation receipt required before real commands")
        receipt_path = Path(receipt_binding["path"])
        if digest(receipt_path) != receipt_binding["sha256"]:
            raise SafetyError("simulation receipt binding changed")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("task") != task or receipt.get("strict_success") is not True or receipt.get("shared_runtime_sha256") != runtime_hash() or receipt.get("mask_source") != "actual_SAM2":
            raise SafetyError("same-runtime full SAM simulation acceptance absent")
    cfg = copy.deepcopy(json.loads(config_path.read_text())["runtime"])
    camera_cal, right_cal = load_calibration(cfg["left_calibration"], cfg["right_calibration"])
    cfg.update(
        T_left_from_camera=camera_cal.tolist(),
        T_left_from_right=right_cal.tolist(),
        mode="execute",
    )
    _, _, leaf_path, bundle, _, graph, roles = build(task)
    bundle["graph"] = graph
    bundle["paths"]["graph"] = leaf_path
    cfg["known_initial_gripper_commands"] = bundle["initial_g"].tolist()
    cfg["vision"].update(role_boxes={}, prompt_frame_sha256=None, prompt_frame_output=str(output / "INITIAL_RGB.png"), prompt_reply_path=str(output / "INITIAL_BOXES.json"))
    if output.exists():
        raise SafetyError("fresh real/shadow output required")
    output.mkdir(parents=True)
    write(output / "EFFECTIVE_CONFIG.json", cfg)
    bound = {str(path.resolve()): digest(path) for path in SOURCES + (leaf_path, config_path, scope_path, output / "EFFECTIVE_CONFIG.json")}
    if mode == "real":
        receipt_path = Path(authority["passed_same_runtime_simulation"]["path"])
        bound[str(receipt_path.resolve())] = digest(receipt_path)
    freeze = {
        "schema": "graph-self-adapter.multitask-articulated-real-freeze.v22",
        "status": "result_free_frozen",
        "task": task,
        "mode": mode,
        "runtime_sha256": runtime_hash(),
        "bindings": [{"path": path, "sha256": sha} for path, sha in sorted(bound.items())],
        "operator_execution_intent": "explicit run.sh real invocation under the bound task authority",
        "initial_gripper_state_confirmed": cfg.get("initial_gripper_state_confirmed", False),
        "hardware_safety_reviewed_flag": cfg.get("hardware_safety_reviewed", False),
        "hardware_qualification_inferred": False,
    }
    write(output / "EXECUTION_FREEZE.json", freeze)
    if any(digest(path) != sha for path, sha in bound.items()):
        raise SafetyError("real interface binding changed before connection")

    from prove.xtrainer.io import NativeRPC, DashboardStop, SafetyLayer, feedback_fk_check
    from prove.xtrainer.perception import ZEDCamera
    from real_feedback_stream import FramedFeedbackStream
    from operator_launcher import startup_vision_class
    from multitask_ordinary_runtime import SharedSAMVision
    from multitask_special_judge import SpecialSemanticJudge
    import numpy as np

    streams = [FramedFeedbackStream(cfg["robot_ips"][arm], cfg["sdk_file"]) for arm in ("left", "right")]
    stops = [DashboardStop(cfg["robot_ips"][arm]) for arm in ("left", "right")] if mode == "real" else []
    rpc = NativeRPC()
    warm = startup_vision_class(SharedSAMVision, SafetyError)
    vision = warm(ZEDCamera(serial=cfg["vision"].get("serial", 38968158)), np.asarray(cfg["T_left_from_camera"]), cfg["vision"], evidence_root=output / "SAM_MASK_EVIDENCE")
    class Robot:
        @property
        def commands(self):
            return rpc.commands
        def samples(self):
            return [stream.snapshot() for stream in streams]
        def fk(self, *args):
            return rpc.fk(*args)
        def command(self, *args):
            if mode != "real":
                raise SafetyError("read-only shadow cannot command")
            return rpc.command(*args)
    try:
        for stream in streams:
            stream.connect()
        for stop in stops:
            stop.connect()
        rpc.connect()
        samples = [stream.snapshot() for stream in streams]
        SafetyLayer(cfg["safety"]).samples(samples)
        comparison = feedback_fk_check(rpc, samples, cfg["known_initial_gripper_commands"])
        if not comparison["passed"]:
            raise SafetyError("actual native FK/feedback inconsistency")
        vision.connect()
        if mode == "shadow":
            observation = vision.capture()
            samples = Robot().samples()
            SafetyLayer(cfg["safety"]).samples(samples)
            result = {
                "task": task,
                "mode": "shadow",
                "hardware_commands": 0,
                "runtime_sha256": runtime_hash(),
                "required_roles": roles,
                "roles": {key: None if value is None else np.asarray(value).tolist() for key, value in observation.roles.items()},
                "actual_aperture_m": [sample.aperture for sample in samples],
                "native_FK_feedback": comparison,
                "hardware_qualified": False,
            }
        else:
            spec = cfg["stage_judge"]
            judge = SpecialSemanticJudge(output / "REAL_AGENT_HANDOFFS", task=task, graph_sha=digest(leaf_path), identity=cfg["root_identity"], key_file=spec["key_file"], maximum_calls=spec["maximum_calls"])
            result = ArticulatedExecutorV22(bundle, Robot(), vision, judge, stops, cfg, output).run()
            result.update(runtime_sha256=runtime_hash(), native_FK_feedback=comparison, hardware_qualified=False, physical_qualification_inferred=False)
        write(output / "RESULT.json", result)
        return result
    finally:
        rpc.close()
        vision.close()
        for stream in streams:
            stream.close()
        for stop in stops:
            stop.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("mode", nargs="?", default="check", choices=("check", "prepare", "sim", "shadow", "real"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    ready = check(args.task)
    if args.mode == "check":
        print(json.dumps(ready, indent=2))
        return ready
    package, owner, leaf_path, _, _, graph, _ = build(args.task)
    source_config = args.config or REPO / "evaluations/multitask_runtime_v20" / args.task / "CONFIG_PROPOSED.json"
    if args.mode in ("shadow", "real"):
        real_scope = args.scope or owner / "REAL_SCOPE.json"
        output = args.output or owner / "real_runs" / (time.strftime("%Y%m%d_%H%M%S") + "_" + args.mode)
        return real_interface(args.task, args.mode, source_config, real_scope, output)
    scope = args.scope or owner / "repair_v22_1/TRIAL_SCOPE.json"
    authority = json.loads(scope.read_text())
    if authority.get("task") != args.task or authority.get("execution_authorized") is not True:
        raise SafetyError("exact task simulation authority required")
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    campaign = scope.parent
    output = campaign / "runs" / stamp
    control = campaign / "control" / stamp
    if output.exists() or control.exists():
        raise SafetyError("fresh task output required")
    control.mkdir(parents=True)
    route = control / "TASK_GRAPH.json"
    write(route, graph)
    claim = reserve(scope, campaign / "FORMAL_RUN_LEDGER.json", output, control)
    reservation = control / "FORMAL_RUN_RESERVATION.json"
    write(reservation, claim)
    cfg = copy.deepcopy(json.loads(source_config.read_text()))
    cfg.update(special_route_graph=str(route), formal_run_reservation=str(reservation))
    cfg["runtime"]["vision"].update(prompt_frame_output=str(output / "INITIAL_RGB.png"), prompt_reply_path=str(output / "INITIAL_BOXES.json"), prompt_frame_sha256=None, role_boxes={})
    config = control / "CONFIG.json"
    write(config, cfg)
    _, _, _, bound = bindings(config, cfg, "sim")
    for path in (source_config, scope, package / "run.sh"):
        bound[str(path.resolve())] = digest(path)
    frozen = {
        "schema": "graph-self-adapter.shared-cup-freeze.v1",
        "status": "result_free_frozen",
        "task": args.task,
        "scope_sha256": digest(scope),
        "runtime_sha256": runtime_hash(),
        "bindings": [{"path": path, "sha256": sha} for path, sha in sorted(bound.items())],
        "jobs": [{"task": args.task, "mode": "sim", "output": str(output)}],
        "formal_reservation": str(reservation),
        "single_native_initial_restore": True,
        "mid_episode_restore_teacher_playback": False,
    }
    freeze = control / "EXECUTION_FREEZE.json"
    write(freeze, frozen)
    print(json.dumps({"task": args.task, "mode": args.mode, "output": str(output), "freeze": str(freeze), "runtime_sha256": runtime_hash()}, indent=2), flush=True)
    if args.mode == "prepare":
        return frozen
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return execute(args.task, config, scope, output, freeze)


if __name__ == "__main__":
    main()
