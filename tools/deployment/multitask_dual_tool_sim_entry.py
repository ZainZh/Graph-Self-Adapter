#!/usr/bin/env python3
"""Portable check/shadow/sim/real entry for the two active V22 graphs."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prove.xtrainer.agent import digest
from prove.xtrainer.io import SafetyError, load_calibration
from multitask_dual_tool_route import check_task, load_task_bundle, route_task
import direct_white_entry as backend

RUNNABLE = ("sorting_cube_bimanual", "sweep_objects")
SOURCES = (
    Path(__file__), HERE / "multitask_dual_tool_route.py", HERE / "multitask_dual_tool_runtime.py",
    HERE / "multitask_dual_tool_one_command.py", HERE / "multitask_dual_tool_shadow.py",
    HERE / "multitask_special_route.py", HERE / "multitask_readonly_preflight.py",
    HERE / "multitask_special_runtime.py", HERE / "multitask_ordinary_runtime.py",
    HERE / "multitask_special_judge.py", HERE / "compact_special_judge.py",
    HERE / "object_centered_runtime.py", HERE / "sim_robot_io.py", HERE / "direct_white_entry.py",
    REPO / "prove/xtrainer/multitask_bundle.py", REPO / "prove/xtrainer/runtime.py",
    REPO / "tests/test_multitask_dual_tool_v22.py",
    REPO / "evaluations/multitask_parallel_v22/dual_tool/TASK_ASSET_INVENTORY.json",
)


def runtime_hash():
    return hashlib.sha256("".join(digest(path) for path in SOURCES).encode()).hexdigest()


def bindings(config_path, config, mode):
    task = config["task"]
    if task not in RUNNABLE or mode not in ("sim", "shadow", "real"):
        raise SafetyError("task/mode is outside the active V22 portable leaf")
    ready = check_task(task)
    bundle = load_task_bundle(task)
    route_path = Path(config["dual_tool_route_graph"])
    actual = json.loads(route_path.read_text())
    if actual != route_task(task, bundle["graph"]):
        raise SafetyError("exact task-local route differs")
    bundle["graph"] = actual
    bundle["paths"]["graph"] = route_path
    runtime = copy.deepcopy(config["runtime"])
    if runtime.get("hardware_reference_speed_scale") != 1.0 or runtime.get("vision", {}).get("role_boxes"):
        raise SafetyError("fresh initial grounding and reference clock 1.0 required")
    from multitask_special_judge import validate_config
    validate_config(runtime["stage_judge"])
    required = sum(s["phase"] in ("Grasp", "Complete") for s in actual["stages"])
    if runtime["stage_judge"]["maximum_calls"] != required:
        raise SafetyError("exact declared task semantic budget required")
    camera, right = load_calibration(runtime["left_calibration"], runtime["right_calibration"])
    runtime.update(T_left_from_camera=camera.tolist(), T_left_from_right=right.tolist())
    paths = list(SOURCES) + [config_path, route_path, Path(runtime["sdk_file"]),
        Path(runtime["left_calibration"]), Path(runtime["right_calibration"]),
        Path(runtime["vision"]["sam_utility"]), Path(runtime["vision"]["sam_checkpoint"])]
    if mode == "sim":
        paths.extend((Path(config["formal_run_reservation"]), Path(config["sim"]["physics_source"])))
        scene = Path(config["sim"]["scene_run"])
        for name in ("scene.xml", "reconstruction.json", "motor_targets.npz", "reference.npz", "dynamics.npz", "physics_validation.json"):
            paths.append(scene / name)
        generation = "v02" if task == "sorting_cube_bimanual" else "v01"
        catalog = Path(config.get("external_asset_catalog", REPO / f"evaluations/twin_rsi_{generation}/EXTERNAL_ASSET_BINDINGS.json"))
        paths.append(catalog)
        asset_catalog = json.loads(catalog.read_text())
        for path, sha in asset_catalog["task_assets"][task]["external_asset_hashes"].items():
            if digest(path) != sha:
                raise SafetyError("task scene asset changed")
            paths.append(Path(path))
        for evaluator in asset_catalog.get("external_evaluator_sources", []):
            if digest(evaluator["path"]) != evaluator["sha256"]:
                raise SafetyError("postseal evaluator changed")
            paths.append(Path(evaluator["path"]))
    bound = dict(bundle["bindings"])
    bound.update({str(Path(path).resolve()): digest(path) for path in paths})
    return {"dependencies": {}, "ready": ready}, runtime, bundle, bound


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=RUNNABLE)
    parser.add_argument("mode", nargs="?", default="check", choices=("check", "shadow", "sim", "real"))
    for name in ("config", "scope", "execution-freeze", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    if args.mode == "check":
        report = check_task(args.task)
        report["shared_runtime_sha256"] = runtime_hash()
        print(json.dumps(report, indent=2))
        return report
    if not args.config:
        raise SafetyError("explicit task portable config required")
    cfg = json.loads(args.config.read_text())
    if cfg.get("task") != args.task:
        raise SafetyError("configuration belongs to another task")
    if args.mode == "shadow":
        if not args.scope or not args.output:
            raise SafetyError("read-only shadow requires explicit task scope and fresh output")
        scope = json.loads(args.scope.read_text())
        if scope.get("task") != args.task or scope.get("read_only_shadow_authorized") is not True:
            raise SafetyError("exact task read-only shadow scope is not authorized")
        report, runtime_cfg, bundle, bound = bindings(args.config, cfg, "shadow")
        runtime_cfg["vision"].update(role_boxes={}, prompt_frame_sha256=None,
            prompt_frame_output=str(args.output / "INITIAL_RGB.png"),
            prompt_reply_path=str(args.output / "INITIAL_BOXES.json"))
        bound[str(args.scope.resolve())] = digest(args.scope)
        freeze = {"schema":"graph-self-adapter.multitask-readonly-freeze.v1", "status":"result_free_frozen",
            "task":args.task, "mode":"shadow", "shared_runtime_sha256":runtime_hash(),
            "bindings":[{"path":path,"sha256":sha} for path,sha in sorted(bound.items())],
            "job":{"task":args.task,"mode":"shadow","output":str(args.output.resolve())},
            "hardware_commands":0, "api_calls":0}
        from multitask_dual_tool_shadow import run_readonly_shadow
        return run_readonly_shadow(task=args.task, runtime=runtime_cfg,
            required_roles=report["ready"]["roles"]["SAM_roles"], output=args.output, freeze=freeze)
    if not all((args.scope, args.output)):
        raise SafetyError("explicit task scope and fresh output required")
    original_load = backend.load
    def load(path, name):
        if Path(path).name == "direct_white_runtime.py":
            from multitask_dual_tool_runtime import DualToolExecutor, SharedSAMVision
            return SimpleNamespace(SharedCupExecutor=DualToolExecutor, SharedSAMVision=SharedSAMVision)
        return original_load(path, name)
    from multitask_special_judge import SpecialSemanticJudge
    def judge(*judge_args, **judge_kwargs):
        return SpecialSemanticJudge(*judge_args, **judge_kwargs,
            key_file=cfg["runtime"]["stage_judge"]["key_file"],
            maximum_calls=cfg["runtime"]["stage_judge"]["maximum_calls"])
    backend.TASK = args.task
    backend.VARIANT = "multitask_dual_tool_object_centered_v22"
    backend.bindings = bindings
    backend.runtime_hash = runtime_hash
    backend.load = load
    backend.RealRootJudge = judge
    if args.mode == "sim":
        if not args.execution_freeze:
            raise SafetyError("result-free simulator freeze required")
        return backend.main(["sim", "--config", str(args.config), "--scope", str(args.scope),
            "--execution-freeze", str(args.execution_freeze), "--output", str(args.output)])
    scope = json.loads(args.scope.read_text())
    if scope.get("task") != args.task or scope.get("hardware_motion_commands_authorized") is not True:
        raise SafetyError("exact task hardware scope is not authorized")
    gate = scope.get("passed_simulation_report")
    if not gate or digest(gate["path"]) != gate["sha256"]:
        raise SafetyError("same-runtime full-task simulation receipt required")
    promoted = json.loads(Path(gate["path"]).read_text())
    if promoted.get("strict_success") is not True or promoted.get("shared_runtime_sha256") != runtime_hash() or promoted.get("task") != args.task:
        raise SafetyError("task-specific same-runtime promotion absent")
    if args.output.exists():
        raise SafetyError("fresh real output required")
    _, runtime_cfg, bundle, bound = bindings(args.config, cfg, "real")
    runtime_cfg["mode"] = "execute"
    runtime_cfg["vision"].update(role_boxes={}, prompt_frame_sha256=None,
        prompt_frame_output=str(args.output / "INITIAL_RGB.png"),
        prompt_reply_path=str(args.output / "INITIAL_BOXES.json"))
    bound[str(args.scope.resolve())] = digest(args.scope)
    bound[str(Path(gate["path"]).resolve())] = gate["sha256"]
    args.output.mkdir(parents=True)
    from prove.xtrainer.agent import write
    write(args.output / "EXECUTION_FREEZE.json", {
        "schema":"graph-self-adapter.multitask-real-freeze.v1", "status":"result_free_frozen",
        "task":args.task, "mode":"real", "shared_runtime_sha256":runtime_hash(),
        "bindings":[{"path":path,"sha256":sha} for path,sha in sorted(bound.items())],
        "job":{"task":args.task,"mode":"real","output":str(args.output.resolve())},
        "operator_motion_declaration":"explicit run.sh real plus existing task hardware scope",
        "legacy_hardware_safety_reviewed_flag_required":False,
    })
    from prove.xtrainer.io import NativeRPC, DashboardStop, SafetyLayer, feedback_fk_check
    from prove.xtrainer.perception import ZEDCamera
    framed = original_load(HERE / "real_feedback_stream.py", "_dual_tool_feedback")
    startup = original_load(HERE / "operator_launcher.py", "_dual_tool_startup")
    from multitask_dual_tool_runtime import DualToolExecutor, SharedSAMVision
    streams = [framed.FramedFeedbackStream(runtime_cfg["robot_ips"][arm], runtime_cfg["sdk_file"]) for arm in ("left", "right")]
    stops = [DashboardStop(runtime_cfg["robot_ips"][arm]) for arm in ("left", "right")]
    rpc = NativeRPC()
    import numpy as np
    vision = None
    try:
        for stream in streams: stream.connect()
        for stop in stops: stop.connect()
        rpc.connect()
        samples = [stream.snapshot() for stream in streams]
        SafetyLayer(runtime_cfg["safety"]).samples(samples)
        comparison = feedback_fk_check(rpc, samples, runtime_cfg["known_initial_gripper_commands"])
        if not comparison["passed"]:
            raise SafetyError("actual native TCP/FK consistency failure")
        vision_type = startup.startup_vision_class(SharedSAMVision, SafetyError)
        vision = vision_type(ZEDCamera(), np.asarray(runtime_cfg["T_left_from_camera"]), runtime_cfg["vision"], evidence_root=args.output / "SAM_MASK_EVIDENCE")
        vision.connect()
        class RealRobot:
            @property
            def commands(self): return rpc.commands
            def samples(self): return [stream.snapshot() for stream in streams]
            def fk(self, *values): return rpc.fk(*values)
            def command(self, *values): return rpc.command(*values)
        stage_judge = judge(args.output / "REAL_AGENT_HANDOFFS", task=args.task,
            graph_sha=digest(bundle["paths"]["graph"]), identity=runtime_cfg["root_identity"])
        executor = DualToolExecutor(bundle, RealRobot(), vision, stage_judge, stops, runtime_cfg, args.output)
        vision.stage_context = lambda: executor.stage
        result = executor.run()
        result.update(shared_runtime_sha256=runtime_hash(), mask_source="actual_SAM2",
            task_specific_graph_model_control=True, native_feedback_FK_check=comparison,
            Gemini_boundary_switches_do_not_override_local_holding=True)
        write(args.output / "RESULT.json", result)
        return result
    finally:
        rpc.close()
        if vision: vision.close()
        for stream in streams: stream.close()
        for stop in stops: stop.close()


if __name__ == "__main__":
    main()
