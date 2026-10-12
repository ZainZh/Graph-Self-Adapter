#!/usr/bin/env python3
"""Manual-GUI ALL_CODE_STAGE operator trial; check is device/API free by default.

This unqualified engineering deployment is allowed only by the explicit user
instruction to deploy and run the current two-cup hardware trial without the
cancelled same-runtime simulation. It is not a scientific promotion.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import promoted_gemini_real_entry as promoted
import all_code_stage_sim_entry as candidate
from all_code_stage_route import validate_route
from prove.xtrainer.agent import digest, write
from prove.xtrainer.io import SafetyError
from wait_for_manual_sam import wait_and_annotate


SOURCES = (
    HERE / "all_code_stage_route.py", HERE / "all_code_stage_trajectory.py",
    HERE / "all_code_stage_runtime.py", HERE / "all_code_stage_sim_entry.py",
    HERE / "all_code_stage_real_entry.py", HERE / "manual_sam_first_frame.py",
    HERE / "all_code_stage_operator_entry.py",
    HERE / "wait_for_manual_sam.py", HERE / "preview_saved_sam_masks.py",
    REPO / "tests/test_all_code_stage_runtime.py",
    REPO / "tests/test_manual_sam_first_frame.py",
)


def runtime_hash():
    return candidate.runtime_hash()


def prepare(package_root, config_path, receipt_sha):
    report, cfg, bundle = promoted.prepare(package_root, config_path, receipt_sha)
    route = validate_route(bundle["graph"])
    return report, cfg, bundle, route


def _manual_gate(work, frame, reply, timeout, stops):
    box = {}
    def target():
        try:
            box["result"] = work()
        except BaseException as error:
            box["error"] = error
    worker = threading.Thread(target=target, name="all-code-stage-runtime", daemon=True)
    worker.start()
    try:
        payload = wait_and_annotate(frame, output_path=reply, timeout_s=timeout)
    except BaseException:
        for stop in stops:
            try:
                stop.stop()
            except Exception:
                pass
        raise
    while worker.is_alive():
        worker.join(.25)
    if "error" in box:
        raise box["error"]
    return box["result"], payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "shadow", "execute"), nargs="?", default="check")
    parser.add_argument("--package", type=Path, default=HERE.parents[2])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-initial-grippers-open", action="store_true")
    args = parser.parse_args(argv)
    root = args.package.resolve()
    config_path = args.config or root / "local/CONFIG_REAL.json"
    scope_path = args.scope or root / "local/SCOPE_REAL.json"
    approved = args.receipt_sha256 or (root / "local/ROOT_RECEIPT_SHA256.txt").read_text().strip()
    report, cfg, bundle, route = prepare(root, config_path, approved)
    shared_sha = runtime_hash()
    source_bindings = {str(path.resolve()): digest(path) for path in SOURCES}
    if args.mode == "check":
        result = {**report, "mode": "check", "package_inputs_verified": True,
                  "candidate_runtime_sha256": shared_sha, "route": route,
                  "manual_first_frame": {"frame": "RUN/INITIAL_RGB.png", "reply": "RUN/INITIAL_BOXES.json"},
                  "device_connections": 0, "hardware_commands": 0,
                  "candidate_source_bindings": source_bindings}
        print(json.dumps(result, indent=2)); return result
    authority = json.loads(scope_path.read_text())
    if authority.get("task") != "put_mug_on_box" or authority.get("candidate_runtime_sha256") != shared_sha:
        raise SafetyError("independent scope does not bind this ALL_CODE_STAGE runtime")
    authority_file = root / authority.get("deployment_authority_file", "")
    if not authority_file.is_file() or digest(authority_file) != authority.get("deployment_authority_sha256"):
        raise SafetyError("explicit deployment authority binding absent")
    deployment_authority = json.loads(authority_file.read_text())
    if deployment_authority.get("user_instruction") != "别跑模拟了，你直接部署" or deployment_authority.get("automatic_hardware_motion_during_deployment") is not False:
        raise SafetyError("direct-deploy authority content mismatch")
    if args.mode == "shadow" and authority.get("read_only_shadow_authorized") is not True:
        raise SafetyError("read-only shadow not authorized")
    if args.mode == "execute":
        if authority.get("hardware_motion_commands_authorized") is not True or authority.get("hardware_session_authorized") is not True:
            raise SafetyError("hardware motion not authorized")
        if authority.get("same_runtime_full_simulation_passed") is not False or authority.get("user_explicit_skip_simulation") is not True:
            raise SafetyError("operator scope must explicitly record absent same-runtime simulation pass")
        if authority.get("qualification") != "unvalidated_operator_engineering_trial":
            raise SafetyError("unvalidated operator-trial qualification absent")
        if authority.get("explicit_user_direct_deploy_authorized") is not True or authority.get("authorization_source") != "user: 别跑模拟了，你直接部署; existing current two-cup whole-task motion authorization retained":
            raise SafetyError("explicit direct-deploy operator authority absent")
        if not args.confirm_initial_grippers_open:
            raise SafetyError("operator must explicitly confirm both initial grippers are open")
    output = args.output or root / "runs" / (time.strftime("%Y%m%d_%H%M%S") + "_" + args.mode)
    if output.exists():
        raise SafetyError("fresh real output required")
    output.mkdir(parents=True)
    cfg = copy.deepcopy(cfg); cfg["mode"] = "execute"
    cfg["initial_gripper_state_confirmed"] = bool(args.confirm_initial_grippers_open)
    frame = output / "INITIAL_RGB.png"; reply = output / "INITIAL_BOXES.json"
    cfg["vision"].update(role_boxes={}, prompt_frame_sha256=None,
                         prompt_frame_output=str(frame), prompt_reply_path=str(reply))
    write(output / "EFFECTIVE_CONFIG.json", cfg)
    receipt = json.loads((root / "PROMOTION_RECEIPT.json").read_text())
    bound = {str(root / row["relative_path"]): row["sha256"] for row in receipt["files"]}
    bound.update(source_bindings)
    bound.update({str(config_path.resolve()): digest(config_path), str(scope_path.resolve()): digest(scope_path),
                  str(output / "EFFECTIVE_CONFIG.json"): digest(output / "EFFECTIVE_CONFIG.json")})
    freeze = {"schema": "graph-self-adapter.all-code-stage-operator-freeze.v1", "status": "result_free_frozen",
              "candidate_runtime_sha256": shared_sha, "base_receipt_sha256": approved,
              "bindings": [{"path": path, "sha256": sha} for path, sha in sorted(bound.items())],
              "job": {"mode": args.mode, "task": "put_mug_on_box", "output": str(output)},
              "manual_first_frame_required": True, "same_runtime_full_simulation_passed": False,
              "explicit_user_direct_deploy_authorized": True,
              "qualification": "unvalidated_operator_engineering_trial",
              "user_explicit_skip_simulation": True, "physical_qualification_inferred": False}
    write(output / "EXECUTION_FREEZE.json", freeze)
    if any(digest(path) != sha for path, sha in bound.items()):
        raise SafetyError("real prepared binding changed")
    print(json.dumps({"output": str(output), "mode": args.mode,
                      "candidate_runtime_sha256": shared_sha,
                      "manual_frame": str(frame), "manual_reply": str(reply)}, indent=2), flush=True)

    from prove.xtrainer.io import NativeRPC, DashboardStop, SafetyLayer, feedback_fk_check
    from prove.xtrainer.perception import ZEDCamera
    from real_feedback_stream import FramedFeedbackStream
    from operator_launcher import startup_vision_class
    from gemini_stage_judge import GeminiStageJudge
    import all_code_stage_runtime as runtime
    import numpy as np

    streams = [FramedFeedbackStream(cfg["robot_ips"][arm], cfg["sdk_file"]) for arm in ("left", "right")]
    stops = [DashboardStop(cfg["robot_ips"][arm]) for arm in ("left", "right")] if args.mode == "execute" else []
    rpc = NativeRPC(); vision = None
    warm = startup_vision_class(runtime.SharedSAMVision, SafetyError)
    vision = warm(ZEDCamera(serial=cfg["vision"].get("serial", 38968158)),
                  np.asarray(cfg["T_left_from_camera"]), cfg["vision"],
                  evidence_root=output / "SAM_MASK_EVIDENCE")
    class Robot:
        @property
        def commands(self): return rpc.commands
        def samples(self): return [stream.snapshot() for stream in streams]
        def fk(self, *values): return rpc.fk(*values)
        def command(self, *values):
            if args.mode != "execute": raise SafetyError("read-only shadow cannot command")
            return rpc.command(*values)
    try:
        for stream in streams: stream.connect()
        for stop in stops: stop.connect()
        rpc.connect(); samples = [stream.snapshot() for stream in streams]
        SafetyLayer(cfg["safety"]).samples(samples)
        comparison = None
        if args.mode == "execute":
            comparison = feedback_fk_check(rpc, samples, cfg["known_initial_gripper_commands"])
            if not comparison["passed"]: raise SafetyError("actual native FK/feedback inconsistency")
        vision.connect()
        def work():
            if args.mode == "shadow":
                from gemini_readonly_preflight import preflight
                return preflight(Robot(), vision, SafetyLayer(cfg["safety"]), cfg["known_initial_gripper_commands"])
            judge = GeminiStageJudge(output / "REAL_AGENT_HANDOFFS", task="put_mug_on_box",
                                     graph_sha=digest(root / "data/TASK_GRAPH.json"),
                                     key_file=cfg["stage_judge"]["key_file"],
                                     maximum_calls=cfg["stage_judge"]["maximum_calls"])
            value = runtime.SharedCupExecutor(bundle, Robot(), vision, judge, stops, cfg, output).run()
            value.update(stage_judge_backend="OpenRouter", stage_judge_identity=judge.identity,
                         stage_judge_API_records=judge.api_records)
            return value
        result, grounding = _manual_gate(work, frame, reply,
                                         cfg["vision"].get("prompt_wall_timeout_sec", 300), stops)
        result.update(execution_variant="all_code_stage_trajectory_v01_gemini_judge",
                      shared_runtime_sha256=shared_sha, manual_grounding_frame_sha256=grounding["frame_sha256"],
                      native_FK_feedback=comparison, simulation_reexecuted=False,
                      same_runtime_full_simulation_passed=False,
                      qualification="unvalidated_operator_engineering_trial",
                      user_explicit_skip_simulation=True,
                      operator_engineering_trial=True, physical_qualification_inferred=False)
        write(output / "RESULT.json", result); return result
    except BaseException as error:
        for stop in stops:
            try: stop.stop()
            except Exception: pass
        write(output / "PREFLIGHT_FAILURE.json", {"mode": args.mode, "failure": str(error),
              "hardware_commands": rpc.commands, "manual_grounding_completed": reply.exists(),
              "physical_qualification_inferred": False})
        raise
    finally:
        rpc.close()
        if vision: vision.close()
        for stream in streams: stream.close()
        for stop in stops: stop.close()


if __name__ == "__main__":
    main()
