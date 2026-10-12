#!/usr/bin/env python3
"""V24 shared autonomous cup entry; simulation and real use the same runtime."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import direct_white_entry as base
import all_code_stage_sim_entry as parent
from autonomous_live_view import LiveView
from cup_v25_support_stage_judge import GeminiStageJudge, MODEL_IDENTITY, validate_config
from prove.xtrainer.agent import digest, write
from prove.xtrainer.io import SafetyError


TASK = base.TASK
VARIANT = "cup_v25_combined_visible_support"
SOURCES = (
    HERE / "cup_v25_combined_support_runtime.py", HERE / "cup_v25_combined_support_entry.py",
    HERE / "cup_v25_combined_support_one_command.py",
    HERE / "cup_v25_visible_support_runtime.py", HERE / "cup_v25_visible_support_geometry.py",
    HERE / "cup_v25_support_stage_judge.py",
    HERE / "three_task_v25.py", HERE / "three_task_v25_public_io.py",
    REPO / "tests/test_three_task_v25_public_io.py", REPO / "tests/test_cup_v25_stage_judge.py",
    REPO / "tests/test_cup_v25_real_capability.py",
    REPO / "tests/test_cup_v25_visible_support.py",
    HERE / "cup_v25_real_capability_entry.py", HERE / "cup_v25_real_capability_one_command.py",
    HERE / "cup_v25_real_capability_sim_io.py",
    HERE / "cup_v25_compact_judge.py", HERE / "cup_v25_stage_judge.py",
    HERE / "cup_v25_entry.py", HERE / "cup_v25_one_command.py",
    HERE / "autonomous_sam_grounding.py", HERE / "autonomous_stage_runtime.py",
    HERE / "autonomous_live_view.py", HERE / "autonomous_stage_entry.py",
    REPO / "tests/test_autonomous_sam_grounding.py",
    REPO / "tests/test_autonomous_stage_runtime.py",
    HERE / "compact_stage_judge_v24.py", HERE / "gemini_stage_judge_v24.py",
    HERE / "compact_stage_judge_v24_transition.py", HERE / "gemini_stage_judge_v24_transition.py",
    HERE / "autonomous_stage_runtime_box_support.py",
    HERE / "autonomous_stage_runtime_white_move.py",
    HERE / "autonomous_stage_runtime_grasp_transition.py",
    HERE / "autonomous_stage_runtime_initial_part.py",
    HERE / "autonomous_stage_runtime_object_yaw.py",
    HERE / "compact_stage_judge_v24_grasp_transition.py", HERE / "gemini_stage_judge_v24_grasp_transition.py",
)


def runtime_hash():
    return hashlib.sha256((parent.runtime_hash() + "".join(digest(path) for path in SOURCES)).encode()).hexdigest()


def bindings(config_path, cfg, mode):
    if mode != "sim": raise SafetyError("autonomous simulator entry cannot connect hardware")
    runtime = cfg["runtime"]; validate_config(runtime["stage_judge"])
    if runtime["autonomous_grounding"]["maximum_calls"] != 2: raise SafetyError("initial grounding budget must be exactly two calls")
    vision = runtime["vision"]
    if vision.get("role_boxes"): raise SafetyError("autonomous grounding requires empty initial role boxes")
    if vision.get("autonomous_grounding_maximum_calls") != 2: raise SafetyError("vision grounding budget mismatch")
    if vision.get("autonomous_grounding_key_file") != runtime["stage_judge"]["key_file"]: raise SafetyError("grounding key binding mismatch")
    report, effective, bundle, bound = parent.bindings(config_path, cfg, mode)
    bound.update({str(path.resolve()): digest(path) for path in SOURCES})
    return report, effective, bundle, bound


def load(path, name):
    if Path(path).name == "sim_robot_io.py":
        return parent._original_load(HERE / "cup_v25_real_capability_sim_io.py", name)
    selected = HERE / "cup_v25_combined_support_runtime.py" if Path(path).name == "direct_white_runtime.py" else path
    return parent._original_load(selected, name)


def install(config):
    judge_cfg = config["runtime"]["stage_judge"]; validate_config(judge_cfg)
    def factory(*args, **kwargs):
        return GeminiStageJudge(*args, **kwargs, key_file=judge_cfg["key_file"], maximum_calls=judge_cfg["maximum_calls"])
    base.bindings = bindings; base.runtime_hash = runtime_hash; base.load = load; base.VARIANT = VARIANT; base.RealRootJudge = factory


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args(argv); cfg = json.loads(args.config.read_text()); install(cfg)
    runtime = cfg["runtime"]; vision = runtime["vision"]
    expected = {"prompt_frame_output": args.output / "INITIAL_RGB.png", "prompt_reply_path": args.output / "INITIAL_BOXES.json",
                "live_view_root": args.output / "LIVE", "autonomous_grounding_evidence_root": args.output / "AUTONOMOUS_GROUNDING"}
    for field, path in expected.items():
        if Path(vision.get(field, "")).resolve() != path.resolve(): raise SafetyError(f"fresh autonomous path mismatch: {field}")
    view = LiveView(vision["live_view_root"], host=runtime["live_view"]["host"], port=runtime["live_view"]["port"])
    url = view.start(); print(json.dumps({"live_view": url, "manual_input": False}), flush=True)
    try:
        result = base.main(argv)
        if isinstance(result, dict):
            result.update(stage_judge_backend="OpenRouter", stage_judge_identity="Gemini with explicit current Codex Root fallback",
                          initial_grounding_source="autonomous Gemini proposal validated by actual SAM+same-frame RGBD",
                          manual_grounding_requests=0, live_view_url=url,
                          actual_aperture_capability=None, capability_profile="current_real_SDK_no_measured_aperture",
                          simulated_mode_status_is_explicit_own_robot_proxy=True)
            write(args.output / "RESULT.json", result)
        return result
    finally: view.close()


if __name__ == "__main__": main()
