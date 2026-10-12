#!/usr/bin/env python3
"""Scope-frozen ALL_CODE_STAGE simulator entry with manual first-frame SAM."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import direct_white_entry as base
from gemini_stage_judge import GeminiStageJudge, MODEL_IDENTITY, validate_config
from prove.xtrainer.agent import digest, write
from prove.xtrainer.io import SafetyError
from all_code_stage_route import validate_route


TASK = base.TASK
VARIANT = "all_code_stage_trajectory_v01_gemini_judge"
_original_bindings = base.bindings
_original_hash = base.runtime_hash
_original_load = base.load
SOURCES = (
    HERE / "all_code_stage_trajectory.py",
    HERE / "all_code_stage_runtime.py",
    HERE / "all_code_stage_route.py",
    HERE / "all_code_stage_sim_entry.py",
    HERE / "manual_sam_first_frame.py",
    HERE / "wait_for_manual_sam.py",
    HERE / "preview_saved_sam_masks.py",
    REPO / "tests/test_all_code_stage_runtime.py",
    REPO / "tests/test_manual_sam_first_frame.py",
)


def runtime_hash():
    return hashlib.sha256((_original_hash() + "".join(digest(path) for path in SOURCES)).encode()).hexdigest()


def bindings(config_path, cfg, mode):
    if mode != "sim":
        raise SafetyError("ALL_CODE_STAGE simulator entry cannot connect hardware")
    validate_config(cfg["runtime"]["stage_judge"])
    report, effective, bundle, bound = _original_bindings(config_path, cfg, mode)
    effective["all_code_stage_route"] = validate_route(bundle["graph"])
    bound.update({str(path.resolve()): digest(path) for path in SOURCES})
    return report, effective, bundle, bound


def load(path, name):
    selected = HERE / "all_code_stage_runtime.py" if Path(path).name == "direct_white_runtime.py" else path
    return _original_load(selected, name)


def install(config):
    judge_config = config["runtime"]["stage_judge"]
    validate_config(judge_config)
    def factory(*args, **kwargs):
        return GeminiStageJudge(*args, **kwargs, key_file=judge_config["key_file"],
                                maximum_calls=judge_config["maximum_calls"])
    base.bindings = bindings
    base.runtime_hash = runtime_hash
    base.load = load
    base.VARIANT = VARIANT
    base.RealRootJudge = factory


def manual_grounding_interface(output):
    output = Path(output).resolve()
    return {
        "frame": str(output / "INITIAL_RGB.png"),
        "reply": str(output / "INITIAL_BOXES.json"),
        "launcher": [sys.executable, str(HERE / "wait_for_manual_sam.py"),
                     str(output / "INITIAL_RGB.png"), "--output", str(output / "INITIAL_BOXES.json")],
        "cancel_or_timeout": "fail_closed; no motion may begin without a valid exact-frame reply",
    }


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args, _ = parser.parse_known_args(argv)
    config = json.loads(args.config.read_text())
    install(config)
    vision = config["runtime"]["vision"]
    if vision.get("role_boxes"):
        raise SafetyError("manual current first-frame SAM requires empty role_boxes")
    if args.output:
        expected_frame = str((args.output / "INITIAL_RGB.png").resolve())
        expected_reply = str((args.output / "INITIAL_BOXES.json").resolve())
        if str(Path(vision.get("prompt_frame_output", "")).resolve()) != expected_frame or str(Path(vision.get("prompt_reply_path", "")).resolve()) != expected_reply:
            raise SafetyError("manual SAM paths must be this fresh run's INITIAL_RGB/INITIAL_BOXES")
    result = base.main(argv)
    if isinstance(result, dict):
        result.update(stage_judge_backend="OpenRouter", stage_judge_identity=MODEL_IDENTITY,
                      initial_grounding_source="operator manual boxes/foreground seed on exact current INITIAL_RGB",
                      manual_grounding_interface=manual_grounding_interface(args.output) if args.output else None)
        if args.output and args.output.exists():
            write(args.output / "RESULT.json", result)
    return result


if __name__ == "__main__":
    main()
