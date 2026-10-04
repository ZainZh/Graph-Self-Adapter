"""One-shot P59 B2 constant-safe sidecar and offline evaluation pipeline.

This pipeline is namespaced away from the sealed B1 artifacts.  It waits for
the unique B2 main checkpoint, trains the preregistered FullHistory sidecar,
and runs the existing offline evaluators.  The normalization rule is a
scientific preprocessing revision, not an execution-only or mathematically
equivalent replay of the entire B1 experiment.  No stage performs physics and
no failed stage is retried.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.task_graph import p59_post_training_pipeline as shared


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "evaluations/fast_adapter_p59_control_latent_v01"
DATASET = STUDY / "data_v01/TRAINING_DATASET.pt"
MAIN = STUDY / "training_constant_safe_v01"
P50 = ROOT / "evaluations/fast_adapter_p50_precision_policy_v01/offline_v01/precision_action.pt"
SIDECAR = STUDY / "full_history_constant_safe_v01"
MAIN_EVAL = STUDY / "offline_evaluation_constant_safe_v01"
SIDECAR_EVAL = STUDY / "full_history_evaluation_constant_safe_v01"
PIPELINE = STUDY / "post_training_constant_safe_pipeline_v01"
AMENDMENT = STUDY / "AMENDMENT_B2_CONSTANT_NORMALIZATION.md"
DATASET_SHA256 = shared.DATASET_SHA256
P50_SHA256 = shared.P50_SHA256
DEVICE = "cuda:0"
NORMALIZATION_RULE = "scale_i = 1 if train_std_i <= 1e-4 else train_std_i"


def run_stage(name: str, arguments: list[str], events: Path) -> None:
    log = PIPELINE / f"{name}.log"
    shared.append_event(events, {
        "event": "stage_start", "stage": name, "time": time.time(),
        "arguments": arguments, "b2_constant_safe": True,
    })
    with log.open("xb") as stream:
        result = subprocess.run(arguments, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    shared.append_event(events, {
        "event": "stage_exit", "stage": name, "time": time.time(),
        "returncode": result.returncode, "log_sha256": shared.sha(log),
    })
    if result.returncode:
        raise RuntimeError(f"stage failed without retry: {name}; see {log}")


def wait_for_main(timeout_seconds: int, poll_seconds: int, events: Path) -> Path:
    started = time.monotonic()
    seal, abort = MAIN / "TRAINING_SEAL.json", MAIN / "ABORT.json"
    shared.append_event(events, {
        "event": "wait_constant_safe_main", "timeout_seconds": timeout_seconds,
        "poll_seconds": poll_seconds, "time": time.time(),
    })
    while time.monotonic() - started <= timeout_seconds:
        if abort.exists():
            raise RuntimeError("constant-safe main training wrote ABORT; sidecar/evaluation stopped")
        if seal.exists():
            return seal
        time.sleep(poll_seconds)
    raise TimeoutError("constant-safe main training did not seal within the three-hour bound")


def verify_main_freeze(path: Path) -> dict:
    """Accept the established schema while enforcing B2's immutable assets."""
    freeze = json.loads(path.read_text())
    if freeze.get("schema") != "prove.p59-training-freeze.v1":
        raise ValueError("unknown constant-safe main training freeze schema")
    if freeze.get("dataset_sha256") != DATASET_SHA256:
        raise ValueError("constant-safe main freeze binds a different dataset")
    if freeze.get("p50_checkpoint_sha256") != P50_SHA256:
        raise ValueError("constant-safe main freeze binds a different P50 checkpoint")
    return freeze


def pipeline(timeout_seconds: int, poll_seconds: int) -> dict:
    PIPELINE.mkdir(parents=True, exist_ok=True)
    start_path = PIPELINE / "PIPELINE_START.json"
    events = PIPELINE / "PIPELINE_EVENTS.jsonl"
    if any(path.exists() for path in
           (start_path, events, PIPELINE / "PIPELINE_SEAL.json", PIPELINE / "PIPELINE_ABORT.json")):
        raise FileExistsError("constant-safe post-training pipeline already started; retry is forbidden")
    if shared.sha(DATASET) != DATASET_SHA256 or shared.sha(P50) != P50_SHA256:
        raise ValueError("frozen dataset or P50 hash differs before constant-safe pipeline start")
    if not AMENDMENT.is_file():
        raise FileNotFoundError("B2 normalization amendment is missing")
    shared.dump_new(start_path, {
        "schema": "prove.p59-post-training-pipeline-start.v1",
        "purpose": "B2 single-factor constant-safe normalization follow-up",
        "source_sha256": shared.sha(__file__),
        "shared_execution_helpers_sha256": shared.sha(shared.__file__),
        "amendment_sha256": shared.sha(AMENDMENT),
        "dataset_sha256": shared.sha(DATASET), "p50_sha256": shared.sha(P50),
        "normalization_rule": NORMALIZATION_RULE,
        "normalization_revision": "B2",
        "execution_only": False,
        "scientific_method_changed": True,
        "mathematically_equivalent_to_b1_entire_experiment": False,
        "wait_timeout_seconds": timeout_seconds, "poll_seconds": poll_seconds,
        "device": DEVICE, "cuda_visible_devices_modified": False,
        "automatic_retries": 0, "physics_rollouts": 0, "time": time.time(),
    })
    events.touch(exist_ok=False)
    try:
        main_seal = wait_for_main(timeout_seconds, poll_seconds, events)
        main_checkpoint = MAIN / "p59_model.pt"
        main_freeze = MAIN / "TRAINING_FREEZE.json"
        shared.verify_seal(main_seal, main_checkpoint, "prove.p59-training-seal.v1")
        verify_main_freeze(main_freeze)

        SIDECAR.mkdir(parents=True, exist_ok=False)
        common_train = [
            "--dataset", str(DATASET), "--main-checkpoint", str(main_checkpoint),
            "--main-seal", str(main_seal), "--p50-checkpoint", str(P50),
            "--output", str(SIDECAR),
        ]
        run_stage("full_history_prepare", [
            sys.executable, "-m", "prove.task_graph.p59.full_history_train", "prepare",
            *common_train], events)
        run_stage("full_history_train", [
            sys.executable, "-m", "prove.task_graph.p59.full_history_train", "train",
            *common_train, "--device", DEVICE], events)
        sidecar_checkpoint = SIDECAR / "full_history.pt"
        sidecar_seal = SIDECAR / "FULL_HISTORY_SEAL.json"
        sidecar_freeze = SIDECAR / "FULL_HISTORY_FREEZE.json"
        shared.verify_seal(sidecar_seal, sidecar_checkpoint,
                           "prove.p59-full-history-seal.v1")

        MAIN_EVAL.mkdir(parents=True, exist_ok=False)
        run_stage("main_offline_evaluate", [
            sys.executable, "-m", "prove.task_graph.p59.evaluate",
            "--dataset", str(DATASET), "--checkpoint", str(main_checkpoint),
            "--training-seal", str(main_seal), "--training-freeze", str(main_freeze),
            "--p50-checkpoint", str(P50),
            "--output", str(MAIN_EVAL / "EVALUATION.json"), "--device", DEVICE], events)

        SIDECAR_EVAL.mkdir(parents=True, exist_ok=False)
        common_eval = [
            "--dataset", str(DATASET), "--main-checkpoint", str(main_checkpoint),
            "--main-seal", str(main_seal), "--main-training-freeze", str(main_freeze),
            "--p50-checkpoint", str(P50), "--sidecar-checkpoint", str(sidecar_checkpoint),
            "--sidecar-seal", str(sidecar_seal),
            "--sidecar-training-freeze", str(sidecar_freeze),
            "--output", str(SIDECAR_EVAL),
        ]
        run_stage("full_history_evaluate_prepare", [
            sys.executable, "-m", "prove.task_graph.p59.full_history_evaluate", "prepare",
            *common_eval], events)
        run_stage("full_history_evaluate", [
            sys.executable, "-m", "prove.task_graph.p59.full_history_evaluate", "evaluate",
            *common_eval, "--device", DEVICE], events)

        main_result = MAIN_EVAL / "EVALUATION.json"
        sidecar_result = SIDECAR_EVAL / "FULL_HISTORY_EVALUATION.json"
        seal = {
            "schema": "prove.p59-post-training-pipeline-seal.v1",
            "normalization_rule": NORMALIZATION_RULE,
            "normalization_revision": "B2",
            "execution_only": False,
            "scientific_method_changed": True,
            "mathematically_equivalent_to_b1_entire_experiment": False,
            "amendment_sha256": shared.sha(AMENDMENT),
            "dataset_sha256": shared.sha(DATASET), "p50_sha256": shared.sha(P50),
            "main_training_freeze_sha256": shared.sha(main_freeze),
            "main_training_seal_sha256": shared.sha(main_seal),
            "main_checkpoint_sha256": shared.sha(main_checkpoint),
            "sidecar_training_freeze_sha256": shared.sha(sidecar_freeze),
            "sidecar_training_seal_sha256": shared.sha(sidecar_seal),
            "sidecar_checkpoint_sha256": shared.sha(sidecar_checkpoint),
            "main_evaluation_sha256": shared.sha(main_result),
            "sidecar_evaluation_sha256": shared.sha(sidecar_result),
            "events_sha256": shared.sha(events), "automatic_retries": 0,
            "physics_rollouts": 0, "completed_time": time.time(),
        }
        shared.dump_new(PIPELINE / "PIPELINE_SEAL.json", seal)
        return seal
    except Exception as error:
        abort = {
            "schema": "prove.p59-post-training-pipeline-abort.v1",
            "normalization_revision": "B2", "execution_only": False,
            "error_type": type(error).__name__, "reason": str(error),
            "automatic_retry": False, "physics_rollouts": 0, "time": time.time(),
        }
        abort_path = PIPELINE / "PIPELINE_ABORT.json"
        if not abort_path.exists():
            shared.dump_new(abort_path, abort)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or not 1 <= args.poll_seconds <= 60:
        raise ValueError("invalid bounded wait settings")
    pipeline(args.timeout_seconds, args.poll_seconds)


if __name__ == "__main__":
    main()
