"""One-shot P59 post-training sidecar and offline evaluation pipeline.

This command waits for the unique repaired main run, stops on its ABORT, then
trains the preregistered FullHistory sidecar and evaluates both sealed models.
It performs no physics and never retries a failed stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "evaluations/fast_adapter_p59_control_latent_v01"
DATASET = STUDY / "data_v01/TRAINING_DATASET.pt"
MAIN = STUDY / "training_v01_device_repair"
P50 = ROOT / "evaluations/fast_adapter_p50_precision_policy_v01/offline_v01/precision_action.pt"
SIDECAR = STUDY / "full_history_v01"
MAIN_EVAL = STUDY / "offline_evaluation_v01"
SIDECAR_EVAL = STUDY / "full_history_evaluation_v01"
PIPELINE = STUDY / "post_training_pipeline_v01"
DATASET_SHA256 = "a101b767aa4f99214e2fae6ffe6dc0d272a0fa15eaac284c9b658bbb8e7b01c1"
P50_SHA256 = "96fdcf6a8ed49e53615bdbff8d3f9cf69ffed85353c18f0584b31e024b4b4fc7"


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def append_event(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def verify_seal(seal_path: Path, checkpoint: Path, schema: str) -> dict:
    seal = json.loads(seal_path.read_text())
    if seal.get("schema") != schema or seal.get("checkpoint_sha256") != sha(checkpoint):
        raise ValueError(f"invalid seal/checkpoint binding: {seal_path}")
    return seal


def run_stage(name: str, arguments: list[str], events: Path) -> None:
    log = PIPELINE / f"{name}.log"
    append_event(events, {"event": "stage_start", "stage": name, "time": time.time(),
                          "arguments": arguments})
    with log.open("xb") as stream:
        result = subprocess.run(arguments, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    append_event(events, {"event": "stage_exit", "stage": name,
                          "time": time.time(), "returncode": result.returncode,
                          "log_sha256": sha(log)})
    if result.returncode:
        raise RuntimeError(f"stage failed without retry: {name}; see {log}")


def wait_for_main(timeout_seconds: int, poll_seconds: int, events: Path) -> Path:
    started = time.monotonic()
    seal, abort = MAIN / "TRAINING_SEAL.json", MAIN / "ABORT.json"
    append_event(events, {"event": "wait_main", "timeout_seconds": timeout_seconds,
                          "poll_seconds": poll_seconds, "time": time.time()})
    while time.monotonic() - started <= timeout_seconds:
        if abort.exists():
            raise RuntimeError("main repaired training wrote ABORT; sidecar/evaluation stopped")
        if seal.exists():
            return seal
        time.sleep(poll_seconds)
    raise TimeoutError("main repaired training did not seal within the three-hour bound")


def pipeline(timeout_seconds: int, poll_seconds: int) -> dict:
    PIPELINE.mkdir(parents=True, exist_ok=True)
    start_path = PIPELINE / "PIPELINE_START.json"
    events = PIPELINE / "PIPELINE_EVENTS.jsonl"
    if any(path.exists() for path in
           (start_path, events, PIPELINE / "PIPELINE_SEAL.json", PIPELINE / "PIPELINE_ABORT.json")):
        raise FileExistsError("post-training pipeline already started; automatic retry is forbidden")
    if sha(DATASET) != DATASET_SHA256 or sha(P50) != P50_SHA256:
        raise ValueError("frozen dataset or P50 hash differs before pipeline start")
    dump_new(start_path, {
        "schema": "prove.p59-post-training-pipeline-start.v1",
        "source_sha256": sha(__file__), "dataset_sha256": sha(DATASET),
        "p50_sha256": sha(P50), "wait_timeout_seconds": timeout_seconds,
        "poll_seconds": poll_seconds, "device": "cuda:2",
        "cuda_visible_devices_modified": False, "automatic_retries": 0,
        "physics_rollouts": 0, "time": time.time(),
    })
    events.touch(exist_ok=False)
    try:
        main_seal = wait_for_main(timeout_seconds, poll_seconds, events)
        main_checkpoint = MAIN / "p59_model.pt"
        main_freeze = MAIN / "TRAINING_FREEZE.json"
        verify_seal(main_seal, main_checkpoint, "prove.p59-training-seal.v1")
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
            *common_train, "--device", "cuda:2"], events)
        sidecar_checkpoint = SIDECAR / "full_history.pt"
        sidecar_seal = SIDECAR / "FULL_HISTORY_SEAL.json"
        sidecar_freeze = SIDECAR / "FULL_HISTORY_FREEZE.json"
        verify_seal(sidecar_seal, sidecar_checkpoint, "prove.p59-full-history-seal.v1")

        # Developer material is first opened only after both final checkpoints seal.
        MAIN_EVAL.mkdir(parents=True, exist_ok=False)
        run_stage("main_offline_evaluate", [
            sys.executable, "-m", "prove.task_graph.p59.evaluate",
            "--dataset", str(DATASET), "--checkpoint", str(main_checkpoint),
            "--training-seal", str(main_seal), "--training-freeze", str(main_freeze),
            "--p50-checkpoint", str(P50),
            "--output", str(MAIN_EVAL / "EVALUATION.json"), "--device", "cuda:2"], events)

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
            *common_eval, "--device", "cuda:2"], events)

        main_result = MAIN_EVAL / "EVALUATION.json"
        sidecar_result = SIDECAR_EVAL / "FULL_HISTORY_EVALUATION.json"
        seal = {
            "schema": "prove.p59-post-training-pipeline-seal.v1",
            "main_training_seal_sha256": sha(main_seal),
            "main_checkpoint_sha256": sha(main_checkpoint),
            "sidecar_training_seal_sha256": sha(sidecar_seal),
            "sidecar_checkpoint_sha256": sha(sidecar_checkpoint),
            "main_evaluation_sha256": sha(main_result),
            "sidecar_evaluation_sha256": sha(sidecar_result),
            "events_sha256": sha(events), "automatic_retries": 0,
            "physics_rollouts": 0, "completed_time": time.time(),
        }
        dump_new(PIPELINE / "PIPELINE_SEAL.json", seal)
        return seal
    except Exception as error:
        abort = {"schema": "prove.p59-post-training-pipeline-abort.v1",
                 "error_type": type(error).__name__, "reason": str(error),
                 "automatic_retry": False, "physics_rollouts": 0, "time": time.time()}
        abort_path = PIPELINE / "PIPELINE_ABORT.json"
        if not abort_path.exists():
            dump_new(abort_path, abort)
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
