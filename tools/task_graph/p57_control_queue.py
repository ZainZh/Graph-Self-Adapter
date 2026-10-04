#!/usr/bin/env python3
"""Finite four-GPU queue for the frozen P57 B 20-cell development comparison.

Prepare only writes a result-free command manifest. Run executes exactly once
per cell, in five bounded batches, with no retries or replacement conditions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from tools.task_graph import p57_evaluate_geometry as evaluation

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "p57.geometry-control-queue.v1"
GPUS = (0, 1, 2, 3)
TIMEOUT = 1800


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare(binding_path, runtime_python):
    binding_path = Path(binding_path).resolve()
    binding = json.loads(binding_path.read_text())
    evaluation.verify(binding)
    python = Path(runtime_python).absolute()  # Preserve venv symlinks.
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("P57 runtime Python is missing or not executable")
    jobs = [dict(index=cell["index"], entry=cell["entry"], arm=cell["arm"],
                 gpu=cell["index"] % len(GPUS), timeout_seconds=TIMEOUT)
            for cell in binding["cells"]]
    if len(jobs) != 20:
        raise ValueError("P57 B requires exactly 20 frozen cells")
    return dict(schema=SCHEMA, binding=dict(path=str(binding_path), sha256=sha(binding_path)),
                runtime_python=str(python), queue_source_sha256=sha(__file__),
                evaluator_source_sha256=sha(evaluation.__file__),
                cwd=str(ROOT), gpus=list(GPUS), jobs=jobs,
                batches=[list(range(start, start+len(GPUS))) for start in range(0,20,len(GPUS))],
                automatic_retries=0, stop_after_engineering_error=True,
                command_template=[str(python), "-m", "tools.task_graph.p57_evaluate_geometry",
                                  "run", "--binding", str(binding_path), "--index", "<job.index>",
                                  "--output", "<job.episode>"])


def verify(plan):
    if (plan.get("schema") != SCHEMA or plan.get("gpus") != list(GPUS) or
            plan.get("automatic_retries") != 0 or plan.get("stop_after_engineering_error") is not True or
            plan.get("batches") != [list(range(start,start+4)) for start in range(0,20,4)]):
        raise ValueError("Changed P57 finite queue contract")
    if sha(__file__) != plan.get("queue_source_sha256") or sha(evaluation.__file__) != plan.get("evaluator_source_sha256"):
        raise ValueError("P57 queue/evaluator source changed after freeze")
    if sha(plan["binding"]["path"]) != plan["binding"]["sha256"]:
        raise ValueError("P57 evaluation binding changed after queue freeze")
    binding = json.loads(Path(plan["binding"]["path"]).read_text())
    evaluation.verify(binding)
    jobs = plan.get("jobs")
    if (len(jobs) != 20 or [j["index"] for j in jobs] != list(range(20)) or
            any(j != dict(index=c["index"], entry=c["entry"], arm=c["arm"],
                          gpu=c["index"] % 4, timeout_seconds=TIMEOUT)
                for j,c in zip(jobs,binding["cells"]))):
        raise ValueError("P57 queue jobs differ from frozen 20 cells")
    python = Path(plan["runtime_python"])
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Frozen runtime Python unavailable")
    if plan["cwd"] != str(ROOT):
        raise ValueError("P57 queue repository root differs")
    return jobs


def _one(job, plan, output, manifest_sha):
    folder = output/f"cell_{job['index']:03d}"
    folder.mkdir()
    episode = folder/"episode"
    argv = [plan["runtime_python"], "-m", "tools.task_graph.p57_evaluate_geometry",
            "run", "--binding", plan["binding"]["path"], "--index", str(job["index"]),
            "--output", str(episode)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(job["gpu"]), MUJOCO_EGL_DEVICE_ID=str(job["gpu"]),
        MUJOCO_GL="egl", LIBERO_CONFIG_PATH=str(episode/"libero_config"),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    row = dict(job=job, argv=argv, cwd=plan["cwd"], physical_gpu=job["gpu"],
               libero_config_path=env["LIBERO_CONFIG_PATH"], manifest_sha256=manifest_sha,
               attempt=1, automatic_retry=False, started_unix=time.time())
    write(folder/"ATTEMPT.json", row)
    with (folder/"process.log").open("x") as log:
        try:
            process = subprocess.Popen(argv, cwd=plan["cwd"], env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                row.update(returncode=process.wait(timeout=TIMEOUT), execution_status="finished")
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                row.update(returncode=None, execution_status="engineering_timeout")
        except OSError as error:
            row.update(returncode=None, execution_status="launch_error", error=str(error))
    row["finished_unix"] = time.time()
    row["log_sha256"] = sha(folder/"process.log")
    row["artifacts"] = {name: sha(episode/name) for name in ("RESULT.json", "SEAL.json", "ABORT.json")
                        if (episode/name).is_file()}
    row["engineering_ok"] = row["execution_status"] == "finished" and row["returncode"] == 0 and (
        "RESULT.json" in row["artifacts"] and "SEAL.json" in row["artifacts"])
    write(folder/"RECEIPT.json", row)
    return row


def run(manifest_path, expected_sha, output):
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    if sha(manifest_path) != expected_sha:
        raise ValueError("P57 queue manifest SHA differs")
    plan = json.loads(manifest_path.read_text())
    jobs = verify(plan)
    output.mkdir(parents=True, exist_ok=False)
    write(output/"QUEUE.json", dict(manifest=plan, manifest_sha256=expected_sha,
                                    queue_source_sha256=sha(__file__)))
    receipts = []
    for batch_number, batch in enumerate(plan["batches"]):
        if sha(manifest_path) != expected_sha:
            raise ValueError("P57 queue manifest changed between batches")
        verify(plan)
        with ThreadPoolExecutor(max_workers=4) as pool:
            current = list(pool.map(lambda index: _one(jobs[index], plan, output, expected_sha), batch))
        receipts.extend(current)
        batch_result = dict(batch=batch_number, indices=batch,
                            engineering_ok=all(row["engineering_ok"] for row in current),
                            receipts=[dict(index=row["job"]["index"],
                                           receipt_sha256=sha(output/f"cell_{row['job']['index']:03d}"/"RECEIPT.json"))
                                      for row in current])
        write(output/f"BATCH_{batch_number:03d}.json", batch_result)
        if not batch_result["engineering_ok"]:
            break
    summary = dict(schema="p57.geometry-control-queue-result.v1",
                   planned_jobs=20, attempted_jobs=len(receipts),
                   completed_engineering_ok=sum(row["engineering_ok"] for row in receipts),
                   all_batches_completed=len(receipts)==20,
                   automatic_retries=0, scientific_success_not_inferred=True,
                   receipt_sha256=[sha(output/f"cell_{row['job']['index']:03d}"/"RECEIPT.json")
                                   for row in receipts])
    write(output/"QUEUE_RESULT.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Write result-free explicit 20-cell queue")
    prep.add_argument("--binding", type=Path, required=True)
    prep.add_argument("--runtime-python", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    launch = commands.add_parser("run", help="Run five finite four-GPU batches")
    launch.add_argument("--manifest", type=Path, required=True)
    launch.add_argument("--manifest-sha256", required=True)
    launch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        write(args.output, prepare(args.binding, args.runtime_python))
    else:
        print(json.dumps(run(args.manifest, args.manifest_sha256, args.output)))


if __name__ == "__main__":
    main()
