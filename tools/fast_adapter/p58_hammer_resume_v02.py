#!/usr/bin/env python3
"""One-time P58 V2 engineering continuation after the old BC loader failed.

Only the 21 previously unstarted non-BC cells are scheduled. No model,
condition, criterion or already completed cell is changed.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.fast_adapter import p58_hammer_runner_v02 as runner


INDICES = (8,9,10,11,12,13,15,16,17,18,19,20,21,23,24,25,26,27,28,29,31)


def continue_non_bc(freeze, review, previous_queue, output, python):
    frozen = runner.verify(freeze)
    runner.verify_review(freeze, review)
    prior = runner.base.read(previous_queue)
    if (prior.get("schema") != "p58.hammer-development-queue-result.v2"
            or prior.get("freeze_sha256") != runner.base.sha(freeze)
            or prior.get("launched") != 8
            or [row.get("index") for row in prior["results"]] != list(range(8))
            or [row.get("result", {}).get("status") for row in prior["results"][:6]] != ["completed"]*6
            or prior["results"][6].get("result", {}).get("status") != "engineering_error"
            or prior["results"][6].get("result", {}).get("executed_task_steps") != 0
            or prior["results"][7].get("result", {}).get("status") != "completed"):
        raise ValueError("Unexpected original eight-cell engineering stop receipt")
    if tuple(i for i,c in enumerate(frozen["cells"]) if i >= 8 and c["arm"] != "p30v2_visual_bc") != INDICES:
        raise ValueError("Unstarted non-BC index set changed")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runner.base.write(output/"RESUME_PLAN.json", dict(schema="p58.hammer-development-nonbc-resume.v1",
        freeze_sha256=runner.base.sha(freeze), review_sha256=runner.base.sha(review),
        previous_queue=dict(path=str(previous_queue), sha256=runner.base.sha(previous_queue)),
        indices=list(INDICES), workers=2, automatic_retries=0,
        original_index6_engineering_error_retained=True))
    rows = []
    for start in range(0, len(INDICES), 2):
        processes = []
        for index in INDICES[start:start+2]:
            command = [str(python), "-m", "tools.fast_adapter.p58_hammer_runner_v02", "run",
                "--freeze", str(freeze), "--review", str(review),
                "--index", str(index), "--output", str(output/f"cell_{index:02d}")]
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            environment.setdefault("MUJOCO_GL", "egl")
            environment.setdefault("PYOPENGL_PLATFORM", "egl")
            environment["PYTHONPATH"] = os.pathsep.join((
                str(runner.ROOT/"data/runtime/fast-adapter-p28-deps"), str(runner.ROOT),
                environment.get("PYTHONPATH", "")))
            log = (output/f"cell_{index:02d}.log").open("x")
            process = subprocess.Popen(command, cwd=runner.ROOT, env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((index, command, process, log, time.monotonic()))
        for index, command, process, log, begun in processes:
            try:
                code = process.wait(timeout=max(.001, 1800.-(time.monotonic()-begun)))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = -9
            finally:
                log.close()
            directory = output/f"cell_{index:02d}"
            result = runner.base.read(directory/"RESULT.json") if (directory/"RESULT.json").exists() else None
            rows.append(dict(index=index, command=command, returncode=code, result=result,
                log_sha256=runner.base.sha(output/f"cell_{index:02d}.log"),
                seal_sha256=runner.base.sha(directory/"SEAL.json") if (directory/"SEAL.json").exists() else None))
        if any(row["returncode"] or row["result"] is None
               or row["result"].get("status") == "engineering_error" for row in rows[-len(processes):]):
            break
    runner.base.write(output/"RESUME_RESULT.json", dict(schema="p58.hammer-development-nonbc-resume-result.v1",
        freeze_sha256=runner.base.sha(freeze), launched=len(rows), indices=[r["index"] for r in rows],
        stopped_on_engineering_error=len(rows) < len(INDICES)
        or any(r["returncode"] or r["result"] is None or r["result"].get("status") == "engineering_error" for r in rows),
        rows=rows))
    return rows


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--previous-queue", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    args=parser.parse_args()
    rows=continue_non_bc(args.freeze,args.review,args.previous_queue,args.output,args.python)
    print(dict(launched=len(rows), indices=[r["index"] for r in rows]), flush=True)
