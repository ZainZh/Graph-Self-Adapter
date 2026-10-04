#!/usr/bin/env python3
"""One explicitly authorized BC dependency repair: indices 6,14,22,30 only."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.fast_adapter import p58_hammer_runner_v02 as runner


INDICES = (6, 14, 22, 30)
OVERLAY = runner.ROOT/"data/runtime/p58-vision-deps"


def run(freeze, review, original_queue, nonbc_result, visual_receipt, output, python):
    frozen = runner.verify(freeze)
    runner.verify_review(freeze, review)
    prior = runner.base.read(original_queue)
    continued = runner.base.read(nonbc_result)
    visual = runner.base.read(visual_receipt)
    if (prior.get("schema") != "p58.hammer-development-queue-result.v2"
            or prior.get("freeze_sha256") != runner.base.sha(freeze)
            or prior.get("launched") != 8
            or prior["results"][6].get("result", {}).get("status") != "engineering_error"
            or prior["results"][6].get("result", {}).get("executed_task_steps") != 0):
        raise ValueError("Original index6 no-action library error changed")
    if (continued.get("schema") != "p58.hammer-development-nonbc-resume-result.v1"
            or continued.get("freeze_sha256") != runner.base.sha(freeze)
            or continued.get("launched") != 21
            or continued.get("stopped_on_engineering_error") is not False
            or len(continued.get("indices", [])) != 21):
        raise ValueError("Non-BC continuation incomplete")
    if (visual.get("schema") != "p58.hammer-historical-visual-environment.v1"
            or visual.get("physics_steps") != 0
            or visual.get("feature_shape") != [1,142]
            or visual.get("feature_finite") is not True
            or visual.get("model_loaded_on_cuda") is not True
            or visual.get("overlay") != "data/runtime/p58-vision-deps"
            or visual.get("transformers") != "4.56.2"
            or visual.get("tokenizers") != "0.22.1"
            or visual.get("dino_sha256") != frozen["historical_visual"]["files"]["model.safetensors"]
            or visual.get("pca_sha256") != frozen["models"]["p30v2_visual_pca"]["sha256"]):
        raise ValueError("Historical BC compatibility receipt mismatch")
    if [frozen["cells"][i]["arm"] for i in INDICES] != ["p30v2_visual_bc"]*4:
        raise ValueError("Repair list contains a non-BC arm")
    if Path(python).resolve() != Path(visual["interpreter"]).resolve():
        raise ValueError("BC repair must preserve original simulator interpreter")
    if not OVERLAY.is_dir():
        raise FileNotFoundError(OVERLAY)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    runner.base.write(output/"BC_REPAIR_PLAN.json", dict(schema="p58.hammer-bc-engineering-repair.v1",
        freeze_sha256=runner.base.sha(freeze), review_sha256=runner.base.sha(review),
        original_queue_sha256=runner.base.sha(original_queue),
        nonbc_resume_sha256=runner.base.sha(nonbc_result),
        visual_env_receipt_sha256=runner.base.sha(visual_receipt),
        indices=list(INDICES), index6_attempt="one explicit engineering retry after zero actions",
        indices_14_22_30="first attempts", retained_original_index6_error=True,
        unchanged_physics=True, unchanged_candidate=True, automatic_retries=0,
        original_simulator_interpreter=str(python), overlay=str(OVERLAY)))
    rows=[]
    for start in range(0,4,2):
        launched=[]
        for index in INDICES[start:start+2]:
            command=[str(python),"-m","tools.fast_adapter.p58_hammer_runner_v02","run",
                "--freeze",str(freeze),"--review",str(review),"--index",str(index),
                "--output",str(output/f"cell_{index:02d}")]
            environment=os.environ.copy()
            environment.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
                               PYTHONDONTWRITEBYTECODE="1")
            environment.setdefault("MUJOCO_GL","egl")
            environment.setdefault("PYOPENGL_PLATFORM","egl")
            environment["PYTHONPATH"]=os.pathsep.join((str(OVERLAY),
                str(runner.ROOT/"data/runtime/fast-adapter-p28-deps"),str(runner.ROOT),
                environment.get("PYTHONPATH","")))
            log=(output/f"cell_{index:02d}.log").open("x")
            process=subprocess.Popen(command,cwd=runner.ROOT,env=environment,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            launched.append((index,command,process,log,time.monotonic()))
        for index,command,process,log,begun in launched:
            try:
                code=process.wait(timeout=max(.001,1800.-(time.monotonic()-begun)))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait();code=-9
            finally:log.close()
            directory=output/f"cell_{index:02d}"
            result=runner.base.read(directory/"RESULT.json") if (directory/"RESULT.json").exists() else None
            rows.append(dict(index=index,attempt=2 if index==6 else 1,command=command,
                returncode=code,result=result,log_sha256=runner.base.sha(output/f"cell_{index:02d}.log"),
                seal_sha256=runner.base.sha(directory/"SEAL.json") if (directory/"SEAL.json").exists() else None))
        if any(row["returncode"] or row["result"] is None
               or row["result"].get("status")=="engineering_error" for row in rows[-len(launched):]):
            break
    runner.base.write(output/"BC_REPAIR_RESULT.json",dict(schema="p58.hammer-bc-engineering-repair-result.v1",
        freeze_sha256=runner.base.sha(freeze),launched=len(rows),indices=[row["index"] for row in rows],
        stopped_on_engineering_error=len(rows)<4 or any(row["returncode"] or row["result"] is None
            or row["result"].get("status")=="engineering_error" for row in rows),rows=rows))
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("freeze","review","original-queue","nonbc-result","visual-receipt","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--python",default=sys.executable)
    a=parser.parse_args()
    rows=run(a.freeze,a.review,a.original_queue,a.nonbc_result,a.visual_receipt,a.output,a.python)
    print(dict(launched=len(rows),indices=[r["index"] for r in rows]),flush=True)
