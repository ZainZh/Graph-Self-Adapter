"""Durable three-GPU batch launcher, no model/agent calls during execution."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT=Path('evaluations/fast_adapter_p37_full_success_v01')


def main():
    logdir=ROOT/'logs';logdir.mkdir(parents=True,exist_ok=True)
    started=ROOT/'STARTED.json'
    if started.exists():raise RuntimeError('Existing launch record: inspect before any restart')
    children=[];info=[];logs=[]
    for gpu in (0,1,2):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',
            PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
        f=(logdir/f'worker{gpu}.log').open('x');logs.append(f)
        p=subprocess.Popen([sys.executable,'-u','-m','tools.fast_adapter.p37_success',
            '--worker',str(gpu),'--dino','assets/dinov3'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);info.append(dict(gpu=gpu,pid=p.pid,log=str(logdir/f'worker{gpu}.log')))
    started.write_text(json.dumps(dict(started_at=time.time(),launcher_pid=os.getpid(),workers=info),indent=2)+'\n')
    status=[p.wait() for p in children]
    for f in logs:f.close()
    result=dict(finished_at=time.time(),worker_exit_codes=status)
    if all(c==0 for c in status):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (logdir/'report.log').open('x') as f:
            r=subprocess.run([sys.executable,'-u','-m','tools.fast_adapter.p37_report'],stdout=f,stderr=subprocess.STDOUT,env=env)
        result['report_exit_code']=r.returncode
    (ROOT/'QUEUE_FINISHED.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
