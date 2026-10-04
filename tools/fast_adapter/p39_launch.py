"""Bounded four-GPU P39 jobs; failures stop dependents, never overwrite outputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('evaluations/fast_adapter_p39_tokens_v01')

def env(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(gpu),
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
    return e
def run(args,gpu,tag):
    with (ROOT/'logs'/(tag+'.log')).open('x') as f:
        subprocess.run([sys.executable,'-u','-m','tools.fast_adapter.p39_tokens']+args,env=env(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)
def worker(gpu):
    kinds=[('global','history'),('local',),('relation',),('fused',)][gpu]
    for kind in kinds:
        for seed in (11,22):run(['train','--kind',kind,'--seed',str(seed)],gpu,f'train_{kind}_{seed}')
def main():
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    with (ROOT/'TRAIN_START.json').open('x') as f:json.dump(dict(time=time.time(),pid=os.getpid()),f)
    jobs=[]
    for gpu in range(4):
        with (ROOT/'logs'/f'collect{gpu}.log').open('x') as f:
            p=subprocess.Popen([sys.executable,'-u','-m','tools.fast_adapter.p39_tokens','collect','--shard',str(gpu)],env=env(gpu),stdout=f,stderr=subprocess.STDOUT)
        jobs.append(p)
    codes=[p.wait() for p in jobs]
    if any(codes):raise RuntimeError(('collection',codes))
    run(['merge'],0,'merge');run(['effects'],0,'effects')
    jobs=[]
    for gpu in range(4):
        with (ROOT/'logs'/f'worker{gpu}.log').open('x') as f:
            jobs.append(subprocess.Popen([sys.executable,'-u','-m','tools.fast_adapter.p39_launch','--worker',str(gpu)],stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs]
    (ROOT/'TRAIN_DONE.json').write_text(json.dumps(dict(time=time.time(),exits=codes)))
    if any(codes):raise RuntimeError(('training',codes))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--worker',type=int);v=a.parse_args()
    main() if v.worker is None else worker(v.worker)
