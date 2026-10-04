"""P38 short-contact campaign; bounded workers and persistent receipts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path('evaluations/fast_adapter_p38_causal_v01/push')
KINDS=('motion','appearance','combined','history')


def env(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
         MUJOCO_EGL_DEVICE_ID=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',
         TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
    return e


def command(args):return [sys.executable,'-m','tools.fast_adapter.p38_push']+args


def run(args,gpu,tag):
    with (R/'logs'/(tag+'.log')).open('x') as f:
        subprocess.run(command(args),env=env(gpu),stdout=f,stderr=subprocess.STDOUT,check=True)


def worker(gpu,phase):
    kind=KINDS[gpu]
    for seed in (11,22):
        if phase=='dev':run(['train','--kind',kind,'--seed',str(seed)],gpu,f'{kind}_{seed}_train')
        run(['eval','--kind',kind,'--seed',str(seed),'--phase',phase],gpu,f'{kind}_{seed}_{phase}')
    if gpu==0:
        run(['eval','--kind','references','--phase',phase],gpu,'references_'+phase)


def main(phase):
    (R/'logs').mkdir(parents=True,exist_ok=True)
    receipt=R/(phase+'_START.json')
    if receipt.exists():raise RuntimeError('Existing campaign: inspect instead of duplicating')
    receipt.write_text(json.dumps({'time':time.time(),'pid':os.getpid()}))
    if phase=='dev':
        procs=[]
        for gpu in range(4):
            f=(R/'logs'/f'collect{gpu}.log').open('x')
            procs.append(subprocess.Popen(command(['collect','--shard',str(gpu)]),env=env(gpu),stdout=f,stderr=subprocess.STDOUT));f.close()
        exits=[p.wait() for p in procs]
        if any(exits):raise RuntimeError(('collection',exits))
        run(['merge'],0,'merge');run(['jacobian'],0,'jacobian')
    procs=[]
    for gpu in range(4):
        f=(R/'logs'/f'{phase}_worker{gpu}.log').open('x')
        p=subprocess.Popen([sys.executable,'-m','tools.fast_adapter.p38_push_launch','--worker',str(gpu),'--phase',phase],stdout=f,stderr=subprocess.STDOUT)
        procs.append(p);f.close()
    (R/(phase+'_WORKERS.json')).write_text(json.dumps([p.pid for p in procs]))
    exits=[p.wait() for p in procs]
    (R/(phase+'_DONE.json')).write_text(json.dumps({'time':time.time(),'exits':exits}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--phase',choices=['dev','final'],default='dev');a=p.parse_args()
    if a.worker is None:main(a.phase)
    else:worker(a.worker,a.phase)
