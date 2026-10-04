"""Durable bounded P38 queue. Never terminates unrelated GPU processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('evaluations/fast_adapter_p38_causal_v01')
JOBS = (('structured', 'history'), ('effect',), ('absolute',), ('relation',))


def execute(args, log_name, gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    logs = ROOT/'logs'; logs.mkdir(exist_ok=True, parents=True)
    with (logs/(log_name+'.log')).open('x') as log:
        return subprocess.run([sys.executable, '-m', 'tools.fast_adapter.p38_mapping']+args,
                              stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def worker(gpu):
    for kind in JOBS[gpu]:
        for seed in (11, 22, 33):
            key = f'{kind}_seed{seed}'
            for mode in ('train', 'eval'):
                rc = execute([mode, '--kind', kind, '--seed', str(seed)], key+'_'+mode, gpu)
                if rc:
                    raise RuntimeError((key, mode, rc))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'STARTED.json').exists():
        raise RuntimeError('A queue already exists; inspect it instead of duplicating.')
    (ROOT/'STARTED.json').write_text(json.dumps({'pid': os.getpid(), 'time': time.time()}))
    rc = execute(['controller'], 'controller', 0)
    if rc:
        raise RuntimeError(('controller', rc))
    procs = []
    for gpu in range(4):
        log = (ROOT/'logs'/f'worker{gpu}.log').open('x')
        p = subprocess.Popen([sys.executable, '-m', 'tools.fast_adapter.p38_launch', '--worker', str(gpu)],
                             stdout=log, stderr=subprocess.STDOUT)
        procs.append(p); log.close()
    (ROOT/'WORKERS.json').write_text(json.dumps([{'gpu': g, 'pid': p.pid} for g, p in enumerate(procs)]))
    exits = [p.wait() for p in procs]
    (ROOT/'DEV_QUEUE_DONE.json').write_text(json.dumps({'time': time.time(), 'exit_codes': exits}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--worker', type=int)
    a = p.parse_args()
    if a.worker is None: main()
    else: worker(a.worker)
