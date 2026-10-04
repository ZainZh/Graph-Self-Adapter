"""Bounded local training queue; no agent monitoring and no outcome-based branching."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(workers):
    root=Path('evaluations/fast_adapter_p35_resolution_views_v01')
    for task in ('mug-handle-lift','hammer-block'):
        if not (root/task/'data/RECEIPT.json').exists():
            raise FileNotFoundError(f'{task}: input extraction must finish first')
    out=root/'queue';out.mkdir(exist_ok=False)
    env=dict(os.environ, MUJOCO_GL='egl', PYTHONPATH='data/runtime/fast-adapter-p28-deps:.', OMP_NUM_THREADS='2')
    jobs=[(t,v,s) for t in ('mug-handle-lift','hammer-block')
          for v in ('g224','g448','g224w','g448w') for s in (11,22)]
    def run(job):
        task,view,seed=job;begin=time.time();p=out/f'{task}_{view}_{seed}.log'
        with p.open('x') as log:
            result=subprocess.run([sys.executable,'-m','tools.fast_adapter.p35_train',
                '--task',task,'--view',view,'--seed',str(seed)],env=env,stdout=log,stderr=subprocess.STDOUT)
        row={'task':task,'view':view,'seed':seed,'exit_code':result.returncode,'seconds':time.time()-begin,'log':str(p)}
        print('P35_QUEUE',json.dumps(row),flush=True)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows=list(pool.map(run,jobs))
    (out/'RESULT.json').write_text(json.dumps({'workers':workers,'jobs':rows},indent=2)+'\n')
    if any(r['exit_code'] for r in rows):raise RuntimeError('A fixed training cell failed; inspect its log before validation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('bounded queue permits 1 to 4 workers')
    main(a.workers)
