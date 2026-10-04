#!/usr/bin/env python3
"""Bounded 18-cell P52 local evaluation queue; invoked explicitly by root."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def validate(binding):
    expected=[dict(index=i*3+j,entry=i,arm=arm) for i in range(6) for j,arm in enumerate(('bc','fixed','online'))]
    if binding.get('schema')!='p52.local-policy-pilot.v1' or binding.get('cells')!=expected:
        raise ValueError('Expected exact six-entry bc/fixed/online Cartesian scope')
    if len(binding['entries'])!=6 or binding['max_actions']!=180 or binding['automatic_retries']!=0:
        raise ValueError('Evaluation budget changed')
    if binding['goal_hold_frames']!=3 or binding['initial_restores']!=1:
        raise ValueError('Evaluation termination/reset contract changed')
    return expected


def worker(gpu,cells,args,root):
    records=[]
    for cell in cells:
        if sha(args.binding)!=args.binding_sha256:raise ValueError('Binding changed during queue')
        folder=args.output/f"cell_{cell['index']:02d}_{cell['arm']}";folder.mkdir()
        output=folder/'episode'
        argv=[str(args.python),'-m','tools.task_graph.p52_policy_pilot','run',
              '--binding',str(args.binding),'--index',str(cell['index']),'--output',str(output)]
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),
            MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            LIBERO_CONFIG_PATH=str(output/'libero_config'))
        receipt=dict(cell=cell,physical_gpu=gpu,attempt=1,argv=argv,binding_sha256=args.binding_sha256,
                     started_unix=time.time(),automatic_retry=False)
        write(folder/'ATTEMPT.json',receipt)
        with (folder/'process.log').open('x') as log:
            try:
                process=subprocess.Popen(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:receipt.update(returncode=process.wait(timeout=1800),execution_status='finished')
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                    receipt.update(returncode=None,execution_status='engineering_timeout',scientific_failure=False)
            except OSError as error:
                receipt.update(returncode=None,execution_status='launch_error',error=str(error),scientific_failure=False)
        receipt['finished_unix']=time.time()
        receipt['artifacts']={name:dict(path=str(output/name),sha256=sha(output/name))
            for name in ('RESULT.json','SEAL.json','ABORT.json') if (output/name).is_file()}
        if receipt['execution_status']!='finished':write(folder/'QUEUE_ABORT.json',receipt)
        receipt['log_sha256']=sha(folder/'process.log');write(folder/'RECEIPT.json',receipt);records.append(receipt)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--binding-sha256',required=True)
    p.add_argument('--python',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.binding=a.binding.resolve();a.output=a.output.resolve();a.python=a.python.absolute()
    if sha(a.binding)!=a.binding_sha256:raise ValueError('Binding hash mismatch')
    cells=validate(json.loads(a.binding.read_text()));root=Path(__file__).resolve().parents[2]
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'QUEUE.json',dict(binding=str(a.binding),binding_sha256=a.binding_sha256,cells=cells,
        queue_source_sha256=sha(__file__),physical_gpus=[0,1,2],timeout_seconds=1800,attempts_per_cell=1))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,gpu,[c for c in cells if c['entry']%3==gpu],a,root) for gpu in range(3)]
        records=[row for future in futures for row in future.result()]
    write(a.output/'QUEUE_RESULT.json',dict(cells=sorted(records,key=lambda r:r['cell']['index']),scientific_success_not_inferred=True))


if __name__=='__main__':main()
