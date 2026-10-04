#!/usr/bin/env python3
"""Execute only a hashed explicit P53 job list; no sampling, retries or training."""
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
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def validate(plan):
    if plan['schema']!='p53.explicit-queue.v1' or plan['attempts_per_job']!=1:raise ValueError('Wrong queue contract')
    if not isinstance(plan['workers'],int) or not 1<=plan['workers']<=6:raise ValueError('At most six collectors')
    jobs=plan['jobs']
    if not jobs or len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Nonempty unique condition list required')
    for job in jobs:
        if not job['id'] or Path(job['id']).name!=job['id'] or job['id'] in ('.','..'):raise ValueError('Unsafe job ID')
        if not isinstance(job['seed'],int):raise ValueError('Explicit integer seed required')
        if not 0<job['timeout_seconds']<=1800:raise ValueError('Finite engineering timeout <=1800 required')
        if not isinstance(job['argv'],list) or not job['argv'] or not all(isinstance(x,str) for x in job['argv']):raise ValueError('Explicit argv required')
        if '--output' in job['argv']:raise ValueError('Queue owns output path')
    if not plan['bound_files']:raise ValueError('Source/binding hashes required')
    for path,digest in plan['bound_files'].items():
        if sha(path)!=digest:raise ValueError('Bound input changed: '+path)
    return jobs


def worker(index,jobs,args,plan):
    gpu=index%3;rows=[]
    for job in jobs:
        if sha(args.manifest)!=args.manifest_sha256:raise ValueError('Queue manifest changed')
        folder=args.output/job['id'];folder.mkdir();episode=folder/'episode'
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',
            LIBERO_CONFIG_PATH=str(episode/'libero_config'),PYTHONHASHSEED=str(job['seed']),
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        argv=job['argv']+['--output',str(episode)]
        row=dict(job=job,worker=index,physical_gpu=gpu,argv=argv,initdir=env['LIBERO_CONFIG_PATH'],attempt=1,
                 started_unix=time.time(),manifest_sha256=args.manifest_sha256,automatic_retry=False)
        write(folder/'ATTEMPT.json',row)
        with (folder/'process.log').open('x') as log:
            try:
                proc=subprocess.Popen(argv,cwd=plan['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:row.update(returncode=proc.wait(timeout=job['timeout_seconds']),execution_status='finished')
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    row.update(returncode=None,execution_status='engineering_timeout',scientific_failure=False)
            except OSError as error:row.update(returncode=None,execution_status='launch_error',error=str(error),scientific_failure=False)
        row.update(finished_unix=time.time(),log_sha256=sha(folder/'process.log'))
        row['artifacts']={n:dict(path=str(episode/n),sha256=sha(episode/n)) for n in ('RESULT.json','SEAL.json','ABORT.json') if (episode/n).is_file()}
        if row['execution_status']!='finished':write(folder/'QUEUE_ABORT.json',row)
        write(folder/'RECEIPT.json',row);rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.manifest=a.manifest.resolve();a.output=a.output.resolve()
    if sha(a.manifest)!=a.manifest_sha256:raise ValueError('Manifest hash mismatch')
    plan=json.loads(a.manifest.read_text());jobs=validate(plan);a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'QUEUE.json',dict(manifest=plan,manifest_sha256=a.manifest_sha256,queue_source_sha256=sha(__file__)))
    with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures=[pool.submit(worker,i,jobs[i::plan['workers']],a,plan) for i in range(plan['workers'])]
        rows=[r for future in futures for r in future.result()]
    write(a.output/'QUEUE_RESULT.json',dict(attempts=rows,expected_jobs=len(jobs),scientific_success_not_inferred=True,automatic_retries=0))


if __name__=='__main__':main()
