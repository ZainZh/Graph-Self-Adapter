"""Finite manifest-bound P70 subprocess queue; no simulator imports.

Jobs: id, cwd, argv (literal list with {output}/{freeze}/{review} placeholders),
freeze {path,sha256}, review {path,sha256}. Manifest: jobs, gpu_ids, schema.
A separate Root queue review binds the entire manifest and this source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')


def preflight(manifest_path, review_path):
    manifest=read(manifest_path);review=read(review_path)
    if manifest.get('schema')!='prove.p70-queue.v1':raise ValueError('Unknown queue schema')
    if not (review.get('approved') is True and review.get('reviewer')=='root' and review.get('manifest_sha256')==sha(manifest_path) and review.get('queue_sha256')==sha(__file__)):
        raise ValueError('Exact Root queue approval required')
    jobs=manifest['jobs'];gpus=manifest['gpu_ids']
    if not jobs or len({j['id'] for j in jobs})!=len(jobs):raise ValueError('Unique finite jobs required')
    if not gpus or len(set(gpus))!=len(gpus) or any(type(g) is not int or g not in range(4) for g in gpus):raise ValueError('Invalid GPU allocation')
    for job in jobs:
        if not job['id'].replace('_','').replace('-','').isalnum():raise ValueError('Invalid job id')
        if not Path(job['cwd']).is_dir() or not isinstance(job['argv'],list) or not all(isinstance(a,str) for a in job['argv']):raise ValueError('Invalid job command')
        if '{output}' not in job['argv']:raise ValueError('Each job must receive its isolated output')
        for key in ('freeze','review'):
            if sha(job[key]['path'])!=job[key]['sha256']:raise ValueError('Changed '+key)
        signed=read(job['review']['path'])
        if signed.get('approved') is not True or signed.get('freeze_sha256')!=job['freeze']['sha256']:raise ValueError('Job review does not bind freeze')
    return manifest


def verify_output(output,job):
    if (output/'ABORT.json').exists():raise ValueError('Child ABORT')
    seal=read(output/'SEAL.json');result=read(output/'RESULT.json')
    digest=sha(output/'SEAL.json')
    if result.get('seal_sha256')!=digest:raise ValueError('Result must bind its seal')
    bound=seal.get('freeze_sha256',seal.get('binding_sha256'))
    if bound!=job['freeze']['sha256']:raise ValueError('Seal freeze binding mismatch')
    for name,expected in seal['files'].items():
        path=(output/name).resolve()
        if not path.is_relative_to(output.resolve()) or sha(path)!=expected:raise ValueError('Sealed file mismatch')
    return digest


def run(manifest_path,review_path,output,popen=subprocess.Popen,sleep=time.sleep):
    manifest=preflight(manifest_path,review_path)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    dump(output/'QUEUE_MANIFEST.json',dict(manifest=manifest,manifest_sha256=sha(manifest_path),review_sha256=sha(review_path),queue_sha256=sha(__file__),automatic_retry=False))
    pending=list(manifest['jobs']);active={};completed=[];errors=[]
    while pending or active:
        for gpu,(job,process,log,directory) in list(active.items()):
            if (directory/'ABORT.json').exists() and not errors:errors.append(dict(id=job['id'],reason='ABORT observed'))
            status=process.poll()
            if status is None:continue
            log.close();del active[gpu]
            try:
                if status:raise ValueError('Child exit '+str(status))
                digest=verify_output(directory,job)
                completed.append(dict(id=job['id'],gpu=gpu,seal_sha256=digest))
            except Exception as exc:errors.append(dict(id=job['id'],reason=str(exc)))
        if not errors:
            for gpu in manifest['gpu_ids']:
                if gpu in active or not pending:continue
                job=pending.pop(0);directory=output/job['id'];config=output/'worker_configs'/job['id'];config.mkdir(parents=True)
                substitutions={'{output}':str(directory),'{freeze}':job['freeze']['path'],'{review}':job['review']['path']}
                command=[substitutions.get(a,a) for a in job['argv']]
                env=dict(os.environ,PYTHONPATH='data/runtime/p58-vision-deps:.',CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBERO_CONFIG_PATH=str(config),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                log=(output/(job['id']+'.log')).open('xb')
                try:active[gpu]=(job,popen(command,cwd=job['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT),log,directory)
                except Exception as exc:log.close();errors.append(dict(id=job['id'],reason=str(exc)));break
        if errors and not active:break
        if active:sleep(.5)
    receipt=dict(completed=completed,unsubmitted=[j['id'] for j in pending],errors=errors,all_complete=len(completed)==len(manifest['jobs']) and not errors,automatic_retry=False)
    dump(output/'QUEUE_COMPLETION.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','run']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.command=='preflight':print(json.dumps(dict(ready=True,jobs=len(preflight(a.manifest,a.review)['jobs']))));return 0
    if a.output is None:p.error('run requires --output')
    return 0 if run(a.manifest,a.review,a.output)['all_complete'] else 1


if __name__=='__main__':raise SystemExit(main())
