"""Result-free preflight and bounded forty-cell P69 final queue.

Manifest versions require version,cwd,python,freeze,review,explore_freeze,
explore_queue. Each test Root review must bind this queue's exact source hash.
No task metrics or private state values are used by this scheduler.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def verify_cell(directory, cell, digest):
    directory = Path(directory)
    if (directory/'ABORT.json').exists():
        raise ValueError('Cell has ABORT: '+str(directory))
    seal = read(directory/'SEAL.json'); result = read(directory/'RESULT.json')
    if seal['cell'] != cell or result['cell'] != cell or seal['binding_sha256'] != digest:
        raise ValueError('Cell binding mismatch')
    if result['seal_sha256'] != sha(directory/'SEAL.json'):
        raise ValueError('Result seal mismatch')
    for name, digest in seal['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != digest:
            raise ValueError('Sealed artifact mismatch')
    return sha(directory/'SEAL.json')


def preflight(manifest):
    versions = manifest['versions']
    if [v['version'] for v in versions] != ['R0','R1','R2','R3']:
        raise ValueError('Exactly frozen R0 through R3 required')
    pairing = None; jobs = []; bindings = []
    for item in versions:
        frozen = read(item['freeze']); review = read(item['review'])
        digest = sha(item['freeze']); cwd = Path(item['cwd']).resolve()
        if (frozen.get('schema') != 'prove.p69-full-rsi-freeze.v1' or frozen.get('status') != 'frozen_before_results'
                or frozen['version'] != item['version'] or frozen['split'] != 'test'
                or Path(frozen['version_source_root']).resolve() != cwd):
            raise ValueError('Wrong final version freeze')
        if (review.get('schema') != 'prove.p69-root-review.v1' or review.get('freeze_sha256') != digest
                or any(review.get(k) is not True for k in ('approved','root_reviewed','physical_execution_authorized'))
                or review.get('queue_sha256') != sha(__file__)):
            raise ValueError('Missing exact Root authorization')
        cells = frozen['cells']
        if len(cells) != 10 or [c['index'] for c in cells] != list(range(10)):
            raise ValueError('Final denominator must be ten per version')
        identity = [(c['init_id'],c['state_sha256'],c['execution_seed']) for c in cells]
        if len({str(c['init_id']) for c in cells}) != 10 or len({c['state_sha256'] for c in cells}) != 10:
            raise ValueError('Distinct paired initial states required')
        for i,c in enumerate(cells):
            if c['split'] != 'test' or c['version'] != item['version'] or c['arm'] != 'continuous' or c['execution_seed'] != 69030+i:
                raise ValueError('Final cell contract drift')
        if pairing is not None and pairing != identity:
            raise ValueError('Versions must pair identical test states and seeds')
        pairing = identity
        if not Path(item['python']).is_file():
            raise ValueError('Missing interpreter')
        runner = cwd/'tools/task_graph/p69_full_rsi_runner.py'
        if str(runner) not in frozen['sources']:
            raise ValueError('Version runner is not frozen')
        for path,digest0 in frozen['sources'].items():
            if not Path(path).resolve().is_relative_to(cwd) or sha(path) != digest0:
                raise ValueError('Version source changed')
        for asset in frozen['files'].values():
            if sha(asset['path']) != asset['sha256']:
                raise ValueError('Frozen asset changed')
        explore = read(item['explore_freeze']); completion = read(Path(item['explore_queue'])/'QUEUE_COMPLETION.json')
        if explore['version'] != item['version'] or explore['split'] != 'explore' or len(explore['cells']) != 10 or completion != dict(completed=list(range(10)),unsubmitted=[],aborted=False,automatic_retry=False):
            raise ValueError('All four explore cohorts must be complete')
        if explore['files'] != frozen['files'] or explore['sources'] != frozen['sources'] or explore['graph'] != frozen['graph']:
            raise ValueError('Final candidate differs from explored version')
        for cell in explore['cells']:
            verify_cell(Path(item['explore_queue'])/f"cell_{cell['index']:03d}",cell,sha(item['explore_freeze']))
        bindings.append(dict(version=item['version'],freeze_sha256=digest,review_sha256=sha(item['review']),explore_freeze_sha256=sha(item['explore_freeze'])))
        jobs.extend(dict(item=item,cell=cell,freeze_sha256=digest) for cell in cells)
    return jobs,bindings


def run(manifest, output, popen=subprocess.Popen, sleep=time.sleep):
    jobs, bindings = preflight(manifest)  # All versions before any subprocess/output.
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    dump(output/'QUEUE_MANIFEST.json',dict(schema='prove.p69-final-queue.v1',versions=bindings,total=40,queue_sha256=sha(__file__),manifest=manifest,automatic_retry=False))
    pending=list(jobs); active={}; completed=[]; errors=[]
    while pending or active:
        for gpu,(job,process,log,directory) in list(active.items()):
            if (directory/'ABORT.json').exists() and not errors:
                errors.append(dict(version=job['item']['version'],index=job['cell']['index'],reason='ABORT observed'))
            code=process.poll()
            if code is None:continue
            log.close();del active[gpu]
            try:
                if code:raise ValueError('child exit '+str(code))
                seal=verify_cell(directory,job['cell'],job['freeze_sha256'])
                completed.append(dict(version=job['item']['version'],index=job['cell']['index'],seal_sha256=seal))
            except Exception as exc:
                errors.append(dict(version=job['item']['version'],index=job['cell']['index'],reason=str(exc)))
        if not errors:
            for gpu in range(4):
                if gpu in active or not pending:continue
                job=pending.pop(0);item=job['item'];directory=output/item['version']/f"cell_{job['cell']['index']:03d}"
                directory.parent.mkdir(exist_ok=True)
                log=(directory.parent/(directory.name+'.log')).open('xb')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONPATH='data/runtime/p58-vision-deps:.',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                command=[item['python'],'-m','tools.task_graph.p69_full_rsi_runner','run-cell','--freeze',item['freeze'],'--approval',item['review'],'--index',str(job['cell']['index']),'--output',str(directory.resolve())]
                try:active[gpu]=(job,popen(command,cwd=item['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT),log,directory)
                except Exception as exc:
                    log.close();errors.append(dict(version=item['version'],index=job['cell']['index'],reason=str(exc)));break
        if errors and not active:break
        if active:sleep(.5)
    receipt=dict(completed=completed,unsubmitted=[dict(version=j['item']['version'],index=j['cell']['index']) for j in pending],errors=errors,all_complete=len(completed)==40 and not errors,automatic_retry=False)
    dump(output/'QUEUE_COMPLETION.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','run']);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();manifest=read(a.manifest)
    if a.command=='preflight':
        jobs,bindings=preflight(manifest);print(json.dumps(dict(ready=True,total=len(jobs),versions=bindings)));return 0
    if a.output is None:p.error('run requires --output')
    return 0 if run(manifest,a.output)['all_complete'] else 1


if __name__=='__main__':raise SystemExit(main())
