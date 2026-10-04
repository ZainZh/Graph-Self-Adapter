#!/usr/bin/env python3
"""Run an explicitly bound P52 24-job queue. No generation, retries or shell."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(plan):
    if plan.get('schema') != 'prove.p52-collection-queue.v1':
        raise ValueError('Unexpected queue schema')
    jobs = plan['jobs']
    if len(jobs) != 24 or plan['attempts_per_job'] != 1:
        raise ValueError('Exactly 24 conditions, one attempt each required')
    if len({j['id'] for j in jobs}) != 24 or len({j['seed'] for j in jobs}) != 24:
        raise ValueError('Independent job IDs and seeds required')
    for i, job in enumerate(jobs):
        if job['id'] != f'job_{i:02d}' or not isinstance(job['seed'], int):
            raise ValueError('Ordered job IDs and integer seeds required')
        if not isinstance(job['argv'], list) or not job['argv'] or not all(isinstance(a, str) for a in job['argv']):
            raise ValueError('Explicit string argv required')
        if job['timeout_seconds'] <= 0:
            raise ValueError('Finite job timeout required')
        if '--output' in job['argv']:
            raise ValueError('Queue supplies independent --output paths')
    if not plan['bound_files']:
        raise ValueError('Result-free source/binding hashes required')
    for path, digest in plan['bound_files'].items():
        if sha(path) != digest:
            raise ValueError(f'Bound input changed: {path}')
    return jobs


def run_worker(worker, jobs, root, cwd):
    worker_root = root / f'worker_{worker}'
    worker_root.mkdir()
    results = []
    for job in jobs:
        job_root = worker_root / job['id']
        job_root.mkdir()
        output = job_root / 'episode'
        config = output / 'libero_config'
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(worker // 2), MUJOCO_EGL_DEVICE_ID=str(worker // 2), MUJOCO_GL='egl',
                   LIBERO_CONFIG_PATH=str(config.resolve()), PYTHONHASHSEED=str(job['seed']),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        argv = job['argv'] + ['--output', str(output.resolve())]
        attempt = dict(job=job, worker=worker, physical_gpu=worker // 2, attempt=1,
                       argv=argv, output=str(output), LIBERO_CONFIG_PATH=str(config),
                       started_unix=time.time())
        (job_root / 'ATTEMPT.json').write_text(json.dumps(attempt, indent=2)+'\n')
        with (job_root / 'process.log').open('x') as log:
            try:
                completed = subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=job['timeout_seconds'], check=False)
                attempt.update(returncode=completed.returncode, execution_status='finished')
            except subprocess.TimeoutExpired:
                attempt.update(returncode=None, execution_status='timeout', scientific_failure=False)
                (job_root / 'QUEUE_ABORT.json').write_text(json.dumps(dict(reason='engineering_timeout', timeout_seconds=job['timeout_seconds'], automatic_retry=False))+'\n')
            except OSError as exc:
                attempt.update(returncode=None, execution_status='launch_error', error=str(exc), scientific_failure=False)
                (job_root / 'QUEUE_ABORT.json').write_text(json.dumps(dict(reason='launch_error', error=str(exc), automatic_retry=False))+'\n')
        attempt['finished_unix'] = time.time()
        for name in ('RESULT.json', 'SEAL.json', 'ABORT.json'):
            if (output / name).is_file():
                attempt[name] = {'path': str(output / name), 'sha256': sha(output / name)}
        (job_root / 'ATTEMPT_RESULT.json').write_text(json.dumps(attempt, indent=2)+'\n')
        results.append(attempt)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('first', 'remaining'), required=True)
    args = parser.parse_args()
    if sha(args.manifest) != args.manifest_sha256:
        raise ValueError('Queue manifest changed')
    plan = json.loads(args.manifest.read_text())
    jobs = validate(plan)
    indexed_jobs = list(enumerate(jobs))
    selected = indexed_jobs[:1] if args.phase == 'first' else indexed_jobs[1:]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'QUEUE_MANIFEST.json').write_bytes(args.manifest.read_bytes())
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(run_worker, i, [job for index, job in selected if index % 6 == i], root, plan['cwd']) for i in range(6)]
        results = [row for future in futures for row in future.result()]
    (root / 'QUEUE_RESULT.json').write_text(json.dumps(dict(attempts=sorted(results, key=lambda r:r['job']['id']),
        retries=0, scientific_success_not_inferred=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
