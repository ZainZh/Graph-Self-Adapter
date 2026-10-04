"""Finite P69 explore queue: one canary then at most one process per GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(freeze, approval, output):
    frozen = json.loads(freeze.read_text())
    review = json.loads(approval.read_text())
    digest = hashlib.sha256(freeze.read_bytes()).hexdigest()
    if frozen['split'] != 'explore' or len(frozen['cells']) != 10:
        raise ValueError('Only the predefined ten-cell explore cohort is supported')
    if review.get('approved') is not True:
        raise ValueError('Root approval required; runner verifies exact freeze binding')
    if [c['index'] for c in frozen['cells']] != list(range(10)) or any(c['arm'] != 'continuous' for c in frozen['cells']):
        raise ValueError('Unexpected cell assignments')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'QUEUE_IDENTITY.json').write_text(json.dumps(dict(freeze=str(freeze), freeze_sha256=digest, canary=0, indices=list(range(10)), automatic_retry=False), indent=2)+'\n')
    active = {}; pending = list(range(10)); completed = []; stopped = False
    def start(index, gpu):
        directory = output / f'cell_{index:03d}'
        log = (output / f'cell_{index:03d}.log').open('xb')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        command = [sys.executable, '-m', 'tools.task_graph.p69_full_rsi_runner', 'run-cell', '--freeze', str(freeze), '--approval', str(approval), '--index', str(index), '--output', str(directory)]
        try:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        except BaseException:
            log.close()
            raise
        active[gpu] = (index, process, log, directory)
    while pending or active:
        if any(output.glob('cell_*/ABORT.json')):
            stopped = True
        for gpu, (index, process, log, directory) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close(); del active[gpu]
            if code or not (directory / 'RESULT.json').is_file() or not (directory / 'SEAL.json').is_file():
                stopped = True
                (output / 'QUEUE_ABORT.json').write_text(json.dumps(dict(index=index, returncode=code, automatic_retry=False))+'\n')
            else:
                seal = json.loads((directory / 'SEAL.json').read_text())
                result = json.loads((directory / 'RESULT.json').read_text())
                valid = seal['binding_sha256'] == digest and seal['cell'] == frozen['cells'][index] and result['cell'] == frozen['cells'][index]
                valid = valid and result['seal_sha256'] == hashlib.sha256((directory / 'SEAL.json').read_bytes()).hexdigest()
                valid = valid and all(hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected for name, expected in seal['files'].items())
                if not valid:
                    stopped = True
                    (output / 'QUEUE_ABORT.json').write_text(json.dumps(dict(index=index, error='seal mismatch', automatic_retry=False))+'\n')
                else:
                    completed.append(index)
        if not stopped:
            for gpu in range(4):
                if gpu in active or not pending:
                    continue
                if pending[0] != 0 and 0 not in completed:
                    break
                index = pending.pop(0)
                try:
                    start(index, gpu)
                except Exception as exc:
                    stopped = True
                    (output / 'QUEUE_ABORT.json').write_text(json.dumps(dict(index=index, error=repr(exc), automatic_retry=False))+'\n')
                    break
        if stopped and not active:
            break
        time.sleep(0.5)
    receipt = dict(completed=sorted(completed), unsubmitted=pending, aborted=stopped, automatic_retry=False)
    (output / 'QUEUE_COMPLETION.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return 1 if stopped else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('freeze', 'approval', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args=p.parse_args()
    return run(args.freeze.resolve(),args.approval.resolve(),args.output.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
