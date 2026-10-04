"""Launch exactly four authorized finite workers and persist their process IDs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--kind', choices=['engineering', 'batch'], required=True)
    args = parser.parse_args()
    freeze = Path(args.freeze)
    config = json.loads(freeze.read_text())
    if config['status'] != 'frozen' or not config['root_authorized']:
        raise ValueError('Root-authorized frozen configuration required')
    for source, expected in config['sources'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed source: {source}')
    launch = freeze.parent / 'LAUNCH.json'
    # Exclusive creation prevents accidental second launch, including a partial
    # launch. Any infrastructure repair must be reviewed, not silently retried.
    with launch.open('x') as receipt:
        receipt.write(json.dumps(dict(status='starting', timestamp=time.time())))
    processes = []
    for rank in range(4):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(rank), MUJOCO_GL='egl', MUJOCO_EGL_DEVICE_ID='0',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        if args.kind == 'engineering':
            command = [sys.executable, '-m', 'tools.fast_adapter.p45_engineering_worker_v02', str(freeze.parent), str(rank)]
        else:
            command = [sys.executable, '-m', 'tools.fast_adapter.p45_batch_worker', '--freeze', str(freeze), '--rank', str(rank)]
        with (freeze.parent / f'worker_{rank}.log').open('x') as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(dict(rank=rank, gpu=rank, pid=process.pid, command=command))
    record = dict(status='launched', timestamp=time.time(), freeze=str(freeze),
                  freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(), workers=processes)
    launch.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
