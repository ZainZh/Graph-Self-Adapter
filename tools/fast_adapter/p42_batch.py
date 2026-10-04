"""Independent, result-free-frozen replication of the unchanged P41 prototype."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from tools.fast_adapter.p41_run import rollout, js, sha

ROOT = Path('evaluations/fast_adapter_p42_geometry_scale_v01')


def verify():
    freeze = json.loads((ROOT / 'FREEZE.json').read_text())
    for path, digest in freeze['sources'].items():
        assert sha(path) == digest, path
    assert sha(ROOT / 'PROTOCOL.json') == freeze['protocol_sha256']
    return freeze, json.loads((ROOT / 'PROTOCOL.json').read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['cup', 'insert'], required=True)
    args = p.parse_args()
    freeze, protocol = verify()
    started = time.time()
    cells = [(split, seed, arm) for split in protocol['splits']
             for seed in protocol['seeds'] for arm in protocol['arms']]
    records = []
    for split, seed, arm in cells:
        out = ROOT / 'independent_v01' / args.task / f'{split}_{seed}_{arm}'
        if out.exists():
            raise RuntimeError(f'Existing episode is immutable; no automatic rerun: {out}')
        # The original runner's PRE_RUN/RESULT scope remains "development".
        # This immutable external contract supplies the independent cohort scope.
        contract = dict(task=args.task, split=split, seed=seed, arm=arm,
                        scope='independent_frozen_prototype_replication',
                        freeze_sha256=sha(ROOT / 'FREEZE.json'), start_time=time.time())
        try:
            result = rollout(args.task, split, seed, arm, out, video=False)
            record = dict(task=args.task, split=split, seed=seed, arm=arm,
                          status='complete', result=str(out / 'RESULT.json'))
        except Exception as exc:
            # Keep the original partial files and continue the bounded list.
            # Never remove an algorithmic failure or silently replace a seed.
            out.mkdir(parents=True, exist_ok=True)
            error = dict(type=type(exc).__name__, message=str(exc),
                         traceback=traceback.format_exc(), time=time.time())
            js(out / 'EXCEPTION.json', error)
            record = dict(task=args.task, split=split, seed=seed, arm=arm,
                          status='execution_exception', error=str(out / 'EXCEPTION.json'))
        js(out / 'P42_EPISODE_CONTRACT.json', contract)
        records.append(record)
        js(ROOT / 'independent_v01' / f'{args.task}_PROGRESS.json',
           dict(done=len(records), total=len(cells), started=started, time=time.time(),
                complete=sum(r['status'] == 'complete' for r in records),
                exceptions=sum(r['status'] != 'complete' for r in records)))
    js(ROOT / 'independent_v01' / f'{args.task}_DONE.json',
       dict(rows=records, started=started, ended=time.time(),
            worker_pid=os.getpid(), cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
            mujoco_egl_device_id=os.getenv('MUJOCO_EGL_DEVICE_ID')))


if __name__ == '__main__':
    main()
