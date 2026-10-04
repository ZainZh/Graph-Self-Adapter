"""Finite P45 collection/training queue; no retries or candidate changes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    path = Path(args.freeze)
    frozen = json.loads(path.read_text())
    assert frozen['status'] == 'frozen' and frozen['root_authorized']
    assert frozen['scope'] == 'p45_finite_batch'
    assert 0 <= args.rank < frozen['workers'] <= 4
    out = path.parent / f'worker_{args.rank}'
    out.mkdir(exist_ok=False)
    binding = dict(path=str(path), sha256=digest(path))
    (out / 'START.json').write_text(json.dumps(dict(freeze=binding, started=time.time())))
    completed = []
    for index, job in enumerate(frozen['jobs']):
        if index % frozen['workers'] != args.rank:
            continue
        for source, expected in frozen['sources'].items():
            if digest(source) != expected:
                raise ValueError(f'Frozen queue source changed: {source}')
        command = [sys.executable, '-m', job['module'], *job['arguments']]
        with (out / f'job_{index:04d}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record = dict(index=index, returncode=result.returncode, finished=time.time())
        (out / f'job_{index:04d}.json').write_text(json.dumps(record))
        print(json.dumps(record), flush=True)
        if result.returncode:
            (out / 'FAILED.json').write_text(json.dumps(dict(freeze=binding, **record)))
            raise SystemExit(result.returncode)
        completed.append(index)
    (out / 'DONE.json').write_text(json.dumps(dict(freeze=binding, jobs=completed, finished=time.time())))


if __name__ == '__main__':
    main()
