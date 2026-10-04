"""Root-owned complete input freeze around the unchanged P43 evaluator.

No simulator is constructed until all files and the exact CLI scope match.
This adds a result-free independent gate without changing development sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(path):
    frozen = json.loads(Path(path).read_text())
    if frozen['phase'] != 'independent_validation' or not frozen['root_authorized']:
        raise ValueError('A root-authorized independent freeze is required')
    required = {'checkpoint', 'pca', 'out', 'task', 'model_dir'}
    if set(frozen['execution']) != required:
        raise ValueError('Unexpected or missing execution fields')
    for filename, digest in frozen['files'].items():
        if sha(filename) != digest:
            raise ValueError(f'Frozen input changed: {filename}')
    for field, digest_key in [('checkpoint', 'checkpoint_sha256'), ('pca', 'pca_sha256')]:
        source = frozen['execution'][field]
        if frozen['files'].get(source) != frozen[digest_key]:
            raise ValueError(f'Unbound {field}')
    for source in frozen['sources']:
        if source not in frozen['files']:
            raise ValueError(f'Unbound source: {source}')
    for source in frozen['visual_weight_files']:
        if source not in frozen['files']:
            raise ValueError(f'Unbound visual weights: {source}')
    if len(frozen['seeds']) != len(frozen['splits']):
        raise ValueError('Missing seed/split identity')
    if len(set(zip(frozen['splits'], frozen['seeds']))) != len(frozen['seeds']):
        raise ValueError('Repeated validation condition')
    if frozen['arms'] != ['fixed', 'continuous']:
        raise ValueError('Independent arms differ from root protocol')
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    frozen = validate(args.freeze)
    print(json.dumps({'verified': True, 'freeze_sha256': sha(args.freeze),
                      'files': len(frozen['files']), 'time': time.time()}), flush=True)
    if args.verify_only:
        return
    execution = frozen['execution']
    if Path(execution['out']).exists():
        raise FileExistsError('Preserve prior independent outcomes; no implicit restart')
    command = [sys.executable, '-m', 'tools.fast_adapter.p43_evaluate',
               '--scope', 'independent_validation', '--freeze', args.freeze,
               '--arms', ','.join(frozen['arms']),
               '--seeds', ','.join(map(str, frozen['seeds'])),
               '--splits', ','.join(frozen['splits'])]
    for key, value in execution.items():
        command.extend(['--' + key.replace('_', '-'), str(value)])
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
