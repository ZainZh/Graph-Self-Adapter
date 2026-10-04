"""Read-only HDF5 inventory. Run with ``python -m tools.task_graph.inspect_demo``."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, np.ndarray):
        return [json_value(x) for x in value.tolist()] if value.ndim else json_value(value.item())
    if isinstance(value, np.generic):
        return json_value(value.item())
    return value


def inspect_demo(path: Path, *, include_sha256: bool = True) -> dict:
    """Inventory dataset shapes/attributes without reading image or state arrays."""
    result = {'format': 'prove.demo-inventory.v1', 'path': str(path),
              'size_bytes': path.stat().st_size, 'groups': {}, 'datasets': {}}
    if include_sha256:
        result['sha256'] = sha256_file(path)
    with h5py.File(path, 'r') as stream:
        def visit(name, obj):
            attrs = {key: json_value(value) for key, value in obj.attrs.items()}
            if isinstance(obj, h5py.Dataset):
                result['datasets'][name] = {'shape': list(obj.shape), 'dtype': str(obj.dtype), 'attrs': attrs}
            else:
                result['groups'][name] = {'attrs': attrs}
        visit('/', stream)
        stream.visititems(visit)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--skip-sha256', action='store_true', help='Metadata inventory only; export always hashes source')
    args = parser.parse_args()
    print(json.dumps(inspect_demo(args.dataset, include_sha256=not args.skip_sha256), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
