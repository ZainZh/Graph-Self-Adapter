#!/usr/bin/env python3
"""Offline caddy source/public-channel audit; never constructs a simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prove.task_graph.libero_visible_grounding import capability_report, caddy_source_audit


def audit_demo(path):
    import h5py
    with h5py.File(path, 'r') as handle:
        xml = str(handle['data/demo_0'].attrs['model_file'])
    report = capability_report(caddy_source_audit(xml))
    report['development_source'] = dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                      field='data/demo_0.attrs.model_file',
                                      usage='source topology only; no object poses, goal projections or held-out data')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_demo(args.demo), indent=2, allow_nan=False))
