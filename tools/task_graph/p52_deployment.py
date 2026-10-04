#!/usr/bin/env python3
"""Create a P52 deployment inventory only; never upload or launch a worker.

Static imports do not identify dynamic dependencies/assets: pass these explicitly
with --artifact. LIBERO Python sources are included via --libero-root. The
collector's own binding remains the execution authority.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def source_closure(root, entries):
    root = Path(root).resolve()
    pending = list(entries)
    result = set()
    while pending:
        relative = Path(pending.pop())
        path = (root / relative).resolve()
        relative = path.relative_to(root).as_posix()
        if relative in result:
            continue
        tree = ast.parse(path.read_text())
        result.add(relative)
        package = list(Path(relative).with_suffix('').parts[:-1])
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level > len(package):
                    raise ValueError(f'Relative import escapes package: {relative}')
                base = package[:len(package) - node.level + 1] if node.level else []
                module = '.'.join(base + (node.module.split('.') if node.module else []))
                if module:
                    modules += [module] + [module + '.' + a.name for a in node.names]
        # Include entrypoint package initializers as well as imported packages.
        modules.append('.'.join(Path(relative).with_suffix('').parts))
        for module in modules:
            parts = module.split('.')
            if parts[0] not in ('prove', 'tools'):
                continue
            candidates = [Path(*parts).with_suffix('.py')]
            candidates += [Path(*parts[:i]) / '__init__.py' for i in range(1, len(parts) + 1)]
            pending.extend(p.as_posix() for p in candidates if (root / p).is_file())
    return sorted(result)


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', action='append', required=True, help='Repository-relative Python entry; repeat as needed')
    parser.add_argument('--artifact', action='append', default=[], help='Explicit dynamic dependency, BDDL, data or weight file')
    parser.add_argument('--libero-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entries = args.entry + ['tools/task_graph/p52_deployment.py']
    sources = {p: record(ROOT / p) for p in source_closure(ROOT, entries)}
    libero = args.libero_root.resolve()
    if not (libero / 'libero/libero/__init__.py').is_file():
        raise ValueError('Expected canonical P48 libero.libero source root')
    external = {p.relative_to(libero).as_posix(): record(p) for p in sorted((libero / 'libero').rglob('*.py'))}
    manifest = {
        'scope': 'P52 deployment inventory; not an execution freeze',
        'repository_sources': sources, 'libero_root': str(libero), 'libero_python_sources': external,
        'explicit_artifacts': {str(Path(p).resolve()): record(p) for p in args.artifact},
        'limitations': 'Static closure excludes dynamic imports, assets and runtime packages unless explicitly listed; collector binding must verify them.',
        'collection_queue_proposal': {
            'status': 'pending_teacher_qualification_and_seed_variation_freeze', 'jobs_maximum': 24,
            'automatic_retry': False, 'execution_enabled': False,
            'workers': [{'worker': i, 'gpu': i // 2, 'cpu_threads': 1,
                         'LIBERO_CONFIG_PATH': f'workers/worker_{i}/libero_config',
                         'output_root': f'workers/worker_{i}/jobs'} for i in range(6)],
            'job_seed_and_variation': None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'repository_sources': len(sources), 'libero_python_sources': len(external)}))


if __name__ == '__main__':
    main()
