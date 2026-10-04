"""Combine three already-frozen P47 task queues into one finite four-GPU batch."""
import argparse
import json
from pathlib import Path

from tools.fast_adapter.p45_common import sha
from tools.fast_adapter.p47_evaluate import validate_freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='evaluations/fast_adapter_p47_representation_solver_v01')
    parser.add_argument('--phase', choices=('development', 'main'), required=True)
    args = parser.parse_args()
    root = Path(args.root)
    jobs, sources, evaluations = [], {}, []
    total = 0
    for task in ('cup', 'insert', 'slider'):
        folder = root / args.phase / task
        path = folder / 'EVALUATION_FREEZE.json'
        frozen = validate_freeze(path)
        batch = json.loads((folder / 'FREEZE.json').read_text())
        if batch['workers'] != 4 or batch['sources'] != frozen['sources']:
            raise ValueError('Every task must use the same four-worker frozen protocol')
        if any(path in sources and sources[path] != digest
               for path, digest in frozen['sources'].items()):
            raise ValueError('Shared task source bindings must agree before combining queues')
        sources.update(frozen['sources'])
        jobs.extend(batch['jobs'])
        total += len(frozen['cells'])
        evaluations.append(dict(path=str(path), sha256=sha(path)))
    expected = 78 if args.phase == 'development' else 1248
    if total != expected or len(jobs) != 12:
        raise ValueError('Unexpected P47 scope')
    # Launching and scheduling code is pinned too; do not alter after start.
    for source in ('tools/fast_adapter/p47_batch_plan.py',
                   'tools/fast_adapter/p45_batch_worker.py',
                   'tools/fast_adapter/p45_launch_frozen.py'):
        sources[source] = sha(source)
    output = root / (args.phase + '_batch')
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status='frozen', root_authorized=True, scope='p45_finite_batch',
                  campaign='p47_representation_solver', phase=args.phase, workers=4,
                  sources=sources, jobs=jobs, evaluation_freezes=evaluations,
                  n_physical_cells=total, retry_policy='no automatic retries')
    (output / 'FREEZE.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(path=str(output / 'FREEZE.json'), cells=total, jobs=len(jobs))))


if __name__ == '__main__':
    main()
