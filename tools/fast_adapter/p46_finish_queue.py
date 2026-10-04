"""Finish the already authorized finite P46 queue without an LLM monitor.

Wait for the exact four workers, bind the predeclared supplemental donors,
launch 120 supplemental cells, then aggregate all six sealed cohorts. No retry,
training, candidate selection, new conditions or outcome-dependent allocation.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('evaluations/fast_adapter_p46_representation_validation_v01')
TASKS = ('cup', 'insert', 'slider')


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def wait_batch(folder, deadline):
    """Inspect only queue receipts, not task outcomes; preserve failures."""
    expected = binding(folder / 'FREEZE.json')
    while True:
        done = 0
        for rank in range(4):
            worker = folder / f'worker_{rank}'
            if (worker / 'FAILED.json').exists():
                raise RuntimeError(f'Frozen worker failed: {worker}; no automatic retry')
            if (worker / 'DONE.json').exists():
                if read(worker / 'DONE.json')['freeze'] != expected:
                    raise ValueError('Worker completion belongs to a different freeze')
                done += 1
        if done == 4:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Finite finishing window expired; workers and evidence left untouched')
        time.sleep(30)


def combine_supplemental():
    sources, jobs, parents = {}, [], []
    for task in TASKS:
        path = ROOT / 'supplemental' / task / 'FREEZE.json'
        frozen = read(path)
        if not frozen['root_authorized'] or len(frozen['jobs']) != 4:
            raise ValueError('Expected four authorized supplemental partitions')
        for source, digest in frozen['sources'].items():
            if source in sources and sources[source] != digest:
                raise ValueError('Source changed between frozen task cohorts')
            sources[source] = digest
        jobs.extend(frozen['jobs'])
        parents.append(binding(path))
    folder = ROOT / 'supplemental_batch'
    folder.mkdir(exist_ok=False)
    write(folder / 'FREEZE.json', dict(status='frozen', root_authorized=True,
          scope='p45_finite_batch', campaign='p46_representation_validation', workers=4,
          sources=sources, jobs=jobs, parent_batches=parents, n_physical_cells=120,
          timestamp=time.time()))
    return folder


def run(*arguments):
    subprocess.run([sys.executable, '-m', *arguments], check=True)


def main():
    if not os.environ.get('PROVE_PANDA_ASSET_ROOT'):
        raise ValueError('Panda asset environment must be set before queue finishing')
    receipt = ROOT / 'QUEUE_FINISHER.json'
    with receipt.open('x') as stream:
        json.dump(dict(status='started', pid=os.getpid(), timestamp=time.time(),
                       implementation=binding(__file__), main_batch=binding(ROOT/'main_batch/FREEZE.json'),
                       plan=binding(ROOT/'PLAN.md'), max_additional_physical_cells=120), stream)
    start = read(receipt)
    deadline = time.monotonic() + 4 * 3600
    try:
        wait_batch(ROOT / 'main_batch', deadline)
        for task in TASKS:
            run('tools.fast_adapter.p46_prepare_validation', '--mode', 'supplemental', '--task', task)
        folder = combine_supplemental()
        run('tools.fast_adapter.p45_launch_frozen', '--freeze', str(folder/'FREEZE.json'), '--kind', 'batch')
        wait_batch(folder, deadline)
        arguments = ['tools.fast_adapter.p46_aggregate']
        for phase in ('main', 'supplemental'):
            for task in TASKS:
                directory = ROOT / phase / task
                arguments += ['--cohort', str(directory/'EVALUATION_FREEZE.json'), str(directory/'cells')]
        run(*arguments, '--out', str(ROOT/'summary'))
        write(receipt, dict(start, status='complete', finished=time.time(),
                           summary=binding(ROOT/'summary/RESULTS.json')))
    except Exception as error:
        write(receipt, dict(start, status='failed_preserved', finished=time.time(),
                           error_type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    main()
