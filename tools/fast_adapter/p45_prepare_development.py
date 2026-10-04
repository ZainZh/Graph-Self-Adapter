"""Freeze the exact adopted 144-cell development matrix for a qualified task."""
import argparse
import importlib
import json
from pathlib import Path
import time

from tools.fast_adapter.p45_common import ROOT, TASK_BASE, js, sha
from tools.fast_adapter.p45_evaluate import SOURCES, validate_freeze


def main():
    p = argparse.ArgumentParser(); p.add_argument('--task', required=True, choices=list(TASK_BASE))
    a = p.parse_args(); task = a.task
    gate_path = ROOT / 'engineering' / f'{task}_GATE.json'
    gate = json.loads(gate_path.read_text()); assert gate['passed']
    backend = gate['environment']
    data = ROOT / 'data' / task
    assert (data / 'READY.json').exists()
    # The encoding freeze also binds the precise visual assets and all
    # transitive environment/observer code used for this task's collection.
    data_manifest = json.loads((data / 'train/MANIFEST.json').read_text())
    encode_binding = data_manifest['encoding_freeze']
    assert sha(encode_binding['path']) == encode_binding['sha256']
    encoded = json.loads(Path(encode_binding['path']).read_text())
    sources = dict(encoded['sources'])
    for path in SOURCES + ['tools/fast_adapter/p45_aggregate.py',
                          'tools/fast_adapter/p45_prepare_development.py']:
        sources[path] = sha(path)
    for key in ('module', 'configuration_module', 'controller_module'):
        path = backend[key].replace('.', '/') + '.py'
        sources[path] = sha(path)
    for path, digest in sources.items():
        assert sha(path) == digest, path
    files = dict(encoded['visual_assets'])
    models = {}
    for rep in ('geometry', 'absolute_fusion', 'relative_fusion'):
        models[rep] = {}
        for seed in (11, 22):
            folder = ROOT / 'models' / task / rep / f'seed{seed}'
            diagnostic = json.loads((folder / 'DIAGNOSTIC.json').read_text())
            checkpoint = folder / 'checkpoint.pt'
            assert sha(checkpoint) == diagnostic['checkpoint_sha256']
            files[str(checkpoint)] = sha(checkpoint)
            projection = None
            if rep != 'geometry':
                projection = data_manifest['pca'][rep]
                assert sha(projection['path']) == projection['sha256']
                files[projection['path']] = projection['sha256']
            models[rep][str(seed)] = dict(checkpoint=dict(path=str(checkpoint), sha256=sha(checkpoint)), pca=projection)
    for path in (gate_path, data / 'TRAIN_INPUTS.json', data / 'train/MANIFEST.json', data / 'dev/MANIFEST.json'):
        files[str(path)] = sha(path)
    cells = []
    for rep in models:
        for model_seed in (11, 22):
            for index in range(12):
                for arm in ('fixed', 'continuous'):
                    cells.append(dict(task=task, split='id' if index < 6 else 'geometry',
                        condition_seed=TASK_BASE[task] + 400 + index, rep=rep,
                        model_seed=model_seed, arm=arm,
                        video=rep == 'absolute_fusion' and model_seed == 11 and index in (0, 6)))
    directory = ROOT / 'development' / task
    directory.mkdir(parents=True, exist_ok=False)
    freeze = directory / 'EVALUATION_FREEZE.json'
    assets = importlib.import_module(backend['controller_module']).source_assets()
    frozen = dict(schema='prove.p45-evaluation.v1', status='frozen', root_authorized=True,
        phase='development', timestamp=time.time(), steps_per_episode=240,
        execution=dict(model_dir='assets/dinov3'), sources=sources, files=files,
        robot_assets=assets, engineering_gates={task:dict(path=str(gate_path),sha256=sha(gate_path))},
        environments={task:backend}, tasks={task:dict(models=models)}, cells=cells)
    js(freeze, frozen)
    validate_freeze(freeze)
    jobs = [dict(module='tools.fast_adapter.p45_evaluate', arguments=[
        '--freeze', str(freeze), '--cell-indices', ','.join(map(str, range(rank, len(cells), 4))),
        '--out', str(directory / 'cells'), '--device', 'cuda']) for rank in range(4)]
    js(directory / 'FREEZE.json', dict(status='frozen', root_authorized=True,
        scope='p45_finite_batch', workers=4, sources=sources, jobs=jobs,
        evaluation_freeze=dict(path=str(freeze),sha256=sha(freeze))))
    print(json.dumps(dict(task=task,cells=len(cells),batch_freeze=str(directory/'FREEZE.json'))))


if __name__ == '__main__':
    main()
