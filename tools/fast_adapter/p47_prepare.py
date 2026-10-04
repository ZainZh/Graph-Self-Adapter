"""Prepare result-free P47 batches; this module never launches a rollout.

P45 checkpoints, visual assets, engineering qualifications and P46 physical
backends are inherited by hash. Opened P45 development conditions are used only
for implementation validation. Fresh conditions require the root's complete
comparison-validity review; task success never selects tasks or candidates.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from prove.fast_adapter.p47_neural import REPRESENTATIONS, SOLVER_CONTRACT
from tools.fast_adapter.p45_common import ROOT as P45_ROOT, sha
from tools.fast_adapter.p47_aggregate import ANALYSIS_SOURCES, analysis_contract
from tools.fast_adapter.p47_evaluate import (
    CAMPAIGN, MAIN_BASE, SOURCES, TASKS, VISION_FILES, binding, check_binding,
    make_cells, validate_development_eligibility, validate_freeze,
)

ROOT = Path('evaluations/fast_adapter_p47_representation_solver_v01')
P46_ROOT = Path('evaluations/fast_adapter_p46_representation_validation_v01')
BASE = MAIN_BASE
MODELS = (11, 22)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main_cells(task):
    return make_cells(task, 'independent_validation')


def development_cells(task):
    return make_cells(task, 'development')


def complete_development_manifest(root):
    """Check all 78 filesets before final preparation; do not open outcomes."""
    required = ('PRE_RUN.json', 'TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'SEAL.json', 'RESULT.json')
    paths = {}
    for task in TASKS:
        path = Path(root) / 'development' / task / 'EVALUATION_FREEZE.json'
        frozen = read(path)
        if (frozen.get('schema') != 'prove.p47-evaluation.v1'
                or frozen.get('campaign') != CAMPAIGN or frozen.get('phase') != 'development'
                or frozen.get('cells') != development_cells(task)):
            raise ValueError('Development does not match the prescribed opened 26-cell task matrix')
        for index in range(26):
            directory = path.parent / 'cells' / f'cell_{index:06d}'
            if (any(not (directory / name).is_file() for name in required)
                    or (directory / 'ABORT.json').exists()):
                raise RuntimeError('P47 development incomplete; no RESULT opened and no final freeze created')
        paths[task] = path
    return paths


def inherit_task(task, p46_freeze, p45_development):
    """Copy only runtime facts, not P46 hypotheses, gates, results or donors."""
    previous = read(p46_freeze)
    if (previous.get('schema') != 'prove.p45-evaluation.v1'
            or previous.get('campaign') != 'p46_representation_validation'
            or previous.get('cohort_role') != 'main' or previous.get('status') != 'frozen'
            or previous.get('root_authorized') is not True
            or previous.get('phase') != 'independent_validation'
            or set(previous.get('tasks', {})) != {task}):
        raise ValueError('Expected the historical single-task P46 main freeze')
    development = read(p45_development)
    opened = list(dict.fromkeys(cell['condition_seed'] for cell in development['cells']
                               if cell['task'] == task and cell['split'] == 'geometry'))
    wanted = list(dict.fromkeys(cell['condition_seed'] for cell in development_cells(task)))
    if (development.get('schema') != 'prove.p45-evaluation.v1'
            or development.get('phase') != 'development' or opened[:2] != wanted):
        raise ValueError('P47 development must reuse the first two declared P45 geometry conditions')
    for key in ('environments', 'tasks', 'robot_assets', 'engineering_gates', 'steps_per_episode', 'execution'):
        if development[key] != previous[key]:
            raise ValueError('Historical P45/P46 runtime identity mismatch')
    result = {key: copy.deepcopy(previous[key]) for key in (
        'steps_per_episode', 'execution', 'robot_assets', 'engineering_gates', 'environments', 'tasks')}
    result['tasks'][task]['models'] = {
        rep: copy.deepcopy(previous['tasks'][task]['models'][rep]) for rep in REPRESENTATIONS}
    if any(set(result['tasks'][task]['models'][rep]) != {'11', '22'} for rep in REPRESENTATIONS):
        raise ValueError('P47 requires both frozen P45 model seeds for G and R')
    keep = set(VISION_FILES)
    keep.update(str(Path(result['execution']['model_dir']) / name)
                for name in ('model.safetensors', 'config.json', 'preprocessor_config.json'))
    keep.add(result['engineering_gates'][task]['path'])
    for models in result['tasks'][task]['models'].values():
        for candidate in models.values():
            keep.add(candidate['checkpoint']['path'])
            if candidate['pca'] is not None:
                keep.add(candidate['pca']['path'])
    if not keep <= set(previous['files']):
        raise ValueError('Historical freeze is missing inherited runtime files')
    result['files'] = {path: previous['files'][path] for path in sorted(keep)}
    for path, digest in result['files'].items():
        if sha(path) != digest:
            raise ValueError(f'Inherited frozen runtime file changed: {path}')
    # Preserve the historical runtime closure and its hashes. Old preparation /
    # analysis modules are not P47 candidates; bind the P47 tools separately.
    obsolete = {'tools/fast_adapter/p46_prepare_validation.py', 'tools/fast_adapter/p46_aggregate.py',
                'tools/fast_adapter/p45_prepare_final.py', 'tools/fast_adapter/p45_prepare_development.py'}
    result['sources'] = {path: digest for path, digest in previous['sources'].items() if path not in obsolete}
    for path, digest in result['sources'].items():
        if sha(path) != digest:
            raise ValueError(f'Inherited frozen runtime source changed: {path}')
    for path in dict.fromkeys([*SOURCES, *ANALYSIS_SOURCES]):
        result['sources'][path] = sha(path)
    result['source_p46_freeze'] = binding(p46_freeze)
    result['source_p45_development'] = binding(p45_development)
    for item in (result['source_p46_freeze'], result['source_p45_development']):
        result['files'][item['path']] = item['sha256']
    return result


def write_execution_freezes(out, frozen, workers=4):
    """Use the existing finite worker protocol with a new evaluator module."""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('P47 supports one to four finite workers')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    freeze_path = out / 'EVALUATION_FREEZE.json'
    write(freeze_path, frozen)
    validate_freeze(freeze_path)
    jobs = [dict(module='tools.fast_adapter.p47_evaluate', arguments=[
        '--freeze', str(freeze_path), '--cell-indices', ','.join(map(str, range(rank, len(frozen['cells']), workers))),
        '--out', str(out / 'cells'), '--device', 'cuda']) for rank in range(workers)]
    batch = out / 'FREEZE.json'
    write(batch, dict(status='frozen', root_authorized=True, scope='p45_finite_batch',
                     campaign=CAMPAIGN, workers=workers, sources=frozen['sources'], jobs=jobs,
                     evaluation_freeze=binding(freeze_path)))
    return dict(evaluation_freeze=str(freeze_path), batch_freeze=str(batch), cells=len(frozen['cells']))


def prepare(task, phase, root=ROOT, workers=4, p46_root=P46_ROOT, p45_root=P45_ROOT):
    if task not in TASKS or phase not in ('development', 'independent_validation'):
        raise ValueError('Unknown P47 task or phase')
    root = Path(root)
    out = root / ('development' if phase == 'development' else 'main') / task
    if out.exists():
        raise FileExistsError(f'Preserve existing P47 output: {out}')
    plan = root / 'PLAN.md'
    contract = root / 'ANALYSIS_CONTRACT.json'
    if read(contract) != analysis_contract():
        raise ValueError('P47 analysis file differs from the fixed implementation contract')
    eligibility_path = root / 'DEVELOPMENT_ELIGIBILITY.json'
    if phase == 'independent_validation':
        if not eligibility_path.is_file():
            raise ValueError('Missing root DEVELOPMENT_ELIGIBILITY; no independent freeze permitted')
        complete_development_manifest(root)
    frozen = inherit_task(task, Path(p46_root) / 'main' / task / 'EVALUATION_FREEZE.json',
                          Path(p45_root) / 'development' / task / 'EVALUATION_FREEZE.json')
    frozen.update(schema='prove.p47-evaluation.v1', status='frozen', root_authorized=True,
        campaign=CAMPAIGN, phase=phase, cohort_role='development' if phase == 'development' else 'main',
        timestamp=time.time(), prespecified_tasks=list(TASKS), cells=make_cells(task, phase),
        solver_contract=copy.deepcopy(SOLVER_CONTRACT), analysis_contract=read(contract),
        analysis_contract_file=binding(contract), plan=binding(plan),
        condition_contract='opened_P45_development' if phase == 'development' else 'fresh_geometry_OOD',
        task_inclusion='All three prespecified; no success-rate admission or outcome-dependent substitution')
    for path in (plan, contract):
        frozen['files'][str(path)] = sha(path)
    if phase == 'independent_validation':
        item = binding(eligibility_path)
        frozen['development_eligibility'] = item
        frozen['files'][item['path']] = item['sha256']
        reviewed = validate_development_eligibility(item, frozen)
        for review in reviewed['development'].values():
            for key in ('evaluation_freeze', 'evidence'):
                artifact = review[key]
                frozen['files'][artifact['path']] = artifact['sha256']
    return dict(task=task, phase=phase, prepared=True, **write_execution_freezes(out, frozen, workers))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('development', 'independent_validation'), required=True)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--task', choices=TASKS, help='Omit to prepare all three prespecified tasks')
    parser.add_argument('--p46-root', default=str(P46_ROOT))
    parser.add_argument('--p45-root', default=str(P45_ROOT))
    args = parser.parse_args()
    tasks = (args.task,) if args.task else TASKS
    results = [prepare(task, args.phase, args.root, args.workers, args.p46_root, args.p45_root) for task in tasks]
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
