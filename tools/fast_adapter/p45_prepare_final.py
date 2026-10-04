"""Read a COMPLETE P45 task development cohort and prepare, never run, final.

Admission lists all 144 expected cells and checks completion files before any
RESULT is opened. Failed development gates produce a sealed negative decision
and no independent rollout freeze. Historical files are only read.

The optional supplemental scope is declared before main evaluation: the first
eight OOD conditions, F prefix3/mismatched for both checkpoints, and exactly one
model-free multiview cell per condition. Mismatched donor assignment is a fixed
one-step cyclic shift of these eight conditions; it transfers only the donor's
causal delta-z. A later supplemental freeze must bind the actual donor seals.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from prove.fast_adapter.p45_neural import load_model
from tools.fast_adapter import p45_aggregate
from tools.fast_adapter.p45_common import ROOT, TASK_BASE, sha
from tools.fast_adapter.p45_evaluate import parameters_sha, validate_freeze

SOURCE = 'tools/fast_adapter/p45_prepare_final.py'
REPRESENTATIONS = ('geometry', 'absolute_fusion', 'relative_fusion')
MODELS = (11, 22)
FINAL_BASE = {'cup': 458000, 'insert': 459000, 'slider': 460000}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def binding(path):
    return dict(path=str(path), sha256=sha(path))


def development_cells(task):
    return [dict(task=task, split='id' if index < 6 else 'geometry',
                 condition_seed=TASK_BASE[task] + 400 + index, rep=rep, model_seed=model,
                 arm=arm, video=False)
            for rep in REPRESENTATIONS for model in MODELS for index in range(12)
            for arm in ('fixed', 'continuous')]


def complete_development_manifest(task, freeze_path, results_root):
    """Only the freeze and filesystem metadata are read until all cells exist."""
    frozen = read(freeze_path)
    if (frozen.get('schema') != 'prove.p45-evaluation.v1' or frozen.get('phase') != 'development'
            or frozen.get('status') != 'frozen' or frozen.get('root_authorized') is not True
            or set(frozen.get('tasks', {})) != {task}):
        raise ValueError('Expected a single-task root-authorized P45 development freeze')
    expected = {p45_aggregate._identity(cell) for cell in development_cells(task)}
    actual = [p45_aggregate._identity(cell) for cell in frozen['cells']]
    if len(actual) != 144 or len(set(actual)) != 144 or set(actual) != expected:
        raise ValueError('Development must contain exactly the adopted 144 cells, including all 12 conditions')
    required = ('RESULT.json', 'SEAL.json', 'PRE_RUN.json', 'TRACE.npy', 'OBS.npy', 'TERMINAL.npy')
    incomplete = []
    for index in range(144):
        directory = Path(results_root) / f'cell_{index:06d}'
        missing = [name for name in required if not (directory / name).is_file()]
        if (directory / 'ABORT.json').exists():
            missing.append('ABORT present')
        if missing:
            incomplete.append(dict(cell_index=index, missing=missing))
    if incomplete:
        raise RuntimeError(f'Development incomplete in {len(incomplete)}/144 cells; no RESULT opened')
    return frozen


def checkpoint_hashes(frozen, task):
    models = frozen['tasks'][task]['models']
    if set(models) != set(REPRESENTATIONS) or any(set(models[rep]) != {'11', '22'} for rep in REPRESENTATIONS):
        raise ValueError('Development must bind all six prescribed checkpoints')
    hashes, receipts = {}, []
    for rep in REPRESENTATIONS:
        for seed in MODELS:
            candidate = models[rep][str(seed)]
            model, package = load_model(candidate['checkpoint']['path'], 'cpu')
            if (model.task != task or model.representation != rep or package['seed'] != seed
                    or package['updates'] != 2000 or model.width != 96 or model.latent_dim != 8):
                raise ValueError('Checkpoint task, representation, model seed or training budget mismatch')
            if rep != 'geometry' and package['pca'][rep]['sha256'] != candidate['pca']['sha256']:
                raise ValueError('Development projection differs from checkpoint training projection')
            digest = parameters_sha(model)
            hashes[rep, seed] = digest
            receipts.append(dict(rep=rep, model_seed=seed, checkpoint=candidate['checkpoint'],
                                 pca=candidate.get('pca'), runtime_parameters_sha256=digest))
    return hashes, receipts


def audit_runtime(rows, frozen, task, expected_hashes):
    failures = []
    for row in rows:
        if row['status'] != 'completed':
            failures.append(dict(cell_index=row['cell_index'], reason='Evidence seal or identity failed'))
            continue
        directory = Path(row['path'])
        cell, result = row['cell'], row['result']
        expected = expected_hashes[cell['rep'], cell['model_seed']]
        seal, pre = read(directory / 'SEAL.json'), read(directory / 'PRE_RUN.json')
        candidate = frozen['tasks'][task]['models'][cell['rep']][str(cell['model_seed'])]
        if (seal['neural_parameters_before'] != expected or seal['neural_parameters_after'] != expected
                or result.get('neural_weights_unchanged') is not True or pre['candidate'] != candidate):
            failures.append(dict(cell_index=row['cell_index'], reason='Runtime/checkpoint parameter binding mismatch'))
        trace = np.load(directory / 'TRACE.npy', allow_pickle=True)
        if len(trace) != frozen['steps_per_episode']:
            failures.append(dict(cell_index=row['cell_index'], reason='Physical action budget mismatch'))
        adaptations = [entry['adaptation'] for entry in trace if entry.get('adaptation') is not None]
        for metric in ('same_state_action_position_delta', 'same_state_action_rotation_delta'):
            values = [entry[metric] for entry in adaptations]
            observed = max(values + [0.])
            reported = result.get('max_' + metric)
            if (not np.isfinite(values).all() or reported is None or not np.isfinite(reported)
                    or not np.isclose(observed, reported, atol=1e-9, rtol=1e-7)):
                failures.append(dict(cell_index=row['cell_index'], reason='Stored action effect does not match trace', metric=metric))
    return failures


def assess_development(rows, task, runtime_failures):
    groups = {}
    for model in MODELS:
        selected = {row['cell']['condition_seed']: {} for row in rows
                    if row['cell']['rep'] == 'absolute_fusion' and row['cell']['model_seed'] == model
                    and row['cell']['split'] == 'geometry'}
        for row in rows:
            cell = row['cell']
            if cell['rep'] == 'absolute_fusion' and cell['model_seed'] == model and cell['split'] == 'geometry':
                selected[cell['condition_seed']][cell['arm']] = row
        fixed = online = rescue = harm = observed = 0
        max_position = max_rotation = 0.
        for pair in selected.values():
            if (set(pair) != {'fixed', 'continuous'}
                    or any(entry['status'] != 'completed' for entry in pair.values())):
                continue
            observed += 1
            a, b = pair['continuous']['result'], pair['fixed']['result']
            online += int(a['validated_success'])
            fixed += int(b['validated_success'])
            rescue += int(a['validated_success'] and not b['validated_success'])
            harm += int(b['validated_success'] and not a['validated_success'])
            max_position = max(max_position, a['max_same_state_action_position_delta'])
            max_rotation = max(max_rotation, a['max_same_state_action_rotation_delta'])
        groups[str(model)] = dict(n_planned=6, n_observed_pairs=observed,
            fixed_validated_successes=fixed, online_validated_successes=online,
            paired_rescue=rescue, paired_harm=harm,
            no_regression=bool(observed == 6 and online >= fixed),
            max_online_action_position_delta=max_position, max_online_action_rotation_delta=max_rotation)
    checks = dict(
        all_144_seals_and_outcomes_valid=len(rows) == 144 and all(row['status'] == 'completed' for row in rows),
        all_runtime_weights_and_trace_action_effects_verified=not runtime_failures,
        both_models_ood_no_regression=all(group['no_regression'] for group in groups.values()),
        at_least_one_paired_validated_rescue=sum(group['paired_rescue'] for group in groups.values()) >= 1,
        online_action_delta_exceeds_1e_6=any(max(group['max_online_action_position_delta'],
            group['max_online_action_rotation_delta']) > 1e-6 for group in groups.values()),
    )
    return dict(task=task, passed=all(checks.values()), checks=checks, main_F_ood_by_model=groups,
                runtime_audit_failures=runtime_failures,
                failed_checks=[key for key, passed in checks.items() if not passed],
                denominator_rule='All 144 physical cells retained; mechanical anomalies are validated failures, never filtered',
                action_gate='Any F online OOD position or rotation effect >1e-6; both model peaks reported')


def final_cells(task):
    base = FINAL_BASE[task]
    cells = []
    for rep in REPRESENTATIONS:
        for model in MODELS:
            for split, start, count in (('id', base, 16), ('geometry', base+100, 32)):
                for index in range(count):
                    for arm in ('fixed', 'continuous'):
                        cells.append(dict(task=task, split=split, condition_seed=start+index,
                            rep=rep, model_seed=model, arm=arm,
                            video=rep == 'absolute_fusion' and model == 11 and index == 0))
    return cells


def supplemental_scope(task):
    seeds = list(range(FINAL_BASE[task]+100, FINAL_BASE[task]+108))
    cells = []
    for index, seed in enumerate(seeds):
        for model in MODELS:
            for arm in ('prefix3', 'mismatched'):
                cell = dict(task=task, split='geometry', condition_seed=seed,
                            rep='absolute_fusion', model_seed=model, arm=arm, video=False)
                if arm == 'mismatched':
                    cell['donor_assignment'] = dict(task=task, split='geometry', condition_seed=seeds[(index+1) % 8],
                        rep='absolute_fusion', model_seed=model, arm='continuous',
                        quantity='delta_z=z_donor-z0_donor', causal_lookup='latest donor t <= receiver t')
                cells.append(cell)
        cells.append(dict(task=task, split='geometry', condition_seed=seed, rep='geometry',
                          model_seed=None, arm='multiview', video=False))
    return dict(status='declared_not_executable_until_separate_donor_seal_freeze',
                conditions=seeds, n_physical_cells=40, cells=cells,
                donor_rule='Fixed first-eight OOD cyclic shift +1; same task/model/representation, independent condition',
                donor_access='Only sealed public continuous-arm delta-z; no target truth, no outcomes, no future donor t')


def prepare(task, dev_freeze, dev_results, out, gate_only=False):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Preserve existing preparation output: {out}')
    complete_development_manifest(task, dev_freeze, dev_results)
    # This expensive hash/checkpoint audit happens only after the no-peeking
    # completion guard. It does not initialize a simulator or a visual backbone.
    frozen = validate_freeze(dev_freeze)
    hashes, checkpoint_receipts = checkpoint_hashes(frozen, task)
    rows, provenance, phase = p45_aggregate.load_cohorts([(dev_freeze, dev_results)])
    failures = audit_runtime(rows, frozen, task, hashes)
    summary = p45_aggregate.aggregate(rows, phase, provenance)
    gate = assess_development(rows, task, failures)
    out.mkdir(parents=True, exist_ok=False)
    summary_dir = out / 'dev_summary'
    write(summary_dir / 'RESULTS.json', summary)
    (summary_dir / 'REPORT_ZH.md').write_text(p45_aggregate.markdown(summary))
    gate.update(schema='prove.p45-development-gate.v1', status='complete', timestamp=time.time(),
                environment=frozen['environments'][task], engineering_gate=frozen['engineering_gates'][task],
                engineering_gate_passed_and_backend_verified=True,
                development_freeze=binding(dev_freeze), summary=binding(summary_dir / 'RESULTS.json'),
                checkpoint_parameter_receipts=checkpoint_receipts, preparation_source=binding(SOURCE),
                final_preparation_requested=not gate_only)
    write(out / 'GATE.json', gate)
    if not gate['passed'] or gate_only:
        return dict(task=task, passed=gate['passed'], gate=str(out / 'GATE.json'), final_prepared=False,
                    gate_only=gate_only, failed_checks=gate['failed_checks'])
    final = copy.deepcopy(frozen)
    final.update(phase='independent_validation', timestamp=time.time(), cells=final_cells(task),
                 development_gates={task: binding(out / 'GATE.json')},
                 supplemental_scope=supplemental_scope(task))
    final['sources'][SOURCE] = sha(SOURCE)
    for path in (Path(dev_freeze), out / 'GATE.json', summary_dir / 'RESULTS.json'):
        final['files'][str(path)] = sha(path)
    freeze_path = out / 'EVALUATION_FREEZE.json'
    write(freeze_path, final)
    validate_freeze(freeze_path)
    jobs = [dict(module='tools.fast_adapter.p45_evaluate', arguments=[
        '--freeze', str(freeze_path), '--cell-indices', ','.join(map(str, range(rank, 576, 4))),
        '--out', str(out / 'cells'), '--device', 'cuda']) for rank in range(4)]
    batch_path = out / 'FREEZE.json'
    write(batch_path, dict(status='frozen', root_authorized=True, scope='p45_finite_batch',
                          workers=4, sources=final['sources'], jobs=jobs,
                          evaluation_freeze=binding(freeze_path)))
    return dict(task=task, passed=True, gate=str(out / 'GATE.json'), final_prepared=True,
                main_cells=576, declared_supplemental_cells=40, batch_freeze=str(batch_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=list(TASK_BASE))
    parser.add_argument('--dev-freeze')
    parser.add_argument('--dev-results')
    parser.add_argument('--out')
    parser.add_argument('--gate-only', action='store_true',
                        help='Audit and record the original task gate only, even when it passes')
    args = parser.parse_args()
    torch.set_num_threads(1)
    dev_freeze = Path(args.dev_freeze) if args.dev_freeze else ROOT / 'development' / args.task / 'EVALUATION_FREEZE.json'
    dev_results = Path(args.dev_results) if args.dev_results else dev_freeze.parent / 'cells'
    out = Path(args.out) if args.out else ROOT / 'independent' / args.task
    print(json.dumps(prepare(args.task, dev_freeze, dev_results, out, gate_only=args.gate_only), ensure_ascii=False))


if __name__ == '__main__':
    main()
