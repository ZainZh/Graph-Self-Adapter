"""Prepare P46 prospectively; preserve every P45 gate and frozen artifact.

Main admission is technical qualification only: passed engineering, complete
144-cell development, intact evidence, and correct frozen weights/actions.
The original F success-rate gate is recorded as history and never selects tasks.
All three tasks and the global R hypothesis are declared in every P46 freeze.

Supplemental preparation requires the complete main cohort. It reads seals and
public traces, never main RESULT contents, and binds the already declared cyclic
donors without outcome-based replacement. This script launches no computation.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

import torch

from tools.fast_adapter import p45_aggregate
from tools.fast_adapter import p45_prepare_final as prior
from tools.fast_adapter.p45_common import ROOT as P45_ROOT
from tools.fast_adapter.p45_evaluate import donor_deltas, validate_freeze

ROOT = Path('evaluations/fast_adapter_p46_representation_validation_v01')
TASKS = ('cup', 'insert', 'slider')
BASE = {'cup': 470000, 'insert': 471000, 'slider': 472000}
REPRESENTATIONS = ('geometry', 'absolute_fusion', 'relative_fusion')
MODELS = (11, 22)
CAMPAIGN = 'p46_representation_validation'
NEW_SOURCES = ('tools/fast_adapter/p46_prepare_validation.py',
               'tools/fast_adapter/p46_aggregate.py',
               'tools/fast_adapter/p45_prepare_final.py',
               'tools/fast_adapter/p45_batch_worker.py')
read, write, binding, sha = prior.read, prior.write, prior.binding, prior.sha


def main_cells(task):
    cells = []
    for representation in REPRESENTATIONS:
        for model in MODELS:
            for split, start, count in (('id', BASE[task], 16), ('geometry', BASE[task] + 100, 32)):
                for index in range(count):
                    for arm in ('fixed', 'continuous'):
                        cells.append(dict(task=task, split=split, condition_seed=start + index,
                            rep=representation, model_seed=model, arm=arm,
                            video=representation == 'relative_fusion' and model == 11 and index == 0))
    return cells


def supplemental_scope(task):
    seeds = list(range(BASE[task] + 100, BASE[task] + 108))
    cells = []
    for index, seed in enumerate(seeds):
        for model in MODELS:
            for arm in ('prefix3', 'mismatched'):
                cell = dict(task=task, split='geometry', condition_seed=seed,
                            rep='relative_fusion', model_seed=model, arm=arm, video=False)
                if arm == 'mismatched':
                    cell['donor_assignment'] = dict(task=task, split='geometry',
                        condition_seed=seeds[(index + 1) % 8], rep='relative_fusion', model_seed=model,
                        arm='continuous', quantity='delta_z=z_donor-z0_donor',
                        causal_lookup='latest donor t <= receiver t')
                cells.append(cell)
        cells.append(dict(task=task, split='geometry', condition_seed=seed,
                          rep='geometry', model_seed=None, arm='multiview', video=False))
    return dict(status='declared_not_executable_until_separate_donor_seal_freeze',
                conditions=seeds, n_physical_cells=40, cells=cells,
                donor_rule='First eight OOD conditions, fixed cyclic +1, same R checkpoint',
                donor_access='Sealed public delta-z only; no outcomes, target truth, or future donor t')


def validate_original_gate(path, task, dev_freeze, environment):
    gate = read(path)
    if (gate.get('schema') != 'prove.p45-development-gate.v1' or gate.get('status') != 'complete'
            or gate.get('task') != task or type(gate.get('passed')) is not bool
            or gate.get('environment') != environment
            or gate.get('development_freeze', {}).get('sha256') != sha(dev_freeze)):
        raise ValueError('Original P45 gate identity/backend/development binding mismatch')
    summary = gate['summary']
    if sha(summary['path']) != summary['sha256']:
        raise ValueError('Original P45 gate summary changed')
    return gate


def eligibility(rows, runtime_failures):
    """No success rate, rescue count or representation-dependent selection."""
    checks = dict(all_144_development_cells_verified=len(rows) == 144 and
                  all(row['status'] == 'completed' for row in rows),
                  frozen_weights_and_action_audit_valid=not runtime_failures)
    return dict(passed=all(checks.values()), checks=checks, runtime_audit_failures=runtime_failures,
                success_rate_inclusion_gate=False, original_F_gate_used_for_inclusion=False)


def write_execution_freezes(out, frozen):
    out = Path(out)
    freeze_path = out / 'EVALUATION_FREEZE.json'
    write(freeze_path, frozen)
    validate_freeze(freeze_path)
    count = len(frozen['cells'])
    jobs = [dict(module='tools.fast_adapter.p45_evaluate', arguments=[
        '--freeze', str(freeze_path), '--cell-indices', ','.join(map(str, range(rank, count, 4))),
        '--out', str(out / 'cells'), '--device', 'cuda']) for rank in range(4)]
    batch = out / 'FREEZE.json'
    write(batch, dict(status='frozen', root_authorized=True, scope='p45_finite_batch',
                     campaign=CAMPAIGN, workers=4, sources=frozen['sources'], jobs=jobs,
                     evaluation_freeze=binding(freeze_path)))
    return dict(evaluation_freeze=str(freeze_path), batch_freeze=str(batch), cells=count)


def prepare_main(task, dev_freeze, dev_results, original_gate, plan, out):
    from tools.fast_adapter.p46_aggregate import analysis_contract
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Preserve existing P46 output: {out}')
    # The existing strict completion guard opens no partial RESULT contents.
    prior.complete_development_manifest(task, dev_freeze, dev_results)
    frozen = validate_freeze(dev_freeze)
    plan_binding = binding(plan)
    gate = validate_original_gate(original_gate, task, dev_freeze, frozen['environments'][task])
    expected, checkpoints = prior.checkpoint_hashes(frozen, task)
    rows, provenance, phase = p45_aggregate.load_cohorts([(dev_freeze, dev_results)])
    failures = prior.audit_runtime(rows, frozen, task, expected)
    admission = eligibility(rows, failures)
    summary = p45_aggregate.aggregate(rows, phase, provenance)
    out.mkdir(parents=True, exist_ok=False)
    summary_path = out / 'source_development' / 'RESULTS.json'
    write(summary_path, summary)
    (summary_path.parent / 'REPORT_ZH.md').write_text(p45_aggregate.markdown(summary))
    admission.update(schema='prove.p46-technical-eligibility.v1', task=task, timestamp=time.time(),
        environment=frozen['environments'][task], engineering_gate=frozen['engineering_gates'][task],
        engineering_gate_passed_and_backend_verified=True, development_freeze=binding(dev_freeze),
        original_P45_gate=binding(original_gate), original_P45_F_gate_passed=gate['passed'],
        development_summary=binding(summary_path), checkpoint_parameter_receipts=checkpoints,
        plan=plan_binding, prespecified_tasks=list(TASKS), primary_representation='relative_fusion')
    eligibility_path = out / 'ELIGIBILITY.json'
    write(eligibility_path, admission)
    if not admission['passed']:
        return dict(task=task, prepared=False, eligibility=str(eligibility_path),
                    interpretation='Technical qualification gap; task remains prespecified, never outcome-excluded')
    result = copy.deepcopy(frozen)
    result.update(campaign=CAMPAIGN, cohort_role='main', phase='independent_validation', timestamp=time.time(),
        prespecified_tasks=list(TASKS), analysis_contract=analysis_contract(), cells=main_cells(task),
        plan=plan_binding, technical_eligibility={task: binding(eligibility_path)},
        P45_original_gates={task: binding(original_gate)},
        selection_disclosure=dict(primary='relative_fusion', selected_using='opened cup and insert development',
                                  slider_development_used_for_representation_selection=False,
                                  task_inclusion='All three prespecified; no success-rate admission gate'),
        supplemental_scope=supplemental_scope(task))
    for source in NEW_SOURCES:
        result['sources'][source] = sha(source)
    for path in (Path(plan), Path(dev_freeze), Path(original_gate), summary_path, eligibility_path):
        result['files'][str(path)] = sha(path)
    output = write_execution_freezes(out, result)
    return dict(task=task, prepared=True, eligibility=str(eligibility_path),
                original_P45_F_gate_passed=gate['passed'], **output)


def complete_main_manifest(task, main_freeze, main_results):
    """Require completion by file existence, without reading any main result."""
    frozen = read(main_freeze)
    if (frozen.get('campaign') != CAMPAIGN or frozen.get('cohort_role') != 'main'
            or frozen.get('prespecified_tasks') != list(TASKS)
            or frozen.get('phase') != 'independent_validation'):
        raise ValueError('Expected a frozen P46 main cohort with all three prespecified tasks')
    if frozen['cells'] != main_cells(task) or frozen['supplemental_scope'] != supplemental_scope(task):
        raise ValueError('Main cell list or supplemental donor assignments differ from the prospective scope')
    required = ('RESULT.json', 'SEAL.json', 'PRE_RUN.json', 'TRACE.npy', 'OBS.npy', 'TERMINAL.npy')
    for index in range(576):
        directory = Path(main_results) / f'cell_{index:06d}'
        if any(not (directory / name).is_file() for name in required) or (directory / 'ABORT.json').exists():
            raise RuntimeError('Main cohort is incomplete; no main RESULT opened and no donor selected')
    return frozen


def audit_main_seals(frozen, main_freeze, main_results, parameter_hashes):
    """Validate all trajectory seals and checkpoints without success labels."""
    freeze_digest = sha(main_freeze)
    for index, cell in enumerate(frozen['cells']):
        directory = Path(main_results) / f'cell_{index:06d}'
        seal, pre = read(directory / 'SEAL.json'), read(directory / 'PRE_RUN.json')
        if pre['cell'] != cell or pre['freeze']['sha256'] != freeze_digest:
            raise ValueError('Main pre-run identity differs from its frozen cell')
        if not {'TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'PRE_RUN.json'} <= set(seal['files']):
            raise ValueError('Incomplete main trajectory seal')
        for name, digest in seal['files'].items():
            if Path(name).name != name or sha(directory / name) != digest:
                raise ValueError('Main sealed evidence changed')
        expected = parameter_hashes[cell['rep'], cell['model_seed']]
        if seal['neural_parameters_before'] != expected or seal['neural_parameters_after'] != expected:
            raise ValueError('Main runtime parameters differ from their fixed checkpoint')
        candidate = frozen['tasks'][cell['task']]['models'][cell['rep']][str(cell['model_seed'])]
        if pre['candidate'] != candidate:
            raise ValueError('Main candidate binding mismatch')


def bind_supplemental_cells(task, frozen, main_results):
    lookup = {p45_aggregate._identity(cell): index for index, cell in enumerate(frozen['cells'])}
    cells, artifacts = [], {}
    for declared in frozen['supplemental_scope']['cells']:
        cell = copy.deepcopy(declared)
        assignment = cell.pop('donor_assignment', None)
        if assignment is not None:
            identity = tuple(assignment[key] for key in p45_aggregate.IDENTITY)
            directory = Path(main_results) / f'cell_{lookup[identity]:06d}'
            trace, seal = directory / 'TRACE.npy', directory / 'SEAL.json'
            cell['donor'] = dict(binding(trace), condition_seed=assignment['condition_seed'], seal=binding(seal))
            candidate = frozen['tasks'][task]['models']['relative_fusion'][str(cell['model_seed'])]
            # The frozen runtime helper verifies donor task/checkpoint/identity,
            # then extracts only delta-z. It never reads a donor result label.
            donor_deltas(cell, candidate)
            artifacts[str(trace)] = sha(trace)
            artifacts[str(seal)] = sha(seal)
            artifacts[str(directory / 'PRE_RUN.json')] = sha(directory / 'PRE_RUN.json')
        cells.append(cell)
    return cells, artifacts


def prepare_supplemental(task, main_freeze, main_results, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Preserve existing P46 output: {out}')
    complete_main_manifest(task, main_freeze, main_results)
    frozen = validate_freeze(main_freeze)
    hashes, _ = prior.checkpoint_hashes(frozen, task)
    audit_main_seals(frozen, main_freeze, main_results, hashes)
    cells, artifacts = bind_supplemental_cells(task, frozen, main_results)
    supplement = copy.deepcopy(frozen)
    supplement.update(cohort_role='supplemental', cells=cells, timestamp=time.time(),
                      main_cohort=binding(main_freeze), donor_selection_uses_outcomes=False,
                      donor_binding_rule='Unchanged prospective cyclic +1 assignment; causal delta-z only')
    supplement['files'].update(artifacts)
    supplement['files'][str(main_freeze)] = sha(main_freeze)
    out.mkdir(parents=True, exist_ok=False)
    output = write_execution_freezes(out, supplement)
    return dict(task=task, prepared=True, main_result_contents_opened=False,
                donor_assignment_changed=False, **output)


def main():
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('main', 'supplemental'), default='main')
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--dev-freeze')
    parser.add_argument('--dev-results')
    parser.add_argument('--original-gate')
    parser.add_argument('--plan', default=str(ROOT / 'PLAN.md'))
    parser.add_argument('--main-freeze')
    parser.add_argument('--main-results')
    parser.add_argument('--out')
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out) if args.out else ROOT / args.mode / args.task
    if args.mode == 'main':
        dev = Path(args.dev_freeze) if args.dev_freeze else P45_ROOT / 'development' / args.task / 'EVALUATION_FREEZE.json'
        results = Path(args.dev_results) if args.dev_results else dev.parent / 'cells'
        gate = Path(args.original_gate) if args.original_gate else dev.parent / 'original_gate' / 'GATE.json'
        result = prepare_main(args.task, dev, results, gate, args.plan, out)
    else:
        freeze = Path(args.main_freeze) if args.main_freeze else ROOT / 'main' / args.task / 'EVALUATION_FREEZE.json'
        results = Path(args.main_results) if args.main_results else freeze.parent / 'cells'
        result = prepare_supplemental(args.task, freeze, results, out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
