"""P46 prespecified three-task aggregation using immutable P45 evidence checks.

The runner still writes prove.p45-evaluation.v1. P46 metadata prevents accidental
pooling with P45, and all three tasks must be complete before formal inference.
Supplemental R controls are descriptive; a multiview reference is one physical
cell with model_seed=None even when compared to both neural checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tools.fast_adapter import p45_aggregate as p45

ROOT = Path('evaluations/fast_adapter_p46_representation_validation_v01')
CAMPAIGN = 'p46_representation_validation'
TASKS = p45.TASKS
MODELS = p45.MODELS
REPRESENTATIONS = ('geometry', 'absolute_fusion', 'relative_fusion')
FINAL_BASE = {'cup': 470000, 'insert': 471000, 'slider': 472000}
BOOTSTRAP_SAMPLES = p45.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = p45.BOOTSTRAP_SEED
EVALUATION_SOURCES = p45.EVALUATION_SOURCES
grouped_counts = p45.grouped_counts
signflip_pvalue = p45.signflip_pvalue
bootstrap_interval = p45.bootstrap_interval
holm = p45.holm


def analysis_contract():
    """The exact result-free contract also embedded by P46 preparation."""
    return dict(
        schema='prove.p46-analysis-contract.v1',
        prespecified_tasks=list(TASKS), primary_representation='relative_fusion',
        primary_family_size=3, secondary_family_size=6,
        primary_endpoint='validated_success', native_audit_endpoint='sustained_native_success',
        denominator='All declared physical cells, including mechanical anomalies and unknown outcomes',
        unknown_policy='Missing/aborted/integrity-invalid are unknown; report rate bounds, never measured failures',
        sampling_unit='Independent condition; average fixed model seeds 11/22 within each condition',
        primary_family='3 task-specific OOD R-continuous minus R-fixed; all three tasks required',
        secondary_family='6 task-specific OOD continuous R-F and F-G comparisons; all three tasks required',
        pvalue='Exact two-sided condition paired sign-flip; separate Holm families of 3 and 6',
        ci='Unadjusted percentile 95% condition bootstrap; not a family-adjusted interval',
        bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
        required_id_conditions_per_task=16, required_ood_conditions_per_task=32,
        model_seeds=list(MODELS), condition_seed_bases=dict(FINAL_BASE),
        inference_scope='Conditional on two fixed trained checkpoints; complete prespecified three-task matrix',
        supplemental='First eight OOD conditions: R prefix3/mismatched and geometry multiview; descriptive only',
        multiview='One physical cell per condition, model_seed=null; comparisons reuse it without duplicating its count')


def expected_main_identities(task, split):
    start, count = (FINAL_BASE[task], 16) if split == 'id' else (FINAL_BASE[task]+100, 32)
    return {(task, split, seed, rep, model, arm)
            for seed in range(start, start+count) for rep in REPRESENTATIONS
            for model in MODELS for arm in ('fixed', 'continuous')}


def expected_supplemental_identities(task):
    identities = set()
    for seed in range(FINAL_BASE[task]+100, FINAL_BASE[task]+108):
        identities.add((task, 'geometry', seed, 'geometry', None, 'multiview'))
        identities.update((task, 'geometry', seed, 'relative_fusion', model, arm)
                          for model in MODELS for arm in ('prefix3', 'mismatched'))
    return identities


def _validate_scope(frozen):
    """Check P46 metadata before delegating any evidence/result read to P45."""
    if (frozen.get('campaign') != CAMPAIGN
            or frozen.get('prespecified_tasks') != list(TASKS)
            or frozen.get('analysis_contract') != analysis_contract()):
        raise ValueError('Expected the frozen P46 campaign, all three prespecified tasks and exact analysis contract')
    if frozen.get('cohort_role') not in ('main', 'supplemental'):
        raise ValueError('P46 cohort_role must be main or supplemental')
    tasks = set(frozen.get('tasks', {}))
    if len(tasks) != 1 or not tasks <= set(TASKS):
        raise ValueError('Each P46 freeze must describe exactly one prespecified task')
    task, = tasks
    scope = frozen.get('supplemental_scope', {})
    seeds = list(range(FINAL_BASE[task]+100, FINAL_BASE[task]+108))
    expected_supplemental = expected_supplemental_identities(task)
    try:
        scoped = [p45._identity(cell) for cell in scope['cells']]
    except (KeyError, TypeError) as exc:
        raise ValueError('Missing P46 supplemental scope cells') from exc
    if (scope.get('conditions') != seeds or scope.get('n_physical_cells') != 40
            or len(scoped) != 40 or set(scoped) != expected_supplemental):
        raise ValueError('P46 supplemental scope must prescribe the first eight OOD R controls and one multiview each')
    for cell in scope['cells']:
        if cell['arm'] == 'mismatched':
            assignment = cell.get('donor_assignment', {})
            expected_donor = seeds[(seeds.index(cell['condition_seed'])+1) % len(seeds)]
            if (assignment.get('task') != task or assignment.get('split') != 'geometry'
                    or assignment.get('condition_seed') != expected_donor
                    or assignment.get('rep') != 'relative_fusion'
                    or assignment.get('model_seed') != cell['model_seed']
                    or assignment.get('arm') != 'continuous'
                    or assignment.get('quantity') != 'delta_z=z_donor-z0_donor'
                    or assignment.get('causal_lookup') != 'latest donor t <= receiver t'):
                raise ValueError('P46 mismatched scope requires the prespecified causal R donor cycle')
    allowed = (expected_main_identities(task, 'id') | expected_main_identities(task, 'geometry')
               if frozen['cohort_role'] == 'main' else expected_supplemental)
    if (frozen['cohort_role'] == 'supplemental'
            and {p45._identity(cell) for cell in frozen['cells']} != expected_supplemental):
        raise ValueError('An executable supplemental freeze must contain all 40 prespecified cells')
    for cell in frozen['cells']:
        if p45._identity(cell) not in allowed:
            raise ValueError('Cell is outside its P46 task, role, seed, representation or model scope')
        if frozen['cohort_role'] == 'supplemental' and cell['arm'] == 'mismatched':
            donor = cell.get('donor', {})
            expected_donor = seeds[(seeds.index(cell['condition_seed'])+1) % len(seeds)]
            if (donor.get('condition_seed') != expected_donor or not donor.get('path')
                    or not donor.get('sha256') or not donor.get('seal', {}).get('path')
                    or not donor.get('seal', {}).get('sha256') or 'donor_assignment' in cell):
                raise ValueError('Executable mismatched cells must bind their prespecified sealed donor')


def load_cohorts(cohorts):
    cohorts = list(cohorts)
    metadata = []
    for freeze_path, _ in cohorts:
        frozen = p45._read(freeze_path)
        _validate_scope(frozen)
        metadata.append(dict(campaign=CAMPAIGN, cohort_role=frozen['cohort_role'],
                             prespecified_tasks=list(TASKS),
                             supplemental_scope=frozen['supplemental_scope']))
    rows, provenance, phase = p45.load_cohorts(cohorts)
    for item, extra in zip(provenance, metadata):
        item.update(extra)
    return rows, provenance, phase


def paired_counts(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[tuple(row['cell'][key] for key in p45.GROUP)][row['cell']['condition_seed']] = row
    comparisons = []
    for key, left in sorted(groups.items(), key=lambda item: repr(item[0])):
        task, split, rep, model, arm = key
        if arm != 'continuous':
            continue
        comparators = [('fixed', rep, model)]
        if rep == 'relative_fusion':
            comparators += [('prefix3', rep, model), ('mismatched', rep, model),
                            ('multiview', 'geometry', None)]
        for other_arm, other_rep, other_model in comparators:
            right = groups.get((task, split, other_rep, other_model, other_arm), {})
            if not right and other_arm != 'fixed':
                continue
            conditions = set(left) | set(right) if other_arm == 'fixed' else set(right)
            comparisons.append(dict(task=task, split=split, rep=rep, model_seed=model,
                left_arm='continuous', right_arm=other_arm, right_rep=other_rep,
                right_model_seed=other_model,
                reference_reused_without_extra_physical_cells=other_arm == 'multiview',
                **p45.pair_summary(left, right, conditions)))
    return comparisons


def supplemental_status(rows):
    tasks = []
    for task in TASKS:
        expected = expected_supplemental_identities(task)
        declared = [row for row in rows if p45._identity(row['cell']) in expected]
        completed = sum(row['status'] == 'completed' for row in declared)
        tasks.append(dict(task=task, n_prespecified_cells=40, n_declared_cells=len(declared),
            n_completed_cells=completed, n_not_started_cells=40-len(declared),
            status=('not_started_pending' if not declared else
                    'complete' if completed == 40 else 'incomplete')))
    declared = sum(task['n_declared_cells'] for task in tasks)
    completed = sum(task['n_completed_cells'] for task in tasks)
    return dict(n_prespecified_cells=120, n_declared_cells=declared,
        n_completed_cells=completed, n_not_started_cells=120-declared,
        status=('not_started_pending' if not declared else 'complete' if completed == 120
                else 'partially_declared_pending' if declared == completed else 'incomplete'),
        formal_family='descriptive_only', by_task=tasks)


def aggregate(rows, phase, provenance=None):
    if phase not in ('development', 'independent_validation'):
        raise ValueError('Unknown P46 evaluation phase')
    index = {p45._identity(row['cell']): row for row in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate physical cell in P46 aggregation')
    complete = bool(rows) and all(row['status'] == 'completed' for row in rows)
    gaps, missing_count = [], 0
    if phase == 'independent_validation':
        for task in TASKS:
            for split, count in (('id', 16), ('geometry', 32)):
                expected = expected_main_identities(task, split)
                actual = {identity for identity in index if identity[0] == task
                          and identity[1] == split and identity[5] in ('fixed', 'continuous')}
                missing, unexpected = expected-set(index), actual-expected
                missing_count += len(missing)
                if missing or unexpected:
                    gaps.append(dict(task=task, split=split, expected_conditions=count,
                        declared_conditions=len({identity[2] for identity in actual}),
                        undeclared_required_cells=len(missing), unexpected_main_cells=len(unexpected)))
            supplemental = {identity for identity in index
                            if identity[0] == task and identity[5] not in ('fixed', 'continuous')}
            if supplemental:
                absent = expected_supplemental_identities(task)-supplemental
                if absent:
                    gaps.append(dict(task=task, cohort_role='supplemental',
                                     undeclared_required_cells=len(absent)))
        unexpected = [identity for identity in index
                      if identity[0] not in TASKS or identity[1] not in ('id', 'geometry')
                      or (identity[5] not in ('fixed', 'continuous')
                          and identity not in expected_supplemental_identities(identity[0]))]
        if unexpected:
            gaps.append(dict(unexpected_cells=len(unexpected)))
    formal_ready = phase == 'independent_validation' and complete and not gaps
    supplemental = supplemental_status(rows)
    status = 'complete' if complete and not gaps else 'incomplete_no_formal_claims'
    if formal_ready and supplemental['status'] != 'complete':
        status = 'main_complete_supplemental_pending'
    primary, secondary = [], []
    if phase == 'independent_validation':
        for task in TASKS:
            primary.append(p45.contrast(rows, task, ('relative_fusion', 'continuous'),
                ('relative_fusion', 'fixed'), f'{task}:R-online_minus_R-fixed', formal_ready))
            secondary.append(p45.contrast(rows, task, ('relative_fusion', 'continuous'),
                ('absolute_fusion', 'continuous'), f'{task}:R-online_minus_F-online', formal_ready))
            secondary.append(p45.contrast(rows, task, ('absolute_fusion', 'continuous'),
                ('geometry', 'continuous'), f'{task}:F-online_minus_G-online', formal_ready))
        holm(primary)
        holm(secondary)
    return dict(schema='prove.p46-summary.v1', campaign=CAMPAIGN, phase=phase,
        prespecified_tasks=list(TASKS), provenance=provenance or [],
        status=status, supplemental_scope=supplemental,
        n_planned_cells=len(rows), n_prespecified_main_cells=1728,
        n_declared_main_cells=sum(row['cell']['arm'] in ('fixed', 'continuous') for row in rows),
        n_undeclared_required_main_cells=missing_count,
        cell_status_counts=dict(Counter(row['status'] for row in rows)),
        formal_inference_ready=formal_ready, formal_scope_gaps=gaps,
        analysis_contract=analysis_contract(), groups=grouped_counts(rows), paired=paired_counts(rows),
        primary_comparisons=primary, secondary_comparisons=secondary,
        evidence=[{key: row[key] for key in ('cell', 'cell_index', 'path', 'status', 'binding', 'seal', 'abort', 'error')
                   if key in row} for row in rows])


def markdown(report):
    text = p45.markdown(report).replace('# P45 配对结果汇总', '# P46 三任务配对结果汇总', 1)
    introduction = ('三个预定任务为 cup、insert、slider；主实验共 1,728 个回合。'
                    '主要比较为 R 在线与 R 固定，次要比较为在线 R−F 与 F−G。'
                    '任一预定任务或必要回合缺失时不作正式显著性结论。\n\n'
                    f"尚未声明的必要主实验回合：{report['n_undeclared_required_main_cells']}。"
                    '补充对照只作描述分析，多视图参考不会按神经模型数量重复计数。\n\n'
                    f"补充范围预声明 120 回合，已声明执行 {report['supplemental_scope']['n_declared_cells']}，"
                    f"已完成 {report['supplemental_scope']['n_completed_cells']}；"
                    f"状态：{report['supplemental_scope']['status']}。主统计完整不代表补充范围完成。\n\n")
    return text.replace('\n\n', '\n\n'+introduction, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', nargs=2, action='append', required=True, metavar=('FREEZE', 'RESULTS_ROOT'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Preserve existing aggregation output: {out}')
    rows, provenance, phase = load_cohorts(args.cohort)
    report = aggregate(rows, phase, provenance)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'RESULTS.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (out / 'REPORT_ZH.md').write_text(markdown(report))
    print(json.dumps(dict(out=str(out), status=report['status'], cells=len(rows),
                          formal_ready=report['formal_inference_ready'])))


if __name__ == '__main__':
    main()
