"""P47 representation/solver analysis of sealed, frozen physical evaluations.

This module never starts a simulator. Missing, aborted and invalid evidence is
unknown, and two checkpoints are averaged within each independent condition.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.fast_adapter import p45_aggregate as p45

ROOT = Path('evaluations/fast_adapter_p47_representation_solver_v01')
CAMPAIGN = 'p47_representation_solver'
TASKS = ('cup', 'insert', 'slider')
MODELS = (11, 22)
REPRESENTATIONS = ('geometry', 'relative_fusion')
ARMS = ('fixed', 'continuous', 'direct')
FINAL_BASE = {'cup': 480100, 'insert': 481100, 'slider': 482100}
DEVELOPMENT_BASE = {'cup': 455406, 'insert': 456406, 'slider': 457406}
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 470047
ANALYSIS_SOURCES = ('tools/fast_adapter/p47_aggregate.py',
                    'tools/fast_adapter/p45_aggregate.py',
                    'tools/fast_adapter/p45_common.py',
                    'tools/fast_adapter/p45_evaluate.py')
EVALUATION_SOURCES = list(dict.fromkeys(p45.EVALUATION_SOURCES + [
    'tools/fast_adapter/p47_evaluate.py', *ANALYSIS_SOURCES]))
grouped_counts = p45.grouped_counts
signflip_pvalue = p45.signflip_pvalue
holm = p45.holm


def analysis_contract():
    """Exact prospectively bound contract; root owns its serialized artifact."""
    return dict(
        schema='prove.p47-analysis-contract.v1', campaign=CAMPAIGN,
        prespecified_tasks=list(TASKS), representations=list(REPRESENTATIONS),
        neural_arms=list(ARMS), model_seeds=list(MODELS),
        primary_endpoint='validated_success', native_audit_endpoint='sustained_native_success',
        denominator='All 1248 prespecified physical cells, including mechanical anomalies and unknown outcomes',
        unknown_policy='Missing/aborted/integrity-invalid/undeclared are unknown; report rate bounds, never measured failures',
        sampling_unit='Independent condition; average fixed model seeds 11/22 within each condition',
        primary_A=dict(size=3, comparison='R-continuous minus G-continuous for each task',
                       role='representation'),
        primary_B=dict(size=6, comparison='continuous minus direct for each task and G/R representation',
                       role='solver'),
        support_C=dict(size=6, comparison='continuous minus fixed for each task and G/R representation',
                       role='replication'),
        pvalue='Exact two-sided condition paired sign-flip; separate Holm families A=3, B=6, C=6',
        ci='Unadjusted percentile 95% condition bootstrap; not a family-adjusted interval',
        bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_label_seed='470047 plus first four SHA256(label) bytes interpreted little-endian',
        required_ood_conditions_per_task=32, condition_seed_bases=dict(FINAL_BASE),
        n_prespecified_physical_cells=1248, physical_cells_per_condition=13,
        development_conditions_per_task=2, development_seed_bases=dict(DEVELOPMENT_BASE),
        development_physical_cells=78, development_scope='Opened P45 development conditions; exploratory, never formal',
        inference_scope='Conditional on two fixed trained checkpoints; all three tasks and all 1248 cells required',
        reference_comparisons='G/R continuous and direct minus multiview; descriptive only',
        multiview='One geometry physical cell per condition, model_seed=null; reuse without duplicating physical counts',
        task_dependence='Descriptive only; significance in one task and not another is not a task difference test',
        claim_boundary='No universal DINO benefit, gradient superiority, arbitrary-object transfer, real-robot or full-RSI claim')


def expected_identities(task, phase='independent_validation'):
    if task not in TASKS or phase not in ('development', 'independent_validation'):
        raise ValueError('Unknown P47 task or phase')
    start = (FINAL_BASE if phase == 'independent_validation' else DEVELOPMENT_BASE)[task]
    count = 32 if phase == 'independent_validation' else 2
    identities = {(task, 'geometry', seed, rep, model, arm)
                  for seed in range(start, start + count) for rep in REPRESENTATIONS
                  for model in MODELS for arm in ARMS}
    identities.update((task, 'geometry', seed, 'geometry', None, 'multiview')
                      for seed in range(start, start + count))
    return identities


def _check_binding(binding, files=None):
    if not isinstance(binding, dict) or not {'path', 'sha256'} <= set(binding):
        raise ValueError('Expected an explicit path/hash artifact binding')
    if files is not None and files.get(binding['path']) != binding['sha256']:
        raise ValueError('Artifact binding is absent from frozen file inventory')
    if p45.sha(binding['path']) != binding['sha256']:
        raise ValueError(f"Bound artifact changed: {binding['path']}")


def _validate_scope(frozen):
    if (frozen.get('schema') != 'prove.p47-evaluation.v1'
            or frozen.get('status') != 'frozen' or frozen.get('root_authorized') is not True
            or frozen.get('campaign') != CAMPAIGN
            or frozen.get('prespecified_tasks') != list(TASKS)
            or frozen.get('analysis_contract') != analysis_contract()
            or frozen.get('phase') not in ('development', 'independent_validation')):
        raise ValueError('Expected exact frozen, authorized P47 scope and analysis contract')
    tasks = set(frozen.get('tasks', {}))
    if len(tasks) != 1 or not tasks <= set(TASKS):
        raise ValueError('Each P47 freeze must contain exactly one prespecified task')
    task, = tasks
    cells = frozen.get('cells', [])
    identities = [p45._identity(cell) for cell in cells]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate physical cell in P47 freeze')
    if set(identities) != expected_identities(task, frozen['phase']):
        raise ValueError('P47 freeze must contain the exact complete task matrix')
    if type(frozen.get('steps_per_episode')) is not int or not 10 <= frozen['steps_per_episode'] <= 240:
        raise ValueError('P47 requires a fixed 10..240 action budget')
    for key in ('plan', 'analysis_contract_file'):
        _check_binding(frozen.get(key), frozen.get('files', {}))
    if p45._read(frozen['analysis_contract_file']['path']) != analysis_contract():
        raise ValueError('Bound P47 analysis contract content differs from the frozen contract')
    for source in ANALYSIS_SOURCES:
        if frozen.get('sources', {}).get(source) != p45.sha(source):
            raise ValueError(f'Current analysis helper is not bound to the frozen source: {source}')
    return task


def load_cohorts(cohorts):
    """Verify public seals before reading post-seal evaluator outcomes.

    Source snapshots are verified once per unique frozen inventory. This is
    evidence integrity checking, not physical replay or outcome recomputation.
    """
    # Lazy import prevents preparation's use of analysis_contract() from
    # creating an evaluator/aggregator import cycle.
    from tools.fast_adapter.p47_evaluate import SOURCES as runtime_sources

    rows, provenance, identities = [], [], set()
    phases, shared_sources, task_contracts, candidates, snapshot_checks = set(), {}, {}, {}, {}
    plan_digests = set()
    for freeze_path, results_root in cohorts:
        frozen = p45._read(freeze_path)
        task = _validate_scope(frozen)
        phase = frozen['phase']
        phases.add(phase)
        plan_digests.add((frozen['plan']['sha256'], frozen['analysis_contract_file']['sha256']))
        environment = frozen['environments'][task]
        relevant = set(runtime_sources) | set(ANALYSIS_SOURCES) | {
            environment[key].replace('.', '/') + '.py'
            for key in ('module', 'configuration_module', 'controller_module')}
        if not relevant <= set(frozen['sources']):
            raise ValueError('Frozen P47 evaluation source inventory is incomplete')
        for source, digest in frozen['sources'].items():
            relative = Path(source)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Frozen source paths must be repository-relative')
            if source in shared_sources and shared_sources[source] != digest:
                raise ValueError('Cannot pool changed source versions in one P47 campaign')
            shared_sources[source] = digest
        contract = dict(environment=environment, steps=frozen['steps_per_episode'],
                        solver_contract=frozen.get('solver_contract'))
        if task in task_contracts and task_contracts[task] != contract:
            raise ValueError('Cannot pool changed environment, solver or action-budget versions')
        task_contracts[task] = contract
        freeze_binding = p45._binding(freeze_path)
        provenance.append(dict(freeze=freeze_binding, results_root=str(results_root),
            campaign=CAMPAIGN, task=task, phase=phase, plan=frozen['plan'],
            analysis_contract_file=frozen['analysis_contract_file'], sources=frozen['sources'],
            environment=environment, steps_per_episode=frozen['steps_per_episode']))
        for index, cell in enumerate(frozen['cells']):
            identity = p45._identity(cell)
            if identity in identities:
                raise ValueError('Duplicate physical cell across P47 cohorts')
            identities.add(identity)
            reference = cell['arm'] == 'multiview'
            candidate = (dict(method='analytic_visible_multiview', checkpoint=None, pca=None)
                         if reference else frozen['tasks'][task]['models'][cell['rep']][str(cell['model_seed'])])
            if not reference:
                key = task, cell['rep'], cell['model_seed']
                digest = (candidate['checkpoint']['sha256'],
                          candidate['pca']['sha256'] if candidate.get('pca') else None)
                if key in candidates and candidates[key] != digest:
                    raise ValueError('Cannot pool different checkpoints or projections under one model identity')
                if key not in candidates:
                    for asset in (candidate['checkpoint'], candidate.get('pca')):
                        if asset is not None:
                            _check_binding(asset, frozen['files'])
                candidates[key] = digest
            directory = Path(results_root) / f'cell_{index:06d}'
            record = dict(cell=cell, cell_index=index, path=str(directory), status='missing', result=None)
            if (directory / 'ABORT.json').exists():
                record.update(status='aborted', abort=p45._binding(directory / 'ABORT.json'))
            elif (directory / 'RESULT.json').exists():
                try:
                    pre = p45._read(directory / 'PRE_RUN.json')
                    seal = p45._read(directory / 'SEAL.json')
                    if (pre['freeze']['sha256'] != freeze_binding['sha256'] or pre['cell'] != cell
                            or pre['cell_index'] != index or pre['phase'] != phase
                            or pre['steps'] != frozen['steps_per_episode']
                            or pre['environment'] != environment or pre['candidate'] != candidate):
                        raise ValueError('Pre-run cell/freeze/candidate/environment binding mismatch')
                    if (seal.get('schema') != 'prove.p45-trajectory-seal.v1'
                            or seal.get('campaign') != CAMPAIGN
                            or seal.get('freeze', {}).get('sha256') != freeze_binding['sha256']):
                        raise ValueError('Expected a trajectory seal bound to this P47 campaign and freeze')
                    if not {'PRE_RUN.json', 'TRACE.npy', 'OBS.npy', 'TERMINAL.npy'} <= set(seal['files']):
                        raise ValueError('Incomplete physical/public trajectory seal')
                    for name, digest in seal['files'].items():
                        if Path(name).name != name or p45.sha(directory / name) != digest:
                            raise ValueError('Sealed evidence file missing or changed')
                    snapshot = pre['source_snapshot']
                    snapshot_key = (snapshot, tuple(sorted(frozen['sources'].items())))
                    if snapshot_key not in snapshot_checks:
                        snapshot_checks[snapshot_key] = all(
                            p45.sha(Path(snapshot) / source) == digest
                            for source, digest in frozen['sources'].items())
                    if not snapshot_checks[snapshot_key]:
                        raise ValueError('Frozen source snapshot differs from source inventory')
                    before, after = seal['neural_parameters_before'], seal['neural_parameters_after']
                    if before != after or (reference and before is not None) or (
                            not reference and (not isinstance(before, str) or len(before) != 64)):
                        raise ValueError('Neural parameters changed or parameter evidence is invalid')
                    result = p45._read(directory / 'RESULT.json')
                    if (p45._identity(result) != identity or result['cell_index'] != index
                            or result['phase'] != phase or result['steps'] != frozen['steps_per_episode']):
                        raise ValueError('Result identity/phase/action budget differs from its frozen cell')
                    for key in ('success', 'validated_success', 'mechanical_valid', 'sustained_native_success'):
                        if type(result[key]) is not bool:
                            raise ValueError(f'Outcome is missing a boolean {key}')
                    if result['validated_success'] != (result['success'] and result['mechanical_valid']):
                        raise ValueError('Validated success disagrees with mechanical validity and task success')
                    if result.get('task_success_denominator_included') is not True:
                        raise ValueError('A declared cell was excluded from its task denominator')
                    if not reference and result.get('neural_weights_unchanged') is not True:
                        raise ValueError('Neural-weight freeze audit is absent')
                    record.update(status='completed', result=result,
                                  binding=p45._binding(directory / 'RESULT.json'),
                                  seal=p45._binding(directory / 'SEAL.json'))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    record.update(status='integrity_invalid', error=str(exc))
            rows.append(record)
    if len(phases) != 1:
        raise ValueError('Development and independent validation must be reported separately')
    if len(plan_digests) != 1:
        raise ValueError('Cannot pool different P47 plans or analysis contracts')
    return rows, provenance, phases.pop()


def bootstrap_interval(differences, label):
    values = np.asarray(differences, dtype=float)
    if len(values) < 2:
        return None
    seed = BOOTSTRAP_SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(BOOTSTRAP_SAMPLES, len(values)))].mean(1)
    return np.quantile(draws, [.025, .975]).tolist()


def _comparisons():
    for task in TASKS:
        yield 'primary_A', task, ('relative_fusion', 'continuous'), ('geometry', 'continuous'), f'{task}:R-continuous_minus_G-continuous'
        for rep, short in (('geometry', 'G'), ('relative_fusion', 'R')):
            for family, other in (('primary_B', 'direct'), ('support_C', 'fixed')):
                yield family, task, (rep, 'continuous'), (rep, other), f'{task}:{short}-continuous_minus_{short}-{other}'
            for arm in ('continuous', 'direct'):
                yield 'reference', task, (rep, arm), ('geometry', 'multiview'), f'{task}:{short}-{arm}_minus_multiview'


def paired_counts(rows, phase='independent_validation'):
    groups = defaultdict(dict)
    for row in rows:
        groups[tuple(row['cell'][key] for key in p45.GROUP)][row['cell']['condition_seed']] = row
    comparisons = []
    for family, task, left, right, label in _comparisons():
        conditions = {identity[2] for identity in expected_identities(task, phase)}
        for model in MODELS:
            reference = right[1] == 'multiview'
            right_model = None if reference else model
            a = groups.get((task, 'geometry', left[0], model, left[1]), {})
            b = groups.get((task, 'geometry', right[0], right_model, right[1]), {})
            comparisons.append(dict(family=family, task=task, split='geometry', label=label,
                rep=left[0], model_seed=model, left_arm=left[1], right_arm=right[1],
                right_rep=right[0], right_model_seed=right_model,
                reference_reused_without_extra_physical_cells=reference,
                **p45.pair_summary(a, b, conditions)))
    return comparisons


def contrast(rows, task, left, right, label, formal_ready, phase='independent_validation', reference=False):
    index = {p45._identity(row['cell']): row for row in rows}
    conditions = sorted({identity[2] for identity in expected_identities(task, phase)})
    base = dict(task=task, split='geometry', label=label, left=dict(rep=left[0], arm=left[1]),
        right=dict(rep=right[0], arm=right[1]), expected_model_seeds=list(MODELS),
        n_conditions_planned=len(conditions), n_conditions_complete=0,
        status='incomplete', p_raw=1., difference=None, bootstrap_95_ci=None,
        descriptive_only=reference or phase == 'development',
        reference_reused_without_extra_physical_cells=reference)
    values, native, complete_conditions = [], [], []
    for condition in conditions:
        a = [index.get((task, 'geometry', condition, left[0], model, left[1])) for model in MODELS]
        b = [index.get((task, 'geometry', condition, right[0], model, right[1]))
             for model in ((None,) if reference else MODELS)]
        if any(row is None or row['status'] != 'completed' for row in a + b):
            continue
        values.append([np.mean([row['result']['validated_success'] for row in side]) for side in (a, b)])
        native.append([np.mean([row['result']['sustained_native_success'] for row in side]) for side in (a, b)])
        complete_conditions.append(condition)
    base['n_conditions_complete'] = len(values)
    base['n_conditions_unknown'] = len(conditions) - len(values)
    if not values:
        return base
    values, native = np.asarray(values), np.asarray(native)
    base['observed_complete_condition_difference'] = float((values[:, 0] - values[:, 1]).mean())
    base['condition_values'] = [dict(condition_seed=condition, left=float(value[0]), right=float(value[1]))
                                for condition, value in zip(complete_conditions, values)]
    if len(values) != len(conditions):
        return base
    differences = values[:, 0] - values[:, 1]
    base.update(status='descriptive_complete', rate_left=float(values[:, 0].mean()),
                rate_right=float(values[:, 1].mean()), difference=float(differences.mean()),
                native_rate_left=float(native[:, 0].mean()), native_rate_right=float(native[:, 1].mean()))
    if formal_ready and not reference:
        base.update(status='complete', p_raw=signflip_pvalue(differences),
                    bootstrap_95_ci=bootstrap_interval(differences, label))
    return base


def aggregate(rows, phase, provenance=None):
    if phase not in ('development', 'independent_validation'):
        raise ValueError('Unknown P47 evaluation phase')
    index = {p45._identity(row['cell']): row for row in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate physical cell in P47 aggregation')
    expected = set().union(*(expected_identities(task, phase) for task in TASKS))
    missing, unexpected = expected - set(index), set(index) - expected
    # Entire undeclared tasks are retained in every denominator and bound.
    all_rows = list(rows) + [dict(cell=dict(zip(p45.IDENTITY, identity)), status='undeclared', result=None)
                             for identity in sorted(missing, key=repr)]
    gaps = [dict(task=task, undeclared_required_cells=sum(item[0] == task for item in missing))
            for task in TASKS if any(item[0] == task for item in missing)]
    if unexpected:
        gaps.append(dict(unexpected_cells=len(unexpected)))
    complete = bool(rows) and not gaps and all(row['status'] == 'completed' for row in all_rows)
    formal_ready = complete and phase == 'independent_validation'
    families = {family: [] for family in ('primary_A', 'primary_B', 'support_C', 'reference')}
    for family, task, left, right, label in _comparisons():
        families[family].append(contrast(all_rows, task, left, right, label, formal_ready,
                                         phase=phase, reference=family == 'reference'))
    for family in ('primary_A', 'primary_B', 'support_C'):
        holm(families[family])
    status = ('complete' if formal_ready else 'development_complete_exploratory' if complete
              else 'incomplete_no_formal_claims')
    return dict(schema='prove.p47-summary.v1', campaign=CAMPAIGN, phase=phase, status=status,
        prespecified_tasks=list(TASKS), provenance=provenance or [],
        analysis_sources={source: p45.sha(source) for source in ANALYSIS_SOURCES},
        analysis_contract=analysis_contract(), n_prespecified_physical_cells=len(expected),
        n_planned_cells=len(all_rows), n_declared_cells=len(rows),
        n_undeclared_required_cells=len(missing), n_unexpected_cells=len(unexpected),
        n_completed_cells=sum(row['status'] == 'completed' for row in all_rows),
        n_unknown_cells=sum(row['status'] != 'completed' for row in all_rows),
        cell_status_counts=dict(Counter(row['status'] for row in all_rows)),
        formal_inference_ready=formal_ready, formal_scope_gaps=gaps,
        groups=grouped_counts(all_rows), paired=paired_counts(all_rows, phase),
        primary_A_comparisons=families['primary_A'], primary_B_comparisons=families['primary_B'],
        support_C_comparisons=families['support_C'], reference_comparisons=families['reference'],
        evidence=[{key: row[key] for key in ('cell', 'cell_index', 'path', 'status', 'binding', 'seal', 'abort', 'error')
                   if key in row} for row in all_rows])


def markdown(report):
    lines = ['# P47 表征与求解器配对结果', '',
        f"阶段：{report['phase']}；状态：{report['status']}。"
        f"预定 {report['n_prespecified_physical_cells']} 个物理回合，已声明 {report['n_declared_cells']}，"
        f"已完成 {report['n_completed_cells']}，未知 {report['n_unknown_cells']}。", '',
        '主端点为任务成功且机械有效。缺失、中止、证据异常和未声明回合保持未知；机械异常保留在分母。'
        '开发条件已开放，只作探索描述。多视图每条件仅一个物理回合，模型种子为空。', '',
        '|任务|表征|模型|方法|已确认成功/计划|未知|机械异常|成功率界限|',
        '|---|---|---|---|---:|---:|---:|---|']
    for row in report['groups']:
        bounds = '–'.join(f'{value:.1%}' for value in row['validated_rate_bounds'])
        model = row['model_seed'] if row['model_seed'] is not None else '单次解析参考'
        lines.append(f"|{row['task']}|{row['rep']}|{model}|{row['arm']}|{row['confirmed_successes_over_planned']}|{row['n_unknown']}|{row['mechanical_anomalies']}|{bounds}|")
    lines += ['', '以条件为统计单位，先平均两个固定模型。A 表征主比较3项、B 求解器主比较6项、'
              'C 复现支持比较6项分别作 Holm 校正；95%条件 bootstrap 区间未作多重比较调整。'
              '任一预定回合不完整时不作正式显著性结论。', '',
              '|比较族|比较|完整/计划条件|差值|95%区间|Holm p|状态|',
              '|---|---|---:|---:|---|---:|---|']
    for family in ('primary_A', 'primary_B', 'support_C', 'reference'):
        for row in report[family + '_comparisons']:
            delta = '—' if row['difference'] is None else f"{row['difference'] * 100:+.2f}pp"
            interval = ('—' if row['bootstrap_95_ci'] is None else
                        ', '.join(f'{value * 100:+.2f}pp' for value in row['bootstrap_95_ci']))
            pvalue = f"{row['p_holm']:.6g}" if row['status'] == 'complete' and 'p_holm' in row else '不作推断'
            lines.append(f"|{family}|{row['label']}|{row['n_conditions_complete']}/{row['n_conditions_planned']}|{delta}|{interval}|{pvalue}|{row['status']}|")
    lines += ['', '|模型内配对比较|模型|已观测/计划|救回|误伤|未知|左/右机械异常|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in report['paired']:
        lines.append(f"|{row['label']}|{row['model_seed']}|{row['n_observed_pairs']}/{row['n_planned_pairs']}|{row['rescue']}|{row['harm']}|{row['n_unknown_pairs']}|{row['mechanical_anomalies_left']}/{row['mechanical_anomalies_right']}|")
    lines += ['', '多视图比较仅描述，不重复计入物理样本。任务依赖仅描述；一个任务显著而另一个不显著，'
              '不构成任务差异检验。结果仅针对两个固定训练 checkpoint，不能据此声称 DINO 普遍有效、'
              '梯度普遍优于多视图拟合、任意物体迁移、真机或完整 RSI。', '', '来源：']
    for item in report['provenance']:
        lines.append(f"- `{item['freeze']['path']}`（SHA256 `{item['freeze']['sha256']}`）；"
                     f"计划 `{item['plan']['path']}`；统计合同 `{item['analysis_contract_file']['path']}`。")
    return '\n'.join(lines) + '\n'


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
    (out / 'RESULTS.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (out / 'REPORT_ZH.md').write_text(markdown(report))
    print(json.dumps(dict(out=str(out), status=report['status'], cells=report['n_planned_cells'],
                          formal_ready=report['formal_inference_ready'])))


if __name__ == '__main__':
    main()
