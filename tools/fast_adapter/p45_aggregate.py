"""P45 complete-denominator summaries and prespecified condition-level statistics.

Report schema prove.p45-summary.v1:
  primary: 3 task-specific OOD F-continuous minus F-fixed comparisons;
  secondary: 6 OOD continuous comparisons, R-F and F-G for each task.
  Endpoint is validated_success. sustained_native_success is an audit endpoint.
  Two fixed neural models (11/22) are averaged WITHIN each condition, then
  conditions are resampled. Exact two-sided paired sign-flip tests use these
  half-integer differences. Holm controls the separate families of 3 and 6.
  Percentile 95% bootstrap intervals are condition-based and unadjusted.

Every declared cell remains in its planned denominator. Missing, aborted and
integrity-invalid cells are unknown outcomes, never measured task failures.
Any incomplete declared cohort disables all formal significance statements.
Unrun tasks retain p=1 in their prespecified families. Development is descriptive.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.fast_adapter.p45_common import sha
from tools.fast_adapter.p45_evaluate import SOURCES as EVALUATION_SOURCES

TASKS = ('cup', 'insert', 'slider')
MODELS = (11, 22)
IDENTITY = ('task', 'split', 'condition_seed', 'rep', 'model_seed', 'arm')
GROUP = ('task', 'split', 'rep', 'model_seed', 'arm')
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 450045


def _read(path):
    return json.loads(Path(path).read_text())


def _binding(path):
    return dict(path=str(path), sha256=sha(path))


def _identity(cell):
    return tuple(cell[key] for key in IDENTITY)


def load_cohorts(cohorts):
    rows, provenance, identities, task_contracts, candidates = [], [], set(), {}, {}
    phases = set()
    for freeze_path, results_root in cohorts:
        frozen = _read(freeze_path)
        if frozen.get('schema') != 'prove.p45-evaluation.v1' or frozen.get('status') != 'frozen':
            raise ValueError('Aggregation requires the exact frozen evaluation manifest')
        phase = frozen['phase']
        phases.add(phase)
        freeze_binding = _binding(freeze_path)
        provenance.append(dict(freeze=freeze_binding, results_root=str(results_root)))
        for index, cell in enumerate(frozen['cells']):
            identity = _identity(cell)
            if identity in identities:
                raise ValueError('Duplicate physical cell across aggregation inputs')
            identities.add(identity)
            task = cell['task']
            environment = frozen['environments'][task]
            relevant_sources = set(EVALUATION_SOURCES) | {
                environment[key].replace('.', '/') + '.py'
                for key in ('module', 'configuration_module', 'controller_module')}
            if not relevant_sources <= set(frozen['sources']):
                raise ValueError('Frozen evaluation source inventory is incomplete')
            contract = dict(environment=environment, sources=dict(frozen['sources']),
                            steps=frozen['steps_per_episode'])
            if task in task_contracts:
                previous = task_contracts[task]
                shared = set(previous['sources']) & set(contract['sources'])
                if (previous['environment'] != contract['environment'] or previous['steps'] != contract['steps']
                        or any(previous['sources'][name] != contract['sources'][name] for name in shared)):
                    raise ValueError('Cannot pool a task across changed backend/source/action-budget versions')
                # Additional bound modules are allowed; any common dependency
                # (including an imported earlier backend version) must match.
                contract['sources'] = dict(previous['sources'], **contract['sources'])
            task_contracts[task] = contract
            if cell['arm'] != 'multiview':
                key = task, cell['rep'], cell['model_seed']
                candidate = frozen['tasks'][task]['models'][cell['rep']][str(cell['model_seed'])]
                digest = (candidate['checkpoint']['sha256'],
                          candidate['pca']['sha256'] if candidate.get('pca') else None)
                if key in candidates and candidates[key] != digest:
                    raise ValueError('Cannot pool different checkpoints/projections under one model identity')
                candidates[key] = digest
            directory = Path(results_root) / f'cell_{index:06d}'
            row = dict(cell=cell, cell_index=index, path=str(directory), status='missing', result=None)
            if (directory / 'ABORT.json').exists():
                row.update(status='aborted', abort=_binding(directory / 'ABORT.json'))
            elif (directory / 'RESULT.json').exists():
                try:
                    result = _read(directory / 'RESULT.json')
                    pre = _read(directory / 'PRE_RUN.json')
                    seal = _read(directory / 'SEAL.json')
                    if pre['freeze']['sha256'] != freeze_binding['sha256'] or pre['cell'] != cell or result['cell_index'] != index:
                        raise ValueError('Result/pre-run cell or freeze binding mismatch')
                    if _identity(result) != identity or result['phase'] != phase:
                        raise ValueError('Result identity/phase differs from its frozen cell')
                    for name, digest in seal['files'].items():
                        if Path(name).name != name or sha(directory / name) != digest:
                            raise ValueError('Sealed evidence file missing or changed')
                    if not {'PRE_RUN.json', 'TRACE.npy', 'OBS.npy', 'TERMINAL.npy'} <= set(seal['files']):
                        raise ValueError('Incomplete physical/public trajectory seal')
                    if seal['neural_parameters_before'] != seal['neural_parameters_after']:
                        raise ValueError('Neural parameters changed during evaluation')
                    for key in ('success', 'validated_success', 'mechanical_valid', 'sustained_native_success'):
                        if type(result[key]) is not bool:
                            raise ValueError(f'Outcome is missing a boolean {key}')
                    if result['validated_success'] != (result['success'] and result['mechanical_valid']):
                        raise ValueError('Validated success does not match mechanical validity and task success')
                    if result.get('task_success_denominator_included') is not True:
                        raise ValueError('A declared cell was excluded from its task denominator')
                    row.update(status='completed', result=result, binding=_binding(directory / 'RESULT.json'),
                               seal=_binding(directory / 'SEAL.json'))
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row.update(status='integrity_invalid', error=str(exc))
            rows.append(row)
    if len(phases) != 1 or not phases <= {'development', 'independent_validation'}:
        raise ValueError('Development and independent validation must be reported separately')
    return rows, provenance, phases.pop()


def grouped_counts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row['cell'][key] for key in GROUP)].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        known = [row['result'] for row in group if row['status'] == 'completed']
        n, observed = len(group), len(known)
        k = sum(record['validated_success'] for record in known)
        statuses = Counter(row['status'] for row in group)
        result.append(dict(zip(GROUP, key), n_planned=n, n_observed=observed, n_unknown=n-observed,
            k_validated=k, k_native=sum(record['sustained_native_success'] for record in known),
            k_task_success=sum(record['success'] for record in known),
            confirmed_successes_over_planned=f'{k}/{n}',
            validated_rate=k/n if observed == n else None,
            validated_rate_bounds=[k/n, (k+n-observed)/n],
            ordinary_task_failures=sum(record['mechanical_valid'] and not record['success'] for record in known),
            mechanical_anomalies=sum(not record['mechanical_valid'] for record in known),
            statuses=dict(statuses), physical_rollouts_not_duplicated=True))
    return result


def pair_summary(left, right, conditions):
    rescue = harm = both_success = both_failure = mechanical_left = mechanical_right = observed = 0
    for condition in conditions:
        a, b = left.get(condition), right.get(condition)
        if a is None or b is None or a['status'] != 'completed' or b['status'] != 'completed':
            continue
        observed += 1
        ya, yb = a['result']['validated_success'], b['result']['validated_success']
        rescue += int(ya and not yb)
        harm += int(yb and not ya)
        both_success += int(ya and yb)
        both_failure += int(not ya and not yb)
        mechanical_left += int(not a['result']['mechanical_valid'])
        mechanical_right += int(not b['result']['mechanical_valid'])
    return dict(n_planned_pairs=len(conditions), n_observed_pairs=observed,
                n_unknown_pairs=len(conditions)-observed, rescue=rescue, harm=harm,
                both_success=both_success, both_failure=both_failure,
                mechanical_anomalies_left=mechanical_left, mechanical_anomalies_right=mechanical_right,
                observed_net_rescue=rescue-harm)


def paired_counts(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[tuple(row['cell'][key] for key in GROUP)][row['cell']['condition_seed']] = row
    result = []
    for key, left in sorted(groups.items(), key=lambda item: repr(item[0])):
        task, split, representation, model, arm = key
        if arm != 'continuous':
            continue
        comparators = [('fixed', representation, model)]
        if representation == 'absolute_fusion':
            comparators += [('prefix3', representation, model), ('mismatched', representation, model),
                            ('multiview', 'geometry', None)]
        for other_arm, other_rep, other_model in comparators:
            right = groups.get((task, split, other_rep, other_model, other_arm), {})
            if not right and other_arm != 'fixed':
                continue
            conditions = set(left) | set(right) if other_arm == 'fixed' else set(right)
            result.append(dict(task=task, split=split, rep=representation, model_seed=model,
                left_arm='continuous', right_arm=other_arm, right_rep=other_rep, right_model_seed=other_model,
                reference_reused_without_extra_physical_cells=other_arm == 'multiview',
                **pair_summary(left, right, conditions)))
    return result


def signflip_pvalue(differences):
    """Exact two-sided condition sign-flip distribution, no model replication."""
    values = np.asarray(differences, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return 1.
    integer = np.rint(values * 2).astype(int)
    if not np.allclose(integer, values * 2):
        raise ValueError('Expected means over exactly two fixed binary-outcome models')
    distribution = {0: 1.}
    for weight in np.abs(integer[integer != 0]):
        updated = defaultdict(float)
        for value, probability in distribution.items():
            updated[value-int(weight)] += probability*.5
            updated[value+int(weight)] += probability*.5
        distribution = updated
    observed = abs(int(integer.sum()))
    return float(min(1., sum(probability for value, probability in distribution.items() if abs(value) >= observed)))


def bootstrap_interval(differences, label):
    values = np.asarray(differences, dtype=float)
    if len(values) < 2:
        return None
    seed = BOOTSTRAP_SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(BOOTSTRAP_SAMPLES, len(values)))].mean(1)
    return np.quantile(draws, [.025, .975]).tolist()


def holm(records):
    order = sorted(range(len(records)), key=lambda i: records[i]['p_raw'])
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., records[index]['p_raw'] * (len(records)-rank)))
        records[index]['p_holm'] = previous
        delta = records[index].get('difference')
        eligible = records[index]['status'] == 'complete'
        records[index]['positive_difference_supported'] = bool(eligible and delta > 0 and previous < .05)
        records[index]['negative_difference_supported'] = bool(eligible and delta < 0 and previous < .05)
    return records


def contrast(rows, task, left, right, label, formal_ready):
    # left/right are (representation, arm); model identity is never a replicate.
    index = {_identity(row['cell']): row for row in rows}
    conditions = sorted({row['cell']['condition_seed'] for row in rows
                         if row['cell']['task'] == task and row['cell']['split'] == 'geometry'
                         and (row['cell']['rep'], row['cell']['arm']) in (left, right)})
    base = dict(task=task, split='geometry', label=label, left=dict(rep=left[0], arm=left[1]),
                right=dict(rep=right[0], arm=right[1]), expected_model_seeds=list(MODELS),
                n_conditions_planned=len(conditions), n_conditions_complete=0,
                status='not_run' if not conditions else 'incomplete', p_raw=1.,
                difference=None, bootstrap_95_ci=None)
    values, native, complete_conditions = [], [], []
    for condition in conditions:
        group = [index.get((task, 'geometry', condition, rep, model, arm))
                 for rep, arm in (left, right) for model in MODELS]
        if any(row is None or row['status'] != 'completed' for row in group):
            continue
        outcomes = np.asarray([row['result']['validated_success'] for row in group], float).reshape(2, 2).mean(1)
        native_outcomes = np.asarray([row['result']['sustained_native_success'] for row in group], float).reshape(2, 2).mean(1)
        values.append(outcomes)
        native.append(native_outcomes)
        complete_conditions.append(condition)
    base['n_conditions_complete'] = len(values)
    if not formal_ready or len(conditions) != 32 or len(values) != len(conditions):
        return base
    values, native = np.asarray(values), np.asarray(native)
    differences = values[:, 0] - values[:, 1]
    base.update(status='complete', rate_left=float(values[:, 0].mean()), rate_right=float(values[:, 1].mean()),
                difference=float(differences.mean()), p_raw=signflip_pvalue(differences),
                bootstrap_95_ci=bootstrap_interval(differences, label),
                native_rate_left=float(native[:, 0].mean()), native_rate_right=float(native[:, 1].mean()),
                condition_values=[dict(condition_seed=condition, left=float(value[0]), right=float(value[1]))
                                  for condition, value in zip(complete_conditions, values)])
    return base


def aggregate(rows, phase, provenance=None):
    complete = all(row['status'] == 'completed' for row in rows)
    index = {_identity(row['cell']): row for row in rows}
    scope_gaps = []
    if phase == 'independent_validation':
        for task in sorted({row['cell']['task'] for row in rows}):
            for split, expected_count in (('id', 16), ('geometry', 32)):
                seeds = {row['cell']['condition_seed'] for row in rows
                         if row['cell']['task'] == task and row['cell']['split'] == split
                         and row['cell']['arm'] in ('fixed', 'continuous')}
                missing = sum((task, split, seed, rep, model, arm) not in index
                              for seed in seeds for rep in ('geometry', 'absolute_fusion', 'relative_fusion')
                              for model in MODELS for arm in ('fixed', 'continuous'))
                if len(seeds) != expected_count or missing:
                    scope_gaps.append(dict(task=task, split=split, expected_conditions=expected_count,
                                           declared_conditions=len(seeds), undeclared_required_cells=missing))
    formal_ready = bool(rows) and phase == 'independent_validation' and complete and not scope_gaps
    primary, secondary = [], []
    if phase == 'independent_validation':
        for task in TASKS:
            primary.append(contrast(rows, task, ('absolute_fusion', 'continuous'), ('absolute_fusion', 'fixed'),
                                    f'{task}:F-online_minus_F-fixed', formal_ready))
            secondary.append(contrast(rows, task, ('relative_fusion', 'continuous'), ('absolute_fusion', 'continuous'),
                                      f'{task}:R-online_minus_F-online', formal_ready))
            secondary.append(contrast(rows, task, ('absolute_fusion', 'continuous'), ('geometry', 'continuous'),
                                      f'{task}:F-online_minus_G-online', formal_ready))
        holm(primary)
        holm(secondary)
    return dict(
        schema='prove.p45-summary.v1', phase=phase, provenance=provenance or [],
        status='complete' if complete and not scope_gaps else 'incomplete_no_formal_claims',
        n_planned_cells=len(rows), cell_status_counts=dict(Counter(row['status'] for row in rows)),
        formal_inference_ready=formal_ready,
        formal_scope_gaps=scope_gaps,
        analysis_contract=dict(
            primary_endpoint='validated_success', native_audit_endpoint='sustained_native_success',
            denominator='All declared physical cells, including mechanical anomalies and unknown outcomes',
            unknown_policy='Missing/aborted/integrity-invalid are unknown; report rate bounds, never measured failures',
            sampling_unit='Independent condition; average fixed model seeds 11/22 within each condition',
            primary_family='3 task-specific OOD F-continuous minus F-fixed; unrun task p=1',
            secondary_family='6 task-specific OOD continuous R-F and F-G comparisons; unrun task p=1',
            pvalue='Exact two-sided condition paired sign-flip; separate Holm families of 3 and 6',
            ci='Unadjusted percentile 95% condition bootstrap; not a family-adjusted interval',
            bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
            required_ood_conditions_per_task=32, inference_scope='Conditional on two fixed trained checkpoints',
            multiview='One physical cell per condition, model_seed=null; comparisons reuse it without duplicating its count'),
        groups=grouped_counts(rows), paired=paired_counts(rows),
        primary_comparisons=primary, secondary_comparisons=secondary,
        evidence=[{key: row[key] for key in ('cell', 'cell_index', 'path', 'status', 'binding', 'seal', 'abort', 'error')
                   if key in row} for row in rows],
    )


def markdown(report):
    lines = ['# P45 配对结果汇总', '',
             f"阶段：{report['phase']}；状态：{report['status']}。分母包含所有 {report['n_planned_cells']} 个计划物理回合。", '',
             '主端点为任务成功且机械有效；缺失、执行中止和证据异常保留为未知，不解释成已测得任务失败。', '',
             '|任务|分布|表示|模型|方法|已确认成功/计划|已观测|未知|机械异常|',
             '|---|---|---|---|---|---:|---:|---:|---:|']
    for row in report['groups']:
        lines.append(f"|{row['task']}|{row['split']}|{row['rep']}|{row['model_seed'] if row['model_seed'] is not None else '单次解析参考'}|{row['arm']}|{row['confirmed_successes_over_planned']}|{row['n_observed']}|{row['n_unknown']}|{row['mechanical_anomalies']}|")
    lines += ['', '|任务|分布|表示/模型|在线相对方法|配对观测/计划|救回|误伤|',
              '|---|---|---|---|---:|---:|---:|']
    for row in report['paired']:
        lines.append(f"|{row['task']}|{row['split']}|{row['rep']}/{row['model_seed']}|{row['right_arm']}|{row['n_observed_pairs']}/{row['n_planned_pairs']}|{row['rescue']}|{row['harm']}|")
    if report['phase'] == 'independent_validation':
        lines += ['', '正式统计以条件为单位，先平均两个固定模型；95%区间未经多重比较调整，p值按3项主要/6项次要分别Holm校正。', '',
                  '|比较|完整条件|差值|95%条件bootstrap区间|Holm p|状态|', '|---|---:|---:|---|---:|---|']
        for row in report['primary_comparisons'] + report['secondary_comparisons']:
            delta = '—' if row['difference'] is None else f"{row['difference']*100:+.2f}pp"
            interval = '—' if row['bootstrap_95_ci'] is None else ', '.join(f'{x*100:+.2f}pp' for x in row['bootstrap_95_ci'])
            lines.append(f"|{row['label']}|{row['n_conditions_complete']}|{delta}|{interval}|{row['p_holm']:.6g}|{row['status']}|")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', nargs=2, action='append', required=True, metavar=('FREEZE', 'RESULTS_ROOT'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    rows, provenance, phase = load_cohorts(args.cohort)
    report = aggregate(rows, phase, provenance)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'RESULTS.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (out / 'REPORT_ZH.md').write_text(markdown(report))
    print(json.dumps(dict(out=str(out), status=report['status'], cells=len(rows), formal_ready=report['formal_inference_ready'])))


if __name__ == '__main__':
    main()
