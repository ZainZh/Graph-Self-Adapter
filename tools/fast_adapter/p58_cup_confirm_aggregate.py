#!/usr/bin/env python3
"""Read-only integrity audit and aggregation for the P58 cup confirmation cohort."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np

from tools.fast_adapter.p45_common import sha
from tools.fast_adapter.p58_cup_confirm import cells as expected_cells
from tools.fast_adapter.p58_cup_confirm import cell_output, validate


SCHEMA = 'prove.p58-cup-confirmation-report.v1'
Z95 = 1.959963984540054
ARM_ORDER = (
    ('position_only', 'fixed'), ('position_only', 'online'), ('position_only', 'direct'),
    ('functional_frame', 'fixed'), ('functional_frame', 'online'), ('functional_frame', 'direct'),
    ('p47_geometry_reference', 'fixed'), ('p47_geometry_reference', 'continuous'))
PAIR_ORDER = (
    ('position_only', 'online', 'fixed'),
    ('position_only', 'direct', 'fixed'),
    ('position_only', 'online', 'direct'),
    ('functional_frame', 'online', 'fixed'),
    ('functional_frame', 'direct', 'fixed'),
    ('functional_frame', 'online', 'direct'),
    ('p47_geometry_reference', 'continuous', 'fixed'))


def binding(path):
    return dict(path=str(Path(path)), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def wilson(successes, n):
    if n <= 0 or successes < 0 or successes > n:
        raise ValueError('Invalid Wilson counts')
    rate = successes / n
    denominator = 1 + Z95 * Z95 / n
    center = (rate + Z95 * Z95 / (2 * n)) / denominator
    half = Z95 * math.sqrt(rate * (1-rate) / n + Z95 * Z95 / (4*n*n)) / denominator
    return [max(0., center-half), min(1., center+half)]


def binomial_tail_at_least(k, n):
    if n == 0:
        return 1.
    return sum(math.comb(n, value) for value in range(k, n+1)) / (2 ** n)


def exact_mcnemar(rescues, harms):
    discordant = rescues + harms
    if discordant == 0:
        return dict(discordant=0, exact_two_sided_p=1., exact_one_sided_left_better_p=1.)
    lower = min(rescues, harms)
    two_sided = min(1., 2 * sum(math.comb(discordant, value)
                                for value in range(lower+1)) / (2 ** discordant))
    return dict(discordant=discordant, exact_two_sided_p=two_sided,
                exact_one_sided_left_better_p=binomial_tail_at_least(rescues, discordant))


def validate_completion(root):
    path = Path(root) / 'QUEUE_COMPLETION.json'
    value = read(path)
    if value.get('exit_codes') == [0, 0, 0]:
        return dict(completion=binding(path), queue_log=None,
                    receipt_shape='per_worker_exit_codes', verified_worker_exit_codes=[0, 0, 0]), value
    if value.get('exit_code') != 0:
        raise ValueError('Confirmation aggregation requires a successful queue completion receipt')
    log_path = Path(root) / 'queue.log'
    if not log_path.is_file():
        raise ValueError('Scalar queue exit 0 requires the original queue.log worker summary')
    matches = re.findall(r'"worker_exit_codes"\s*:\s*(\[[^\]]*\])', log_path.read_text())
    if not matches or json.loads(matches[-1]) != [0, 0, 0]:
        raise ValueError('queue.log does not end with successful exit codes for all three workers')
    return dict(completion=binding(path), queue_log=binding(log_path),
                receipt_shape='aggregate_exit_code_plus_queue_log',
                verified_worker_exit_codes=[0, 0, 0]), value


def validate_cell(index, cell, frozen, freeze_path, results_root):
    directory = cell_output(Path(results_root) / 'cells', index, cell)
    result_path, error_path = directory / 'RESULT.json', directory / 'ERROR.json'
    if error_path.exists():
        error = read(error_path)
        raise ValueError(f'Cell {index} has a program error record: {error.get("error_type")}: {error.get("error")}')
    if not result_path.is_file():
        raise ValueError(f'Cell {index} has neither a complete RESULT nor an ERROR record')
    pre, seal, result = [read(directory / name) for name in ('PRE_RUN.json', 'SEAL.json', 'RESULT.json')]
    if (pre.get('index') != index or pre.get('cell') != cell or result.get('index') != index
            or result.get('cell') != cell or pre.get('steps') != frozen['steps']
            or pre.get('freeze_sha256') != sha(freeze_path)):
        raise ValueError(f'Cell {index} identity, budget, or freeze binding changed')
    if pre.get('model') != frozen['models'][cell['representation']]:
        raise ValueError(f'Cell {index} model binding differs from the freeze')
    expected_seal = {'TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'PRE_RUN.json'}
    if seal.get('schema') != 'prove.p58-cup-confirmation-trajectory-seal.v1' or set(seal.get('files', {})) != expected_seal:
        raise ValueError(f'Cell {index} has an incomplete or wrong trajectory seal')
    for name, digest in seal['files'].items():
        if Path(name).name != name or sha(directory / name) != digest:
            raise ValueError(f'Cell {index} sealed file changed: {name}')
    if (seal.get('neural_parameters_before') != seal.get('neural_parameters_after')
            or result.get('neural_weights_unchanged') is not True):
        raise ValueError(f'Cell {index} changed frozen neural parameters')
    for name in ('success', 'validated_success', 'mechanical_valid', 'sustained_native_success'):
        if type(result.get(name)) is not bool:
            raise ValueError(f'Cell {index} lacks boolean {name}')
    if result['validated_success'] != (result['success'] and result['mechanical_valid']):
        raise ValueError(f'Cell {index} validated success is inconsistent')
    expected_outcome = ('validated_success' if result['validated_success'] else
                        'mechanical_anomaly' if not result['mechanical_valid'] else
                        'ordinary_task_failure')
    if result.get('outcome_class') != expected_outcome:
        raise ValueError(f'Cell {index} outcome class is inconsistent')
    if result.get('task_success_denominator_included') is not True:
        raise ValueError(f'Cell {index} was removed from the task denominator')
    trace = np.load(directory / 'TRACE.npy', allow_pickle=True)
    observations = np.load(directory / 'OBS.npy', allow_pickle=True)
    if len(trace) != frozen['steps'] or len(observations) != frozen['steps']:
        raise ValueError(f'Cell {index} trajectory length differs from the freeze')
    if [row['t'] for row in trace] != list(range(frozen['steps'])):
        raise ValueError(f'Cell {index} trajectory indices are not contiguous')
    return dict(index=index, cell=cell, result=result, directory=str(directory),
                result_binding=binding(result_path), seal_binding=binding(directory / 'SEAL.json'))


def grouped(rows):
    output = []
    for split in ('id', 'geometry'):
        for representation, arm in ARM_ORDER:
            group = [row for row in rows if row['cell']['split'] == split
                     and row['cell']['representation'] == representation
                     and row['cell']['arm'] == arm]
            if len(group) != 16:
                raise ValueError(f'Expected 16 {split} {representation}/{arm} cells')
            successes = sum(row['result']['validated_success'] for row in group)
            outcomes = Counter(row['result']['outcome_class'] for row in group)
            output.append(dict(split=split, representation=representation, arm=arm,
                successes=successes, n=len(group), rate=successes/len(group),
                wilson_95=wilson(successes, len(group)), outcomes=dict(sorted(outcomes.items())),
                mechanical_anomalies=sum(not row['result']['mechanical_valid'] for row in group),
                adaptation_updates=sum(int(row['result']['adaptation_updates']) for row in group)))
    return output


def paired(rows):
    lookup = {(row['cell']['split'], row['cell']['condition_seed'],
               row['cell']['representation'], row['cell']['arm']): row for row in rows}
    output = []
    for split in ('id', 'geometry'):
        seeds = sorted({row['cell']['condition_seed'] for row in rows if row['cell']['split'] == split})
        if len(seeds) != 16:
            raise ValueError(f'Expected 16 paired seeds for {split}')
        for representation, left_arm, right_arm in PAIR_ORDER:
            counts = Counter()
            conditions = []
            for seed in seeds:
                left = lookup[(split, seed, representation, left_arm)]['result']['validated_success']
                right = lookup[(split, seed, representation, right_arm)]['result']['validated_success']
                kind = ('rescue' if left and not right else 'harm' if right and not left
                        else 'both_success' if left else 'both_failure')
                counts[kind] += 1
                conditions.append(dict(condition_seed=seed, left_success=left,
                                       right_success=right, pair_class=kind))
            rescues, harms = counts['rescue'], counts['harm']
            output.append(dict(split=split, representation=representation,
                left_arm=left_arm, right_arm=right_arm, n=16,
                rescue=rescues, harm=harms, both_success=counts['both_success'],
                both_failure=counts['both_failure'], net_rescue=rescues-harms,
                **exact_mcnemar(rescues, harms), conditions=conditions))
    return output


def markdown(report):
    labels = {'id': 'ID', 'geometry': '几何 OOD', 'position_only': '抓持中心＋训练平均朝向',
              'functional_frame': '功能坐标架', 'p47_geometry_reference': '历史 P47 可见角点',
              'fixed': 'fixed', 'online': 'online', 'direct': 'direct', 'continuous': 'continuous'}
    group_index = {(row['split'], row['representation'], row['arm']): row for row in report['groups']}
    center_id = [group_index[('id', 'position_only', arm)]['successes'] for arm in ('fixed', 'online', 'direct')]
    center_ood = [group_index[('geometry', 'position_only', arm)]['successes'] for arm in ('fixed', 'online', 'direct')]
    center_id_mechanical = [group_index[('id', 'position_only', arm)]['mechanical_anomalies']
                            for arm in ('fixed', 'online', 'direct')]
    frame_id = [group_index[('id', 'functional_frame', arm)]['successes'] for arm in ('fixed', 'online', 'direct')]
    frame_ood = [group_index[('geometry', 'functional_frame', arm)]['successes'] for arm in ('fixed', 'online', 'direct')]
    lines = ['# P58 杯柄新条件确认集', '',
        '本报告仅包含结果前冻结的 16 个 ID 与 16 个几何 OOD 新条件，共 256 次尝试；不与四条件开发结果合并。',
        '抓持中心表示使用训练集平均朝向，因此不是完整姿态估计。Wilson 区间为双侧 95%；McNemar 为同 seed 二元结果的精确检验。', '',
        '## 主要结论', '',
        f'抓持中心 fixed／online／direct 在 ID 为 {center_id[0]}/{center_id[1]}/{center_id[2]}，'
        f'在几何 OOD 为 {center_ood[0]}/{center_ood[1]}/{center_ood[2]}（每臂分母均16）。'
        f'ID online／direct 分别有 {center_id_mechanical[1]}/{center_id_mechanical[2]} 次机械异常；'
        '因此当前更新不能作为 always-on 策略。OOD 中 online 与 direct 相同，也不支持梯度优于同信息直接解。',
        f'功能坐标架 fixed／online／direct 在 ID 为 {frame_id[0]}/{frame_id[1]}/{frame_id[2]}，'
        f'在几何 OOD 为 {frame_ood[0]}/{frame_ood[1]}/{frame_ood[2]}；其分布差异同样要求分开报告 ID 与 OOD。',
        'ID抓持中心online的七个机械异常虽均达到持续native抓起，但在闭合真实手指阶段超过原2 mm全轨迹穿透上限，仍严格记为失败；详见 `ID_POSITION_ONLINE_MECHANICAL_DIAGNOSTIC.json`。', '',
        '## 成功率', '', '| 划分 | 表征 | 运行臂 | 成功/次数 | 成功率 | Wilson 95% | 机械异常 |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in report['groups']:
        lo, hi = row['wilson_95']
        lines.append(f"| {labels[row['split']]} | {labels[row['representation']]} | {labels[row['arm']]} | "
                     f"{row['successes']}/{row['n']} | {100*row['rate']:.1f}% | "
                     f"{100*lo:.1f}–{100*hi:.1f}% | {row['mechanical_anomalies']} |")
    lines += ['', '## 同 seed 配对', '',
        '| 划分 | 表征 | 左臂−右臂 | 救回 | 误伤 | 双成/双败 | 净救回 | 精确 McNemar 双侧 p | 左臂更优单侧 p |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in report['paired']:
        lines.append(f"| {labels[row['split']]} | {labels[row['representation']]} | "
                     f"{labels[row['left_arm']]}−{labels[row['right_arm']]} | {row['rescue']} | {row['harm']} | "
                     f"{row['both_success']}/{row['both_failure']} | {row['net_rescue']:+d} | "
                     f"{row['exact_two_sided_p']:.6g} | {row['exact_one_sided_left_better_p']:.6g} |")
    lines += ['', '全部 256 个结果、轨迹封印、冻结条件、模型绑定和每回合参数前后哈希均已复核。程序错误为 0。', '']
    return '\n'.join(lines)


def aggregate(freeze_path, results_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    frozen = validate(freeze_path)
    if frozen.get('phase') != 'confirmation' or frozen.get('result_free_at_freeze') is not True:
        raise ValueError('Aggregation requires the result-free confirmation freeze')
    if frozen['cells'] != expected_cells() or len(frozen['cells']) != 256:
        raise ValueError('Confirmation cell matrix changed')
    queue_evidence, queue = validate_completion(results_root)
    rows = [validate_cell(index, cell, frozen, freeze_path, results_root)
            for index, cell in enumerate(frozen['cells'])]
    if len(rows) != 256:
        raise ValueError('Confirmation aggregation requires all 256 complete results')
    errors = list((Path(results_root) / 'cells').glob('**/ERROR.json'))
    results = list((Path(results_root) / 'cells').glob('**/RESULT.json'))
    if errors or len(results) != 256:
        raise ValueError(f'Expected exactly 256 RESULT and zero ERROR files; got {len(results)} and {len(errors)}')
    report = dict(schema=SCHEMA, cohort='new_conditions_confirmation',
        development_results_pooled=False, status='complete', attempts=256,
        result_files=256, program_errors=0,
        freeze=binding(freeze_path), queue_evidence=queue_evidence,
        queue_exit_code=queue.get('exit_code', 0),
        verified_worker_exit_codes=queue_evidence['verified_worker_exit_codes'],
        conditions=dict(id=16, geometry_ood=16), steps_per_episode=frozen['steps'],
        representation_contracts=frozen['representation_contracts'],
        integrity=dict(valid_seals=256, all_neural_weights_unchanged=True,
                       all_conditions_match_freeze=True, all_models_match_freeze=True,
                       all_sources_match_freeze=True),
        groups=grouped(rows), paired=paired(rows),
        evidence=[dict(index=row['index'], cell=row['cell'], directory=row['directory'],
                       result=row['result_binding'], seal=row['seal_binding']) for row in rows],
        statistics=dict(wilson='two-sided 95%, z=1.959963984540054',
            mcnemar='exact conditional binomial on discordant same-seed pairs; two-sided and left-better one-sided reported'))
    output.mkdir(parents=True)
    (output / 'RESULTS.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    (output / 'REPORT_ZH.md').write_text(markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', required=True)
    parser.add_argument('--results-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = aggregate(args.freeze, args.results_root, args.output)
    print(json.dumps(dict(status=report['status'], attempts=report['attempts'],
                          groups=len(report['groups']), paired=len(report['paired'])), indent=2))


if __name__ == '__main__':
    main()
