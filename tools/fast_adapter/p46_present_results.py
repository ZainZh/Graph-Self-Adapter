"""Present a complete P46 summary; never run a robot or recompute inference.

Example (from repository root)::

    python3 -m tools.fast_adapter.p46_present_results \
      --results evaluations/fast_adapter_p46_representation_validation_v01/summary/RESULTS.json \
      --out evaluations/fast_adapter_p46_representation_validation_v01/presentation_v01

Optional --video-index accepts a JSON list (or {"videos": [...]}) of dictionaries
with task, path, description, manifest. Paths are relative to that index, or
absolute local files. Manifests use prove.p45-video-derivative.v1; provenance and
the video hash must be present. Evidence paths are resolved under --evidence-root
(the repository by default), without importing simulation or model code.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[2]
TASKS = {'cup': '杯柄抓持抬升', 'insert': '已持物插入', 'slider': '滑动把手拉开'}
REPS = {'geometry': 'G · 可见几何', 'absolute_fusion': 'F · 绝对视觉融合',
        'relative_fusion': 'R · 角色相对融合'}
SPLITS = {'id': 'ID · 同分布', 'geometry': 'OOD · 几何变化'}
MODELS = (11, 22)
BASES = {'cup': 470000, 'insert': 471000, 'slider': 472000}
GROUP = ('task', 'split', 'rep', 'model_seed', 'arm')
IDENTITY = ('task', 'split', 'condition_seed', 'rep', 'model_seed', 'arm')
METHOD_NOTE = ('成功率先在每个物体条件内平均模型 11 / 22，再对条件平均；'
               '两模型不是额外独立物体样本。ID 每任务 16 条件，OOD 每任务 32 条件。'
               '主指标为任务成功且全程机械有效；机械异常仍在分母内。')
CI_NOTE = ('95% 区间为未校正的条件重采样区间，不是家族同时区间；'
           'p 值为条件级双侧配对 sign-flip，主要三项用 Holm3，次要六项用 Holm6。'
           '统计结论仅针对这两个固定训练 checkpoint。')
MECHANISM = ('模型把当前可见信息编码为几何残差，训练得到的低维 z 经射线一致性损失在线更新；'
             '模型权重与执行控制器不更新。固定 z 臂也读取当前视觉。'
             '杯柄和滑块在进入闭合阶段前冻结静态几何估计，随后按锚点执行；固定插槽可继续适配。'
             '三任务各自独立训练外围模型：每任务 96 个训练条件、12 个开发条件，'
             '三种表征各用两个模型种子，每模型 2,000 次外围更新。P46 沿用这些冻结模型，不新增训练。'
             '这是多任务重复验证，不是同一模型跨任务的零样本迁移。')
TOKEN_NOTE = ('G / F / R 使用相同的 95 维输入接口、96 维学习 token 和 8 维持久 latent z。'
              '输入中 47 维来自可见角点、标定相机位姿、机械臂本体位姿、角色框及可见性；'
              '另 48 维为冻结 DINOv3 / LiLa 的训练集 PCA 特征。G 将这 48 维置零；'
              'F 使用绝对图像空间特征；R 对视觉空间矩和角色框进行相对化、归一化。'
              'z 经学习的低秩解码器修正四个三维角点，控制器据此确定接近位置和朝向；'
              '它不是自由生成动作的 VLA，也不是三个 256 维角色 token 的 cross-attention 模型。')
OPTIMIZATION_NOTE = ('每次有效反馈累积到至少 3 帧后，在线臂对 z 做 3 次梯度更新，最多使用前 12 帧加最近 20 帧。'
                     '更新采用学习解码器的 Jacobian 与解析射线构造的阻尼 Gauss–Newton 预条件，步长 0.65，并限制更新范数；'
                     '因此不能将结果泛化为任意预测损失配普通 SGD 都有效。'
                     '外层用训练几何标签，同时监督更新前后的角点误差；没有直接对整段成功率做可微优化，'
                     '也没有在 P46 测试时使用几何标签。')
LIMITS = ('方法依赖显式可见角点、模拟器真实可见语义部件 mask、相机标定、PnP 和固定执行先验；'
          '不是纯 RGB 方法。DINOv3 / LiLa 冻结，几何监督只用于旧训练，运行时不读目标真值。'
          'R 同时包含相对化与归一化改动，不是单因素 attention 消融。'
          '它没有显式学习动作条件后果预测；本轮不证明元训练必要性、任意物体泛化、真机效果或完整 RSI。'
          '完整 Panda 在 MuJoCo 中执行，插入假设开始时已刚性持物；不是官方 LIBERO 成绩。')
ACTION_NOTE = ('同状态动作诊断比较的是进入步长裁剪、IK 与伺服之前的请求目标，'
               '不能单独当作实际运动的物理反事实。两臂有相同公共信息权限与初始观察动作，'
               '后续闭环图像序列可以不同。')
GEOMETRY_NOTE = ('条件是程序化几何变体，不是未知真实类别资产。杯柄 / 滑块把手宽度从 ID 的 25–35 mm '
                 '扩大到 OOD 的 37–50 mm，高度从 27–38 mm 扩大到 45–60 mm；'
                 'yaw 从 ±10° 变为正负两侧 20–40°，物体高度也从 60–80 mm 增至 90–120 mm。'
                 '插入物横截面全尺寸从 20–28 × 14–20 mm 增至 32–40 × 22–28 mm，'
                 '长度从 45–60 mm 增至 65–80 mm；roll / pitch 从 ±5° 增至 ±18°，yaw 从 ±10° 增至 ±40°。'
                 '插槽每侧间隙仍为 1.5–3 mm。两分布的 XY 随机偏移同为 ±25 mm，'
                 '因此不声称训练范围外的目标 XY 位置泛化。质量、摩擦、力度与执行能力保持固定；本轮不研究力适配。')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(cell):
    return tuple(cell[key] for key in IDENTITY)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def pct(value):
    return f'{100 * value:.1f}%'


def pp(value):
    return f'{100 * value:+.1f}'


def pvalue(value):
    return f'{value:.4f}' if value >= .0001 else f'{value:.2g}'


def relative_link(path, out):
    return quote(os.path.relpath(Path(path).resolve(), Path(out).resolve()), safe='/')


def validate_report(report):
    # This guard precedes every read of an evidence result or video manifest.
    check(report.get('schema') == 'prove.p46-summary.v1'
          and report.get('campaign') == 'p46_representation_validation', 'Expected a P46 summary')
    check(report.get('status') == 'complete' and report.get('formal_inference_ready') is True
          and report.get('phase') == 'independent_validation',
          'Presentation requires status=complete and formal_inference_ready=true; unfinished outcomes are not opened')
    check(report.get('prespecified_tasks') == list(TASKS), 'All three prespecified tasks are required')
    check(report.get('supplemental_scope', {}).get('status') == 'complete'
          and report['supplemental_scope'].get('n_completed_cells') == 120,
          'All 120 supplemental cells must also be complete')
    check(report.get('n_declared_main_cells') == 1728 and report.get('n_planned_cells') == 1848
          and report.get('n_undeclared_required_main_cells') == 0
          and report.get('cell_status_counts') == {'completed': 1848}
          and not report.get('formal_scope_gaps'), 'The full 1848-cell P46 scope must be complete')
    groups = {}
    for group in report['groups']:
        key = tuple(group[key] for key in GROUP)
        check(key not in groups, 'Duplicate summary group')
        n = group['n_planned']
        check(type(n) is int and n > 0 and group['n_observed'] == n and group['n_unknown'] == 0,
              'A presentation group is incomplete')
        for metric in ('k_validated', 'k_native', 'k_task_success', 'mechanical_anomalies'):
            check(type(group[metric]) is int and 0 <= group[metric] <= n, f'Invalid {metric}')
        check(math.isclose(group['validated_rate'], group['k_validated'] / n, abs_tol=1e-12),
              'Group success count and rate disagree')
        groups[key] = group
    expected = set()
    for task in TASKS:
        for split, n in (('id', 16), ('geometry', 32)):
            for rep in REPS:
                for model in MODELS:
                    for arm in ('fixed', 'continuous'):
                        key = task, split, rep, model, arm
                        check(key in groups and groups[key]['n_planned'] == n, 'Incomplete main group matrix')
                        expected.add(key)
        for model in MODELS:
            for arm in ('prefix3', 'mismatched'):
                key = task, 'geometry', 'relative_fusion', model, arm
                check(key in groups and groups[key]['n_planned'] == 8, 'Incomplete supplemental group matrix')
                expected.add(key)
        key = task, 'geometry', 'geometry', None, 'multiview'
        check(key in groups and groups[key]['n_planned'] == 8, 'Missing eight-cell multiview reference')
        expected.add(key)
    check(set(groups) == expected, 'Unexpected P46 summary groups')
    expected_comparisons = {
        'primary_comparisons': [('relative_fusion', 'continuous', 'relative_fusion', 'fixed')],
        'secondary_comparisons': [('relative_fusion', 'continuous', 'absolute_fusion', 'continuous'),
                                  ('absolute_fusion', 'continuous', 'geometry', 'continuous')]}
    for family, pairs in expected_comparisons.items():
        records = report[family]
        seen = set()
        for row in records:
            key = row['task'], row['left']['rep'], row['left']['arm'], row['right']['rep'], row['right']['arm']
            check(key not in seen, 'Duplicate formal comparison')
            seen.add(key)
            check(row['status'] == 'complete' and row['n_conditions_complete'] == 32
                  and row['n_conditions_planned'] == 32 and row['expected_model_seeds'] == list(MODELS),
                  'Formal comparisons require 32 complete conditions and the two fixed models')
            for metric in ('rate_left', 'rate_right', 'p_raw', 'p_holm'):
                check(number(row[metric]) and 0 <= row[metric] <= 1, f'Invalid comparison {metric}')
            check(number(row['difference']) and math.isclose(row['difference'], row['rate_left']-row['rate_right'], abs_tol=1e-12),
                  'Comparison difference does not match its rates')
            interval = row['bootstrap_95_ci']
            check(len(interval) == 2 and all(number(v) for v in interval) and -1 <= interval[0] <= interval[1] <= 1,
                  'Invalid condition bootstrap interval')
            check(row['positive_difference_supported'] == (row['difference'] > 0 and row['p_holm'] < .05)
                  and row['negative_difference_supported'] == (row['difference'] < 0 and row['p_holm'] < .05),
                  'Comparison support flags disagree with Holm-adjusted results')
            for side in ('left', 'right'):
                config = row[side]
                average = sum(groups[(row['task'], 'geometry', config['rep'], model, config['arm'])]['validated_rate']
                              for model in MODELS) / 2
                check(math.isclose(average, row[f'rate_{side}'], abs_tol=1e-12), 'Comparison and group rates disagree')
        check(seen == {(task, *pair) for task in TASKS for pair in pairs}, 'Incomplete comparison family')
    return groups


def main_rows(groups, task):
    rows = []
    for split in SPLITS:
        for rep in REPS:
            arms = {}
            for arm in ('fixed', 'continuous'):
                model_groups = [groups[(task, split, rep, model, arm)] for model in MODELS]
                arms[arm] = dict(rate=sum(g['validated_rate'] for g in model_groups) / 2,
                                 counts={str(m): f"{g['k_validated']}/{g['n_planned']}" for m, g in zip(MODELS, model_groups)},
                                 mechanical=sum(g['mechanical_anomalies'] for g in model_groups))
            rows.append(dict(split=split, rep=rep, **arms))
    return rows


def bound_path(binding, root):
    path = Path(binding['path'])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    check(path.is_file() and sha(path) == binding['sha256'], f'Changed or unavailable bound file: {path}')
    return path


def supplemental_rows(report, groups, root):
    """Read only hash-bound R fixed/online results for the prescribed eight seeds."""
    selected, bindings = {}, []
    for row in report['evidence']:
        cell = row['cell']
        if (cell['task'] in TASKS and cell['split'] == 'geometry'
                and BASES[cell['task']]+100 <= cell['condition_seed'] < BASES[cell['task']]+108
                and cell['rep'] == 'relative_fusion' and cell['arm'] in ('fixed', 'continuous')):
            check(row['status'] == 'completed' and cell['model_seed'] in MODELS, 'Incomplete eight-condition baseline')
            key = identity(cell)
            check(key not in selected, 'Duplicate eight-condition baseline cell')
            result_path = bound_path(row['binding'], root)
            seal_path = bound_path(row['seal'], root)
            result, seal = read_json(result_path), read_json(seal_path)
            check(identity(result) == key and result['phase'] == 'independent_validation', 'Bound result identity mismatch')
            check(type(result['validated_success']) is bool and type(result['mechanical_valid']) is bool
                  and type(result['success']) is bool
                  and result['validated_success'] == (result['success'] and result['mechanical_valid'])
                  and result.get('task_success_denominator_included') is True, 'Invalid baseline endpoint')
            check(seal.get('schema') == 'prove.p45-trajectory-seal.v1'
                  and seal['neural_parameters_before'] == seal['neural_parameters_after']
                  and {'PRE_RUN.json', 'TRACE.npy', 'OBS.npy', 'TERMINAL.npy'} <= set(seal['files']),
                  'Baseline lacks a complete unchanged-weight trajectory seal')
            selected[key] = result['validated_success']
            bindings.append(dict(result=dict(path=str(result_path), sha256=row['binding']['sha256']),
                                 seal=dict(path=str(seal_path), sha256=row['seal']['sha256'])))
    output = {}
    for task in TASKS:
        rows = []
        primary = next(row for row in report['primary_comparisons'] if row['task'] == task)
        condition_values = {row['condition_seed']: row for row in primary['condition_values']}
        for arm, label in (('fixed', 'R · 固定 z'), ('continuous', 'R · 在线 z')):
            counts = {}
            for model in MODELS:
                keys = [(task, 'geometry', seed, 'relative_fusion', model, arm)
                        for seed in range(BASES[task]+100, BASES[task]+108)]
                check(all(key in selected for key in keys), 'Missing same-eight-condition baseline evidence')
                counts[str(model)] = sum(selected[key] for key in keys)
            for seed in range(BASES[task]+100, BASES[task]+108):
                mean = sum(selected[(task, 'geometry', seed, 'relative_fusion', model, arm)] for model in MODELS) / 2
                side = 'left' if arm == 'continuous' else 'right'
                check(seed in condition_values and mean == condition_values[seed][side],
                      'Eight-condition result and aggregate condition values disagree')
            rows.append(dict(label=label, arm=arm, rate=sum(counts.values()) / 16,
                             counts={key: f'{value}/8' for key, value in counts.items()}, physical_cells=16))
        for arm, label in (('prefix3', 'R · 前三次更新后固定'), ('mismatched', 'R · 因果错配 z')):
            counts = {str(model): groups[(task, 'geometry', 'relative_fusion', model, arm)]['k_validated'] for model in MODELS}
            rows.append(dict(label=label, arm=arm, rate=sum(counts.values()) / 16,
                             counts={key: f'{value}/8' for key, value in counts.items()}, physical_cells=16))
        multiview = groups[(task, 'geometry', 'geometry', None, 'multiview')]['k_validated']
        rows.append(dict(label='可见几何 · 普通多视图拟合', arm='multiview', rate=multiview / 8,
                         counts={'single': f'{multiview}/8'}, physical_cells=8))
        output[task] = rows
    check(len(selected) == 96, 'Expected 96 bound same-eight-condition baseline cells')
    return output, bindings


def comparison_label(row):
    if row['right']['arm'] == 'fixed':
        return 'R 在线 − R 固定'
    return 'R 在线 − F 在线' if row['left']['rep'] == 'relative_fusion' else 'F 在线 − G 在线'


def support_text(row):
    if row['positive_difference_supported']:
        return '支持正向差异'
    if row['negative_difference_supported']:
        return '支持负向差异'
    return '差异未确立'


def conclusions(report):
    primary = report['primary_comparisons']
    positives = [TASKS[row['task']] for row in primary if row['positive_difference_supported']]
    negatives = [TASKS[row['task']] for row in primary if row['negative_difference_supported']]
    undecided = [TASKS[row['task']] for row in primary
                 if not row['positive_difference_supported'] and not row['negative_difference_supported']]
    items = []
    if positives:
        items.append('R 的在线收益获得条件级统计支持：'+'、'.join(positives)+'。')
    if negatives:
        items.append('R 在线更新出现有统计支持的退步：'+'、'.join(negatives)+'。')
    if undecided:
        items.append('R 在线收益尚未确立：'+'、'.join(undecided)+'；未确立不等于等效。')
    for row in report['secondary_comparisons']:
        items.append(f"{TASKS[row['task']]}的{comparison_label(row)}：{support_text(row)}（{pp(row['difference'])} 个百分点）。")
    items.append('R−F 与 F−G 是各自预定的比较；不能由二者推导未检验的 R−G 显著性，也不能笼统宣称 DINO 普遍优于几何。')
    return items


def load_videos(index_path):
    if index_path is None:
        return []
    index_path = Path(index_path).resolve()
    source = read_json(index_path)
    entries = source if isinstance(source, list) else source['videos']
    result = []
    for item in entries:
        check(item['task'] in TASKS, 'Video task outside P46')
        paths = {}
        for key in ('path', 'manifest'):
            check('://' not in str(item[key]), 'Videos must use existing local files')
            candidate = Path(item[key])
            paths[key] = (candidate if candidate.is_absolute() else index_path.parent/candidate).resolve()
            check(paths[key].is_file(), f'Missing video {key}')
        manifest = read_json(paths['manifest'])
        check(manifest.get('schema') == 'prove.p45-video-derivative.v1' and manifest.get('status') == 'complete',
              'Video must have a complete derivative provenance manifest')
        check(manifest['output_sha256'].get(paths['path'].name) == sha(paths['path']), 'Video differs from its manifest')
        inputs = manifest.get('inputs', [])
        check(inputs and all(entry['identity']['task'] == item['task'] for entry in inputs), 'Video manifest task mismatch')
        origins = ['reconstruction' in entry.get('verified', {}) for entry in inputs]
        provenance = ('混合来源：含封存状态重建及原始执行 RGB' if any(origins) and not all(origins) else
                      '封存状态重建回放 · 非原始 RGB · 零物理积分' if all(origins) else
                      '封存原始执行 RGB · 注释展示副本')
        result.append(dict(task=item['task'], description=str(item['description']), **paths,
                           provenance=provenance, manifest_sha256=sha(paths['manifest'])))
    return result


def plot_results(report, main, out, font=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    choices = [Path(font)] if font else [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'), Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')]
    available = next((path for path in choices if path.is_file()), None)
    check(available is not None, 'Chinese plotting font unavailable; pass --font')
    font_manager.fontManager.addfont(str(available))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(available)).get_name(),
                         'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharey=True)
    for col, task in enumerate(TASKS):
        for row, split in enumerate(SPLITS):
            ax = axes[row, col]
            chosen = [r for r in main[task] if r['split'] == split]
            for i, item in enumerate(chosen):
                for shift, arm, color, label in ((-.18, 'fixed', '#94a3b8', '固定 z'), (.18, 'continuous', '#167a78', '在线 z')):
                    value = item[arm]['rate']
                    ax.bar(i+shift, value, .32, color=color, label=label if i == 0 else None)
                    ax.text(i+shift, value+.024, pct(value), ha='center', fontsize=8)
            ax.set_title(f'{TASKS[task]} · {SPLITS[split]}', loc='left', fontsize=11)
            ax.set_xticks(range(3), ['G 几何', 'F 绝对融合', 'R 相对融合'])
            ax.set_ylim(0, 1.13)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel('机械有效的任务成功率')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, frameon=False, bbox_to_anchor=(.98, .99))
    plot_title = ('合成数据自检 · 非实验成绩' if report.get('synthetic_test_fixture') else
                  'P46 · 完整 Panda 的三任务独立验证')
    fig.suptitle(plot_title, x=.055, ha='left', y=.98, fontsize=16)
    fig.text(.055, .022, '每个条件内先平均模型 11 / 22；ID 16 条件，OOD 32 条件。机械异常保留分母；不合并跨任务成功率。', fontsize=10)
    fig.tight_layout(rect=(.015, .05, 1, .94))
    for extension in ('png', 'svg'):
        fig.savefig(out/f'success_rates.{extension}', dpi=190)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for y, task in enumerate(TASKS):
        row = next(r for r in report['primary_comparisons'] if r['task'] == task)
        lo, hi = [100*v for v in row['bootstrap_95_ci']]
        delta = 100*row['difference']
        color = '#167a78' if row['positive_difference_supported'] else '#b5473a' if row['negative_difference_supported'] else '#64748b'
        ax.plot([lo, hi], [y, y], color=color, linewidth=2.5)
        ax.scatter([delta], [y], color=color, s=65, zorder=3)
        ax.text(1.02, y, f"{pp(row['difference'])} pp\nHolm3 p={pvalue(row['p_holm'])}", transform=ax.get_yaxis_transform(), va='center', fontsize=10)
    ax.axvline(0, color='#718096', ls='--', lw=1)
    ax.set_yticks(range(3), list(TASKS.values()))
    ax.set_ylim(2.5, -.5)
    ax.set_xlabel('OOD：R 在线 − R 固定（百分点）')
    ax.set_title(('合成数据自检 · 非实验成绩 · ' if report.get('synthetic_test_fixture') else '')+
                 '主要比较 · 条件级差异及 95% 区间', loc='left', pad=15)
    ax.grid(axis='x', alpha=.2)
    fig.text(.04, .025, '区间：条件重采样 10,000 次，未校正 95% 区间；显著性：双侧配对 sign-flip，Holm3。', fontsize=9)
    fig.tight_layout(rect=(0, .065, .87, 1))
    for extension in ('png', 'svg'):
        fig.savefig(out/f'primary_effects.{extension}', dpi=190)
    plt.close(fig)
    return str(available)


def main_table(rows):
    content = []
    for row in rows:
        fixed, online = row['fixed'], row['continuous']
        content.append([SPLITS[row['split']], REPS[row['rep']], f"{pct(fixed['rate'])} → {pct(online['rate'])}",
                        f"{pp(online['rate']-fixed['rate'])} pp", f"{fixed['counts']['11']} → {online['counts']['11']}",
                        f"{fixed['counts']['22']} → {online['counts']['22']}", f"{fixed['mechanical']} → {online['mechanical']}"])
    return ['分布', '表征', '固定 → 在线', '变化', '模型 11 · k/n', '模型 22 · k/n', '机械异常回合'], content


def contrast_table(rows):
    return ['任务', '比较', '差异（pp）', '95% 区间（pp）', '原始 p', 'Holm p', '结论'], [
        [TASKS[r['task']], comparison_label(r), pp(r['difference']),
         f"[{100*r['bootstrap_95_ci'][0]:.1f}, {100*r['bootstrap_95_ci'][1]:.1f}]",
         pvalue(r['p_raw']), pvalue(r['p_holm']), support_text(r)] for r in rows]


def supplement_table(rows):
    return ['方法 · 同 8 条件', '条件平均成功率', '模型 11 · k/n', '模型 22 · k/n', '独立物理回合'], [
        [r['label'], pct(r['rate']), r['counts'].get('11', '单份 '+r['counts'].get('single', '')),
         r['counts'].get('22', '—'), str(r['physical_cells'])] for r in rows]


def html_table(table):
    headers, rows = table
    return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def md_table(table):
    headers, rows = table
    escape = lambda value: str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(map(escape, headers))+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(escape, row))+' |' for row in rows])


def render_documents(report, main, supplements, videos, results_path, out):
    findings = conclusions(report)
    synthetic = report.get('synthetic_test_fixture') is True
    title = ('合成数据自检 · 非实验成绩 · ' if synthetic else '')+'P46：三任务神经几何适配'
    source = relative_link(results_path, out)
    blocks = [f'<header><div class="eyebrow">PROVE · P46 · 独立验证</div><h1>{html.escape(title)}</h1>',
              '<p class="lead">同一冻结模型，在新物体条件上比较固定 z 与在线 z；完整呈现支持、退步和未确立的结果。</p>',
              '<div class="chips"><span>3 个任务</span><span>1,728 主实验回合</span><span>120 补充回合</span><span>所有预定回合完成</span></div></header>',
              '<nav><a href="#conclusion">结论</a><a href="#main">成功率</a><a href="#inference">条件级统计</a><a href="#supplement">补充对照</a><a href="#scope">机制与边界</a></nav>',
              '<section id="conclusion"><h2>结果支持到哪里</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in findings)+'</ul></section>',
              '<section id="main"><h2>固定 → 在线：逐任务成功率</h2><p>'+METHOD_NOTE+'</p>',
              '<figure><img src="success_rates.png" alt="三任务G、F、R在ID和OOD的固定与在线成功率"><figcaption><a href="success_rates.svg">矢量图 SVG</a></figcaption></figure>']
    markdown = ['# '+title, '', *findings, '', METHOD_NOTE, '', '![逐任务成功率](success_rates.png)', '']
    for task in TASKS:
        table = main_table(main[task])
        blocks.extend([f'<h3>{TASKS[task]}</h3>', html_table(table)])
        markdown.extend(['## '+TASKS[task], '', md_table(table), ''])
    blocks.extend(['</section><section id="inference"><h2>OOD 条件级统计</h2><p>'+CI_NOTE+'</p>',
                   '<h3>主要：R 在线 − R 固定 · Holm3</h3>', html_table(contrast_table(report['primary_comparisons'])),
                   '<figure><img src="primary_effects.png" alt="R在线相对固定的条件级成功率差异与95%区间"><figcaption><a href="primary_effects.svg">矢量图 SVG</a></figcaption></figure>',
                   '<h3>次要：在线 R−F / F−G · Holm6</h3>', html_table(contrast_table(report['secondary_comparisons'])), '</section>',
                   '<section id="supplement"><h2>前 8 个 OOD 条件：补充对照</h2><p>以下所有方法使用同一组前 8 条件。固定 / 在线基准由这 8 条件的封存结果重新取子集，'
                   '不使用上方 32 条件的总体率。补充比较仅作描述，多视图每条件仅运行一次，不按两个神经模型重复计数。</p>'])
    markdown.extend(['## OOD 主要比较 · Holm3', '', CI_NOTE, '', md_table(contrast_table(report['primary_comparisons'])), '',
                     '## OOD 次要比较 · Holm6', '', md_table(contrast_table(report['secondary_comparisons'])), '',
                     '## 同前 8 个 OOD 条件的补充对照', '', '固定 / 在线基准仅使用同前 8 条件；全部比较仅作描述，多视图参考不重复计数。', ''])
    for task in TASKS:
        seeds = f"{BASES[task]+100}–{BASES[task]+107}"
        blocks.extend([f'<h3>{TASKS[task]} <small>条件 {seeds}</small></h3>', html_table(supplement_table(supplements[task]))])
        markdown.extend([f'### {TASKS[task]} · {seeds}', '', md_table(supplement_table(supplements[task])), ''])
    control_note = ('prefix3 只允许前三次更新事件，每事件 3 个梯度步；mismatched 使用预定其他条件在不晚于当前时刻的 delta-z。'
                    '它们不能单独证明每一步更新都必要，也不要求梯度方法优于普通多视图拟合。'
                    '表中固定 / 在线行复用主实验子集，未增加物理回合。')
    blocks.extend(['<p class="note">'+control_note+'</p></section>'])
    markdown.extend([control_note, ''])
    if videos:
        blocks.extend(['<section><h2>封存个例视频</h2><p>视频用于审阅选定个例，不能代替全量统计；每段保留来源与选择说明。</p><div class="video-grid">'])
        markdown.extend(['## 封存个例视频', ''])
        for video in videos:
            video_link, manifest_link = relative_link(video['path'], out), relative_link(video['manifest'], out)
            description = html.escape(video['description'])
            blocks.append(f'<article><h3>{TASKS[video["task"]]}</h3><video controls preload="metadata" src="{video_link}"></video>'
                          f'<p>{description}</p><p class="note">{video["provenance"]} · <a href="{manifest_link}">来源清单</a></p></article>')
            markdown.extend([f'- [{TASKS[video["task"]]}：{video["description"]}]({video_link})。{video["provenance"]}；[来源清单]({manifest_link})。'])
        blocks.append('</div></section>')
        markdown.append('')
    blocks.extend(['<section id="scope"><h2>机制与适用边界</h2><p>'+MECHANISM+'</p><h3>Token 与 latent 的实际构造</h3><p>'+TOKEN_NOTE+
                   '</p><p>'+OPTIMIZATION_NOTE+'</p><h3>具体变化的几何范围</h3><p>'+GEOMETRY_NOTE+
                   '</p><h3>结论限制</h3><p>'+LIMITS+'</p><p>'+ACTION_NOTE+'</p></section>',
                   f'<footer><a href="{source}">完整汇总 RESULTS.json</a> · <a href="REPORT_ZH.md">中文报告</a> · <a href="PRESENTATION_MANIFEST.json">展示来源与哈希</a></footer>'])
    markdown.extend(['## 机制与适用边界', '', MECHANISM, '', TOKEN_NOTE, '', OPTIMIZATION_NOTE, '', GEOMETRY_NOTE, '', LIMITS, '', ACTION_NOTE, '',
                     f'[完整汇总]({source}) · [展示来源与哈希](PRESENTATION_MANIFEST.json)', ''])
    css = '''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f7;color:#203344;font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1450px;margin:auto;padding:42px 32px}header{padding:32px;background:#122d40;color:white;border-radius:18px}.eyebrow{font-size:12px;letter-spacing:.15em;color:#95ccd0}h1{font-size:34px;line-height:1.35;margin:14px 0}.lead{max-width:950px;color:#d8e6ed}.chips{display:flex;gap:12px;flex-wrap:wrap}.chips span{background:#254456;border-radius:30px;padding:5px 14px;font-size:13px}nav{display:flex;gap:26px;flex-wrap:wrap;padding:22px 4px}a{color:#116a82}section{background:white;margin:12px 0 26px;padding:28px;border:1px solid #e3e8ed;border-radius:14px}h2{font-size:25px;color:#155e65;margin:0 0 18px}h3{margin:28px 0 12px;font-size:19px}small{font-size:13px;font-weight:400;color:#687888;margin-left:12px}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:12px 14px;border-bottom:1px solid #e3e8ed;text-align:right;white-space:nowrap}th{background:#f2f6f7;color:#446074}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}tr:hover td{background:#f8fbfb}figure{margin:22px 0}img{display:block;width:100%;height:auto}figcaption{font-size:13px;margin:4px}.note{background:#f0f5f7;padding:12px 16px;color:#4f6577;font-size:14px;border-radius:6px}li{margin:6px 0}.video-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}video{width:100%;background:#132c3b;border-radius:8px}footer{padding:14px;color:#65798a;font-size:14px}@media(max-width:750px){main{padding:16px 12px}header,section{padding:20px}h1{font-size:26px}.video-grid{grid-template-columns:1fr}nav{gap:14px}}'''
    (out/'index.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                f'<title>{html.escape(title)}</title><style>{css}</style></head><body><main>'+''.join(blocks)+'</main></body></html>\n', encoding='utf-8')
    (out/'REPORT_ZH.md').write_text('\n'.join(markdown), encoding='utf-8')


def present(results, out, video_index=None, evidence_root=REPO, font=None):
    results, out, evidence_root = Path(results).resolve(), Path(out).resolve(), Path(evidence_root).resolve()
    report = read_json(results)
    groups = validate_report(report)
    main = {task: main_rows(groups, task) for task in TASKS}
    supplements, bindings = supplemental_rows(report, groups, evidence_root)
    videos = load_videos(video_index)
    # A separate immutable presentation directory prevents modifying summary/seals.
    check(not out.exists(), 'Use a new independent presentation directory; existing output is preserved')
    out.mkdir(parents=True)
    font_path = plot_results(report, main, out, font)
    render_documents(report, main, supplements, videos, results, out)
    manifest = dict(schema='prove.p46-presentation.v1', status='complete',
                    source=dict(path=str(results), sha256=sha(results)),
                    source_status=report['status'], formal_inference_ready=report['formal_inference_ready'],
                    generator=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                    synthetic_test_fixture=report.get('synthetic_test_fixture', False),
                    supplemental_baseline_conditions={task: list(range(BASES[task]+100, BASES[task]+108)) for task in TASKS},
                    supplemental_baseline_bindings=bindings, main_tables=main, supplemental_tables=supplements,
                    video_index=None if video_index is None else dict(path=str(Path(video_index).resolve()), sha256=sha(video_index)),
                    videos=[{k: str(v) if isinstance(v, Path) else v for k, v in video.items()} for video in videos],
                    font=font_path, statistics='Copied from complete P46 aggregate; no new inferential family',
                    outputs={path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})
    (out/'PRESENTATION_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return out/'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='Complete P46 summary/RESULTS.json')
    parser.add_argument('--out', required=True, help='New independent presentation directory')
    parser.add_argument('--video-index', help='Optional JSON list of task/path/description/manifest video records')
    parser.add_argument('--evidence-root', default=str(REPO), help='Root used for repository-relative bound evidence paths')
    parser.add_argument('--font', help='Optional Chinese font file')
    args = parser.parse_args()
    print(present(args.results, args.out, args.video_index, args.evidence_root, args.font))


if __name__ == '__main__':
    main()
