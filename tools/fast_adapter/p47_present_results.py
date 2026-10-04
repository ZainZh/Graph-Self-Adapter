"""Chinese P47 presentation from a completed main summary, with no live reads.

Run after main aggregation closes::

    python -m tools.fast_adapter.p47_present_results \
      --results evaluations/fast_adapter_p47_representation_solver_v01/summary/RESULTS.json \
      --out evaluations/fast_adapter_p47_representation_solver_v01/presentation_v01

Optional --video-index uses {"schema":"prove.p47-video-index.v1", "videos":
[{"task":..., "title":..., "path":..., "manifest":..., "kind":...,
"condition_seed":...}]}. Relative media paths are repository-relative under
--evidence-root. A missing index is shown as pending. Repeat --diagnostic for
opened-replay PREDICTIONS.json files; their times are local-solve measurements.
No cell RESULT, observation, model, simulator, or fresh inference is loaded.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

from tools.fast_adapter.p46_present_results import (
    check, html_table, number, pct, pp, pvalue, read_json, relative_link, sha,
)

REPO = Path(__file__).resolve().parents[2]
TASKS = {'cup': '杯柄抓持抬升', 'insert': '已持物插入', 'slider': '滑动把手拉开'}
REPS = {'geometry': 'G · 可见几何', 'relative_fusion': 'R · 角色相对融合'}
ARMS = {'fixed': '固定 z', 'continuous': '梯度更新（每次3步）', 'direct': '同 latent 直接求解'}
MODELS = (11, 22)
GROUP = ('task', 'split', 'rep', 'model_seed', 'arm')
FAMILIES = {
    'primary_A': ('A · 表征主比较', 3, 'R 梯度 − G 梯度'),
    'primary_B': ('B · 求解器主比较', 6, '梯度 − 同 latent 直接求解'),
    'support_C': ('C · 在线适配复现', 6, '梯度 − 固定 z'),
}
METHOD = ('每任务 32 个独立几何条件；神经方法先在条件内平均模型 11 / 22，再平均条件。'
          '因此每个神经配置有 64 个物理回合，但统计样本量为 32 条件。'
          '普通多视图每条件仅运行一次，每任务 32 回合，未复制成两份模型样本。'
          '主指标为任务成功且机械有效；机械异常保留分母，未知结果不能算作已观测失败。')
STATISTICS = ('差值为左侧减右侧，单位为百分点。区间为 10,000 次条件 bootstrap 的未校正 95% 区间；'
              '条件级双侧配对 sign-flip 的 p 值分别在 A 的 3 项、B 的 6 项、C 的 6 项内作 Holm 校正。'
              '未确立差异不等于等效；一个任务显著而另一个不显著，不构成任务差异检验。')
LIMITS = ('模型、DINOv3 / LiLa 与控制器沿用冻结 P45 版本；固定 z 同样读取当前视觉。'
          '梯度臂使用三次阻尼、限幅的 Gauss–Newton 更新；直接臂在相同学习基、当前 offset 和射线损失上'
          '作 float64 最小改变量伪逆求解，数值秩阈值为 1e-6，无阻尼或步长限制。'
          '二者区别包含正则化与更新幅度，不能据此宣称梯度形式本身不可替代。'
          '方法依赖可见语义部件 mask、相机标定、角点与 PnP 几何先验；杯柄和滑块进入操作移动阶段后冻结静态估计。'
          '完整 Panda 在 MuJoCo 中运行，插入从刚性持物开始；三任务分别训练。'
          '结论仅针对两个固定 checkpoint，不证明 DINO 普遍增益、梯度普遍最优、任意物体迁移、真机或完整 RSI。')


def validate_report(report):
    """Guard before any optional diagnostic/media read; never inspect live cells."""
    check(report.get('schema') == 'prove.p47-summary.v1'
          and report.get('campaign') == 'p47_representation_solver'
          and report.get('phase') == 'independent_validation'
          and report.get('status') == 'complete'
          and report.get('formal_inference_ready') is True,
          'Presentation requires the completed P47 independent main summary')
    check(report.get('n_planned_cells') == report.get('n_completed_cells') == 1248
          and report.get('n_unknown_cells') == report.get('n_undeclared_required_cells') == 0
          and report.get('n_unexpected_cells') == 0 and not report.get('formal_scope_gaps')
          and report.get('cell_status_counts') == {'completed': 1248},
          'All 1248 planned main cells must be completed before presentation')
    groups = {}
    for row in report['groups']:
        key = tuple(row[field] for field in GROUP)
        check(key not in groups, 'Duplicate summary group')
        check(row['n_planned'] == row['n_observed'] == 32 and row['n_unknown'] == 0,
              'Every completed group must retain all 32 conditions')
        for metric in ('k_validated', 'k_native', 'k_task_success', 'ordinary_task_failures', 'mechanical_anomalies'):
            check(type(row[metric]) is int and 0 <= row[metric] <= 32, f'Invalid {metric}')
        check(number(row['validated_rate']) and math.isclose(row['validated_rate'], row['k_validated']/32, abs_tol=1e-12),
              'Group rate and count disagree')
        groups[key] = row
    expected = {(task, 'geometry', rep, model, arm)
                for task in TASKS for rep in REPS for model in MODELS for arm in ARMS}
    expected |= {(task, 'geometry', 'geometry', None, 'multiview') for task in TASKS}
    check(set(groups) == expected, 'Expected exactly G/R x two models x three arms and one multiview group per task')
    for family, (_, count, _) in FAMILIES.items():
        comparisons = report[family + '_comparisons']
        expected_pairs = {
            (task, 'relative_fusion' if family == 'primary_A' else rep, 'continuous',
             'geometry' if family == 'primary_A' else rep,
             'continuous' if family == 'primary_A' else 'direct' if family == 'primary_B' else 'fixed')
            for task in TASKS for rep in (('geometry',) if family == 'primary_A' else REPS)}
        seen = set()
        for row in comparisons:
            key = (row['task'], row['left']['rep'], row['left']['arm'], row['right']['rep'], row['right']['arm'])
            check(key not in seen, 'Duplicate comparison')
            seen.add(key)
            check(row['status'] == 'complete' and row['n_conditions_complete'] == row['n_conditions_planned'] == 32
                  and row['expected_model_seeds'] == list(MODELS), 'Incomplete formal comparison')
            for metric in ('rate_left', 'rate_right', 'p_raw', 'p_holm'):
                check(number(row[metric]) and 0 <= row[metric] <= 1, f'Invalid comparison {metric}')
            check(number(row['difference']) and math.isclose(row['difference'], row['rate_left']-row['rate_right'], abs_tol=1e-12),
                  'Comparison difference and rates disagree')
            interval = row['bootstrap_95_ci']
            check(isinstance(interval, list) and len(interval) == 2 and all(number(value) for value in interval)
                  and -1 <= interval[0] <= interval[1] <= 1, 'Invalid 95% interval')
            for side in ('left', 'right'):
                config = row[side]
                rate = sum(groups[(row['task'], 'geometry', config['rep'], model, config['arm'])]['validated_rate']
                           for model in MODELS)/2
                check(math.isclose(rate, row['rate_'+side], abs_tol=1e-12), 'Comparison differs from grouped rates')
            check(row['positive_difference_supported'] == (row['difference'] > 0 and row['p_holm'] < .05)
                  and row['negative_difference_supported'] == (row['difference'] < 0 and row['p_holm'] < .05),
                  'Support flag differs from adjusted inference')
        check(len(comparisons) == count and seen == expected_pairs, 'Unexpected formal comparison family')
    return groups


def task_rows(groups, task):
    output = []
    for rep in REPS:
        for arm in ARMS:
            rows = [groups[(task, 'geometry', rep, model, arm)] for model in MODELS]
            output.append(dict(rep=rep, arm=arm, label=f'{REPS[rep]} · {ARMS[arm]}',
                               rate=sum(row['validated_rate'] for row in rows)/2,
                               successes=sum(row['k_validated'] for row in rows), runs=64, conditions=32,
                               per_model={str(model): f"{row['k_validated']}/32" for model, row in zip(MODELS, rows)},
                               unknown=sum(row['n_unknown'] for row in rows),
                               mechanical=sum(row['mechanical_anomalies'] for row in rows),
                               ordinary_failures=sum(row['ordinary_task_failures'] for row in rows)))
    row = groups[(task, 'geometry', 'geometry', None, 'multiview')]
    output.append(dict(rep='geometry', arm='multiview', label='普通多视图 · 独立单份参考',
                       rate=row['validated_rate'], successes=row['k_validated'], runs=32, conditions=32,
                       per_model={}, unknown=row['n_unknown'], mechanical=row['mechanical_anomalies'],
                       ordinary_failures=row['ordinary_task_failures']))
    return output


def load_videos(index_path, report, evidence_root):
    if index_path is None or not Path(index_path).is_file():
        return []
    source = read_json(index_path)
    check(source.get('schema') == 'prove.p47-video-index.v1', 'Expected P47 video index schema')
    evidence = {(row.get('binding', {}).get('sha256'), row.get('seal', {}).get('sha256')): row
                for row in report['evidence'] if row['status'] == 'completed'}
    videos = []
    for item in source['videos']:
        check(item['task'] in TASKS, 'Unknown video task')
        paths = {}
        for key in ('path', 'manifest'):
            check('://' not in str(item[key]), 'Video paths must be local files')
            path = Path(item[key])
            paths[key] = (path if path.is_absolute() else Path(evidence_root)/path).resolve()
            check(paths[key].is_file(), f'Missing video {key}')
        manifest = read_json(paths['manifest'])
        check(manifest.get('schema') == 'prove.p47-video-derivative.v1'
              and manifest.get('campaign') == 'p47_representation_solver'
              and manifest.get('status') == 'complete', 'Incomplete or non-P47 video manifest')
        check(manifest['output_sha256'].get(paths['path'].name) == sha(paths['path']), 'Video hash differs from its manifest')
        inputs = manifest.get('inputs', [])
        check(bool(inputs), 'Video must bind its input episodes')
        cells = []
        for entry in inputs:
            verified, identity = entry['verified'], entry['identity']
            check(all(isinstance(verified.get(name), str) and len(verified[name]) == 64
                      for name in ('RESULT.json', 'SEAL.json')), 'Video lacks complete result/seal hashes')
            match = evidence.get((verified.get('RESULT.json'), verified.get('SEAL.json')))
            check(match is not None, 'Video outcome/seal hashes are absent from the completed summary')
            cell = match['cell']
            check(all(identity[key] == cell[key] for key in ('task', 'split', 'rep', 'arm'))
                  and identity['seed'] == cell['condition_seed']
                  and cell['task'] == item['task'] and cell['condition_seed'] == item['condition_seed'],
                  'Video identities differ from the summary or index')
            cells.append(cell)
        reconstructed = ['reconstruction' in entry['verified'] for entry in inputs]
        provenance = ('混合来源：含封存状态重建及原始 RGB' if any(reconstructed) and not all(reconstructed) else
                      '封存状态重建 · 非原始 RGB · 零物理积分' if all(reconstructed) else '封存原始执行 RGB · 注释展示副本')
        videos.append(dict(task=item['task'], title=str(item.get('title', '结果对比')),
                           kind=str(item.get('kind', '')), condition_seed=item['condition_seed'],
                           **paths, provenance=provenance, cells=cells))
    return videos


def load_diagnostics(paths):
    rows, sources = [], []
    for path in paths:
        data = read_json(path)
        check(data.get('schema') == 'prove.p47-opened-replay-diagnostic.v1', 'Expected opened P47 replay diagnostic')
        cell = data['cell']
        check(cell['task'] in TASKS and cell['rep'] in REPS, 'Unexpected diagnostic configuration')
        source_rows = data['records']
        check(bool(source_rows), 'Empty diagnostic')
        for arm in ('continuous', 'direct'):
            values = [record['arms'][arm]['solve_seconds'] for record in source_rows]
            check(all(number(value) and value >= 0 for value in values), 'Invalid diagnostic solve time')
            later = ('—' if len(values) < 2 else
                     f'{1000*min(values[1:]):.3f}–{1000*max(values[1:]):.3f}')
            rows.append([TASKS[cell['task']], REPS[cell['rep']], str(cell['model_seed']), ARMS[arm],
                         str(len(values)), f'{1000*values[0]:.3f}', later])
        sources.append(dict(path=str(Path(path).resolve()), sha256=sha(path)))
    return rows, sources


def plot_success(main, out, synthetic=False, font=None):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    candidates = [Path(font)] if font else [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'), Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')]
    font_path = next((path for path in candidates if path.is_file()), None)
    check(font_path is not None, 'Chinese plotting font unavailable; pass --font')
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font_path)).get_name(),
                         'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), sharey=True)
    colors = {'fixed': '#94a3b8', 'continuous': '#147d80', 'direct': '#7963b5'}
    for ax, (task, rows) in zip(axes, main.items()):
        for i, rep in enumerate(REPS):
            for j, arm in enumerate(ARMS):
                row = next(row for row in rows if row['rep'] == rep and row['arm'] == arm)
                x = i + (j-1)*.23
                ax.bar(x, row['rate'], .21, color=colors[arm], label=ARMS[arm] if i == 0 else None)
                offset = .08 if arm == 'continuous' else .025
                ax.text(x, row['rate']+offset, pct(row['rate']), ha='center', fontsize=7.5)
        reference = rows[-1]
        ax.bar(2.25, reference['rate'], .27, facecolor='#eeeeee', edgecolor='#374151', hatch='//', label='普通多视图（单份）')
        ax.text(2.25, reference['rate']+.025, pct(reference['rate']), ha='center', fontsize=7.5)
        ax.set_xticks([0, 1, 2.25], ['G 几何\n64 回合', 'R 相对融合\n64 回合', '多视图参考\n32 回合'])
        ax.set_title(TASKS[task], loc='left')
        ax.set_ylim(0, 1.13)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('机械有效的任务成功率')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.51, .93), ncol=4, frameon=False)
    title = '合成自检数据 · 非实验成绩' if synthetic else 'P47 · 表征与求解器的三任务独立验证'
    fig.suptitle(title, x=.055, ha='left', y=.99, fontsize=16)
    fig.text(.055, .015, '每任务 32 个独立条件；神经方法在条件内平均两个固定模型。每根神经柱 64 回合，多视图柱 32 回合。', fontsize=9)
    fig.tight_layout(rect=(.015, .06, 1, .82))
    for suffix in ('png', 'svg'):
        fig.savefig(out/f'success_rates.{suffix}', dpi=180)
    plt.close(fig)
    return font_path


def render(report, main, videos, diagnostics, results, out):
    escape = html.escape
    synthetic = '<p class="alert">合成数据自检，以下数字不是实验成绩。</p>' if report.get('synthetic_test_fixture') else ''
    body = [f'<h1>P47：表征与求解器的三任务独立验证</h1>{synthetic}',
            f'<p>预定 {report["n_planned_cells"]:,} 回合，完成 {report["n_completed_cells"]:,}，未知 {report["n_unknown_cells"]}。'
            '主要结果与推断来自封存后的完整汇总，本页不重算显著性。</p>',
            f'<p>{METHOD}</p><img class="chart" src="success_rates.svg" alt="三任务 G/R 固定、梯度、直接求解与单份多视图成功率">']
    for task, rows in main.items():
        table = (['配置', '成功率', '成功 / 物理回合', '模型 11', '模型 22', '独立条件', '未知', '机械异常', '普通任务失败'],
                 [[row['label'], pct(row['rate']), f"{row['successes']}/{row['runs']}",
                   row['per_model'].get('11', '单份参考'), row['per_model'].get('22', '—'),
                   32, row['unknown'], row['mechanical'], row['ordinary_failures']] for row in rows])
        body.extend([f'<h2>{TASKS[task]}</h2>', html_table(table)])
    body.extend(['<h2>预定比较</h2>', f'<p>{STATISTICS}</p>'])
    for family, (title, size, description) in FAMILIES.items():
        table_rows = []
        for row in report[family+'_comparisons']:
            interval = ' 至 '.join(pp(value) for value in row['bootstrap_95_ci'])
            support = ('左侧更高' if row['positive_difference_supported'] else
                       '右侧更高' if row['negative_difference_supported'] else '差异尚未确立')
            rep = 'R − G' if family == 'primary_A' else REPS[row['left']['rep']]
            table_rows.append([TASKS[row['task']], rep, pp(row['difference']), interval,
                               pvalue(row['p_holm']), support])
        body.extend([f'<h3>{title} · Holm{size} · {description}</h3>',
                     html_table((['任务', '表征', '差值（pp）', '未校正 95% 区间（pp）', f'Holm{size} p', '结论'], table_rows))])
    body.extend(['<p class="note">普通多视图是额外的独立解析参考；其比较仅作描述，不属于 A/B/C 推断家族。</p>',
                 '<h2>局部求解时间</h2>'])
    if diagnostics:
        body.append(html_table((['任务', '表征', '模型', '求解器', '历史端点数', '首次调用（ms）', '后续调用范围（ms）'], diagnostics)))
        body.append('<p class="note">来自已开放开发观察的离线诊断，3 / 6 / 12 / 全历史端点共用输入与初始 z；'
                    '端点不是独立物体样本。每配置仅四个不同历史端点，未做预热或重复计时；首次调用明显受初始化影响，因此单列首次与后续范围，不能据此估计可靠加速比。'
                    '这是小型局部求解耗时，不是闭环任务速度、完整视觉延迟或硬件无关基准。</p>')
    else:
        body.append('<p class="note">尚未提供独立诊断文件；本页不推断计算速度。</p>')
    body.append('<h2>注释视频</h2>')
    if not videos:
        body.append('<p class="note">视频待补充。结果与统计不依赖视频完成。</p>')
    for video in videos:
        url, manifest = relative_link(video['path'], out), relative_link(video['manifest'], out)
        body.append(f'<article><h3>{escape(video["title"])}</h3>'
                    f'<video controls preload="metadata" src="{url}"></video>'
                    f'<p>{escape(TASKS[video["task"]])} · 条件 {video["condition_seed"]} · {escape(video["kind"])}</p>'
                    f'<p class="note">{escape(video["provenance"])} · <a href="{manifest}">来源与封存绑定</a></p></article>')
    body.extend(['<h2>解释边界</h2>', f'<p>{LIMITS}</p>',
                 '<p>视频为有目的选取的成功或失败个例，不能用于估计成功率；相同历史上的更低几何损失也不保证闭环任务成功。</p>',
                 f'<p class="note"><a href="{relative_link(results, out)}">完整机器汇总</a> · '
                 '<a href="PRESENTATION_MANIFEST.json">展示来源与文件哈希</a> · <a href="success_rates.png">下载图表 PNG</a></p>'])
    css = '''body{margin:0;background:#f5f7fa;color:#172331;font:16px/1.65 system-ui,"Noto Sans CJK SC",sans-serif}
main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:30px;line-height:1.3}h2{margin-top:36px}
h3{font-size:18px}.chart{width:100%;background:white}p{max-width:1100px}.note{color:#586577;font-size:14px}
.table-scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:white;font-size:14px}
th,td{text-align:left;padding:10px;border-bottom:1px solid #e0e5eb;white-space:nowrap}th{background:#edf1f5}
article{padding:18px;background:white;margin:18px 0}video{width:100%;max-height:640px}.alert{background:#fff0cf;padding:12px}a{color:#086c77}'''
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>P47 表征与求解器验证</title><style>'+css+'</style><main>'+''.join(body)+'</main></html>'
    (out/'index.html').write_text(document, encoding='utf-8')


def present(results, out, video_index=None, diagnostics=(), evidence_root=REPO, font=None):
    results, out, evidence_root = Path(results).resolve(), Path(out).resolve(), Path(evidence_root).resolve()
    check(not out.exists(), 'Use a new presentation directory; preserve existing artifacts')
    report = read_json(results)
    groups = validate_report(report)
    main = {task: task_rows(groups, task) for task in TASKS}
    if video_index is None:
        video_index = results.parent.parent/'presentation'/'media'/'VIDEO_INDEX.json'
    videos = load_videos(video_index, report, evidence_root)
    timing_rows, timing_sources = load_diagnostics(diagnostics)
    out.mkdir(parents=True)
    font_path = plot_success(main, out, bool(report.get('synthetic_test_fixture')), font)
    render(report, main, videos, timing_rows, results, out)
    manifest = dict(
        schema='prove.p47-presentation.v1', status='complete',
        source=dict(path=relative_link(results, out), sha256=sha(results)),
        source_status=report['status'], formal_inference_ready=True,
        synthetic_test_fixture=bool(report.get('synthetic_test_fixture')),
        sources={relative_link(path, out): sha(path) for path in (
            Path(__file__).resolve(), REPO/'tools/fast_adapter/p46_present_results.py')},
        font=dict(path=relative_link(font_path, out), sha256=sha(font_path)),
        video_status='available' if videos else 'pending',
        video_index=(dict(path=relative_link(video_index, out), sha256=sha(video_index))
                     if Path(video_index).is_file() else None),
        videos=[dict(task=video['task'], title=video['title'], kind=video['kind'],
                     path=relative_link(video['path'], out), manifest=relative_link(video['manifest'], out),
                     video_sha256=sha(video['path']), manifest_sha256=sha(video['manifest']),
                     cells=video['cells'], provenance=video['provenance']) for video in videos],
        diagnostics=[dict(path=relative_link(source['path'], out), sha256=source['sha256']) for source in timing_sources],
        statistics='Copied from completed P47 summary; no new inference or cell outcome reads',
        outputs={path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()},
    )
    (out/'PRESENTATION_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return out/'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--video-index')
    parser.add_argument('--diagnostic', action='append', default=[])
    parser.add_argument('--evidence-root', default=str(REPO))
    parser.add_argument('--font')
    args = parser.parse_args()
    print(present(args.results, args.out, args.video_index, args.diagnostic, args.evidence_root, args.font))


if __name__ == '__main__':
    main()
