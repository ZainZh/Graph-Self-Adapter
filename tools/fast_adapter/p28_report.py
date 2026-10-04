"""Per-task reporting, including tasks that fail base readiness."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tools.fast_adapter.p28_common import R, TASKS, SEEDS, task_root, save_json, digest
from tools.fast_adapter.p28_eval import EVAL_STRATA
from tools.fast_adapter.p27_statistics import paired_clusters

NAMES = {'pick-place-v3': '抓取并放置物块', 'drawer-open-v3': '拉开抽屉'}
STRATUM_NAMES = {'native': '原生动作响应', 'response_id': '训练范围内的新响应', 'response_ood': '更大动作响应偏差'}
ARMS = [('base', '基座／不更新z'), ('fast_correct', '本环境梯度适配'),
        ('fast_mean', '固定训练平均z'), ('fast_donor', '独立错配反馈z'), ('history', '无梯度History')]


def arrays(result, key='success'):
    by_arm = {}
    for row in sorted(result['rows'], key=lambda x: x['offset']):
        by_arm.setdefault(row['arm'], []).extend(row[key])
    out = {'base': np.array(by_arm['base'], dtype=float)}
    for name, _ in ARMS[1:]:
        out[name] = np.array([by_arm[f'{name}_seed{s}'] for s in SEEDS], dtype=float)
    assert all(x.shape[-1] == 64 for x in out.values())
    return out


def main():
    out = R/'summary'; out.mkdir(exist_ok=True)
    assert json.loads((out/'AUDIT.json').read_text())['passed']
    done = json.loads((R/'CAMPAIGN_DONE.json').read_text())
    records = {}; sources = []
    lines = ['# P28：两个Meta-World机械臂任务的Fast Adapter验证', '',
        '使用官方任务物理与success判据，固定160步query预算及一个全局RGB相机；不是完整MT10/ML1官方成绩。普通运行模型只读图像、本体位置、夹爪开度和自身动作。预训练DINOv3+LiLa固定，PCA只拟合本任务训练图像。专家状态/目标和人为动作响应参数仅用于训练监督。', '',
        '每个通过基座资格的任务：三档各64个新环境条件、两个已训练模型。80个共同基座支持动作、3次梯度更新，随后换独立几何初态执行160步，query中z固定；不得称只需3次物理交互。History读取相同80条反馈。', '']
    for ti, task in enumerate(TASKS):
        gate = json.loads((task_root(task)/'qualification/BASE.json').read_text())
        lines += [f'## {NAMES[task]}（{task}）', '', f"原生基座资格：{gate['successes']}/16；{'通过' if gate['passed'] else '未通过'}。", '']
        if not gate['passed']:
            records[task] = {'status': 'base_not_ready', 'base_qualification': gate}
            lines += ['该任务保留在结果中，未进入适配效能判断；不能把尚未学会任务解释成梯度机制失败。', '']
            continue
        records[task] = {'status': 'completed', 'strata': {}}
        for si, st in enumerate(EVAL_STRATA):
            path = task_root(task)/'evaluation'/st/'RESULT.json'
            result = json.loads(path.read_text()); a = arrays(result); terminal = arrays(result, 'terminal_success')
            comparisons = {}
            for j, control in enumerate(('base', 'fast_mean', 'fast_donor', 'history')):
                primary = st == 'response_ood' and control != 'history'
                comparisons[control] = {**paired_clusters(a['fast_correct'], a[control],
                    seed=28028+100*ti+10*si+j, family=6 if primary else 1),
                    'primary': primary, 'confirmatory': primary}
            entry = {'rates_percent': {k: float(v.mean()*100) for k, v in a.items()},
                     'per_seed_rates_percent': {k: (v.mean(1)*100).tolist() for k, v in a.items() if v.ndim == 2},
                     'terminal_rates_percent': {k: float(v.mean()*100) for k, v in terminal.items()},
                     'comparisons_correct_minus': comparisons, 'condition_clusters': 64,
                     'system_gate': st == 'response_ood' and comparisons['base']['conditional_positive_gate'],
                     'feedback_gate': st == 'response_ood' and all(comparisons[k]['conditional_positive_gate'] for k in ('fast_mean', 'fast_donor'))}
            records[task]['strata'][st] = entry
            sources.append({'path': str(path), 'sha256': digest(path)})
        lines += ['| 方法 | 原生动作响应 | 范围内新响应 | 更大响应偏差 |', '|---|---:|---:|---:|']
        for arm, label in ARMS:
            lines.append('| '+label+' | '+' | '.join(f"{records[task]['strata'][s]['rates_percent'][arm]:.2f}%" for s in EVAL_STRATA)+' |')
        lines += ['', '| 更大响应偏差：正确反馈减去 | 差值（百分点） | 六项比较校正区间 | 两模型方向一致且区间过零门 |', '|---|---:|---|---|']
        for c in ('base', 'fast_mean', 'fast_donor'):
            v = records[task]['strata']['response_ood']['comparisons_correct_minus'][c]
            ci = v['multiplicity_adjusted_ci']
            lines.append(f"| {dict(ARMS)[c]} | {v['delta_pp']:+.2f} | [{ci[0]:.2f},{ci[1]:.2f}] | {'通过' if v['conditional_positive_gate'] else '未通过'} |")
        lines += ['', '原生与范围内响应、History比较以及结束时成功均为次要分析，不能使用其未校正区间代替主要门。', '']
    lines += ['## 解释边界与成本', '',
        '动作响应偏差通过隐藏的水平旋转与轴向增益施加，夹爪指令不变，原任务几何分布与接触物理保持原生。它检验执行校准迁移到真实机械臂接触任务，并不证明摩擦、质量、未知物体或任意任务变化的适配。', '',
        '每任务基座12000步BC＋6000步训练状态补采修正；两个种子各16000步Gradient及同预算History训练。训练K随机1/2/3/5，测试固定K3。训练集、支持与独立query分开，固定平均z只来自64个训练支持环境。', '',
        f"独立评估共{done['query_episodes']}条query、{done['support_episodes']}条支持，另有训练与资格采集。统计单位为独立目标条件及其唯一donor，两个固定模型先在条件内平均；六项主要比较采用20000次配对条件bootstrap与Bonferroni校正（99.1667%近似区间）。不能推广成新训练种子总体结论。", '',
        '[机器统计](RESULTS.json) · [复算](AUDIT.json) · [注释视频](index.html#videos)', '', '![逐任务成功率](success_rates.png)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    save_json(out/'RESULTS.json', {'tasks': records, 'source_bindings': sources,
              'primary_family': '2 tasks x3 OOD comparisons;6 even if a task is unready',
              'inference': 'Approximate condition-cluster paired bootstrap conditional on two fixed model checkpoints'})
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for ti, task in enumerate(TASKS):
        for si, st in enumerate(EVAL_STRATA):
            ax = axes[ti, si]; ax.set_ylim(0, 108)
            ax.set_title(NAMES[task]+'｜'+STRATUM_NAMES[st], fontproperties=font)
            if records[task]['status'] != 'completed':
                ax.text(.5, .5, '基座未就绪，未进入适配评估', transform=ax.transAxes, ha='center', fontproperties=font)
                continue
            vals = [records[task]['strata'][st]['rates_percent'][key] for key, _ in ARMS]
            bars = ax.bar(range(5), vals, color=['#8a949b', '#147e70', '#6f8eae', '#a28daf', '#cd9b63'])
            ax.set_xticks(range(5), ['基座', '梯度适配', '平均z', '错配z', 'History'], fontproperties=font, rotation=15)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, val+1.5, f'{val:.1f}', ha='center', fontsize=10)
            ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('P28：两项原生接触任务＋受控动作响应扰动｜每档64个独立条件', fontproperties=font)
    fig.tight_layout(); fig.savefig(out/'success_rates.png', dpi=170); plt.close(fig)
    print(json.dumps({t: ({s: x['rates_percent'] for s, x in d['strata'].items()} if d['status'] == 'completed' else d['status']) for t, d in records.items()}), flush=True)


if __name__ == '__main__':
    main()
