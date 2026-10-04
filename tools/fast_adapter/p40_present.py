"""Create a task-separated desktop review from closed P40 results only."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('evaluations/fast_adapter_p40_skills_v01')
TASKS = {'cup': '杯柄抓持抬升', 'hammer': '锤头接触推进', 'insert': '物块直线插入'}
SPLITS = {'train': '同分布', 'physical': '物理条件变化', 'geometry': '几何变化', 'joint': '几何＋物理变化', 'paired': '同外观物理配对'}
METHODS = {
    'references/base': '冻结基础控制', 'references/strong_fixed': '开发选定固定力度',
    'relation/continuous': '关系后果反馈', 'local/continuous': '局部DINO',
    'motion/continuous': '变化DINO', 'fused/continuous': '局部DINO＋关系',
    'history/continuous': 'History',
}
ENGLISH = ['Frozen base', 'Dev-selected fixed', 'Relation consequence', 'Local DINO', 'Motion DINO', 'Fused', 'History']


def main():
    final = json.loads((ROOT/'summary/final/RESULTS.json').read_text())
    freeze = json.loads((ROOT/'FINAL_FREEZE.json').read_text())
    assert final['executions'] == freeze['executions']
    out = ROOT/'summary/presentation_v01'
    out.mkdir(exist_ok=False)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharey=True)
    colors = ['#a4adb8', '#d4a94b', '#167f78', '#4477aa', '#8866aa', '#44aabb', '#bb6677']
    x = np.arange(5)
    for ax, (task, title) in zip(axes, TASKS.items()):
        stats = final['tasks'][task]['stats']
        for i, (method, label) in enumerate(zip(METHODS, ENGLISH)):
            ys = [100*stats[s][method]['success_rate'] for s in SPLITS]
            ax.bar(x+(i-3)*.11, ys, .105, label=label, color=colors[i])
        ax.set_title({'cup':'Handle grasp and lift', 'hammer':'Held hammer head / guided block', 'insert':'Held peg / compliant slot'}[task], loc='left')
        ax.set_ylim(0, 110)
        ax.set_ylabel('Task success (%)')
        ax.set_xticks(x, ['ID', 'Physical shift', 'Geometry shift', 'Combined shift', 'Appearance pairs'])
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(ncol=4, fontsize=9, loc='upper center', bbox_to_anchor=(.5, 1.3))
    fig.suptitle('P40 independent evaluation: all frozen candidates', y=.99, fontsize=15)
    fig.text(.02, .008, 'Two model seeds averaged within each condition; 16 conditions per ordinary stratum, 16 appearance pairs. No pooled cross-task rate.', fontsize=10)
    fig.tight_layout(rect=(0, .025, 1, .95))
    fig.savefig(out/'success_rates.png', dpi=170)
    fig.savefig(out/'success_rates.pdf')
    plt.close(fig)

    diagnostics=json.loads((ROOT/'summary/MECHANISM_DIAGNOSTICS.json').read_text())
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for row,task in enumerate(TASKS):
        records=diagnostics['tasks'][task]
        for method,label,color in zip(METHODS,ENGLISH,colors):
            kind,arm=method.split('/')
            if arm=='strong_fixed':arm=freeze['strong_fixed'][task]
            selected=[r for r in records if r['split']=='joint' and r['kind']==kind and r['arm']==arm]
            if not selected:continue
            force=np.mean([r['mean_command_force'] for r in selected],axis=0)
            threshold=np.mean([r['mean_parameters'] for r in selected],axis=0)[:,0]
            axes[row,0].plot(np.arange(len(force)),force,label=label,color=color)
            # Fixed-force arms do not use the nominal latent to determine force.
            if not arm.startswith('fixed'):
                axes[row,1].plot(np.arange(len(threshold)),threshold,label=label,color=color)
        for ax in axes[row]:
            ax.axvline(12,color='gray',ls=':',lw=1)
            ax.set_xlabel('Completed execution steps')
            ax.grid(alpha=.2)
        axes[row,0].set_title(task+': command force, combined shift')
        axes[row,1].set_title(task+': effective threshold estimate')
        axes[row,0].set_ylabel('N (per jaw for cup)')
        axes[row,1].set_ylabel('Effective threshold (N)')
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(out/'adaptation_curves.png',dpi=160)
    fig.savefig(out/'adaptation_curves.pdf')
    plt.close(fig)

    blocks = ['<h1>P40：三类近距离交互的独立验证</h1>',
              '<p>全部冻结候选均保留。测试中只更新当前回合的 z；图像模型、反馈预测器与控制器冻结。每个任务单独统计。</p>',
              '<p class="note">这是受控接触技能：杯子使用浮动夹爪；锤子和插入物块由平移台持有。采用模拟器可见角色／部件mask、相机标定和工具本体状态。不是完整机械臂、RGB-only或标准benchmark成绩。</p>',
              '<p><a href="../ROOT_REPORT_ZH.md">主Agent结论与边界</a> · <a href="../final/RESULTS.json">全部数字及配对统计</a> · <a href="../../FINAL_FREEZE.json">测试前冻结</a></p>',
              '<img src="success_rates.png" alt="三任务各设置成功率">',
              '<h2>执行中的变化</h2><p>下图是几何与物理同时变化时的平均命令力度与有效阈值估计。虚线为共同前12步结束。固定力度不使用latent；这些描述性曲线不构成额外显著性检验。</p>',
              '<img src="adaptation_curves.png" alt="适配过程曲线">']
    for task, title in TASKS.items():
        d = final['tasks'][task]
        blocks += ['<h2>'+title+'</h2>', '<p>开发集预先选定的固定力度：'+d['strong_fixed']+'。模型种子不作为独立物体样本。</p>',
                   '<div class="scroll"><table><thead><tr><th>设置</th>'+''.join('<th>'+x+'</th>' for x in METHODS.values())+'</tr></thead><tbody>']
        for split, label in SPLITS.items():
            s = d['stats'][split]
            blocks += ['<tr><th>'+label+'</th>'+''.join(f'<td>{100*s[k]["success_rate"]:.2f}%</td>' for k in METHODS)+'</tr>']
        blocks += ['</tbody></table></div>', '<h3>同外观、不同物理条件：反馈因果对照</h3>',
                   '<table><tr><th>方法</th><th>成功率</th><th>平均命令力</th></tr>']
        paired = d['stats']['paired']
        controls = {'relation/continuous':'当前物体持续更新', 'relation/mean':'固定训练平均 z', 'relation/first12':'前12步更新后固定', 'relation/donor':'相反物理条件的反馈', 'motion/continuous':'变化DINO持续更新', 'motion/mean':'变化DINO固定训练平均 z', 'motion/donor':'变化DINO错配反馈'}
        for k, label in controls.items():
            if k in paired:
                s = paired[k]
                blocks += [f'<tr><td>{label}</td><td>{100*s["success_rate"]:.2f}%</td><td>{s["mean_command_force"]:.3f} N</td></tr>']
        blocks += ['</table><p class="note">命令力不是能量、接触真值或安全指标。杯子数值为单侧夹持力。精确区间、多重比较及每粒模型计数见原始摘要。</p>']
    catalog = ROOT/'summary/media_v01/VIDEO_INDEX.json'
    if catalog.exists():
        blocks += ['<h2>带注释的封存个例</h2><p>按照预定规则选例，包含失败；视频不代替全量成功率。</p>']
        for v in json.loads(catalog.read_text())['videos']:
            name = v['name']
            title = TASKS[v['task']]+' · '+SPLITS[v['split']]+' · 初态 '+str(v['index'])
            blocks += ['<h3>'+html.escape(title)+'</h3>', f'<video controls preload="metadata" poster="../media_v01/{name}.jpg" src="../media_v01/{name}.mp4"></video>', f'<p><a href="../media_v01/{name}_ANNOTATIONS.json">逐帧注释与来源</a></p>']
    blocks += ['<h2>读数边界</h2><p>16个普通初态／16个外观配对是统计单位，两个训练种子在条件内平均。表征对照同时包含反馈目标与损失构造，不是仅换编码器的消融。几何反馈尤其在插入中包含持有工具的本体位移；不将它解释为DINO识别了遮挡中的物块。</p>']
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>P40 独立结果</title><style>body{font-family:system-ui,sans-serif;max-width:1420px;margin:36px auto;padding:0 28px;color:#203844;background:#f6f8fa;line-height:1.65}h1,h2{color:#116f6a}h2{margin-top:46px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #dce4e8;text-align:right}th:first-child,td:first-child{text-align:left}img,video{max-width:100%;height:auto;background:white;border-radius:10px}video{width:1200px}.note{color:#627582;background:#eaf0f3;padding:12px}.scroll{overflow:auto}a{color:#006e84}</style><body>'+''.join(blocks)+'</body></html>'
    (out/'index.html').write_text(page)
    print(out/'index.html')


if __name__ == '__main__':
    main()
