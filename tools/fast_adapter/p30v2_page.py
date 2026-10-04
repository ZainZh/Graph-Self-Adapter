"""Desktop review page generated from the complete cohort summary and videos."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tools.fast_adapter.p30v2_common import ROOT
from tools.fast_adapter.p30v2_report import ARMS, LABELS, TASK_NAMES, SPLITS


def main():
    out = ROOT/'summary'
    summary = json.loads((out/'SUMMARY.json').read_text())
    videos = json.loads((out/'media/VIDEO_INDEX.json').read_text())['videos']
    acceptance = json.loads((out/'ROOT_ACCEPTANCE.json').read_text())
    tables = summary['tables']; tr = []
    groups = [(task, split) for task in TASK_NAMES for split in SPLITS]
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', size=10)
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    x = np.arange(4); colors = ['#72818e', '#9d7ca3', '#bb9440', '#127f77', '#6177b8']
    for k, arm in enumerate(ARMS):
        rates = [100*next(r['success_rate'] for r in tables if (r['task'],r['split'],r['arm']) == (t,s,arm)) for t,s in groups]
        bars = ax.bar(x+(k-2)*.16, rates, width=.15, color=colors[k], label=LABELS[arm])
        ax.bar_label(bars, labels=[f'{v:.1f}' for v in rates], fontsize=8, padding=3)
    ax.set_xticks(x, [TASK_NAMES[t]+'\n'+SPLITS[s] for t,s in groups], fontproperties=font)
    ax.set_ylabel('Success (%)'); ax.set_ylim(0,112); ax.set_yticks(np.arange(0,101,20))
    ax.legend(prop=font, ncol=5, loc='upper center', bbox_to_anchor=(.5,1.16), frameon=False)
    ax.spines[['top','right']].set_visible(False); ax.set_axisbelow(True); ax.grid(axis='y', alpha=.2)
    fig.savefig(out/'success_rates.png', dpi=160); fig.savefig(out/'success_rates.svg'); plt.close(fig)
    for task, split in groups:
        values = [next(r for r in tables if (r['task'],r['split'],r['arm']) == (task,split,a)) for a in ARMS]
        cells = ''.join(f'<td>{100*r["success_rate"]:.2f}%<small>{r["successes_by_seed"][0]}/16 · {r["successes_by_seed"][1]}/16</small></td>' for r in values)
        tr.append(f'<tr><th>{TASK_NAMES[task]}<small>{SPLITS[split]}</small></th>{cells}</tr>')
    cards = []
    for v in videos:
        rel = Path(v['file']).relative_to(out).as_posix()
        kind = {'rescue':'梯度救回个例','harm':'梯度误伤个例','success':'梯度成功个例','failure':'梯度失败个例'}[v['kind']]
        cards.append(f'<article><h3>{TASK_NAMES[v["task"]]} · {SPLITS[v["split"]]} · {kind}</h3><p>种子{v["seed"]}，条件{v["condition"]}。五列配对；状态和接触注释取自封存记录。</p><video controls preload="metadata" poster="{rel[:-4]}.png"><source src="{rel}" type="video/mp4"></video></article>')
    headline = html.escape(acceptance['conclusion_zh'])
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P30-v2 持续在线适配</title>
<style>body{{margin:0;background:#eef2f3;color:#203540;font:16px/1.7 system-ui,sans-serif}}main{{max-width:1400px;min-width:1080px;margin:32px auto;padding:0 24px}}h1{{font-size:34px;margin:8px 0}}section,article{{background:white;border:1px solid #d5dfe3;border-radius:12px;padding:24px;margin:24px 0}}.lead{{border-left:6px solid #bc6c3f}}small{{display:block;color:#60727b;font-weight:400}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:14px 8px;border-bottom:1px solid #dfe6e8;text-align:right}}th:first-child{{text-align:left}}thead{{background:#f0f5f5}}img,video{{display:block;width:100%;border-radius:8px}}a{{color:#087c78}}.links a{{margin-right:24px}}footer{{color:#64757e}}</style>
<main><small>Fast Adapter · P30-v2 · 开发验证完成</small><h1>每个执行步更新z，是否越执行越好？</h1>
<section class="lead"><b>{headline}</b><p>640个配对执行回合，64个独立物体/初态条件，2粒模型种子。每任务每档16个条件；重复种子不会扩大物体样本数。当前结果属于开发验证。</p><div class="links"><a href="REPORT_ZH.md">完整报告</a><a href="SUMMARY.json">数值与配对区间</a><a href="DIAGNOSTICS.json">事后诊断</a><a href="../PROTOCOL.json">结果前协议</a><a href="../../adapter_dashboard/">Adapter总览</a></div></section>
<section><h2>逐任务成功率</h2><img src="success_rates.svg" alt="五个方法的配对开发成功率"><table><thead><tr><th>任务 / 条件</th>{''.join('<th>'+LABELS[a]+'</th>' for a in ARMS)}</tr></thead><tbody>{''.join(tr)}</tbody></table><p>每格小字为种子11、22各自成功数。基座重复执行仅用于同条件配对。任务判据要求真实夹持或锤头接触与保持；终态达标率另存数值文件。</p></section>
<section><h2>本轮究竟更新了什么</h2><p>视觉保持DINOv3＋LiLa。测试时全部网络权重冻结，仅32维z变化：完成一个控制动作→收到反馈→最近8步滑窗更新一次→用于下一步。200步回合内不重置z。History用GRU状态吸收每次新反馈，不作测试时梯度更新。</p><p>训练按完整200步历史顺序展开，每20步截断反传但保留z；每个模型2400次外层更新。训练是离线物理记录加训练期专家动作监督，尚非在线采集的元强化学习。锤子从真实夹持状态开始，只检验随后水平工具作用，不包含学会抓取锤子。</p><p>动作、latent和128,000个物理状态的成功判据复核完成，所有权重冻结检查通过。没有声称任意真实物体、正式benchmark、真机、π0.5或普遍自提升优势。</p></section>
<h2>有注释的配对视频</h2><p>按固定展示规则保留救回、误伤或失败个例。视频用于解释行为，不代替上方全量成功率。</p>{''.join(cards)}
<footer>资料由封存记录生成；不覆盖P29及旧20步重置原型。</footer></main></html>'''
    (out/'index.html').write_text(page)
    print(out/'index.html')


if __name__ == '__main__':
    main()
