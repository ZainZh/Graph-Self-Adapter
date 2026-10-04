"""Standalone desktop report, independent of the paused project Dashboard."""
import html,json
from pathlib import Path
from tools.fast_adapter.p29_common import R,TASKS
from tools.fast_adapter.p29_report import TASK_NAMES,STRATUM_NAMES,STRATA

def main():
    out=R/'summary';data=json.loads((out/'RESULTS.json').read_text());accept=json.loads((out/'ROOT_ACCEPTANCE.json').read_text())
    e=html.escape;parts=[]
    for task in TASKS:
        d=data['tasks'][task]
        parts += [f'<section><h2>{e(TASK_NAMES[task])}</h2><p>封存版本 {e(d["round"])}；每个物体／初态条件内先平均两个固定模型种子，再统计结果。</p>',
                  '<table><thead><tr><th>测试条件</th><th>基座／K0</th><th>本物体梯度适配</th><th>固定平均z</th><th>另一物体z</th><th>History</th><th>本体反馈适配</th></tr></thead><tbody>']
        for st in STRATA:
            r=d['strata'][st];rates=r['rates_percent']
            parts += [f'<tr><td>{e(STRATUM_NAMES[st])} · {r["conditions"]}</td>'+''.join(f'<td class="{"focus" if a=="fast_correct" else ""}">{rates[a]:.2f}%</td>' for a in ('base','fast_correct','fast_mean','fast_donor','history','robot'))+'</tr>']
        parts+=['</tbody></table>']
        x=d['strata']['geometry_ood']['comparisons'];parts+=['<p class="note">范围外新几何，正确反馈的主要比较（差值百分点／六项比较校正区间）：'+ '；'.join(f'{label} {x[k]["delta_pp"]:+.2f} [{x[k]["multiplicity_adjusted_ci"][0]:.2f}, {x[k]["multiplicity_adjusted_ci"][1]:.2f}]' for k,label in [('base','对基座'),('fast_mean','对平均z'),('fast_donor','对错配z')])+'。</p></section>']
    videos=[]
    for p in sorted((out/'final').glob('*/media_v2/VIDEO_INDEX.json')):
        for v in json.loads(p.read_text())['videos']:
            if v['task'] in TASKS and data['tasks'][v['task']]['round']==p.parent.parent.name:videos.append(v)
    media=[]
    for v in videos:
        path=Path(v['file']).relative_to(out);title=TASK_NAMES[v['task']]+' · '+STRATUM_NAMES[v['stratum']]
        kind={'rescue':'救回个例','harm':'误伤个例','success':'成功个例','failure':'失败个例'}[v['kind']]
        media.append(f'<article><h3>{e(title)} · {kind}</h3><p>模型种子 {v["seed"]}，条件 {v["condition"]}。前段是共同支持采集，后段四列比较基座、本体反馈、History与物体反馈梯度适配。</p><video controls preload="metadata" poster="{path.with_suffix(".png")}"><source src="{path}" type="video/mp4"></video></article>')
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P29 物体泛化验证</title>
<style>body{{margin:0;background:#edf2f3;color:#182c35;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1344px;min-width:1024px;margin:36px auto;padding:0 28px}}h1{{font-size:38px;line-height:1.25;margin:12px 0}}h2{{font-size:24px}}h3{{font-size:20px}}section,article,.lead{{background:#fff;border:1px solid #d5e0e2;border-radius:12px;padding:24px;margin:22px 0}}.eyebrow{{color:#117c75;letter-spacing:.08em;font-weight:700}}.lead{{border-left:5px solid #117c75}}.cards{{display:flex;gap:16px}}.cards div{{background:#dceae7;padding:18px 24px;flex:1;border-radius:10px}}.cards b{{font-size:30px;display:block}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}th,td{{padding:13px 9px;border-bottom:1px solid #dce3e6;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f0f5f5;font-size:14px}}.focus{{background:#e8f5f1;font-weight:700}}.note{{font-size:14px;color:#4c626d}}a{{color:#086b75}}video,img{{max-width:100%;width:100%;border-radius:8px}}.links a{{margin-right:22px}}footer{{color:#52666f;padding:20px 0}}</style><main>
<div class="eyebrow">P29 · 同类物体泛化 · 独立测试</div><h1>杯柄抓取与工具—物体交互</h1>
<p>正常机械臂动作；固定 DINOv3＋LiLa 与任务动作基座，测试时从交互记录更新32维z。</p>
<div class="lead">{e(accept.get('decision_summary_zh','详见下表及主Agent报告。'))}</div>
<div class="cards"><div><b>320</b>独立物体／初态条件</div><div><b>3,520</b>配对任务执行</div><div><b>640</b>共同／错配支持采集</div><div><b>2</b>固定模型种子</div></div>
<p class="links"><a href="ROOT_REPORT_ZH.md">主Agent结论与解释</a><a href="RESULTS_ZH.md">完整数值及区间</a><a href="ROOT_ACCEPTANCE.json">验收记录</a><a href="success_rates.pdf">下载图表PDF</a></p>
<section><h2>成功率</h2><img src="success_rates.png" alt="两项任务、三档物体条件的六种方法成功率对照"><p class="note">主指标要求杯子持续抓稳并抬升15步，或锤头接触后推动木块达标5步。结束时是否仍保持成功另列于完整结果。百分比按条件内两个模型种子平均；不把种子重复当成额外独立物体。</p></section>
{''.join(parts)}
<section><h2>交互输入与适配</h2><p>杯子：actor为杯子局部视觉，第二object槽全部置零并mask。锤子：actor为锤子，object为木块。另读末端位置、夹爪开度和自身动作；不读物体真值位置、接触真值或成功标签。</p><p>先用冻结基座采集160个动作及视觉反馈，再对z做3次梯度更新，随后在同物体的新初态执行200步。160次物理动作与3次梯度更新是不同的计数。History使用同一段反馈；本体反馈对照仍看物体图像，但适配损失只预测本体变化。</p><p class="note">这里是固定色标、同一生成器下的尺寸与接触条件变化，尚非真实杯型／任意外观泛化。锤子任务为水平接触推击，不是冲击力控制或钉钉子。小型动作基座从示范训练后冻结，不是直接测试预训练VLA。</p></section>
<section><h2>带注释的完整过程视频</h2><p>视频由封存的完整物理状态重绘。每任务最多两个个例，包含阶段与成功／失败注释；个例不替代全部样本的成功率。</p></section>{''.join(media)}
<footer>数据、失败候选和训练修订均保留。独立测试不回流调参；本页独立于按用户要求暂缓更新的Dashboard。</footer></main></html>'''
    assert len(videos)==4, f'Expected two annotated videos per task, got {len(videos)}'
    (out/'index.html').write_text(content);print('STANDALONE_INDEX',out/'index.html')
if __name__=='__main__':main()
