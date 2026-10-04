"""Desktop P31 result page generated only from completed numeric evidence."""
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.fast_adapter.p31v2_common import *
from tools.fast_adapter.p31v2_report import LABELS


def main():
    out=ROOT/'summary';s=json.loads((out/'SUMMARY.json').read_text());arms=protocol()['evaluation']['arms']
    index={(r['split'],r['arm']):r for r in s['rows']}
    names=['Base / z0','Old/full','Old/stage','Own/full','Own/stage','Old/stage3','Own/stage3']
    fig,axes=plt.subplots(1,2,figsize=(14,4.4),sharey=True,layout='constrained')
    for ax,split,title in zip(axes,['train','geometry_ood'],['ID: new in-range objects','OOD: larger cup geometry']):
        values=[100*index[split,a]['mean_rate'] for a in arms];xx=np.arange(len(arms))
        ax.bar(xx,values,color=['#758494','#aec2cb','#73b9aa','#91abb9','#187d70','#b4cabf','#57a48a'])
        for i,a in enumerate(arms):
            for j,c in enumerate(index[split,a]['counts']):ax.scatter(i+(j-.5)*.17,c/16*100,c='black',s=17,zorder=5)
            ax.text(i,values[i]+3,f'{values[i]:.1f}',ha='center',fontsize=9)
        ax.set(ylim=(0,110),xticks=xx,xticklabels=names,title=title,ylabel='Success rate (%)');ax.tick_params(axis='x',rotation=24);ax.grid(axis='y',alpha=.2)
    fig.suptitle('P31 development:16 paired conditions per split,2 model seeds (dots)')
    fig.savefig(out/'success_rates.svg');fig.savefig(out/'success_rates.png',dpi=150);plt.close(fig)
    rows=''
    for a in arms:
        cells=[]
        for split in ('train','geometry_ood'):
            r=index[split,a];c=r['counts'];cells.append(f'<td><strong>{r["mean_rate"]*100:.2f}%</strong><small>{c[0]}/16 · {c[1]}/16</small></td>')
        rows+=f'<tr><td>{LABELS[a]}</td>'+''.join(cells)+'</tr>'
    media=json.loads((out/'media/VIDEO_INDEX.json').read_text());videos=''
    for v in media['videos']:
        file=Path(v['file']).relative_to(out);poster=file.with_suffix('.png')
        videos+=f'<section><h3>{"救回" if v["kind"]=="rescue" else "误伤" if v["kind"]=="harm" else "示例"} · {v["split"]} · seed{v["seed"]} 条件{v["condition"]}</h3><video controls preload="metadata" poster="{poster}" src="{file}"></video></section>'
    report=html.escape((out/'REPORT_ZH.md').read_text())
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P31 杯子关键交互适配</title><style>
body{{font:16px/1.65 system-ui;background:#edf3f4;color:#172c35;margin:32px auto;max-width:1480px;padding:0 28px}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px 20px;text-align:left;border-bottom:1px solid #dce5e8}}small{{display:block;color:#607580}}img,video{{width:100%;border-radius:8px}}section{{margin:28px 0;background:white;padding:20px;border-radius:12px}}pre{{white-space:pre-wrap;font:15px/1.8 system-ui}}a{{color:#067767}}</style>
<h1>P31 杯子：关键阶段适配 × 自身轨迹训练</h1><p>448回合开发对照 · 两种训练种子 · 视觉和动作基座冻结 · 全部方法在本机4090执行</p>
<p>先区分减少误伤与超过基座，再检验持续更新是否比三次更新更有效。小样本开发结果，未晋升。</p>
<p><a href="../../adapter_dashboard/">Adapter总览</a> · <a href="REPORT_ZH.md">完整结论</a> · <a href="SUMMARY.json">机器摘要</a> · <a href="../PROTOCOL.json">结果前协议</a></p>
<section><img src="success_rates.svg" alt="逐方法ID与OOD成功率"><table><thead><tr><th>方法</th><th>ID · 两种子平均/各自计数</th><th>OOD · 两种子平均/各自计数</th></tr></thead><tbody>{rows}</tbody></table></section>
{videos}<details><summary>完整报告与限制</summary><pre>{report}</pre></details></html>'''
    (out/'index.html').write_text(page);print('PAGE_COMPLETE',out/'index.html')


if __name__=='__main__':main()
