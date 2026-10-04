"""Local CPU-only table/figure for the sealed P37 success results."""
from pathlib import Path
import html
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('evaluations/fast_adapter_p37_full_success_v01/summary')
REPS=('g224','g448','g224w','g448w','no_explicit_bbox_geometry','p34_existing_bbox_geometry','p36_explicit_relation_geometry')
NAMES=('全局224','全局448','224＋腕部','448＋腕部','无显式框几何','已有框几何','新增显式关系几何')
SHORT=('Global224','Global448','224+wrist','448+wrist','No bbox','Existing bbox','New geometry')


def main():
    d=json.loads((ROOT/'SUMMARY.json').read_text());cells=d['cells']
    fig,axs=plt.subplots(2,2,figsize=(15,8),sharey=True,layout='constrained')
    tables=[]
    for row,task in enumerate(('mug-handle-lift','hammer-block')):
        for col,split in enumerate(('train','geometry_ood')):
            sel=[c for c in cells if c['task']==task and c['split']==split]
            b=next(c for c in sel if c['kind']=='base');ax=axs[row,col];x=np.arange(7)
            for mode,offset,color in (('fixed',-.18,'#94a3b8'),('continuous',.18,'#0d9488')):
                cc=[next(c for c in sel if c['representation']==r and c['mode']==mode) for r in REPS]
                ax.bar(x+offset,[c['rate']*100 for c in cc],width=.34,color=color,label='Fixed z' if mode=='fixed' else 'Continuous z')
                for sx,marker in ((0,'o'),(1,'x')):
                    ax.scatter(x+offset,[c['per_seed'][sx]['successes']/8*100 for c in cc],s=20,marker=marker,color='#14233b',zorder=3)
            ax.axhline(b['rate']*100,color='#be123c',linestyle='--',label=f"Frozen base: {b['rate']:.1%}")
            ax.set_xticks(x,SHORT,rotation=25,ha='right');ax.set_ylim(-3,108);ax.grid(axis='y',alpha=.15)
            ax.set_title(('Mug handle lift' if row==0 else 'Held hammer interaction')+' / '+('ID' if col==0 else 'Geometry OOD'))
            if col==0:ax.set_ylabel('Full-task success (%)')
            ax.legend(fontsize=8,loc='upper right')
            title=('杯柄抓取' if row==0 else '持锤推击木块')+'／'+('同分布' if col==0 else '物体几何OOD')
            lines=[f'<h2>{title}</h2><p>原基座：{b["per_seed"][0]["successes"]}/8 = {b["rate"]:.1%}</p>',
                   '<table><thead><tr><th>表示</th><th>固定z</th><th>持续更新z</th><th>更新相对固定</th><th>更新相对基座</th></tr></thead><tbody>']
            for rep,name in zip(REPS,NAMES):
                a=next(c for c in sel if c['representation']==rep and c['mode']=='fixed')
                c=next(c for c in sel if c['representation']==rep and c['mode']=='continuous')
                def fmt(v):return f"{v['rate']:.2%} <small>（{v['per_seed'][0]['successes']}/8，{v['per_seed'][1]['successes']}/8）</small>"
                lines.append(f'<tr><td>{name}</td><td>{fmt(a)}</td><td>{fmt(c)}</td><td>{(c["rate"]-a["rate"])*100:+.2f} pp</td><td>{(c["rate"]-b["rate"])*100:+.2f} pp</td></tr>')
            tables.append('\n'.join(lines)+ '</tbody></table>')
    fig.suptitle('P37: 8 new conditions per task/setting; bars = two fixed model seeds; dots = individual seeds',fontsize=13)
    fig.savefig(ROOT/'success_comparison.png',dpi=160);fig.savefig(ROOT/'success_comparison.svg');plt.close(fig)
    videos=[];p=ROOT/'media/VIDEO_INDEX.json'
    if p.exists():
        for v in json.loads(p.read_text())['videos']:
            src='media/'+Path(v['file']).name
            label={'rescue':'适配补回个例','harm':'适配误伤个例','base_harm':'相对基座误伤个例','success':'成功个例','failure':'失败个例'}.get(v['reason'],v['reason'])
            videos.append(f'<article><p>{html.escape(v["task"])} / {html.escape(v["split"])} / 条件{v["condition"]}：{html.escape(label)}</p><video controls preload="none" poster="media/{Path(v["cover"]).name}" src="{src}"></video></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P37 完整任务成功率</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#15263c;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px 14px;border-bottom:1px solid #ddd;text-align:left}small{color:#596878}img,video{max-width:100%}article{margin:24px 0}h2{margin-top:32px}a{color:#087a73}</style>
<h1>P37：完整任务成功率验证</h1><p>928个完整回合，32个独立物体／初态条件。每任务、每设置8个条件；两个已训练模型种子属于重复测量。所有模型冻结，只允许在线更新z。</p>
<p>下表括号依次为种子11、22；百分比是两者平均。不能将197次反馈更新、局部动作进展或预测损失视为成功率提升。当前是小规模候选验证。</p>
<p><a href="ROOT_REPORT_ZH.md">主Agent结论</a> · <a href="REPORT_ZH.md">完整说明</a> · <a href="SUMMARY.json">全部数字与配对记录</a> · <a href="AUDIT.json">复核</a> · <a href="SUCCESS_RATES.csv">CSV</a></p>
<img src="success_comparison.png" alt="两任务ID/OOD下，固定z和持续更新z的完整成功率，与原基座比较">'''
    page+='\n'.join(tables)+'<h2>少量带注释视频</h2><p>事后选取448双视角、种子11的少量正反例，用于解释行为。统一全局视角从封存状态重绘，接触和成功注释不输入控制器；个例不代表平均效果。</p>'+'\n'.join(videos)+'</html>'
    (ROOT/'index.html').write_text(page)
    (ROOT/'DISPLAY_RECEIPT.json').write_text(json.dumps(dict(summary_sha256=hashlib.sha256((ROOT/'SUMMARY.json').read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('P37_DISPLAY_COMPLETE')


if __name__=='__main__':main()
