"""All-arm P32 development summary; fixed condition-cluster descriptive intervals."""
import csv
import html
import numpy as np
from tools.fast_adapter.p32_common import *

ARMS=['base','predictive','learned','learned_first3','learned_donor']
LABELS={'base':'冻结基座 / z₀','predictive':'预测损失 · 持续更新','learned':'学到的损失 · 持续更新',
        'learned_first3':'学到的损失 · 仅前三次','learned_donor':'学到的损失 · 其他物体历史'}


def main():
    out=ROOT/'summary';out.mkdir(exist_ok=False)
    result=json.loads((ROOT/'development/RESULT.json').read_text());audit=json.loads((ROOT/'AUDIT.json').read_text())
    assert audit['passed'] and result['physical_episodes']==144
    rows=result['rows'];lookup={(r['seed'],r['split'],r['condition'],r['arm']):r for r in rows}
    groups=[];contrasts=[];rng=np.random.default_rng(323233)
    for split in ('train','geometry_ood'):
        for arm in ARMS:
            per_seed=[]
            for seed in ([None] if arm=='base' else [11,22]):
                rr=[lookup[seed,split,i,arm] for i in range(8)]
                per_seed.append({'seed':seed,'successes':sum(r['success'] for r in rr),'n':8,
                       'ever_bilateral':sum(r['ever_bilateral'] for r in rr),'terminal_successes':sum(r['terminal_success'] for r in rr),
                       'mean_updates':float(np.mean([r['updates'] for r in rr]))})
            groups.append({'split':split,'arm':arm,'per_seed':per_seed,'mean_rate':float(np.mean([r['successes']/8 for r in per_seed])),
                  'distinct_conditions':8,'base_reused':arm=='base'})
        for ref in ('base','predictive','learned_first3','learned_donor'):
            matrix=[];details=[];counts=[]
            for seed in (11,22):
                dd=[];wins=harms=ties=0
                for i in range(8):
                    a=lookup[seed,split,i,'learned'];b=lookup[None if ref=='base' else seed,split,i,ref]
                    delta=int(a['success'])-int(b['success']);dd.append(delta)
                    wins+=delta>0;harms+=delta<0;ties+=delta==0
                    details.append({'seed':seed,'condition':i,'learned':a['success'],'reference':b['success'],'delta':delta})
                matrix.append(dd);counts.append({'seed':seed,'wins':wins,'harms':harms,'ties':ties})
            condition_means=np.asarray(matrix).mean(0);sample=rng.integers(0,8,(20000,8))
            ci=np.quantile(condition_means[sample].mean(1),[.025,.975])*100
            contrasts.append({'split':split,'candidate':'learned','reference':ref,
                 'mean_delta_pp':float(condition_means.mean()*100),'descriptive_condition_bootstrap95_pp':ci.tolist(),
                 'per_seed_pairs':counts,'details':details,
                 'interval_limit':'8 shared condition clusters,2 fixed models; descriptive unadjusted intervals, not a confirmatory multiple-comparison test'})
    summary={'phase':'development','physical_episodes':144,'unique_object_conditions_per_stratum':8,
             'model_seeds':[11,22],'base_physical_episodes':16,'groups':groups,'contrasts':contrasts,
             'directions':audit['directions'],'exposure':audit['update_exposure'],
             'bindings':bindings([ROOT/'PROTOCOL.json',ROOT/'development/RESULT.json',ROOT/'AUDIT.json',Path(__file__)])}
    dump(out/'SUMMARY.json',summary)
    with (out/'SUCCESS_RATES.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['setting','arm','model_seed','successes','episodes','rate_percent','shared_base'])
        for g in groups:
            for r in g['per_seed']:w.writerow([g['split'],g['arm'],r['seed'],r['successes'],r['n'],r['successes']/r['n']*100,g['base_reused']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(1,2,figsize=(14,5.3),layout='constrained')
    colors=['#63788d','#cb9342','#198b7d','#8262af','#ae6678']
    for ax,split,title in zip(axes,('train','geometry_ood'),('同分布新物体','几何范围外新物体')):
        gg=[g for g in groups if g['split']==split];rates=[g['mean_rate']*100 for g in gg]
        ax.barh(range(5),rates,color=colors,height=.58)
        for i,g in enumerate(gg):
            ax.text(rates[i]+1,i,f'{rates[i]:.2f}%',va='center')
            for r in g['per_seed']:ax.scatter(r['successes']/8*100,i,c='black',s=22,zorder=3,marker='|' if r['seed']==11 else '_')
        ax.set_yticks(range(5),[LABELS[a] for a in ARMS],fontproperties=fp)
        ax.set_xlim(0,108);ax.invert_yaxis();ax.set_xlabel('成功率 (%)',fontproperties=fp)
        ax.set_title(title+' · 每模型8条件',fontproperties=fp);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('P32 杯柄抓取开发验证：柱为均值，标记为各模型种子',fontproperties=fp,fontsize=15)
    fig.savefig(out/'success_rates.svg');fig.savefig(out/'success_rates.png',dpi=140);plt.close(fig)
    trs=[]
    for g in groups:
        per='；'.join(('共用基座' if r['seed'] is None else f'种子{r["seed"]}')+f' {r["successes"]}/8' for r in g['per_seed'])
        trs.append(f'<tr><td>{"同分布" if g["split"]=="train" else "几何范围外"}</td><td>{LABELS[g["arm"]]}</td><td>{per}</td><td>{g["mean_rate"]*100:.2f}%</td></tr>')
    body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P32 交互损失验证</title>
<style>body{font:17px/1.65 system-ui;background:#f4f6f8;color:#182d38;max-width:1280px;margin:38px auto;padding:0 28px}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px 16px;border-bottom:1px solid #dde4e9;text-align:left}img{width:100%;background:white;border-radius:12px}a{color:#116c8a}.note{background:#e7eef3;padding:15px;border-radius:9px}</style>
<h1>P32 · 预测损失与学到的交互损失</h1>
<p>144个物理回合与完整记录复核完成。开发验证；每档8个独立物体条件，两个固定模型种子共享条件，基座只采一次。未晋升。</p>
<p><a href="REPORT_ZH.md">主Agent结论与限制</a> · <a href="SUMMARY.json">完整数值</a> · <a href="SUCCESS_RATES.csv">下载表格</a> · <a href="../PROTOCOL.json">结果前协议</a> · <a href="../AUDIT.json">记录复核</a></p>
<img src="success_rates.svg" alt="各方法同分布及范围外成功率">
<table><thead><tr><th>设置</th><th>方法</th><th>成功数 / 条件数</th><th>均值</th></tr></thead><tbody>'''+''.join(trs)+'''</tbody></table>
<p class="note">视觉、基座与Observer冻结。预测对照加宽预测头匹配参数预算；两组预测辅助项对z停止梯度。因此是新的损失家族比较，不是历史P31逐字复现。其他物体历史对照保持记录内部动作—反馈一致，但接触阶段覆盖可能不同。代理误差与真实成功率分开报告。</p>
<h2>代表性注释视频</h2><div id="videos"></div>
<script>fetch('media/VIDEO_INDEX.json').then(r=>r.json()).then(d=>{for(const x of d.videos){const s=document.createElement('section');const h=document.createElement('h3');h.textContent=x.caption;const v=document.createElement('video');v.controls=true;v.preload='metadata';v.style.width='100%';v.src='media/'+x.file.split('/').pop();s.append(h,v);document.getElementById('videos').append(s)}}).catch(()=>{document.getElementById('videos').textContent='代表视频尚未生成；完整数值不依赖视频。'});</script></html>'''
    (out/'index.html').write_text(body)
    print('SUMMARY_COMPLETE')
    for g in groups:print(g['split'],g['arm'],[r['successes'] for r in g['per_seed']],g['mean_rate'])


if __name__=='__main__':main()
