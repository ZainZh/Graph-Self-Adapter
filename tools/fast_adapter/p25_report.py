"""All-arm descriptive development report; no model selection or automatic final."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tools.fast_adapter.p25_common import R, SEEDS, VARIANTS
from tools.fast_adapter.p18_run import save_json, digest

NAMES={'id':'同分布','position':'槽位置变化','scale':'整体尺度变化'}


def paired(a,b):
    d=np.asarray(a,float)-np.asarray(b,float)
    if d.ndim==1:d=d[None]
    rng=np.random.default_rng(25025)
    si=rng.integers(d.shape[0],size=(2000,d.shape[0]));di=rng.integers(d.shape[1],size=(2000,d.shape[1]))
    boot=d[si[:,:,None],di[:,None,:]].mean((1,2))*100
    return {'delta_pp':float(d.mean()*100),'descriptive_ci95':np.quantile(boot,[.025,.975]).tolist(),
        'seed_deltas_pp':(d.mean(1)*100).tolist(),'wins':int((d>0).sum()),'harms':int((d<0).sum())}


def main():
    assert (R/'CAMPAIGN_DONE.json').exists()
    out=R/'summary';out.mkdir(exist_ok=True);results={};bindings=[]
    for st in NAMES:
        path=R/'development'/st/'RESULT.json';m=json.loads(path.read_text())['metrics']
        arrays={a:np.array(m[a]['success']) for a in ('base','oracle','bounded_oracle')}
        for name in VARIANTS:
            modes=['correct','swapped','mean']+(['k0'] if name=='original' else [])+(['early'] if name=='physical' else [])
            for mode in modes:
                arm=f'{name}_{mode}'
                arrays[arm]=np.array([m[f'{arm}_seed{s}']['success'] for s in SEEDS])
        pairs=[('anchored_correct','original_correct'),('physical_correct','anchored_correct'),
            ('physical_correct','base'),('physical_correct','physical_swapped'),('physical_correct','physical_mean'),
            ('physical_correct','physical_early'),('physical_correct','history_correct'),('bounded_oracle','base')]
        results[st]={'rates_percent':{k:float(v.mean()*100) for k,v in arrays.items()},
            'seed_rates_percent':{k:(v.mean(1)*100).tolist() for k,v in arrays.items() if v.ndim==2},
            'independent_conditions':len(arrays['base']),
            'comparisons':{f'{a}_minus_{b}':paired(arrays[a],arrays[b]) for a,b in pairs}}
        bindings.append({'path':str(path),'sha256':digest(path)})
    save_json(out/'RESULTS.json',{'strata':results,'source_bindings':bindings,'stage':'small_development_only',
        'uncertainty':'12 independent conditions per stratum, two model seeds share them. Bootstrap is descriptive, not confirmatory evidence; no final or promotion.',
        'scope':'All methods retrained with same phase-spanning data. Not a one-factor comparison against P24 results. The physical target changes predictor output and loss scaling as declared.',
        'cost':'32 shared support actions, 3 selected transitions/updates, then independent reset and48-step query. No3-action efficiency claim.'})
    labels=[('base','冻结基座'),('original_k0','原结构K0'),('original_correct','原结构＋反馈'),('anchored_correct','保护基座＋反馈'),
        ('physical_correct','保护基座＋位移反馈'),('physical_early','位移版本：早期反馈'),('physical_swapped','位移版本：交换z'),
        ('physical_mean','位移版本：训练平均z'),('history_correct','History'),('bounded_oracle','同修正范围真值参考'),('oracle','真值参考')]
    lines=['# P25 小规模开发结果','', 'LiLa、PCA与视觉基座不变。两个种子，每版本6000次更新；每类12个新开发条件。结果属于快筛，不是正式最终验证。','',
        '| 方法 | 同分布 | 槽位置变化 | 整体尺度变化 |','|---|---:|---:|---:|']
    for key,label in labels:lines.append('| '+label+' | '+' | '.join(f"{results[st]['rates_percent'][key]:.2f}%" for st in NAMES)+' |')
    lines+=['','保护版本K0与基座严格相同，按预定代数等价复用基座轨迹；原结构K0独立执行。所有correct/swapped/mean与种子完整保留于RESULTS.json。',
        '', '每个环境先由同一冻结基座采集32个动作，从三段时间窗各选择一条实际响应偏差最大的已完成转移；各方法读取相同三条，query换独立起点，z固定48步。接触事实仅在事后诊断，不用于选择。',
        '', '训练复用2304条P24训练期完整query轨迹，每个环境六个独立起点；support与query来自不同轨迹。无新训练物理采集。原结构也在该新材料上重训，不能和P24最终结果做单因素归因。',
        '', '![开发成功率](success_rates.png)', '', '完整审计见AUDIT.json，主Agent解释见ROOT_REPORT_ZH.md。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    selected=[('base','冻结基座'),('original_correct','原结构'),('anchored_correct','保护基座'),('physical_correct','位移反馈'),('history_correct','History')]
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.8),sharey=True)
    colors=['#788895','#b97752','#d5a741','#147c73','#768bc2']
    for ax,(st,label) in zip(axes,NAMES.items()):
        values=[results[st]['rates_percent'][key] for key,_ in selected]
        bars=ax.bar(range(5),values,color=colors)
        ax.set_xticks(range(5),[label for _,label in selected],rotation=20,fontproperties=font)
        ax.set_title(label,fontproperties=font);ax.set_ylim(0,110);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+2,f'{value:.1f}',ha='center')
    axes[0].set_ylabel('成功率（%）',fontproperties=font)
    fig.suptitle('P25 开发快筛｜每类12条件、两个模型种子｜视觉前端保持不变',fontproperties=font)
    fig.tight_layout();fig.savefig(out/'success_rates.png',dpi=170);plt.close(fig)
    print(json.dumps({s:r['rates_percent'] for s,r in results.items()}),flush=True)


if __name__=='__main__':main()
