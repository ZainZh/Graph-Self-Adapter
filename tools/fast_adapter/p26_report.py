"""All-arm small-development results, fixed paired comparisons."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tools.fast_adapter.p26_common import R,SEEDS
from tools.fast_adapter.p25_report import paired
from tools.fast_adapter.p18_run import save_json,digest

NAMES={'id':'同分布','position':'槽位置变化','scale':'整体尺度变化'}
LABELS=[('base','冻结视觉基座 / K0'),('parent_correct','P25父模型'),('oldlabel_correct','旧标签继续训练'),('relabel_correct','修正标签'),('refresh_correct','修正标签＋当前轨迹'),('refresh_swapped','修复版交换z'),('refresh_mean','修复版训练平均z'),('bounded_repaired_oracle','同幅度修正专家参考'),('repaired_oracle','修正专家参考')]


def main():
    assert (R/'summary/AUDIT.json').exists()
    out=R/'summary';out.mkdir(exist_ok=True);results={};bindings=[]
    for st in NAMES:
        path=R/'development'/st/'RESULT.json';m=json.loads(path.read_text())['metrics']
        a={key:np.array(m[key]['success']) for key in ('base','repaired_oracle','bounded_repaired_oracle')}
        for key in ('parent_correct','oldlabel_correct','relabel_correct','refresh_correct','refresh_swapped','refresh_mean'):
            a[key]=np.array([m[f'{key}_seed{s}']['success'] for s in SEEDS])
        comparisons=[('refresh_correct','base'),('refresh_correct','parent_correct'),('relabel_correct','oldlabel_correct'),('refresh_correct','relabel_correct'),('refresh_correct','refresh_swapped'),('refresh_correct','refresh_mean'),('relabel_correct','base')]
        results[st]={'rates_percent':{k:float(v.mean()*100) for k,v in a.items()},'independent_conditions':12,
            'seed_rates_percent':{k:(v.mean(1)*100).tolist() for k,v in a.items() if v.ndim==2},
            'comparisons':{f'{x}_minus_{y}':paired(a[x],a[y]) for x,y in comparisons}}
        bindings.append({'path':str(path),'sha256':digest(path)})
    save_json(out/'RESULTS.json',{'strata':results,'source_bindings':bindings,'stage':'small_development_only','uncertainty':'Two seeds share12 conditions per stratum; intervals descriptive, no confirmatory final or History superiority claim.'})
    lines=['# P26 动作头修复开发结果','','视觉、基础控制和交互适应模块均冻结；每种子仅训练动作头，三个修订版本各6000次更新。三类新开发条件各12个，两个模型种子共享这些条件。','',
        '| 方法 | 同分布 | 槽位置变化 | 整体尺度变化 |','|---|---:|---:|---:|']
    for key,label in LABELS:lines.append('| '+label+' | '+' | '.join(f"{results[s]['rates_percent'][key]:.2f}%" for s in NAMES)+' |')
    lines+=['','旧标签与修正标签使用相同初始动作头、训练数据、抽样和6000更新；补采版本从修正标签4000步分叉，末2000步换入25%当前训练轨迹。比较包含保留的P25父模型，不将增加训练成本写成同总预算算法竞赛。',
        '', '每次执行先共享32步冻结基座支持，选3条反馈更新z，再换独立初态执行48步；query期间z固定。K0与基座严格相同。槽坐标/尺度/隐藏动力学只供训练标签和评价，普通控制不读取。',
        '', '先通过48个保持起点、16个新起点的训练专家资格检查；新增32条共同训练支持与64条当前策略query。旧开发/最终数据未回流训练。本批540条新开发query、36条支持；没有正式final或模型晋升。',
        '', '全部臂和种子的数字见RESULTS.json；复算见AUDIT.json。成功判据保持原末三帧插入判据，未放宽。','', '![成功率](success_rates.png)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    plot=[('base','冻结基座'),('parent_correct','父模型'),('oldlabel_correct','旧标签续训'),('relabel_correct','修正标签'),('refresh_correct','修正＋补采')]
    for ax,(st,label) in zip(axes,NAMES.items()):
        vals=[results[st]['rates_percent'][k] for k,_ in plot];bars=ax.bar(range(len(plot)),vals,color=['#788895','#ae7667','#bcab79','#459a93','#16716b'])
        ax.set_xticks(range(len(plot)),[v for _,v in plot],rotation=23,fontproperties=font);ax.set_title(label,fontproperties=font);ax.set_ylim(0,110);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
        for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+2,f'{val:.1f}',ha='center')
    axes[0].set_ylabel('成功率（%）',fontproperties=font);fig.suptitle('P26：冻结视觉与适应模块，只修动作头｜每档12条件、两个种子',fontproperties=font);fig.tight_layout();fig.savefig(out/'success_rates.png',dpi=170);plt.close(fig)
    print(json.dumps({k:v['rates_percent'] for k,v in results.items()}),flush=True)

if __name__=='__main__':main()
