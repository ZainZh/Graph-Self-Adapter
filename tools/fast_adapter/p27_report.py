"""Predeclared all-arm attribution, fixed-seed environment-cluster intervals."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tools.fast_adapter.p27_common import R,SEEDS,STRATA,N
from tools.fast_adapter.p27_statistics import paired_clusters
from tools.fast_adapter.p18_run import save_json,digest

NAMES={'id':'同分布','position':'槽位置变化','scale':'整体尺度变化'}
LABELS=[('base','冻结基座 / 不更新z'),('parent_correct','P25父模型'),('oldlabel_correct','旧标签续训'),('relabel_correct','修正标签'),('refresh_correct','修正标签＋补采：正确反馈'),('refresh_matchedmean','同采集方式的固定训练平均z'),('refresh_mean','原训练平均z'),('refresh_donor1','独立错配反馈1'),('refresh_donor2','独立错配反馈2'),('refresh_donors','两份错配反馈平均')]
PRIMARY=[('refresh_correct','base'),('refresh_correct','refresh_matchedmean'),('refresh_correct','refresh_donors')]
SECONDARY=[('relabel_correct','oldlabel_correct'),('refresh_correct','relabel_correct'),('refresh_correct','parent_correct'),('refresh_correct','refresh_mean')]


def arrays(metrics):
    out={'base':np.array(metrics['base']['success'],float)}
    for name,_ in LABELS[1:-1]:out[name]=np.array([metrics[f'{name}_seed{s}']['success'] for s in SEEDS],float)
    out['refresh_donors']=(out['refresh_donor1']+out['refresh_donor2'])/2
    return out


def main():
    assert json.loads((R/'summary/AUDIT.json').read_text())['passed']
    out=R/'summary';results={};bindings=[]
    for j,st in enumerate(STRATA):
        p=R/'evaluation'/st/'RESULT.json';a=arrays(json.loads(p.read_text())['metrics'])
        primary={f'{x}_minus_{y}':paired_clusters(a[x],a[y],seed=27027+j*10+k) for k,(x,y) in enumerate(PRIMARY)}
        secondary={f'{x}_minus_{y}':paired_clusters(a[x],a[y],seed=28027+j*10+k,family=1) for k,(x,y) in enumerate(SECONDARY)}
        results[st]={'rates_percent':{k:float(v.mean()*100) for k,v in a.items()},
            'seed_rates_percent':{k:(v.mean(1)*100).tolist() for k,v in a.items() if v.ndim==2},
            'primary':primary,'secondary_exploratory':secondary,
            'system_benefit_confirmed_conditionally':primary['refresh_correct_minus_base']['conditional_positive_gate'],
            'environment_specific_benefit_confirmed_conditionally':all(primary[f'refresh_correct_minus_{k}']['conditional_positive_gate'] for k in ('refresh_matchedmean','refresh_donors')),
            'independent_condition_clusters':N}
        bindings.append({'path':str(p),'sha256':digest(p)})
    pred=json.loads((out/'PREDICTION_DIAGNOSTIC.json').read_text())['records'];pred_summary={}
    for st in STRATA:
        pred_summary[st]={}
        for mode in ('correct','donor1','donor2','mean','matchedmean','k0'):
            pred_summary[st][mode]=float(np.mean([x['condition_mse'] for x in pred if x['stratum']==st and x['mode']==mode]))
    save_json(out/'RESULTS.json',{'strata':results,'prediction_mse_on_base_states':pred_summary,'source_bindings':bindings,
        'stage':'prospective_frozen_checkpoint_attribution','uncertainty':'128 independent target-plus-donor clusters per stratum. Two fixed model checkpoints are averaged inside clusters. Primary95% and99.4444% Bonferroni-adjusted paired bootstrap intervals are approximate; no training-seed population or History claim.',
        'secondary_inference':'Exploratory95% intervals; not additional multiplicity-confirmed primary findings.'})
    lines=['# P27 冻结模型归因结果','','三档各128个新独立环境条件，两个固定模型种子共享条件；每个条件额外配两份独立且不复用的同档donor反馈。所有模型和视觉前端保持P26封存权重，没有重新训练。','',
        '| 方法 | 同分布 | 位置变化 | 尺度变化 |','|---|---:|---:|---:|']
    for key,label in LABELS:lines.append('| '+label+' | '+' | '.join(f"{results[s]['rates_percent'][key]:.2f}%" for s in STRATA)+' |')
    lines+=['','## 九项预定主比较','', '| 设置 | 比较 | 差值（百分点） | 配对95%区间 | 九比较校正区间 | 通过条件门 |','|---|---|---:|---|---|---|']
    compare_names={'base':'冻结基座','refresh_matchedmean':'匹配采集方式的固定平均z','refresh_donors':'两份独立错配反馈平均'}
    for st in STRATA:
        for x,y in PRIMARY:
            q=results[st]['primary'][f'{x}_minus_{y}'];f=lambda v:f'[{v[0]:.2f}, {v[1]:.2f}]'
            lines.append(f"| {NAMES[st]} | 正确反馈 − {compare_names[y]} | {q['delta_pp']:+.2f} | {f(q['paired_ci95'])} | {f(q['multiplicity_adjusted_ci'])} | {'是' if q['conditional_positive_gate'] else '否'} |")
    lines+=['','门要求校正后区间下界>0且两固定模型方向均>0；这是对两个已训练checkpoint的条件结论。≥5个百分点的实用幅度另在机器结果中记录。',
        '', '## 训练改动的复验（次要比较）','', '| 设置 | 修正标签 − 同预算旧标签 | 补采 − 修正标签 | 修复版 − P25父模型 |','|---|---:|---:|---:|']
    for st in STRATA:
        q=results[st]['secondary_exploratory'];keys=['relabel_correct_minus_oldlabel_correct','refresh_correct_minus_relabel_correct','refresh_correct_minus_parent_correct']
        lines.append('| '+NAMES[st]+' | '+' | '.join(f"{q[k]['delta_pp']:+.2f} pp" for k in keys)+' |')
    lines+=['','次要比较保留全部种子和探索性配对95%区间，不作为额外已校正主结论。',
        '', '## 成本与边界','',
        '主方法每个环境仍需32个支持动作、3条选中反馈/3次z更新，之后换独立初态执行48步，query中z固定；不把3次更新写成3次物理动作。不更新z与基座恒等，共用基座物理轨迹并逐输入验证。',
        '', '共6528条query、384条本环境支持、768条donor支持。donor是错配对照的额外实验成本，不进入主方法。固定匹配平均z仅使用P26训练期32条共同基座支持，不含新测试信息；它是冻结平均经验控制，不声称最强固定策略。',
        '', '全部权重与动作/latent/成功判据复算见[AUDIT.json](AUDIT.json)。两个模型种子不能替代更多训练种子；本轮没有同预算重训History，不能声称梯度算法优于History。实验仍是一个自定义无机械臂的平移插入任务的三个几何设置，不是官方LIBERO或真实机器人证明。',
        '', '位移预测诊断来自共同基座轨迹上的真实动作→物块位移，运行时不读槽真值，预测改善不能替代任务成功。[所有原始对比](RESULTS.json)。', '', '![成功率与归因](success_rates.png)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');fig,axes=plt.subplots(1,3,figsize=(14.5,5),sharey=True)
    keys=[('base','基座'),('parent_correct','修复前'),('refresh_correct','正确反馈'),('refresh_matchedmean','固定平均z'),('refresh_donors','错配反馈')]
    for ax,st in zip(axes,STRATA):
        vals=[results[st]['rates_percent'][k] for k,_ in keys];bars=ax.bar(range(len(keys)),vals,color=['#89959d','#ba8c6b','#17786c','#738eb7','#a794b7'])
        ax.set_xticks(range(len(keys)),[label for _,label in keys],rotation=22,fontproperties=font);ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_title(NAMES[st],fontproperties=font)
        for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+1.5,f'{val:.1f}',ha='center')
    axes[0].set_ylabel('成功率（%）',fontproperties=font);fig.suptitle('P27：每档128个新独立条件｜固定两个模型｜全部权重不变',fontproperties=font);fig.tight_layout();fig.savefig(out/'success_rates.png',dpi=175);plt.close(fig)
    print(json.dumps({st:{'rates':r['rates_percent'],'system_gate':r['system_benefit_confirmed_conditionally'],'feedback_gate':r['environment_specific_benefit_confirmed_conditionally']} for st,r in results.items()}),flush=True)

if __name__=='__main__':main()
