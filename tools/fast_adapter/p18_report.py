"""Summarize all P18 final cells without selecting a winning K/checkpoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K=(0,1,2,3,5)
FAMILIES=('gradient_z','history','hybrid')
SEEDS=(11,22,33,44,55)
STRATA=('id','mild','strong')
LABEL={'base':'Base','fixed_z':'Fixed z (same Gradient model)','gradient_z':'Gradient z','history':'History','hybrid':'Hybrid','oracle':'Bounded Oracle'}


def interval(delta, reps=4000):
    """Crossed seed/domain bootstrap, keeping query repeats within domains."""
    rng=np.random.default_rng(831991)
    means=[]
    for _ in range(reps):
        s=rng.integers(0,delta.shape[0],delta.shape[0]);d=rng.integers(0,delta.shape[1],delta.shape[1])
        means.append(float(delta[s][:,d].mean()))
    return {'mean':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'per_seed':delta.mean(1).tolist(),'positive_seeds':int((delta.mean(1)>0).sum())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='evaluations/fast_adapter_p18_visual_v01');a=p.parse_args();root=Path(a.root)
    cells={}; diagnostics={};complete=[]
    for family in FAMILIES:
        for seed in SEEDS:
            path=root/'final'/family/f'seed{seed}'
            receipt=json.loads((path/'EVAL.json').read_text())
            assert receipt['all_model_parameters_unchanged'] and receipt['domains_per_stratum']==128
            complete.append(str(path/'EVAL.json'))
            for line in (path/'raw.jsonl').read_text().splitlines():
                r=json.loads(line);key=(r['stratum'],r['arm'],r['k'],seed)
                cells.setdefault(key,{})[r['domain']]=(float(np.mean(r['query_success'])),float(np.mean(r['query_error'])))
            for r in receipt['details']: diagnostics[(r['stratum'],r['arm'],r['k'],seed)]=r
    def data(st,arm,k,metric=0):
        if arm=='fixed_z':arm='gradient_z';k=0
        if arm in ('base','oracle'):k=0
        return np.array([[cells[(st,arm,k,s)][d][metric] for d in range(128)] for s in SEEDS])
    means={};contrasts={}
    for st in STRATA:
        means[st]={arm:{str(k):{'success':float(data(st,arm,k).mean()),'error':float(data(st,arm,k,1).mean())} for k in K} for arm in LABEL}
        contrasts[st]={}
        pairs={'gradient_vs_own_fixed':('gradient_z',3,'gradient_z',0),'gradient_vs_swap':('gradient_z',3,'gradient_z_swapped',3),'gradient_vs_history':('gradient_z',3,'history',3),'hybrid_vs_history':('hybrid',3,'history',3),'hybrid_vs_own_no_refinement':('hybrid',3,'hybrid_no_refinement',3),'gradient_vs_zero_support_visual':('gradient_z',3,'gradient_z_zero_support_visual',3),'history_vs_zero_support_visual':('history',3,'history_zero_support_visual',3),'hybrid_vs_swap':('hybrid',3,'hybrid_swapped',3)}
        for name,(arm,k,base,bk) in pairs.items():
            contrasts[st][name]={'success_delta':interval(data(st,arm,k)-data(st,base,bk)),'goal_error_delta':interval(data(st,arm,k,1)-data(st,base,bk,1))}
    out=root/'summary';out.mkdir(exist_ok=True)
    result={'complete_final_jobs':len(complete),'means':means,'contrasts':contrasts,'bootstrap':'crossed 5 training seeds x128 held-out domains, query4 retained within domain; 4000 replicates; pointwise 95% CI, no multiple-comparison-adjusted global superiority claim','scope':'GT-ROI-assisted RGB 2D alignment surrogate; no contact insertion or robot validation'}
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for metric,ylabel,name in [('success','Task success (%)','success_vs_interactions'),('error','Final goal error','error_vs_interactions')]:
        fig,axs=plt.subplots(1,3,figsize=(15,4.2),constrained_layout=True)
        for ax,st in zip(axs,STRATA):
            for arm in LABEL:
                y=[means[st][arm][str(k)][metric]*(100 if metric=='success' else 1) for k in K]
                ax.plot(K,y,marker='o',label=LABEL[arm],linestyle='--' if arm in ('base','fixed_z','oracle') else '-')
            ax.set(title=st.upper(),xlabel='Support interactions K',ylabel=ylabel,xticks=K);ax.grid(alpha=.2)
        axs[-1].legend(fontsize=8,loc='best');fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
    for metric,ylabel,name in [('calibration_prediction_mse','Held-out next-u prediction MSE','prediction_loss_vs_interactions'),('z_distance_from_z0','Mean ||z - learned z0||','z_distance_vs_interactions')]:
        fig,axs=plt.subplots(1,3,figsize=(15,4.2),constrained_layout=True)
        for ax,st in zip(axs,STRATA):
            for arm in FAMILIES:
                vals=[np.array([diagnostics[(st,arm,k,s)].get(metric) for s in SEEDS]) for k in K]
                if any(v[0] is None for v in vals):continue
                ax.plot(K,[v.mean() for v in vals],marker='o',label=LABEL[arm])
            ax.set(title=st.upper(),xlabel='Support interactions K',ylabel=ylabel,xticks=K);ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
    lines=['# P18 RGB + DINOv3 Fast Adapter 完整结果','', '完成 3 类可训练方法 × 5 粒模型种子。每档 final 为 128 个隐藏物理域 × 4 个新起点；五个训练种子共享相同测试域，不把重复测试误算为 2560 个独立域。','', 'Base 使用 GT ROI 中心；可学习模块只接收 DINO ROI 特征、前一动作和当前 nominal action。支持阶段使用共同固定校准动作；query 阶段不更新 latent。', '', 'Fixed-z 是 Gradient 同 checkpoint 的 K0 行，不是另训的弱基线。Base、Fixed-z、Oracle 的常数行跨 K 重复展示，不声称重复运行产生额外样本。','']
    for st in STRATA:
        lines+=['## '+st.upper(),'', '| Method | K0 | K1 | K2 | K3 | K5 |','|---|---:|---:|---:|---:|---:|']
        for arm in LABEL:lines.append('| '+LABEL[arm]+' | '+' | '.join(f"{means[st][arm][str(k)]['success']*100:.2f}%" for k in K)+' |')
        lines+=['','配对差异（百分点；括号为 crossed bootstrap 95% CI）：','']
        for name,d in contrasts[st].items():
            x=d['success_delta'];lo,hi=x['ci95'];lines.append(f"- {name}: {x['mean']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]；正向种子 {x['positive_seeds']}/5。")
        lines.append('')
    lines+=['## 解释边界','','这是真实冻结 DINOv3 特征上的二维视觉控制机制测试；GT ROI 与 nominal controller 的几何信息仍由模拟器提供。它不证明无 GT 视觉 grounding、接触插入、真实机器人或完整 RSI。所有 K、三档分布和种子都保留；不按最终结果重选 checkpoint、K 或任务。','', '主Agent的科学验收结论见 ROOT_ACCEPTANCE.json；图表和以上数字从完整 raw.jsonl 重算。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete_final_jobs':len(complete),'summary':str(out/'SUMMARY.json')}))

if __name__=='__main__':main()
