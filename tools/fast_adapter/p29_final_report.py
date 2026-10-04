"""Aggregate selected P29 candidates using the already frozen paired statistics.
Descriptive contact stages and plots supplement, never replace, success outcomes.
"""
import json
from pathlib import Path
import numpy as np
from tools.fast_adapter.p29_common import R, TASKS, SEEDS, task_root, digest, save_json
from tools.fast_adapter.p29_eval import STRATA
from tools.fast_adapter.p29_report import ARMS, LABELS, TASK_NAMES, STRATUM_NAMES, arrays
from tools.fast_adapter.p27_statistics import paired_clusters


def physical_diagnostics(result):
    samples = {a: [] for a in ARMS}
    for row in result['rows']:
        arm = row['arm'].rsplit('_seed', 1)[0]
        side = json.loads(Path(row['path']).with_suffix('.json').read_text())
        for m in side['metrics']:
            samples[arm].append({
                'bilateral_grasp': any(x['bilateral_handle'] for x in m),
                'head_contact_while_grasped': any(x['head_contact_seen'] for x in m),
                'direct_robot_target': any(x['target_direct_contact'] for x in m),
                'any_success': any(x['success'] for x in m),
                'terminal_success': bool(m[-1]['success']),
                'max_lift_m': max(x['lift'] for x in m),
                'max_target_displacement_m': max(x['target_displacement'] for x in m),
                'first_success_step': next((i+1 for i,x in enumerate(m) if x['success']), None)})
    out = {}
    for arm, rows in samples.items():
        vals = {k: float(np.mean([v[k] for v in rows])) for k in rows[0] if k != 'first_success_step'}
        steps = [v['first_success_step'] for v in rows if v['first_success_step'] is not None]
        vals.update(physical_query_episodes=len(rows), successful_first_step_mean=float(np.mean(steps)) if steps else None,
                    scope='descriptive query proportions; paired condition clusters, not pooled episodes, used for inference')
        out[arm] = vals
    return out


def build():
    out = R/'summary';out.mkdir(exist_ok=True)
    assert not (out/'RESULTS.json').exists()
    records = {};sources=[];counts={'queries':0,'support':0,'conditions':0};lines=[]
    for ti,task in enumerate(TASKS):
        root=task_root(task);fp=root/'FINAL_FREEZE.json';f=json.loads(fp.read_text());rnd=f['round']
        audit_path=out/'final'/rnd/task/'AUDIT.json';audit=json.loads(audit_path.read_text());assert audit['passed']
        gate=root/'candidates/base_v03/BASE_GATE.json' if task==TASKS[1] else root/'qualification/BASE.json'
        g=json.loads(gate.read_text());rows={}
        for si,st in enumerate(STRATA):
            p=root/'final'/rnd/st/'RESULT.json';d=json.loads(p.read_text());a=arrays(d);terminal=arrays(d,'terminal_success');comp={}
            for j,c in enumerate(ARMS):
                if c=='fast_correct':continue
                primary=st=='geometry_ood' and c in ('base','fast_mean','fast_donor')
                comp[c]={**paired_clusters(a['fast_correct'],a[c],seed=29029+ti*100+si*10+j,family=6 if primary else 1),'primary':primary}
            rows[st]={'rates_percent':{k:float(v.mean()*100) for k,v in a.items()},
                      'success_counts':{k:int(v.sum()) if v.ndim==1 else v.sum(1).astype(int).tolist() for k,v in a.items()},
                      'per_seed_rates_percent':{k:(v.mean(1)*100).tolist() for k,v in a.items() if v.ndim==2},
                      'terminal_rates_percent':{k:float(v.mean()*100) for k,v in terminal.items()},
                      'comparisons':comp,'conditions':d['conditions'],'contact_diagnostics':physical_diagnostics(d)}
            counts['queries']+=d['query_episodes'];counts['support']+=d['support_episodes'];counts['conditions']+=d['conditions']
            sources.append({'path':str(p),'sha256':digest(p)})
        records[task]={'round':rnd,'base_gate':{'successes':g['successes'],'n':g['n'],'source':str(gate)},'audit':str(audit_path),'strata':rows,
                       'ood_three_primary_gates_pass':all(rows['geometry_ood']['comparisons'][c]['conditional_positive_gate'] for c in ('base','fast_mean','fast_donor'))}
        sources.extend({'path':str(p),'sha256':digest(p)} for p in (fp,audit_path))
    assert counts=={'queries':3520,'support':640,'conditions':320},counts
    save_json(out/'RESULTS.json',{'tasks':records,'counts':counts,'primary_family':6,'fixed_model_seeds':list(SEEDS),'sources':sources,
              'inference':'Two fixed checkpoints averaged within each object/reset cluster. Six predeclared OOD comparisons; approximate paired percentile bootstrap, not population-of-training-seeds inference.',
              'success':'Frozen physical predicates sustained for15 cup/5 hammer steps; success_any is primary, terminal retention separately reported.'})
    plots(records,out)
    lines=['# P29 独立测试：完整数值','','数值为两个固定模型种子的平均成功率；基座每条件只执行一次。标准形状32个新初态；范围内／范围外各64个新物体条件。','']
    for task,d in records.items():
        lines += [f'## {TASK_NAMES[task]}（{d["round"]}）','','| 方法 | 标准形状 | 范围内新物体 | 范围外新几何 |','|---|---:|---:|---:|']
        for arm in ARMS:
            lines += ['|'+LABELS[arm]+'|'+'|'.join(f'{d["strata"][s]["rates_percent"][arm]:.2f}%' for s in STRATA)+'|']
        lines += ['','| 范围外：正确反馈相对 | 差值（百分点） | 区间 | 两个种子分别差值 |','|---|---:|---|---|']
        for c in ('base','fast_mean','fast_donor','history','robot'):
            x=d['strata']['geometry_ood']['comparisons'][c];ci=x['multiplicity_adjusted_ci']
            lines += [f'|{LABELS[c]}|{x["delta_pp"]:+.2f}|[{ci[0]:.2f}, {ci[1]:.2f}]'+('（六项校正）' if x['primary'] else '（探索性95%）')+'|'+', '.join(f'{v:+.2f}' for v in x['seed_deltas_pp'])+'|']
        lines+=['']
    lines += ['没有追加测试样本、挑选种子或用独立结果重训。History和仅本体变化预测是探索性对照；仅本体反馈对照仍读取当前物体图像。',
              '训练平均z与错配z使用同一个Gradient模型，主要用于检验当前物体支持信息的额外价值；这仍不等于证明梯度机制普遍优于History或静态视觉推断。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({t:{s:d['rates_percent'] for s,d in v['strata'].items()} for t,v in records.items()},ensure_ascii=False,indent=2))


def plots(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Base / K0','Current object gradient','Training mean z','Donor object z','History','Robot feedback gradient']
    colors=['#7a8692','#117c75','#e5b45d','#ad809d','#6593c8','#cc795f']
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),sharey=True)
    for ax,(task,d) in zip(axes,records.items()):
        x=np.arange(3);width=.13
        for i,(arm,label,color) in enumerate(zip(ARMS,labels,colors)):
            y=[d['strata'][st]['rates_percent'][arm] for st in STRATA]
            ax.bar(x+(i-2.5)*width,y,width,color=color,label=label)
            for k,val in enumerate(y):
                if val>0:ax.text(x[k]+(i-2.5)*width,val+1,f'{val:.1f}',ha='center',fontsize=7,rotation=90)
        ax.set_xticks(x,['Canonical\n32 resets','New objects, ID\n64 objects','New geometry, OOD\n64 objects'])
        ax.set_title('Cup handle grasp' if task==TASKS[0] else 'Hammer-to-block interaction')
        ax.set_ylim(0,110);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Success rate (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.005))
    fig.suptitle('P29: native robot actuation, two fixed model seeds; held-out object/reset evaluation')
    fig.tight_layout(rect=(0,.12,1,.94));fig.savefig(out/'success_rates.png',dpi=180,bbox_inches='tight');fig.savefig(out/'success_rates.pdf',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':build()
