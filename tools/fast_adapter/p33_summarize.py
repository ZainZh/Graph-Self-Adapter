"""Small-cohort descriptive summaries; object conditions are the sampling units."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.fast_adapter.p33_common import *

LABELS={'base':'冻结基座','cross_fixed':'关系模型·固定z','cross_first3':'关系模型·前三次','cross_continuous':'关系模型·持续',
        'within_fixed':'物体内模型·固定z','within_continuous':'物体内模型·持续'}

def main():
    out=ROOT/'summary';out.mkdir(exist_ok=False)
    raw=json.loads((ROOT/'development/RESULT.json').read_text());rr=raw['rows']
    lookup={(r['seed'],r['split'],r['condition'],r['arm']):r for r in rr}
    arms=protocol()['evaluation']['arms'];rates={};comparisons=[]
    for split in ('train','geometry_ood'):
        rates[split]={}
        for arm in arms:
            seeds=[None] if arm=='base' else [11,22]
            counts={str(s):sum(lookup[s,split,i,arm]['success'] for i in range(8)) for s in seeds}
            terminal={str(s):sum(lookup[s,split,i,arm]['terminal_success'] for i in range(8)) for s in seeds}
            rates[split][arm]={'successes':counts,'per_seed_denominator':8,'mean_rate':float(np.mean(list(counts.values())))*12.5,
                               'terminal_successes':terminal,'terminal_mean_rate':float(np.mean(list(terminal.values())))*12.5}
        for ref in ('base','cross_fixed','cross_first3','within_continuous'):
            values=[];by_seed={}
            for s in (11,22):
                a=np.array([lookup[s,split,i,'cross_continuous']['success'] for i in range(8)],int)
                b=np.array([lookup[None if ref=='base' else s,split,i,ref]['success'] for i in range(8)],int)
                values.append(a-b);by_seed[str(s)]={'wins':int(((a==1)&(b==0)).sum()),'harms':int(((a==0)&(b==1)).sum()),
                       'win_conditions':np.where((a==1)&(b==0))[0].tolist(),'harm_conditions':np.where((a==0)&(b==1))[0].tolist()}
            d=np.mean(values,0);gen=np.random.default_rng(3332026)
            boot=d[gen.integers(0,8,(20000,8))].mean(-1)*100
            comparisons.append({'split':split,'current':'cross_continuous','reference':ref,
               'delta_pp':float(d.mean()*100),'descriptive_95_interval':np.quantile(boot,[.025,.975]).tolist(),'paired':by_seed})
    d={'phase':'development','physical_episodes':len(rr),'unique_conditions':16,'model_seeds':[11,22],
       'rates':rates,'comparisons':comparisons,'preparation_failures':sum(not x['preparation']['passed'] for x in rr if x['arm']=='base'),
       'inference_unit':'8 object conditions per stratum; two model seeds share conditions; base run once',
       'statistics':'20k object-cluster bootstrap, model seeds fixed, descriptive and unadjusted; not a confirmatory test',
       'sources':bindings([ROOT/'development/RESULT.json',ROOT/'AUDIT.json',ROOT/'diagnostics/FORECAST.json'])}
    dump(out/'SUMMARY.json',d)
    with (out/'SUCCESS_RATES.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['setting','arm','seed11_success','seed22_success','per_seed_n','mean_percent','terminal_percent'])
        for split,rows in rates.items():
            for arm,r in rows.items():w.writerow([split,arm,r['successes'].get('11',r['successes'].get('None')),r['successes'].get('22','shared base'),8,r['mean_rate'],r['terminal_mean_rate']])
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),sharey=True)
    names=['Base','Relation fixed z','Relation first 3','Relation continual','Within fixed z','Within continual']
    colors=['#64748b','#94a3b8','#d4a044','#168c8b','#c1b2d1','#8460a0']
    for ax,split in zip(axs,('train','geometry_ood')):
        ax.bar(np.arange(6),[rates[split][a]['mean_rate'] for a in arms],color=colors)
        for i,a in enumerate(arms):
            vals=list(rates[split][a]['successes'].values())
            ax.scatter([i-.09+j*.18 for j in range(len(vals))],np.array(vals)*12.5,color='black',s=25,zorder=3)
            ax.text(i,rates[split][a]['mean_rate']+3,f"{rates[split][a]['mean_rate']:.2f}%",ha='center',fontsize=9)
        ax.set_xticks(np.arange(6),names,rotation=28,ha='right');ax.set_ylim(0,112)
        ax.set_title('ID: new objects in training range' if split=='train' else 'OOD: longer / larger hammer heads')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axs[0].set_ylabel('Episode success (%)')
    fig.suptitle('P33 bounded development | 8 shared conditions per setting | dots: two model seeds')
    fig.tight_layout();fig.savefig(out/'success_rates.png',dpi=160);fig.savefig(out/'success_rates.svg');plt.close(fig)
    print(json.dumps({'rates':rates,'comparisons':comparisons},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
