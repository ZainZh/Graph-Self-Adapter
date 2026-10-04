"""Task-separated P40 estimates from sealed result files, with paired inference."""
import argparse,json,math
from pathlib import Path
import numpy as np
from tools.fast_adapter.p40_evaluate import ROOT,STRATA,js,sha

TASKS=('cup','hammer','insert')
CONTRASTS=[('relation/continuous','references/base'),('local/continuous','references/base'),
 ('fused/continuous','references/base'),('history/continuous','references/base'),
 ('local/continuous','relation/continuous'),('fused/continuous','relation/continuous'),
 ('fused/continuous','local/continuous'),('relation/continuous','history/continuous'),
 ('relation/continuous','relation/mean'),('relation/continuous','relation/first12'),
 ('relation/continuous','relation/donor'),
 ('relation/continuous','references/strong_fixed'),('local/continuous','references/strong_fixed'),
 ('fused/continuous','references/strong_fixed'),
 ('motion/continuous','references/base'),('motion/continuous','local/continuous'),
 ('motion/continuous','relation/continuous'),('motion/continuous','motion/mean'),
 ('motion/continuous','motion/donor'),('motion/continuous','references/strong_fixed')]

def group(files):
    groups={};bindings=[]
    for p in sorted(files):
        d=json.loads(p.read_text());assert d['total']==len(d['rows']);assert d['successes']==sum(x['success'] for x in d['rows'])
        key=(d['split'],d['kind']+'/'+d['arm']);groups.setdefault(key,[]).append(d);bindings.append(dict(path=str(p),sha256=sha(p)))
    stats={};arrays={}
    for (split,method),items in groups.items():
        items=sorted(items,key=lambda d:d['seed']);seeds=[d['seed'] for d in items];assert len(set(seeds))==len(seeds)
        base=[(x['index'],x['seed'],x.get('pair')) for x in items[0]['rows']]
        for d in items:assert [(x['index'],x['seed'],x.get('pair')) for x in d['rows']]==base
        success=np.array([[r['success'] for r in d['rows']] for d in items],float).mean(0)
        progress=np.array([[r['progress'] for r in d['rows']] for d in items]).mean(0)
        force=np.array([[r['force_mean'] for r in d['rows']] for d in items]).mean(0)
        stats.setdefault(split,{})[method]=dict(success_rate=float(success.mean()),conditions=len(success),model_seeds=seeds,
            per_model=[dict(seed=d['seed'],successes=d['successes'],total=d['total']) for d in items],mean_progress=float(progress.mean()),mean_command_force=float(force.mean()),
            observation_aborts=sum(not r['observation_valid'] for d in items for r in d['rows']))
        arrays[(split,method)]=success
    return stats,arrays,bindings

def contrasts(arrays,phase):
    out={};rng=np.random.default_rng(409901);family=len(CONTRASTS)*len(STRATA)
    for split in STRATA:
        for first,second in CONTRASTS:
            if (split,first) not in arrays or (split,second) not in arrays:continue
            d=arrays[(split,first)]-arrays[(split,second)]
            if split=='paired':d=d.reshape(-1,2).mean(-1)
            samples=d[rng.integers(len(d),size=(50000,len(d)))].mean(-1)
            ci=np.quantile(samples,[.025,.975]);adj=np.quantile(samples,[.025/family,1-.025/family])
            positive=int((d>0).sum());negative=int((d<0).sum());n=positive+negative
            sign_p=min(1.,2*sum(math.comb(n,i) for i in range(min(positive,negative)+1))/2**n) if n else 1.
            donor_dependence=second.endswith('/donor') and split!='paired'
            out.setdefault(split,{})[first+' minus '+second]=dict(difference=float(d.mean()),ci95=ci.tolist(),ci_task_family=adj.tolist(),family_comparisons=family,
                independent_units=len(d),positive_units=positive,negative_units=negative,tie_units=int((d==0).sum()),two_sided_sign_p=sign_p,sign_p_bonferroni=min(1.,sign_p*family),
                confirmed_positive=bool(phase=='final' and not donor_dependence and adj[0]>0 and sign_p*family<.05),descriptive_only=donor_dependence,
                note='Ordinary cyclic donor contrasts are descriptive because of cross-condition dependence. Seeds averaged within condition. Paired stratum resamples appearance pairs; zero-width bootstrap is not population certainty.')
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['dev','final'],required=True);a=p.parse_args();out=ROOT/'summary'/a.phase
    if a.phase=='final':
        freeze=json.loads((ROOT/'FINAL_FREEZE.json').read_text())
        for c in freeze['cells']:
            file=ROOT/c['task']/'final'/c['kind']/('seed'+str(c['seed']))/c['split']/(c['arm']+'_RESULT.json')
            assert file.exists(),str(file)
    out.mkdir(parents=True,exist_ok=False)
    result=dict(phase=a.phase,tasks={},analysis='100 prespecified success contrasts per task across five strata, Bonferroni within task; no pooled cross-task rate; strong fixed selected by development macro success, lower command force then earlier index, frozen before final; any test hindsight envelope descriptive only.');total=0
    for task in TASKS:
        files=list((ROOT/task/a.phase).rglob('*_RESULT.json'));stats,arrays,bindings=group(files);n=sum(json.loads(p.read_text())['total'] for p in files);total+=n
        if a.phase=='final':chosen=json.loads((ROOT/'FINAL_FREEZE.json').read_text())['strong_fixed'][task]
        else:
            def rank(i):
                rows=[stats[s]['references/fixed'+str(i)] for s in STRATA]
                return (-np.mean([x['success_rate'] for x in rows]),np.mean([x['mean_command_force'] for x in rows]),i)
            chosen='fixed'+str(min(range(4),key=rank))
        for split in STRATA:
            stats[split]['references/strong_fixed']=dict(stats[split]['references/'+chosen],alias=chosen)
            arrays[(split,'references/strong_fixed')]=arrays[(split,'references/'+chosen)]
        result['tasks'][task]=dict(stats=stats,contrasts=contrasts(arrays,a.phase),strong_fixed=chosen,executions=n,bindings=bindings)
    result['executions']=total;js(out/'RESULTS.json',result)
    lines=['# P40 '+('开发结果（非独立结论）' if a.phase=='dev' else '独立结果'),'','各模型种子在同一物体条件内平均；执行次数不等于独立样本数。','']
    for task,d in result['tasks'].items():
        lines.extend(['## '+task,'','| 设置 | 基础控制 | 关系适配 | 局部DINO | 变化DINO | 融合 | History |','|---|---:|---:|---:|---:|---:|---:|'])
        for split in STRATA:
            s=d['stats'].get(split,{})
            def pct(k):return f'{s[k]["success_rate"]*100:.2f}%' if k in s else '未运行'
            lines.append('| '+split+' | '+' | '.join(pct(k) for k in ('references/base','relation/continuous','local/continuous','motion/continuous','fused/continuous','history/continuous'))+' |')
        lines.extend(['','全部固定力度、平均latent、前12步后固定、错配反馈、配对区间及逐模型计数见 RESULTS.json。',''])
    (out/'TABLES_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({t:d['executions'] for t,d in result['tasks'].items()}))

if __name__=='__main__':main()
