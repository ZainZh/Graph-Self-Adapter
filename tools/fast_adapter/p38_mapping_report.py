"""Batch audit and statistics for sealed P38 mapping records; no rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def process(root,phase):
    files=sorted((root/'mapping').glob(f'*/seed*/{phase}/*/RESULT.json'))
    rows={};audits=[];base_reference={}
    for path in files:
        j=json.loads(path.read_text());raw=torch.load(path.parent/'RUNTIME.pt',weights_only=True)
        seal=json.loads((path.parent/'SEAL.json').read_text())
        assert hashlib.sha256((path.parent/'RUNTIME.pt').read_bytes()).hexdigest()==seal['runtime_hash']
        assert hashlib.sha256((path.parent.parent.parent/'model.pt').read_bytes()).hexdigest()==seal['model_hash']
        truth=torch.load(path.parent/'EVALUATOR.pt',weights_only=True)['dynamics']
        e=truth.reshape(-1,2,3)
        for name,trace in raw['trajectories'].items():
            measured=(trace['state'][:,-1]-trace['goal']).norm(dim=-1)
            assert torch.allclose(measured,torch.tensor(j['rows'][name]['error']),atol=1e-6)
            assert (measured<.03).tolist()==j['rows'][name]['success']
            a=trace['action'];design=torch.cat((a,torch.ones_like(a[...,:1])),-1)
            predicted=torch.einsum('bsj,bij->bsi',design,e)
            actual=trace['state'][:,1:]-trace['state'][:,:-1]
            error=float((actual-predicted).abs().max());assert error<2e-5
            assert torch.isfinite(trace['state']).all() and a.abs().max()<=1.
            audits.append(error)
        split=j['split'];ref=raw['trajectories']['frozen']
        if split in base_reference:
            for k in ('state','action','goal'):assert torch.equal(ref[k],base_reference[split][k])
        else:base_reference[split]=ref
        rows.setdefault((j['kind'],j['split']),[]).append(j)
    summaries={};comparisons=[]
    rng=np.random.default_rng(380919)
    for (kind,split),items in rows.items():
        arm_values={}
        for arm in items[0]['rows']:
            sr=np.array([x['rows'][arm]['success'] for x in items],dtype=float)
            er=np.array([x['rows'][arm]['error'] for x in items])
            arm_values[arm]={'success_rate':float(sr.mean()),'error':float(er.mean()),
                            'per_seed_success':[float(x.mean()) for x in sr]}
        key=kind+'/'+split
        summaries[key]={'seeds':[x['seed'] for x in items],'n_conditions':items[0]['n'],
                        'arms':arm_values,'diagnostics':{k:float(np.mean([x['diagnostics'][k] for x in items])) for k in items[0]['diagnostics'] if k!='loss_curve'}}
        n=items[0]['n'];resamples=rng.integers(0,n,size=(20000,n))
        for a,b in [('correct','frozen'),('correct','swapped'),('online','frozen'),('correct','least_squares')]:
            d=np.array([np.array(x['rows'][a]['success'],float)-np.array(x['rows'][b]['success'],float) for x in items]).mean(0)
            bs=d[resamples].mean(-1)
            comparisons.append({'kind':kind,'split':split,'a':a,'b':b,'delta':float(d.mean()),
                'ci95':np.quantile(bs,[.025,.975]).tolist(),
                'ci_bonferroni16':np.quantile(bs,[.025/16,1-.025/16]).tolist(),
                'scope':'fixed model seeds; independent domain clusters, not a new-model population CI'})
    out=root/'summary';out.mkdir(exist_ok=True,parents=True)
    result={'phase':phase,'summary':summaries,'paired_comparisons':comparisons,
            'audit':{'files':len(files),'trajectories_audited':len(audits),'max_dynamics_replay_error':max(audits),'matched_conditions_and_frozen_actions':True}}
    (out/(phase+'_RESULTS.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# P38 数值机制：'+phase,'','动作头共用、冻结；z0严格等于基座。数值位置是公开观察，环境参数只用于训练/Oracle/封存后评价。',
           '','|表示|设置|冻结z|S3/U3|错配z|在线从零交互|History或LS说明|','|---|---|---:|---:|---:|---:|---|']
    for key,j in summaries.items():
        a=j['arms'];v=[a[k]['success_rate']*100 for k in ('frozen','correct','swapped','online')]
        kind,split=key.split('/');lines.append(f'|{kind}|{split}|'+ '|'.join(f'{x:.2f}%' for x in v)+f'|LS {a["least_squares"]["success_rate"]*100:.2f}%|')
    lines+=['','S3/U3使用3次共同校准交互＋8次query；online没有校准支持，在8次实际执行中更新。两者预算不同，不能直接比较为同成本算法优劣。',
            '所有优化次数消融、诊断、固定模型配对区间由同目录数值文件保留。更多优化步不增加交互，结果不能按测试集挑选成一个赢家。']
    (out/(phase+'_REPORT_ZH.md')).write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['audit']));print('\n'.join(lines[4:]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('evaluations/fast_adapter_p38_causal_v01'));p.add_argument('--phase',default='final');a=p.parse_args();torch.set_num_threads(1);process(a.root,a.phase)
