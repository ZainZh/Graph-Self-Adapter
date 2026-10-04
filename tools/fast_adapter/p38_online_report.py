"""Audit/replay or summarize the sealed P38 continuous numerical confirmation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.causal_mapping import Identification, effect
from tools.fast_adapter import p38_mapping as common

R=common.ROOT/'online_mapping'


def audit():
    from tools.fast_adapter.p38_online_causal_final import run
    actor=common.controller();reports=[]
    for kind in ('structured','effect','history'):
        for seed in (11,22,33):
            source=R/'mapping'/kind/f'seed{seed}/model.pt'
            n=Identification(kind).cuda();n.load_state_dict(torch.load(source,weights_only=True));n.eval().requires_grad_(False)
            folder=R/'causal_pairs/final'/kind/f'seed{seed}'
            mean=torch.load(folder/'TRAIN_MEAN.pt',weights_only=True).cuda()
            for code,split in enumerate(('native','id','ood')):
                dest=folder/split;seal=json.loads((dest/'SEAL.json').read_text())
                assert seal['runtime_hash']==common.sha(dest/'RUNTIME.pt') and seal['model_hash']==common.sha(source)
                raw=torch.load(dest/'RUNTIME.pt',weights_only=True)
                e=torch.load(dest/'EVALUATOR.pt',weights_only=True).cuda()
                result=json.loads((dest/'RESULT.json').read_text())
                for arm,tr in raw.items():
                    replay=run(n,actor,e,tr['state'][:,0].cuda(),tr['goal'].cuda(),arm,common.generator(500000+code),mean)
                    errors={k:float((replay[k]-tr[k]).abs().max()) for k in tr}
                    assert max(errors.values())<3e-5,(kind,seed,split,arm,errors)
                    final=(tr['state'][:,-1]-tr['goal']).norm(dim=-1)
                    assert (final<.03).tolist()==result['rows'][arm]['success']
                    assert torch.allclose(final,torch.tensor(result['rows'][arm]['error']),atol=1e-6)
                    reports.append({'kind':kind,'seed':seed,'split':split,'arm':arm,'errors':errors,'n':len(final)})
            print('audited',kind,seed,flush=True)
    common.save_json(R/'summary/CAUSAL_AUDIT.json',{'passed':True,'replays':reports,
              'noise_reconstruction':'exact original CUDA generator seed500000+stratum; source and weights unchanged'})


def summarize():
    rows={};summaries={};comparisons=[]
    for p in sorted((R/'causal_pairs/final').glob('*/seed*/*/RESULT.json')):
        j=json.loads(p.read_text());rows.setdefault((j['kind'],j['split']),[]).append(j)
    assert len(rows)==9 and all(len(v)==3 for v in rows.values())
    rng=np.random.default_rng(380929)
    for (kind,split),items in rows.items():
        summaries[kind+'/'+split]={arm:{'success_rate':float(np.mean([j['rows'][arm]['success'] for j in items])),
             'error':float(np.mean([j['rows'][arm]['error'] for j in items])),
             'per_seed_success':[int(np.sum(j['rows'][arm]['success'])) for j in items],
             'error_curve':np.mean([j['rows'][arm]['error_curve'] for j in items],axis=0).tolist()} for arm in items[0]['rows']}
        for arm in ('frozen','donor','train_mean','least_squares'):
            d=np.mean([np.array(j['rows']['continuous']['success'],float)-np.array(j['rows'][arm]['success'],float) for j in items],axis=0)
            d=d.reshape(-1,2).mean(1)
            bs=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(-1)
            comparisons.append({'kind':kind,'split':split,'control':arm,'delta':float(d.mean()),
              'ci95':np.quantile(bs,[.025,.975]).tolist(),'ci_bonferroni12':np.quantile(bs,[.025/12,1-.025/12]).tolist(),
              'n_independent_domain_pairs':len(d),'scope':'three fixed models,not new training-seed generality'})
    common.save_json(R/'summary/CAUSAL_RESULTS.json',{'summary':summaries,'comparisons':comparisons,
                'actions_all_arms':8,'independent_domains_per_stratum':512,'support':0,'used_updates':7})
    lines=['# P38 连续训练与连续执行的独立验证','','|表示|设置|冻结z|连续更新|错配反馈|训练均值|History说明|','|---|---|---:|---:|---:|---:|---|']
    for key,arms in summaries.items():
        kind,split=key.split('/');lines.append(f'|{kind}|{split}|'+ '|'.join(f'{100*arms[a]["success_rate"]:.2f}%' for a in ('frozen','continuous','donor','train_mean'))+'|同训练预算单列history|')
    lines+=['','所有臂8次实际动作；没有独立校准轨迹。前7次完成动作后各更新一次，影响后续动作；不把优化次数当作独立回合。',
             '共用同一冻结神经动作头；真参数输入该头不是数学控制上界，另列真实响应解析Oracle检验任务可解性。',
             'History也按相同8步闭环、同4000外层次数重新训练，不用旧的少步History失败来宣称梯度胜出。']
    out=R/'summary/CAUSAL_REPORT_ZH.md';out.write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','report']);a=p.parse_args();torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    if a.mode=='audit':audit()
    else:summarize()
