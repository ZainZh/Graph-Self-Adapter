"""Aggregate sealed P37 full-task outcomes; never selects or tunes a candidate."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
import torch
from tools.fast_adapter.p33_common import dump, digest

ROOT=Path('evaluations/fast_adapter_p37_full_success_v01')
LABELS={'base':'原视觉基座','g224':'全局224','g448':'全局448','g224w':'全局224＋腕部224',
        'g448w':'全局448＋腕部448','no_explicit_bbox_geometry':'无显式框几何',
        'p34_existing_bbox_geometry':'已有框几何','p36_explicit_relation_geometry':'新增显式关系几何'}


def interval(differences):
    # Resample condition clusters, never count model seeds as new objects.
    d=np.asarray(differences,float);rng=np.random.default_rng(3702026)
    means=d[rng.integers(0,len(d),(40000,len(d)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def main():
    manifest=json.loads((ROOT/'MANIFEST.json').read_text());rows=[];audits=[];refs=[];stages=[]
    initial={};pair_max=0.;physical=0
    for job in manifest['jobs']:
        p=ROOT/'runs'/job['id'];done=json.loads((p/'COMPLETE.json').read_text())
        assert done['audit_passed'] and done['outcomes_sha256']==digest(p/'OUTCOMES.json')
        out=json.loads((p/'OUTCOMES.json').read_text());a=json.loads((p/'AUDIT.json').read_text())
        seal=json.loads((p/'TRAJECTORY_SEAL.json').read_text())
        assert seal['trajectory']==out['trajectory']==a['trajectory']
        rawpath=Path(out['trajectory']['path']);assert digest(rawpath)==out['trajectory']['sha256']
        raw=torch.load(rawpath,weights_only=True,map_location='cpu')
        for field in ('visual','proprio','tokens','boxes','actions','base_actions','scores','z_before_action','z_final'):
            if field in raw:assert bool(torch.isfinite(raw[field]).all()),(job['id'],field,'nonfinite')
        key=(job['task'],job['split']);x={k:v[:,0].clone() for k,v in raw.get('states',{}).items()}
        for k in ('visual','proprio'):
            if k in raw:x[k]=raw[k][:,0].clone()
        if key not in initial:initial[key]=(raw['valid'],x)
        else:
            oldvalid,old=initial[key];assert oldvalid==raw['valid']
            for k,v in x.items():
                err=float((v-old[k]).abs().max());pair_max=max(pair_max,err)
                assert err<(1e-5 if k=='visual' else 1e-12),(job['id'],k,err)
        assert len(out['rows'])==8 and raw['horizon']==200 and a['passed']
        for r in out['rows']:
            assert r['updates']==(197 if r['mode']=='continuous' and r['preparation']['passed'] else 0)
        for r in out['rows']:
            mm=out['metrics'][r['source_row']] if r['source_row'] is not None else []
            stages.append(dict(**{k:r[k] for k in ('task','split','kind','representation','seed','mode','condition')},
                ever_bilateral=any(m['bilateral_handle'] for m in mm),
                max_lift=max((m['lift'] for m in mm),default=0.),
                max_hold_streak=max((m['hold_streak'] for m in mm),default=0.),
                ever_head_contact=any(m['head_target_contact'] for m in mm),
                ever_direct_target_contact=any(m['target_direct_contact'] for m in mm),
                max_target_displacement=max((m['target_displacement'] for m in mm),default=0.)))
        rows.extend(out['rows']);audits.append(a);physical+=a['physical_states']
        refs.append(dict(path=str(p/'OUTCOMES.json'),sha256=digest(p/'OUTCOMES.json')))
        del raw
    assert len(rows)==928
    lookup={(r['task'],r['split'],r['kind'],r['representation'],r['seed'],r['mode'],r['condition']):r for r in rows}
    cells=[];comparisons=[]
    groups=defaultdict(list)
    for r in rows:groups[r['task'],r['split'],r['kind'],r['representation'],r['mode']].append(r)
    for (task,split,kind,rep,mode),rr in groups.items():
        per_seed=[]
        for seed in ([None] if kind=='base' else [11,22]):
            sr=[r for r in rr if r['seed']==seed]
            per_seed.append(dict(seed=seed,successes=sum(r['success'] for r in sr),n=len(sr),
                                 terminal_successes=sum(r['terminal_success'] for r in sr)))
        cells.append(dict(task=task,split=split,kind=kind,representation=rep,mode=mode,per_seed=per_seed,
                          rate=float(np.mean([r['success'] for r in rr])),terminal_rate=float(np.mean([r['terminal_success'] for r in rr])),
                          independent_conditions=8,episodes=len(rr),preparation_failures=sum(not r['preparation']['passed'] for r in rr)))
        if kind=='base' or mode!='continuous':continue
        for control in ('fixed','base'):
            pairs=[];clusters=defaultdict(list)
            for r in rr:
                args=(kind,rep,r['seed'],'fixed') if control=='fixed' else ('base','base',None,'fixed')
                b=lookup[(task,split)+args+(r['condition'],)]
                delta=int(r['success'])-int(b['success']);clusters[r['condition']].append(delta)
                pairs.append(dict(condition=r['condition'],seed=r['seed'],candidate=r['success'],control=b['success'],delta=delta))
            ds=[np.mean(v) for _,v in sorted(clusters.items())]
            comparisons.append(dict(task=task,split=split,kind=kind,representation=rep,control=control,
                delta=float(np.mean(ds)),cluster_bootstrap_95_descriptive=interval(ds),
                wins=sum(p['delta']>0 for p in pairs),losses=sum(p['delta']<0 for p in pairs),ties=sum(p['delta']==0 for p in pairs),
                seed_deltas={str(s):float(np.mean([p['delta'] for p in pairs if p['seed']==s])) for s in (11,22)},pairs=pairs))
    summary=dict(cohort='P37',phase='bounded development',episodes=len(rows),independent_conditions=32,
                 cells=cells,comparisons=comparisons,rows=rows,stage_diagnostics=stages,source_bindings=refs,
                 statistical_limit='8 condition clusters per task/setting; two fixed model seeds; exploratory multiple comparisons; descriptive unadjusted intervals, no winner selection or promotion')
    out=ROOT/'summary';dump(out/'SUMMARY.json',summary)
    dump(out/'AUDIT.json',dict(passed=True,completed_batches=len(audits),episodes=len(rows),physical_states=physical,
         initial_pairs_max_error=pair_max,all_trajectory_hashes_verified=True,
         all_weights_unchanged=all(a['weights_frozen'] for a in audits),all_choices_replayed=True,all_runtime_numeric_inputs_and_outputs_finite=True,
         action_max_error=max(a['action_max_error'] for a in audits),z_max_error=max(a['z_max_error'] for a in audits),
         score_max_error=max(a['score_max_error'] for a in audits),full_RGB_reencoding=False))
    with (out/'SUCCESS_RATES.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['task','split','representation','mode','seed','successes','n','terminal_successes']);w.writeheader()
        for c in cells:
            for s in c['per_seed']:w.writerow({**{k:c[k] for k in ('task','split','representation','mode')},**s})
    lines=['# P37：完整任务成功率比较','',
        '冻结原视觉基座与P35／P36模型；两个已训练种子，逐执行步反馈。每任务ID／几何OOD各8个新物体条件，共928回合。',
        '连续更新从第4步反馈开始，每回合最多197次z更新，所有网络权重不变。固定z使用同一模型和同一动作选择器。',
        '以下百分比为两个固定模型种子的平均；分母为8个独立条件，不能把两个种子当成16个独立物体。括号列出种子11／22各自成功数。','']
    def cell(task,split,kind,rep,mode):return next(c for c in cells if (c['task'],c['split'],c['kind'],c['representation'],c['mode'])==(task,split,kind,rep,mode))
    for task in ('mug-handle-lift','hammer-block'):
        for split in ('train','geometry_ood'):
            b=cell(task,split,'base','base','fixed');lines += [f'## {task} / {split}','',f"原基座：{b['per_seed'][0]['successes']}/8（{b['rate']:.1%}）。",'',
              '| 表示 | 固定z | 持续更新z | 更新相对固定 | 持续更新相对基座 |','|---|---:|---:|---:|---:|']
            for kind,reps in (('p35',('g224','g448','g224w','g448w')),('p36',('no_explicit_bbox_geometry','p34_existing_bbox_geometry','p36_explicit_relation_geometry'))):
                f=cell(task,split,kind,reps[0],'fixed')
                for rep in reps:
                    f=cell(task,split,kind,rep,'fixed');c=cell(task,split,kind,rep,'continuous')
                    fmt=lambda x:f"{x['rate']:.1%}（"+'/'.join(str(s['successes']) for s in x['per_seed'])+'，各/8）'
                    lines.append(f"| {LABELS[rep]} | {fmt(f)} | {fmt(c)} | {(c['rate']-f['rate'])*100:+.2f} pp | {(c['rate']-b['rate'])*100:+.2f} pp |")
            lines += ['']
    lines += ['## 证据边界','',
        '- 杯柄抓取需双侧物理接触、抬升并保持；锤子任务从已持工具开始，初始化失败保留。',
        '- 同类程序化几何与固定色标视觉定位，不代表完整官方benchmark、任意真实资产或真机。',
        '- 成功率来自完整200步执行；所有决策／latent与物理成功判据复算，初态配对检查通过。',
        '- 保存了全量数值轨迹与少量RGB，未对所有帧重算DINO特征。',
        '- 小样本多重探索比较，不据此挑选赢家或晋升。损失下降、少量救回和稳定超过基座须分开解释。','',
        '[完整数字和配对记录](SUMMARY.json) · [复核](AUDIT.json) · [逐种子CSV](SUCCESS_RATES.csv)']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print('P37_REPORT_COMPLETE',len(rows),physical,flush=True)


if __name__=='__main__':main()
