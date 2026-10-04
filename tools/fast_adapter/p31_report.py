"""Aggregate the declared complete factorial; pair over unique object conditions."""
import json
import numpy as np
from tools.fast_adapter.p31_common import ROOT, protocol, dump, bindings

LABELS={'base':'冻结基座 / 同模型z不更新','old_full':'旧材料＋全程适配','old_stage':'旧材料＋关键阶段适配',
        'own_full':'自身轨迹＋全程适配','own_stage':'自身轨迹＋关键阶段适配',
        'old_stage_first3':'旧材料＋阶段内仅更新3次','own_stage_first3':'自身轨迹＋阶段内仅更新3次'}


def main():
    cfg=protocol()['evaluation'];arms=cfg['arms'];rows=[]
    for seed in cfg['model_seeds']:
        result=json.loads((ROOT/'development'/f'seed{seed}'/'RESULT.json').read_text());rows+=result['rows']
        assert json.loads((ROOT/'development'/f'seed{seed}'/'AUDIT.json').read_text())['passed']
    assert len(rows)==cfg['total_episodes']
    out=ROOT/'summary';out.mkdir(exist_ok=True)
    summaries=[];contrasts=[];text=['# P31 杯子关键阶段适配与自身轨迹开发结果','',
        '两种训练种子，每种子16个新ID、16个新OOD条件；七臂配对，共448回合。两个种子共享物体条件，不能视作32个独立物体。',
        '冻结DINOv3+LiLa/PCA与动作基座；原预测目标不变。四组共用一致的训练教师，故旧材料组也不是旧P30 checkpoint。',
        '所有执行在同一本机4090完成；新增自身轨迹64条，所有失败保留。每组128条记录、2400次外层更新，但自身轨迹组付出了额外物理采集成本。','']
    for split in cfg['strata']:
        text += ['## '+('同分布新物体（ID）' if split=='train' else '范围外杯子几何（OOD）'),'','| 方法 | 种子11 | 种子22 | 两种子平均成功率 |','|---|---:|---:|---:|']
        values={}
        for arm in arms:
            rr=sorted([r for r in rows if r['split']==split and r['arm']==arm],key=lambda r:(r['seed'],r['condition']))
            a=np.array([r['success'] for r in rr],float).reshape(2,16);values[arm]=a
            counts=a.sum(1).astype(int).tolist()
            summaries.append({'split':split,'arm':arm,'counts':counts,'denominator_per_seed':16,'mean_rate':float(a.mean()),
                'ever_bilateral_counts':[sum(r['ever_bilateral'] for r in rr if r['seed']==s) for s in (11,22)],
                'mean_updates':float(np.mean([r['updates'] for r in rr]))})
            text.append(f'| {LABELS[arm]} | {counts[0]}/16 | {counts[1]}/16 | {100*a.mean():.2f}% |')
        comparisons=[('old_stage','old_full'),('own_stage','own_full'),('own_full','old_full'),
                     ('own_stage','old_stage'),('own_stage','base'),('own_stage','own_stage_first3'),('old_stage','old_stage_first3')]
        text += ['','配对差值（百分点；按16个物体条件聚类重采样，两种子共同保留；95%区间仅为开发描述）：','']
        for hi,lo in comparisons:
            d=values[hi]-values[lo];rng=np.random.default_rng(3131)
            boot=d.mean(0)[rng.integers(0,16,(20000,16))].mean(1)*100
            ci=np.quantile(boot,[.025,.975]).tolist()
            row={'split':split,'comparison':hi+' - '+lo,'delta_pp':float(d.mean()*100),'ci95_pp':ci,
                 'rescues':int((d>0).sum()),'harms':int((d<0).sum())};contrasts.append(row)
            text.append(f'- {LABELS[hi]} − {LABELS[lo]}：{row["delta_pp"]:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]；救回{row["rescues"]}、误伤{row["harms"]}（32个种子-条件配对）。')
        interaction=(values['own_stage']-values['own_full'])-(values['old_stage']-values['old_full'])
        contrasts.append({'split':split,'comparison':'factorial_interaction','delta_pp':float(interaction.mean()*100)})
        text.append('')
    text += ['## 解释边界','',
        '- 小样本开发对照，未经独立最终确认，也没有晋升。不能把恢复到基座水平写成自提升。',
        '- 当前只刷新一次旧Adapter访问过的训练状态，再离线元训练；新模型继续执行后仍可能离开这些状态，不能宣称训练与部署分布完全一致。',
        '- 阶段观察器用了训练几何标签；运行时仅使用视觉、自身状态和上一动作，成功/接触真值不进入控制。',
        '- 在定制MetaWorld物理任务上测试同类杯子的几何变化；不是官方benchmark、自然外观或真机泛化成绩。',
        '- 完整缓存输入的动作、z、门控、更新次数已回放，物理成功判据已重算；没有重编码全部视觉帧。','']
    dump(out/'SUMMARY.json',{'phase':'development','episodes':len(rows),'rows':summaries,'contrasts':contrasts,
         'bindings':bindings([ROOT/'development'/f'seed{s}'/f for s in (11,22) for f in ('RESULT.json','AUDIT.json')])})
    (out/'REPORT_ZH.md').write_text('\n'.join(text))
    print(json.dumps({'rows':summaries,'contrasts':contrasts},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
