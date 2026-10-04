"""One owner for P38 physical per-setting statistics; no model or rollout calls."""
import argparse
import json
from pathlib import Path
import numpy as np

R=Path('evaluations/fast_adapter_p38_causal_v01')


def summarize(phase):
    tables={};raw_groups={}
    for family in ('push','push_contrast'):
        for p in sorted((R/family/phase).rglob('*_RESULT.json')):
            j=json.loads(p.read_text());key=(j['split'],j['kind'],j['arm'])
            raw_groups.setdefault(key,[]).append(j)
    for key,rows in raw_groups.items():
        vals=np.array([[r['success'] for r in row['rows']] for row in rows],dtype=float)
        errs=np.array([[r['error'] for r in row['rows']] for row in rows])
        tables['/'.join(key)]={'success_rate':float(vals.mean()),'mean_error_m':float(errs.mean()),
             'per_model_successes':[int(x.sum()) for x in vals],'n_conditions':vals.shape[1],
             'n_independent_pairs':vals.shape[1]//2 if key[0]=='contrast' else None,'model_seeds':[r['seed'] for r in rows]}
    rng=np.random.default_rng(380526);comparisons=[]
    def compare(split,ka,kb,arm_a='continuous',arm_b='continuous',family_size=12):
        a=np.array([[r['success'] for r in j['rows']] for j in raw_groups[(split,ka,arm_a)]],float).mean(0)
        b=np.array([[r['success'] for r in j['rows']] for j in raw_groups[(split,kb,arm_b)]],float).mean(0)
        delta=a-b
        if split=='contrast':delta=delta.reshape(-1,2).mean(1)
        bs=delta[rng.integers(0,len(delta),size=(20000,len(delta)))].mean(-1)
        comparisons.append({'split':split,'a':ka+'/'+arm_a,'b':kb+'/'+arm_b,
           'delta':float(delta.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),
           'ci_bonferroni12':np.quantile(bs,[.025/family_size,1-.025/family_size]).tolist(),
           'unit':'geometry pair' if split=='contrast' else 'configuration','scope':'fixed trained models'})
    if ('contrast','motion','continuous') in raw_groups:
        for kind in ('motion','appearance','combined'):
            compare('contrast',kind,'references',arm_b='base')
            compare('contrast',kind,kind,arm_b='donor')
            compare('contrast',kind,kind,arm_b='train_mean')
        for ka,kb in (('motion','appearance'),('combined','appearance'),('combined','motion')):
            compare('contrast',ka,kb)
        for split in ('native','id','ood'):
            for kind in ('motion','appearance','combined','history'):
                if (split,kind,'continuous') in raw_groups:compare(split,kind,kind,arm_b='frozen')
    out=R/'summary';out.mkdir(exist_ok=True,parents=True)
    (out/(phase+'_PHYSICAL_RESULTS.json')).write_text(json.dumps({'phase':phase,'table':tables,'comparisons':comparisons},ensure_ascii=False,indent=2)+'\n')
    text=['# P38 近接触物体交互：'+phase,'','真实MuJoCo接触，隐藏的是物体摩擦与阻力。共同解析直接动作修正器，固定RGB颜色检测/相机标定；不是通用视觉机器人或官方benchmark。',
          '','|设置|方法|基础/冻结z|持续更新|前三次后固定|错配反馈|训练平均z|','|---|---|---:|---:|---:|---:|---:|']
    def cell(split,kind,arm):
        j=tables.get('/'.join((split,kind,arm)))
        return '—' if not j else f'{100*j["success_rate"]:.2f}%'
    for split in ('native','id','ood','contrast'):
        for kind in ('motion','appearance','combined','history'):
            base=cell(split,kind,'frozen') if split!='contrast' else cell(split,'references','base')
            text.append(f'|{split}|{kind}|{base}|'+ '|'.join(cell(split,kind,a) for a in ('continuous','first3','donor','train_mean'))+'|')
        text.append(f'|{split}|参考|Base {cell(split,"references","base")}|LS {cell(split,"references","least_squares")}|Oracle {cell(split,"references","oracle")}|—|—|')
    text+=['','对比组每对外观/几何相同，阻力相反；统计单位为几何对。模型种子是重复测量，不能把两个模型结果相加当作独立环境数。',
           '所有方法30次物理动作，前三次固定共同探测计入预算。持续版本30次反馈更新，最后一次不再影响本回合；前三次固定版本仅用3次反馈。',
           '原三档的错配沿全批环形交换，仅作描述；严格因果区间来自对比组内相互交换，按配对簇统计。原参考LS使用完整已完成前缀，对比组LS使用与梯度相同的4步窗口。',
           'DINO版本仍由共同RGB几何反馈控制目标，差异在适配损失的观测目标；不声称纯DINO端到端控制。']
    (out/(phase+'_PHYSICAL_REPORT_ZH.md')).write_text('\n'.join(text)+'\n')
    print('\n'.join(text[4:]));return tables


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='final');a=p.parse_args();summarize(a.phase)
