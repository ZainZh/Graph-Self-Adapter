"""Aggregate all predefined P39 arms without best-seed or best-method selection."""
import argparse
import json
import numpy as np
import torch
from tools.fast_adapter.p39_tokens import ROOT,js,sha
from tools.fast_adapter.p39_evaluate import SPLITS

LABELS={'base':'冻结基座','oracle':'真实参数参考','global':'整图DINO','local':'局部空间DINO','relation':'显式相对关系','fused':'局部DINO＋关系','history':'History'}

def main(phase):
    out=ROOT/'summary';out.mkdir(exist_ok=True)
    splits=SPLITS+(('paired',) if phase=='final' else ())
    table={};arrays={};curves={};bindings=[]
    for split in splits:
        for kind in ('references','global','local','relation','fused','history'):
            for arm in (('base','oracle') if kind=='references' else ('continuous','first3','mean','donor')):
                rows=[];errors=[];details=[];cs=[]
                for seed in ([0] if kind=='references' else [11,22]):
                    folder=ROOT/phase/kind/f'seed{seed}'/split
                    f=folder/(arm+'_RESULT.json');r=json.loads(f.read_text());raw=torch.load(folder/(arm+'_RUNTIME.pt'),weights_only=True,map_location='cpu')
                    row=np.array([x['success'] for x in r['rows']],float);err=np.array([x['error'] for x in r['rows']])
                    rows.append(row);errors.append(err);details.append(dict(model_seed=seed,successes=int(row.sum()),n=len(row),rate=float(row.mean()),error_mm=float(err.mean()*1000)))
                    pos=raw['positions'];cs.append(torch.linalg.vector_norm(pos[:,:,2]-pos[:,:,1],dim=-1).numpy().mean(0)*1000)
                    bindings.append(dict(path=str(f),sha256=sha(f)))
                key=f'{split}/{kind}/{arm}';arr=np.stack(rows).mean(0);arrays[key]=arr;curves[key]=np.stack(cs).mean(0).tolist()
                table[key]=dict(rate=float(arr.mean()),error_mm=float(np.stack(errors).mean()*1000),unique_conditions=len(arr),models=details,visibility_failures=sum(not x['observation_valid'] for seed in ([0] if kind=='references' else [11,22]) for x in json.loads((ROOT/phase/kind/f'seed{seed}'/split/(arm+'_RESULT.json')).read_text())['rows']))
    comparisons={};rng=np.random.default_rng(39071)
    for split in splits:
        pairs=[]
        for k in ('global','local','relation','fused'):
            for ref in ('base','first3','mean','donor'):
                right='references/base' if ref=='base' else k+'/'+ref
                pairs.append((k+'/continuous',right))
        pairs += [('local/continuous','global/continuous'),('relation/continuous','local/continuous'),('fused/continuous','relation/continuous'),('fused/continuous','history/continuous')]
        n=len(arrays[split+'/references/base'])//(2 if split=='paired' else 1);indices=rng.integers(0,n,(50000,n))
        for left,right in pairs:
            d=arrays[split+'/'+left]-arrays[split+'/'+right]
            if split=='paired':d=d.reshape(-1,2).mean(-1)
            boot=d[indices].mean(-1)*100
            comparisons[f'{split}/{left} vs {right}']=dict(delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),family_corrected_ci_pp=np.quantile(boot,[.025/len(pairs),1-.025/len(pairs)]).tolist(),family_size=len(pairs),cluster_unit='matched pair' if split=='paired' else 'condition',positive_conditions=int((d>0).sum()),negative_conditions=int((d<0).sum()),ties=int((d==0).sum()))
    result=dict(phase=phase,table=table,comparisons=comparisons,mask_error_curves_mm=curves,source_bindings=bindings,
       inference='Bootstrap50k over unique conditions after averaging two fixed model seeds. Bonferroni20 comparisons per stratum; five final strata reported separately; paired stratum bootstraps32 matched pairs. These intervals do not represent training-seed population uncertainty; all-identical outcomes may yield degenerate bootstrap intervals.',
       limitation='All controllers receive simulator-mask-derived geometry. Explicit geometry changes adaptation target; visual/local comparisons retain equal PCA size and effect-head architecture. Not a pure end-to-end learned token or robot test.')
    js(out/(phase+'_RESULTS.json'),result)
    lines=[f'# P39 {phase} 完整结果','', '每组列出唯一环境条件数；两个固定训练模型在同一条件内平均，不能把它们当成双倍独立物体。','', '| 设置 | 基座 | 整图DINO | 局部DINO | 相对关系 | 局部＋关系 | History | 参数参考 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in splits:
        keys=['references/base']+[k+'/continuous' for k in ('global','local','relation','fused','history')]+['references/oracle']
        lines.append('| '+s+' | '+' | '.join(f'{table[s+"/"+k]["rate"]*100:.2f}%' for k in keys)+' |')
    lines+=['','train=范围内新条件；resistance=较高隐藏阻力；geometry=更大圆柱与更斜目标方向；joint=几何与阻力同时变化；paired=32对同外观但低/高阻力物块。','', '## 因果与表示对照','', '| 设置／方法 | 持续更新 | 三次后固定 | 训练平均z | 错配反馈 |','|---|---:|---:|---:|---:|']
    for s in splits:
        for k in ('global','local','relation','fused','history'):
            lines.append('| '+s+' / '+LABELS[k]+' | '+' | '.join(f'{table[f"{s}/{k}/{a}"]["rate"]*100:.2f}%' for a in ('continuous','first3','mean','donor'))+' |')
    lines+=['','全部配对区间、逐模型结果、可见误差曲线及来源哈希见同目录机器结果。','', '## 实现边界','',
      '- 这是两轴平移、真实接触的定制MuJoCo任务；圆柱不自由旋转，没有机械臂。',
      '- 实例mask由模拟器提供，相机标定和顶面高度已知；普通适配器不读物体真实位姿、阻力、质量或成功标签。',
      '- DINOv3与LiLa冻结。整图、局部特征各用训练PCA压到16维；局部包含mask加权外观与空间特征矩。',
      '- 关系token描述actor—object和object—goal的相对二维位置。关系损失与视觉损失同时改变了反馈目标，不能把其差距都称为注意力架构贡献。',
      '- z是四维有效响应参数修正，响应形式和控制器已知；只有梯度预条件器通过外层元学习。History是32维GRU对照。',
      '- 视觉后果头用训练mask测量运动监督；外层使用训练环境物理参数生成目标推力。测试期无这些监督标签。',
      '- 同一40步执行预算包含3次共同探测；每步完成后用最近4次交互更新。视频只作封存后的示例。',
      '- 没有证实的差异保留为不确定；负结果与成功案例一起保留。']
    (out/(phase+'_REPORT_ZH.md')).write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='final');a=p.parse_args();main(a.phase)
