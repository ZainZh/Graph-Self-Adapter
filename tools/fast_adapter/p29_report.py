import argparse,json
from pathlib import Path
import numpy as np
from tools.fast_adapter.p29_common import *
from tools.fast_adapter.p29_eval import STRATA
from tools.fast_adapter.p27_statistics import paired_clusters
ARMS=('base','fast_correct','fast_mean','fast_donor','history','robot')
LABELS={'base':'冻结视觉动作基座／K0','fast_correct':'本物体反馈梯度适配','fast_mean':'固定训练平均z','fast_donor':'另一物体反馈z','history':'同反馈History','robot':'仅末端反馈适配'}
TASK_NAMES={'mug-handle-lift':'抓住杯子把手并抬升','hammer-block':'持锤以锤头作用木块'}
STRATUM_NAMES={'canonical':'标准训练形状的新初态','train':'范围内未见过的新物体','geometry_ood':'范围外的新物体几何'}
def arrays(result,key='success'):
    by={}
    for r in sorted(result['rows'],key=lambda x:x['offset']):by.setdefault(r['arm'],[]).extend(r[key])
    a={'base':np.array(by['base'],float)}
    for name in ARMS[1:]:a[name]=np.array([by[f'{name}_seed{s}'] for s in SEEDS],float)
    assert all(v.shape[-1]==result['conditions'] for v in a.values());return a

def report(phase='development',round_name='r1'):
    out=R/'summary'/phase/round_name;out.mkdir(parents=True,exist_ok=True);records={};sources=[]
    lines=[f'# P29：同类物体抓取与工具交互（{phase}/{round_name}）','','机械臂使用正常原生动作映射，无方向旋转或幅度缩放。物体是基于Meta-World Sawyer构建的参数化接触任务，不是完整官方benchmark。',
      '杯子保留actor局部特征，第二object全零并mask；锤子与目标木块为两路实体。区域来自已知色标的RGB分割，DINOv3+LiLa固定；不声称任意颜色/纹理识别。',
      '每任务基座：96条标准物体示范＋64条本策略补采，12000＋6000次更新。适配训练：64个不同物体，160步共同基座支持，两个200步不同初态query，每种方法每种子14000次外层更新。推理仅更新32维z三次，query内固定。','']
    for ti,task in enumerate(TASKS):
        root=task_root(task);gate=root/'qualification/BASE.json'
        if not gate.exists():records[task]={'status':'not_ready'};continue
        g=json.loads(gate.read_text());lines += [f'## {TASK_NAMES[task]}','',f"标准形状开发资格：{g['successes']}/{g['n']}。",'']
        rows={}
        for si,st in enumerate(STRATA):
            p=root/phase/round_name/st/'RESULT.json'
            if not p.exists():continue
            d=json.loads(p.read_text());a=arrays(d);terminal=arrays(d,'terminal_success');comp={}
            for j,c in enumerate(ARMS):
                if c=='fast_correct':continue
                primary=phase=='final' and st=='geometry_ood' and c in ('base','fast_mean','fast_donor')
                comp[c]={**paired_clusters(a['fast_correct'],a[c],seed=29029+ti*100+si*10+j,family=6 if primary else 1),'primary':primary}
            rows[st]={'rates_percent':{k:float(v.mean()*100) for k,v in a.items()},'per_seed_rates_percent':{k:(v.mean(1)*100).tolist() for k,v in a.items() if v.ndim==2},'terminal_rates_percent':{k:float(v.mean()*100) for k,v in terminal.items()},'comparisons':comp,'conditions':d['conditions']}
            sources.append({'path':str(p),'sha256':digest(p)})
        records[task]={'base_gate':g,'status':'complete' if len(rows)==3 else 'partial_or_base_unready','strata':rows}
        lines += ['| 方法 | 标准形状 | 范围内新物体 | 范围外新几何 |','|---|---:|---:|---:|']
        for arm in ARMS:lines += ['|'+LABELS[arm]+'|'+'|'.join(f"{rows[s]['rates_percent'][arm]:.2f}%" if s in rows else '尚无结果' for s in STRATA)+'|']
        lines += ['']
        if 'geometry_ood' in rows:
            for c in ('base','fast_mean','fast_donor','history','robot'):
                v=rows['geometry_ood']['comparisons'][c];ci=v['multiplicity_adjusted_ci']
                lines += [f"- 正确反馈相对{LABELS[c]}：{v['delta_pp']:+.2f}个百分点；{'主要六比较校正' if v['primary'] else '探索性95%'}区间[{ci[0]:.2f},{ci[1]:.2f}]。"]
        lines += ['']
    lines += ['开发结果用于模型诊断/选择，不是独立确认。最终结论以封存候选的新物体与初态测试为准；两个种子在每个物体条件内平均，不将其当成独立样本翻倍。支持采集160步与query200步的成本分别保留。','',
      '抓取成功要求真实把手双侧接触并稳定抬升。工具任务要求锤头与目标接触、目标沿滑轨位移且没有机械臂直接触碰；这是水平工具接触/推击，不能宣称已验证真实冲击锤击或驱钉。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');save_json(out/'RESULTS.json',{'phase':phase,'round':round_name,'tasks':records,'sources':sources,'primary_family':6,'fixed_model_seeds':list(SEEDS)})
    print(json.dumps({t:{s:r['rates_percent'] for s,r in d.get('strata',{}).items()} for t,d in records.items()},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='development',choices=['development','final']);p.add_argument('--round',default='r1');a=p.parse_args();report(a.phase,a.round)
