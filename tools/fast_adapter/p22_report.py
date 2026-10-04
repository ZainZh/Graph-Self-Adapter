"""Report P22 exactly as scoped; no selection of seeds, strata or checkpoints."""
from pathlib import Path
import json,hashlib
import numpy as np
from tools.fast_adapter.p22_data import R
from tools.fast_adapter.p18_run import save_json,digest

def paired(a,b):
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    if d.ndim==1:d=d[None,:]
    rng=np.random.default_rng(220223);boot=[]
    for _ in range(5000):
        seeds=rng.integers(d.shape[0],size=d.shape[0]);domains=rng.integers(d.shape[1],size=d.shape[1])
        boot.append(d[seeds][:,domains].mean()*100)
    return {'delta_pp':float(d.mean()*100),'ci95_pp':np.quantile(boot,[.025,.975]).tolist(),
            'simultaneous_two_endpoint_ci_pp':np.quantile(boot,[.0125,.9875]).tolist(),
            'wins':int((d>0).sum()),'harms':int((d<0).sum()),'seed_deltas_pp':(d.mean(1)*100).tolist()}

def main():
    done=json.loads((R/'CAMPAIGN_DONE.json').read_text());stage='final' if done['final_run'] else 'dev'
    out=R/'summary';out.mkdir(exist_ok=True)
    names={'id':'同分布','position':'槽位置变化','scale':'整体尺度变化'};results={};bindings=[]
    for st in names:
        ref=R/stage/st/'seed0/RESULT.json';refs=json.loads(ref.read_text())['metrics']
        bindings.append({'path':str(ref),'sha256':digest(ref)})
        arrays={k:np.array(refs[k]['success'],dtype=bool) for k in ('base','oracle')}
        if done['final_run']:
            entries=[]
            for seed in (11,22,33):
                path=R/stage/st/f'seed{seed}/RESULT.json';entries.append(json.loads(path.read_text()))
                bindings.append({'path':str(path),'sha256':digest(path)})
            for arm in ('history','u0','u3','swapped'):
                arrays[arm]=np.array([e['metrics'][arm]['success'] for e in entries],dtype=bool)
        result={'rates_percent':{k:float(v.mean()*100) for k,v in arrays.items()},
                'independent_domains':len(arrays['base']),
                'model_seed_rates_percent':{k:(v.mean(1)*100).tolist() for k,v in arrays.items() if v.ndim==2}}
        if done['final_run']:
            result['paired']={f'u3_minus_{k}':paired(arrays['u3'],arrays[k]) for k in ('u0','history','base','swapped')}
        results[st]=result
    save_json(out/'RESULTS.json',{'campaign_status':done['status'],'stage':stage,'strata':results,'source_bindings':bindings,
      'uncertainty':'Paired bootstrap of domain and adapter-seed axes. Shared base seed fixed. Two OOD endpoint adjustment reported.',
      'root_acceptance_required':True})
    lines=['# P22：视觉＋本体感觉的插入验证','',
      '执行模型输入仅为全局RGB、受控物块XYZ和已完成动作历史。槽坐标、尺寸、真值框、隐藏动力学及成功标签不作为模型输入。训练专家使用几何/动力学生成动作监督。', '',
      '训练包含384个独立条件，每个条件5次探测和48步完整示范，共21,120张图像；视觉基座训练12,000次。物块位置模拟刚性抓持下的机械臂本体感觉。', '',
      '| 测试条件 | 视觉基座 | History | 同模型U0 | 同模型U3 | 交换z | 真值参考 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for st,r in results.items():
        cells=[f'{r["rates_percent"][a]:.2f}%' if a in r['rates_percent'] else '未运行' for a in ('base','history','u0','u3','swapped','oracle')]
        lines.append('| '+names[st]+' | '+' | '.join(cells)+' |')
    lines+=['']
    if done['final_run']:
        lines+=['每档32个独立环境条件；学习方法为3个模型种子的平均，不能把96次相关执行当作96个独立环境。','']
        for st,r in results.items():
            q=r['paired']['u3_minus_u0'];lo,hi=q['ci95_pp'];alo,ahi=q['simultaneous_two_endpoint_ci_pp']
            lines.append(f'- {names[st]}：U3−U0 **{q["delta_pp"]:+.2f}个百分点**；95%配对区间[{lo:+.2f}, {hi:+.2f}]；两项OOD比较校正所用区间[{alo:+.2f}, {ahi:+.2f}]。')
        lines+=['','所有模型均执行，未按开发收益选择checkpoint；最终数据已打开后不得继续调参并仍称其为独立测试。']
    else:
        lines+=['**视觉基座资格未通过，本批按预定规则关闭，未训练adapter或运行最终测试。** 上表仅为每档16条件的开发资格结果，不能用于主张TTT有效或无效。']
    lines+=['','边界：同一个定向插入任务、固定相机和物体外观；整体缩放保留间隙比例，未测试新形状或新间隙。LiLa使用外部机器人监督。支持阶段与执行阶段使用不同初态；z在48步执行中固定。不是机械臂或完整RSI结果。','',
      '完整原始轨迹、模型和来源哈希由导出清单索引，正式解释以主Agent验收为准。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'stage':stage,'results':results},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

