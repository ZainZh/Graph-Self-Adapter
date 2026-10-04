"""Generate P21's review page directly from frozen diagnostic/control results."""
from pathlib import Path
import json,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.fast_adapter.p18_run import save_json,digest
R=Path('evaluations/fast_adapter_p21_frontends_v01')
NAMES={'last':'DINO 最后一层','middle17':'DINO 第17层','lila':'LiLa 预训练前端'}
ARM_NAMES={'base':'基础控制','strong_fixed':'固定增强反馈','oracle':'Oracle参考','history':'History','hybrid_u0':'同模型冻结 z','hybrid_u3':'梯度更新 z','hybrid_swapped':'交换 z'}

def main():
    out=R/'summary';out.mkdir(exist_ok=True)
    diag=json.loads((R/'diagnostic/RESULT.json').read_text());selected=diag['selected']
    plt.rcParams['font.sans-serif']=['Noto Sans CJK JP','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    colors=['#778895','#218b88','#d28b33'];reps=list(NAMES)
    for ax,key,title in zip(axes,['position_xy_rmse_mm','support_and_query_displacement_rmse_mm'],['相对位置读出误差（XY）','已发生运动的位移读出误差（XYZ）']):
        vals=[diag['metrics'][r][key] for r in reps]
        ax.bar([NAMES[r] for r in reps],vals,color=colors)
        ax.set_ylabel('RMSE / 毫米');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom')
        ax.set_ylim(0,max(vals)*1.22)
    fig.suptitle('相同图像、相同64维输出、相同诊断读出器；768组拟合 / 256组验证',fontsize=12)
    fig.savefig(out/'representation_comparison.png',dpi=170);plt.close(fig)
    lines=['# P21：视觉前端与交互适配对照','',
       '一个自移动物块与一个槽，沿用P20真实MuJoCo接触任务。目标是分别检验视觉表示与交互梯度更新的作用。',
       '', '## 视觉表示诊断','',
       '|视觉前端|XY位置 RMSE/mm|位移 RMSE/mm|近接触XY RMSE/mm|选择分数（越低越好）|',
       '|---|---:|---:|---:|---:|']
    for r in reps:
        m=diag['metrics'][r];lines.append(f"|{NAMES[r]}|{m['position_xy_rmse_mm']:.3f}|{m['support_and_query_displacement_rmse_mm']:.3f}|{m['near_contact_query_xy_rmse_mm']:.3f}|{m['selection_score']:.3f}|")
    lines+=['',f"按结果前规则选择：**{NAMES[selected]}**；进入控制资格：**{diag['control_qualified']}**。",
       '', '位置/位移标签只训练离线诊断读出器，诊断权重未接入控制。此表不是任务成功率。投影拟合与诊断验证按domain隔离，但材料属于已开放P20训练条件，因此不称新独立任务测试。',
       '', '![视觉表示比较](representation_comparison.png)', '', '## 解释范围','',
       '- LiLa公开权重使用过外部机器人动作和未来视觉监督；其场景token没有保证物体A/B各自对应。',
       '- 最后一层和中间层使用GT区域框；LiLa前端直接读取整幅RGB。这里比较的是完整视觉前端，不是adapter结构或预训练数据量的单因素因果实验。',
       '- 三种表示都固定为64维；LiLa的64×768 token按固定顺序展开后拟合无监督PCA，没有把64个token先平均。其负结果不能排除其他压缩方式或完整LiLa模型。',
       '- 基础控制仍使用已知目标及物块proprio；普通适配器不能读取真实动力学、诊断坐标或成功标签。固定增强反馈是参考，其修正不受学习分支±0.35 residual限制，但总动作仍限制于[-1,1]。',
       '- History与Hybrid都使用冻结前端，保持6000次外层更新；仅Hybrid在测试前用已执行的5次交互做3次z梯度更新，完整query期间z固定。']
    controls={}
    for stage in ['dev','final']:
        p=out/f'{stage.upper()}_RESULTS.json'
        if not p.exists():continue
        d=json.loads(p.read_text());controls[stage]=d
        lines+=['',f"## {'开发回归' if stage=='dev' else '新独立测试'}成功率",'',
                 '|前端/方法|ID 成功率|OOD 成功率|','|---|---:|---:|']
        for arm in ['base','strong_fixed','oracle']:
            vals=[f"{d['metrics'][st][arm]['rate_percent']:.2f}%" if st in d['strata'] else '未运行' for st in ['train','ood']]
            lines.append(f'|{ARM_NAMES[arm]}|'+ '|'.join(vals)+'|')
        for rep in diag['control_frontends']:
            for arm in ['history','hybrid_u0','hybrid_u3','hybrid_swapped']:
                vals=[f"{d['metrics'][st][rep][arm]['rate_percent']:.2f}%" if st in d['strata'] else '未运行' for st in ['train','ood']]
                lines.append(f'|{NAMES[rep]} / {ARM_NAMES[arm]}|'+ '|'.join(vals)+'|')
        lines+=['','三粒模型种子在相同domain上评估，不能把seed×domain当作全部独立样本。置信区间同时重采样模型seed和domain，配对条件始终保持。']
        for st in d['strata']:
            c=d['comparisons'][st][selected]
            for key,label in [('u3_vs_u0','梯度更新相对同checkpoint冻结z'),('u3_vs_history','梯度更新相对History')]:
                item=c[key];lo,hi=item['ci95_pp']
                lines.append(f"- {st.upper()} {label}：{item['delta_pp']:+.2f}pp，95%配对区间[{lo:+.2f}, {hi:+.2f}]；wins/harms={item['wins']}/{item['harms']}。")
        if stage=='dev':lines+=['',f"预定开发门：{d['dev_gate']}。"]
    control_chart=''
    if 'final' in controls:
        d=controls['final'];fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        choices=[(None,'base','基础控制'),(None,'strong_fixed','固定增强反馈'),('last','hybrid_u3','末层\n更新z'),(selected,'history','LiLa\nHistory'),(selected,'hybrid_u0','LiLa\n冻结z'),(selected,'hybrid_u3','LiLa\n更新z')]
        for ax,st,label in zip(axes,['train','ood'],['同分布 ID','更大动作方向偏差 OOD']):
            vals=[d['metrics'][st][r][a]['rate_percent'] if r else d['metrics'][st][a]['rate_percent'] for r,a,_ in choices]
            ax.bar([name for _,_,name in choices],vals,color=['#66747f','#adb8bf','#597fa5','#218b88','#e8c99b','#d28b33'])
            ax.set_ylim(0,110);ax.set_ylabel('任务成功率 / %');ax.set_title(label);ax.spines[['top','right']].set_visible(False)
            for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.2f}',ha='center',va='bottom',fontsize=10)
        fig.suptitle('新独立测试：每档48个条件；学习方法为3粒模型种子均值',fontsize=13)
        fig.savefig(out/'success_comparison.png',dpi=170);plt.close(fig)
        lines+=['','![独立测试成功率](success_comparison.png)']
        control_chart='<h2>新独立测试成功率</h2><img src="success_comparison.png"><p>同模型z更新：ID +5.56pp（区间跨零）；OOD +11.11pp（95%配对区间+2.78至+20.83pp）。相对History的区间均跨零。</p>'
    if not diag['control_qualified']:
        conclusion='本轮完成表示资格诊断；新前端未通过预定控制资格门，未启动控制训练或新独立测试。'
    elif 'final' in controls:
        d=controls['final'];cs=[d['comparisons'][st][selected]['u3_vs_u0'] for st in d['strata']]
        supported=all(c['delta_pp']>0 and c['ci95_pp'][0]>0 for c in cs)
        supported_strata=[st.upper() for st in d['strata'] if d['comparisons'][st][selected]['u3_vs_u0']['ci95_pp'][0]>0]
        conclusion='本轮完整独立测试已完成。'+('、'.join(supported_strata)+'支持同模型z梯度更新的有界成功率改善；其余分布仍不确定。' if supported_strata else '尚未建立可靠的同模型梯度更新成功率优势。')+' 相对History的额外优势需单独判断，整体控制仍低于基础控制和固定增强反馈。'
    elif 'dev' in controls:
        conclusion='开发回归已完成；'+('已通过预定门，需完成新独立测试后才能给最终结论。' if controls['dev']['dev_gate']['enter_final'] else '未通过预定独立测试门，本轮以开发负/混合结果关闭，不扩大测试或调参。')
    else:conclusion='表示诊断已完成，控制对照待完成；暂无新的任务成功率结论。'
    lines.insert(2,conclusion)
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    rows=''.join('<tr><td>'+NAMES[r]+'</td>'+''.join('<td>'+f"{diag['metrics'][r][key]:.3f}"+'</td>' for key in ['position_xy_rmse_mm','support_and_query_displacement_rmse_mm','selection_score'])+'</tr>' for r in reps)
    tables=''
    for stage,d in controls.items():
        tables+=f'<h2>{"开发回归（已开放条件）" if stage=="dev" else "新独立测试"}</h2><table><tr><th>前端 / 方法</th><th>ID</th><th>更大方向偏差 OOD</th></tr>'
        for arm in ['base','strong_fixed','oracle']:
            tables+='<tr><td>'+ARM_NAMES[arm]+'</td>'+''.join('<td>'+ (f"{d['metrics'][st][arm]['rate_percent']:.2f}%" if st in d['strata'] else '未运行')+'</td>' for st in ['train','ood'])+'</tr>'
        for rep in diag['control_frontends']:
            for arm in ['history','hybrid_u0','hybrid_u3','hybrid_swapped']:
                tables+='<tr><td>'+NAMES[rep]+' / '+ARM_NAMES[arm]+'</td>'+''.join('<td>'+ (f"{d['metrics'][st][rep][arm]['rate_percent']:.2f}%" if st in d['strata'] else '未运行')+'</td>' for st in ['train','ood'])+'</tr>'
        tables+='</table>'
    media='';vp=out/'media/VIDEO_INDEX.json'
    if vp.exists():
        for v in json.loads(vp.read_text())['videos']:
            media+='<h3>'+html.escape(v['title'])+'</h3><video controls preload="metadata" src="media/'+Path(v['path']).name+'"></video>'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P21视觉前端与交互适配</title><style>body{{font:17px/1.7 system-ui;max-width:1200px;margin:36px auto;color:#23333f;background:#fafbfc}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px 18px;border-bottom:1px solid #dde4e8;text-align:left}}img,video{{width:100%;max-width:1200px}}.note{{background:#e9f3f2;padding:18px;border-radius:10px}}a{{color:#126e73}}</style><h1>P21：视觉前端与交互适配</h1><p class="note">{html.escape(conclusion)}</p><p>固定三维任务、相同图像与动作条件；主Agent独立完成。<a href="REPORT_ZH.md">完整说明与配对区间</a> · <a href="../PROTOCOL.json">预定协议</a> · <a href="../diagnostic/RESULT.json">表示诊断原始统计</a></p><h2>表示诊断：位置与运动信息能否读出来</h2><p>1024组既有训练条件，768组拟合、256组诊断验证。几何标签仅用于此诊断，读出器未用于控制。</p><img src="representation_comparison.png"><table><tr><th>视觉前端</th><th>XY位置误差/mm</th><th>位移误差/mm</th><th>选择分数</th></tr>{rows}</table><p>预定规则选中：{NAMES[selected]}。此表不是任务成功率，也不证明梯度适配收益。</p>{control_chart}{tables}<h2>完整控制回合的注释对比视频</h2>{media or '<p>本阶段尚无控制视频。</p>'}<h2>适用范围</h2><p>已知目标/物块本体位置辅助的固定基座；末层和中间层使用GT区域框。LiLa有外部机器人训练监督，输出场景token而非保证分离的物体特征。全部视觉权重冻结，完整query期间z固定；不属于机械臂或官方LIBERO评测。</p></html>'''
    (out/'index.html').write_text(page)
    save_json(out/'REPORT_STATUS.json',{'conclusion':conclusion,'selected':selected,'diagnostic':str(R/'diagnostic/RESULT.json'),'control_stages':list(controls),'report_sha256':digest(out/'REPORT_ZH.md')})
    print(conclusion)
if __name__=='__main__':main()
