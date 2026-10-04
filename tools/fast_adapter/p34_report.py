"""Report every construction and seed, including unqualified comparisons."""
import html
import numpy as np
from tools.fast_adapter.p34_common import *

NAMES={'flat':'直接拼接','role_attention':'三角色注意力','relation_mlp':'关系 MLP','relation_attention':'关系注意力'}

def main():
    out=ROOT/'summary';out.mkdir(exist_ok=True)
    forecasts=json.loads((ROOT/'validation/PREDICTION_RESULTS.json').read_text())
    branches=json.loads((ROOT/'branches/RESULTS.json').read_text())
    diagnostic=json.loads((ROOT/'branches/DECISION_DIAGNOSTIC.json').read_text())
    audit=json.loads((ROOT/'AUDIT.json').read_text());assert audit['passed']
    assert not any(g['pass'] for g in branches['gates'].values()),'A qualified task requires full rollouts before closing'
    results={};table=[];detail=[]
    for task in TASKS:
        results[task]={}
        for f in FAMILIES:
            results[task][f]={}
            for mode in ('fixed','first3','continuous'):
                pp=[r for r in forecasts['rows'] if r['task']==task and r['family']==f]
                bb=[r for r in branches['rows'] if r['task']==task and r['family']==f and r['mode']==mode]
                z={'prediction_mean':float(np.mean([r['metrics'][mode]['all']['mean'] for r in pp])),
                   'prediction_by_seed':[r['metrics'][mode]['all']['mean'] for r in pp],
                   'progress_mm':float(np.mean([r['mean_progress_m'] for r in bb])*1000),
                   'regret_mm':float(np.mean([r['mean_regret_m'] for r in bb])*1000),
                   'progress_by_seed_mm':[r['mean_progress_m']*1000 for r in bb]}
                results[task][f][mode]=z
                detail.append([task,NAMES[f],mode,*z['prediction_by_seed'],*z['progress_by_seed_mm']])
    for f in FAMILIES:
        a=results[TASKS[0]][f]['continuous'];b=results[TASKS[1]][f]['continuous']
        table.append([NAMES[f],f"{a['prediction_mean']:.4f}",f"{b['prediction_mean']:.4f}",f"{a['progress_mm']:.3f}",f"{b['progress_mm']:.3f}"])
    header=['构造','抓取预测误差↓','锤子预测误差↓','抓取局部进展 mm↑','锤子局部进展 mm↑']
    mdtable='| '+' | '.join(header)+' |\n|---|---:|---:|---:|---:|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in table)
    summary={'experiment':'P34','status':'complete_bounded_negative_or_inconclusive','full_rollout_episodes':0,
             'training_models':16,'source_episodes_per_task':128,'validation_objects_per_task':16,
             'branch_snapshots_per_task':16,'branch_physical_trials':288,'branch_action_steps':1152,
             'results':results,'gates':branches['gates'],'scope':'development, not OOD/final task success',
             'sources':bindings([ROOT/'validation/PREDICTION_RESULTS.json',ROOT/'branches/RESULTS.json',ROOT/'branches/DECISION_DIAGNOSTIC.json'])}
    dump(out/'SUMMARY.json',summary)
    report=f'''# P34：角色条件化 Interaction Token 的受控比较

结论：角色定义和四种实现已打通，但本轮没有找到关系注意力或关系MLP稳定优于直接拼接的行为证据。抓取有很小的预测改善，未改变实际结果；锤子预测比较由直接拼接领先，动作比较信息不足。按预定规则结束，不晋升、不启动完整新物体回合，也没有新增任务成功率。

## 结果

{mdtable}

以上均为持续适配组、两个固定训练种子的平均。误差为训练尺度归一化、有效观测加mask的物体效果预测MSE，越低越好。毫米数是从同一状态开始保持候选动作四步后的物理进展，**不是成功率**。不同任务的误差尺度不可横向比较。

抓取：关系注意力相对拼接，预测误差0.7318→0.7277，约降低0.57%；两种子方向相同，但幅度小，未作确认性显著性结论。所有24种“编码器×种子×适配模式”在16个状态上得到完全相同的物理进展向量；持续更新相对固定z的动作改变数全部为0。冻结基座单次动作的平均进展3.201 mm，经预测选择后均为5.653 mm；这份局部收益固定z已经具备，不能归给在线梯度或复杂关系结构。

锤子：持续适配的直接拼接预测误差最低。预先按非拼接候选中预测误差选中的三角色注意力，动作进展在种子11优于拼接、种子22劣于拼接；它自己的持续组与固定z组没有任何动作选择变化。只有2/16个状态的候选进展跨度超过2 mm，低于预定至少5个的要求。因此动作侧不足以确立结构优劣。

## 四种构造的区别

- 直接拼接：Actor、Object、Goal各自编码后拼接，公共读出得到U256。
- 三角色注意力：三个256维角色token经过一层自注意力，再读出U256。
- 关系MLP：显式构造Actor–Object和Object–Goal两条关系，使用共享MLP，联合Object节点读出。
- 关系注意力：节点与两条关系边组成五个token，每条边只聚合相关端点；联合Object节点和两条边读出。

四组输入、输出维度、视觉骨干、归一化统计、反馈目标、训练样本顺序和预算、预测解码器、动作候选及打分相同。公共层初始化相同；结构不同的混合层不声称逐参数同初始化。编码器参数为1,148,160／1,149,184／1,152,256／1,151,232，最大差异0.36%。没有用空置填充参数凑容量；四组相同的视觉Goal接口在本轮两个语义Goal任务中均未训练或评估。

## 角色和视觉输入

抓取为夹爪／杯子／抓持并抬升目标；工具交互为已持锤子／木块／推动方向目标。Goal是可学习语义token，与无目标角色或不可见的实体Goal分开，期望抓住不作为已经抓住的事实。物体/部位绑定由当前任务给定，本轮没有Agent自动识别角色或把手定位。

全局RGB224×224；角色裁剪缩放至224×224；冻结DINOv3和LiLa，使用4×4空间特征及LiLa查询上下文。抓取夹爪裁剪只由固定相机标定和自身EEF位置得到，不读取杯子位置；杯子、锤子和木块沿用RGB色标提取。夹爪ROI包含周围像素，不是精确夹爪分割。新角色次序不会改变冻结基座所接收的历史视觉输入次序。

本轮两个任务分别训练模型；并未训练共享多技能网络，也未验证实体Goal、功能部位目标或新Goal泛化。这些接口的单元测试不能当作机器人行为证据。

## 训练与测试时适配

每任务复用64个旧训练物体的128条轨迹：专家执行64条、基座执行64条，每条200步。物体0–47共96条训练；48–63共32条预测验证；没有读旧final或新采示范。已执行动作作为动力学输入，不用专家建议动作作为监督。

每模型600次预测预训练，1000次外层元训练更新（100个轨迹批次，每批200步、每20步截断梯度但保留z数值）；batch8，两种子11/22，四种结构，共16模型，仅用最后checkpoint。U256描述当前交互，32维z是独立快速适配记忆。

预测未来四个已给定动作造成的128维被操作物体特征变化、4维物体图像框变化、2维Actor–Object相对中心变化。组权重0.5/0.25/0.25，只对可用观测归一化。自身位置仍作输入，完全移出预测监督。目标特征来自冻结编码器，不靠可训练目标网络收缩损失。

z根据过去完成的四步转移，用最近8条转移的真实观察误差作梯度更新；网络权重冻结。对照为固定z、只更新前三次、每执行步持续更新。四步延迟后才有首次反馈，不能把“三次更新”叫“三次接触”。离线预测给模型未来记录动作作为待预测计划，未来视觉只用于打分，不进入z更新。

## 同状态动作比较

按结果前规则，选验证物体48–55各两条记录，从20–140步中取公开图像关系距离最小的有效状态，共每任务16个。每状态重放源记录的过去交互更新z，选择一次动作；随后9个候选各固定执行4步，共288条真实短分支、1152个动作步。没有在每个分支中重新选动作，没有重复运行24份相同物理分支。

九候选是基座动作，以及xyz各正负0.25扰动，另加抓取开合正负0.35或锤子xy两个对角扰动。这是相同归一化控制空间的局部搜索预算，不是逐物体厘米脚本。打分为预测物体沿训练RGB所得方向的进展减动作偏离惩罚。所有模型选择在运行物理分支前封存。评价端随后读取实际位移和接触，抓取进展定义为“有双侧把手接触时的非负抬升量”的末值减初值；这些真值不用于选动作或更新z。

预先规定：选中非拼接模型必须在两个训练种子都优于拼接的平均物理遗憾，且不劣于基座，另需至少5个有区分力的状态。抓取两种子均打平拼接；锤子只有一个种子胜出且状态不足。两项均未通过，0条完整新测试回合，不能填写新的ID/OOD成功率。

## 这次发现的实验限制

抓取16个状态有13个已抓住并抬起，其中11个的抬升已超过7 cm（这不等同于满足持续成功判据）。大多数比较只要求继续上抬；所有模型/模式384次选择中357次选择增加向上动作。未建立接触的困难状态占比不足，表示变化很难影响这个简单选择。

锤子多数状态已处于推动后段或短窗口内动作差别不大，14/16个状态的候选物理进展跨度不足2 mm。预测目标平均误差和关键决策区分能力仍然有缺口。

因此下一轮更有针对性的改变应是：在接触前、刚接触、接触失效的公开可观测阶段取样，让“先对准／闭合／保持／移动”的选择有真实分歧；同时检查目标打分是否包含建立或保持目标关系，而不仅是物体往某方向移动。先统一改进这项实验条件，再原样比较四种编码器。该建议未执行，不追加本轮结果后调参。

这些结果仅是色标、固定相机、程序化同类物体上的小型视觉动作基座开发诊断；不是官方benchmark分数、真实机器人或π0.5系统升级。不能由本轮否定关系表示，也不能把术语统一写成已证实收益。

## 复核与材料

6项针对性测试通过：角色与语义Goal缺失处理、容量、物体监督与mask、解析梯度对自动微分、元梯度传播、因果更新与冻结权重。两任务两种子中，拼接及选中模型的连续适配从真实公开历史逐步重放，z与候选分数最大差异均低于2e-5，动作选择全部一致。288条分支的物理进展全部从封存状态重算一致。源文件、冻结权重和原锤子数据hash核对通过。视觉骨干权重与此前LiLa方案一致。

原始预测：[PREDICTION_RESULTS.json](../validation/PREDICTION_RESULTS.json)；物理分支：[RESULTS.json](../branches/RESULTS.json)；诊断：[DECISION_DIAGNOSTIC.json](../branches/DECISION_DIAGNOSTIC.json)；审计：[AUDIT.json](../AUDIT.json)。各任务两个注释短视频展示局部收益和剩余差距，全部明确标为四步分支，见[结果页面](index.html)。
'''
    (out/'REPORT_ZH.md').write_text(report)
    import csv
    with (out/'all_variants.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['task','encoder','mode','prediction_seed11','prediction_seed22','progress_mm_seed11','progress_mm_seed22']);w.writerows(detail)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));xs=np.arange(4);labels=['Flat','Role attention','Relation MLP','Relation attention']
    for ax,task in zip(axes,TASKS):
        fixed=[results[task][f]['fixed']['prediction_mean'] for f in FAMILIES]
        cont=[results[task][f]['continuous']['prediction_mean'] for f in FAMILIES]
        ax.bar(xs-.17,fixed,.34,label='Frozen z',color='#91a6b8');ax.bar(xs+.17,cont,.34,label='Continuous z',color='#16877d')
        ax.set_xticks(xs,labels,rotation=18,ha='right');ax.set_title(task);ax.set_ylabel('Normalized observed-effect MSE (lower better)');ax.legend()
    fig.suptitle('P34: reserved training-range objects; prediction, not task success');fig.tight_layout()
    fig.savefig(out/'prediction_comparison.svg');fig.savefig(out/'prediction_comparison.png',dpi=160);plt.close(fig)
    vids=json.loads((out/'media/VIDEO_INDEX.json').read_text())['videos']
    vt=''.join(f'<article><h3>{html.escape(v["task"])} · {html.escape(v["kind"])}</h3><video controls preload="metadata" poster="media/{Path(v["cover"]).name}" src="media/{Path(v["file"]).name}"></video></article>' for v in vids)
    tb='<tr>'+''.join(f'<th>{h}</th>' for h in header)+'</tr>'+''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in table)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P34角色条件化交互表示比较</title><style>
body{{font:17px/1.65 system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 28px;color:#142934;background:#f6f9fb}}
table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:13px;border-bottom:1px solid #d8e1e8;text-align:left}}
.notice{{background:#fff0dc;border-left:5px solid #ce8b21;padding:18px}}img,video{{width:100%}}article{{padding:15px;background:white;margin:18px 0}}a{{color:#116e80}}
</style><h1>P34：哪种角色关系编码有帮助？</h1><div class="notice"><b>已完成：尚未找到复杂关系编码的稳定行为优势。</b><br>16模型 · 两任务 · 288条四步分支 · 0条完整新物体回合。未通过预定执行门槛，不填写新成功率。</div>
<p>Actor／Object／Goal定义已统一；视觉保持DINOv3＋LiLa。抓取：夹爪／杯子／抓持目标；工具交互：锤子／木块／推动方向。</p>
<table>{tb}</table><p>两训练种子均值。误差越低越好；毫米数为四步局部物理进展，<b>不是成功率</b>。完整逐种子和固定z对照见<a href="all_variants.csv">CSV</a>。</p>
<img src="prediction_comparison.svg" alt="四种编码器固定z与持续更新预测误差比较">
<p>抓取关系注意力预测只改善0.57%，所有构造实际进展完全相同，梯度更新未改变动作。13/16个状态已抓住并抬起；锤子只有2/16个状态能明显区分候选动作。</p>
<p><b>下一步建议（未启动）：</b>让比较覆盖接触建立和失效时刻，检查目标打分是否真的要求建立/保持交互关系，再比较相同四种构造。</p>
<p><a href="REPORT_ZH.md">完整方法与结论</a> · <a href="SUMMARY.json">数字摘要</a> · <a href="../AUDIT.json">验收审计</a> · <a href="../PROTOCOL.json">预定协议</a> · <a href="../../adapter_dashboard/">Adapter Dashboard</a></p>
<h2>局部动作对比：有进展与剩余差距</h2><p>每段只固定执行一个候选四步。展示事后最佳候选作为解释，它没有提供给运行时。</p>{vt}</html>'''
    (out/'index.html').write_text(page)
    dump(out/'ROOT_ACCEPTANCE.json',{'passed':True,'experiment':'P34','status':'complete_negative_or_inconclusive',
        'tests_passed':6,'no_promotion':True,'no_full_rollouts':True,'gate_failures_retained':True,
        'new_terminology_adopted_not_behavior_promoted':True,
        'bindings':bindings([out/'SUMMARY.json',out/'REPORT_ZH.md',out/'index.html',out/'media/VIDEO_INDEX.json',ROOT/'AUDIT.json',Path(__file__)])})
    print('P34_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
