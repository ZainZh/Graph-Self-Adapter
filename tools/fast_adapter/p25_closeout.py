"""Root-reviewed small-trial closure; all results retained, no automatic expansion."""
import json
import time
from pathlib import Path
from tools.fast_adapter.p25_common import R
from tools.fast_adapter.p18_run import digest, save_json


def main():
    out=R/'summary';assert not (out/'ROOT_ACCEPTANCE.json').exists()
    result=json.loads((out/'RESULTS.json').read_text());audit=json.loads((out/'AUDIT.json').read_text())
    assert audit['passed'] and audit['query_episodes_recomputed']==1116
    for b in json.loads((R/'ANALYSIS_FREEZE.json').read_text())['bindings']+result['source_bindings']:
        assert digest(Path(b['path']))==b['sha256']
    videos=json.loads((out/'media/VIDEO_INDEX.json').read_text())['videos']
    for video in videos:
        assert digest(Path(video['file']))==video['sha256']
        for b in video['sources']:assert digest(Path(b['path']))==b['sha256']
    def binding(path):return {'path':str(path),'sha256':digest(path)}
    status='completed_valid_small_development_mixed_no_gain_over_base'
    acceptance={'status':status,'time':time.time(),'root':'GPT-6 Astra, no delegated agents',
        'candidate_promoted':False,'formal_final_started':False,'query_episodes':1116,'support_episodes':36,
        'independent_conditions':36,'models':8,'training_updates_each':6000,
        'audit':binding(out/'AUDIT.json'),'results':binding(out/'RESULTS.json'),
        'posthoc_diagnostic':binding(out/'POSTHOC_COMMON_STATE_DIAGNOSTIC.json'),
        'supported':['Anchored K0 preserves the frozen base exactly, after training and on actual evaluation inputs.',
            'Selected feedback includes later and contact states without runtime access to goal/contact truth.',
            'Physical variant improves scale mean relative to the feature variants, but only ties base and History there.',
            'Correct support improves measured displacement prediction over the training-mean latent on recorded base states.',
            'A privileged controller under the same residual bound demonstrates substantial feasible recovery space.'],
        'not_supported':['Overall improvement over frozen base; ID and position means worsen.',
            'Reliable gradient-method superiority to History or swapped/mean-latent success rates.',
            'A causal proof that contact feedback, new objective, or short-horizon instability explains the entire outcome.',
            'Paper-level confirmation, three-action adaptation efficiency, or a promotion-ready controller.'],
        'next_work_not_started':'Investigate how reduced one-step supervised error fails to translate into stable closed-loop insertion, using recorded contact/terminal segments before proposing another training run.',
        'hardware':'All campaign workers exited; GPU0/1/2 returned to idle; unrelated GPU3 job untouched.',
        'dashboard':'Deferred per user instruction'}
    save_json(out/'ROOT_ACCEPTANCE.json',acceptance)
    labels=[('base','冻结视觉基座'),('original_correct','原结构，读取三条阶段反馈'),('anchored_correct','保护基座＋表示预测'),
        ('physical_correct','保护基座＋位移预测'),('history_correct','同信息History'),('bounded_oracle','同修正幅度真值参考')]
    strata=[('id','同分布'),('position','槽位置变化'),('scale','整体尺度变化')]
    table=['| 方法 | 同分布 | 槽位置变化 | 整体尺度变化 |','|---|---:|---:|---:|'];htmlrows=[]
    for key,label in labels:
        values=[f"{result['strata'][st]['rates_percent'][key]:.2f}%" for st,_ in strata]
        table.append('| '+' | '.join([label,*values])+' |')
        htmlrows.append('<tr><th>'+label+'</th>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    markdown='''# P25 小规模尝试：主Agent结论

**本轮没有获得超过冻结基座的自提升。尺度档出现改进线索，但同分布和位置变化仍有明显误伤；不扩大、不晋升。**

LiLa、DINOv3、原PCA和视觉基础控制均保持冻结。两个随机种子，每个种子的原结构、保护基座版本、位移预测版本和History各训练6000次，共8个模型。没有新增训练物理回合，复用2304条既有训练轨迹。支持与监督query来自同一环境的不同轨迹／独立起点。

三个新开发分区各12条件，共36个独立条件；不同方法和两个种子共享条件，总计1116次query和36条共同支持轨迹。整批训练与开发执行约7分钟，不含实现、回传、审计和视频整理。本批的基座比例与上一批不同是因为条件集合和分母改变，基座没有重新训练。

TABLE

普通学习版本均使用3条反馈；支持采集实际执行32步，随后换独立起点执行48步query，z在query期间固定。原结构也用这批阶段材料重新训练，不能直接拿它与P24结果做单因素比较。保护基座只保证K0严格等于基础控制，不能保证更新后无损害。

## 三项修改的实际效果

动作接入的保护实现了：在任意已训练权重下，z=z0时输出严格等于基座。完整动作复算验证了这项恒等关系。但更新后仍会误伤：位移版本相对基座，同分布为0次救回／6次损害，位置档1次救回／4次损害，尺度档3次救回／3次损害，计数均为模型×条件执行。

反馈时机确实改变了：三档最后一个窗口选中的反馈，分别有7/12、9/12、9/12个后继状态发生接触；中间窗口分别5/12、9/12、8/12。接触仅在封存后统计，选择器没有读取接触标签。因而本轮已经补上“后期反馈完全缺席”的问题，但不能说加入接触就必然提升控制。

位移目标的尺度成功率比保护基座的表示目标高12.5个百分点（45.83%→58.33%），两个种子差值为+16.67、+8.33个百分点；同分布只高4.17个百分点，位置低4.17个百分点。尺度仍只是追平基座与History，并未证明额外任务收益。物理目标也改变预测头维数与声明的损失缩放，这是一个小组合修订，不是无其他差别的单项消融。

## 环境反馈有没有被利用

位移版本正确z的三档结果为66.67%／4.17%／58.33%；交换z为62.50%／0%／54.17%；训练期平均z为66.67%／4.17%／41.67%。尺度相对固定平均z有正向线索，但只有12个环境和两个种子，配对描述区间仍跨零。三档相对交换z的区间也都包含或触及零；不声称已确认环境专属的任务成功率收益。

同一物理checkpoint改用最早三条反馈，三档为70.83%／4.17%／50.00%。阶段反馈相对早期反馈在尺度均值更好，同分布更差，位置相同；并非所有场景越晚反馈就越好。双方都执行同一条32步共同轨迹，不把选用三条反馈写成只用了三次物理动作。

## 额外事后诊断：单步学习与完整执行出现差距

所有开发结果固定后，在封存基座轨迹的相同状态上检查位移版本，没有执行新的动作或更新模型。评价器用真值计算动作参考，再投影到原±0.35修正范围；这些标签只用于事后评价，不回流运行时或训练。

正确反馈z在六个“种子×分区”中，下一步实际位移预测误差都低于固定训练平均z；其单步动作对可达专家参考的平均误差，也都低于基础控制。在同分布上，该动作误差由0.05634降到两个种子平均0.01733，但真正闭环任务成功率由91.67%降到66.67%。这表明不能简单声称“完全没有学到信息”，也不能以单步预测／动作误差的下降替代整段插入成功。

这个诊断将下一步问题缩小到：为什么较好的单步拟合没有转化为自身执行轨迹上的稳定成功。可能涉及状态分布变化、接触阶段误差、动作连续性与最终保持；当前没有分离这些原因。它不是新的独立测试或控制优越性证明。

同修正幅度的真值参考达到100%／66.67%／100%，说明在当前动作范围内存在明显的改进空间。它使用特权信息，只是可行性参考，不是可部署方案或经过证明的最优成功率上界。现有结果不足以把失败主要归因于±0.35的幅度限制。

## 验证与交付

实现检查包括：训练后任意参数下K0恒等、时间窗因果性、真实前一动作配对、完整元梯度有限差分与既有数据的小样本数值检查。实际执行后，1116条query的动作、latent、支持选择和末尾连续三帧成功判据全部复算通过。动作最大差约1.97e-6，latent最大差约2.41e-6。本批保存实际float32输入，解决的是本批记录精度问题；P24原审计例外不追溯修改。

![本批成功率](success_rates.png)

[完整机器结果](RESULTS.json) · [复算](AUDIT.json) · [事后同态诊断](POSTHOC_COMMON_STATE_DIAGNOSTIC.json) · [两个注释视频](index.html#videos)。视频按固定规则选出一次救回和一次误伤，包含32步支持采集与独立query两个阶段；由封存状态重渲染，不是新执行。

本轮按小规模开发验证关闭，不启动formal final或下一批训练。下一步建议先检查已有接触／结束片段中的误伤和闭环稳定性，再决定是否需要短时序训练目标。Dashboard按用户要求继续暂缓。
'''.replace('TABLE','\n'.join(table))
    (out/'ROOT_REPORT_ZH.md').write_text(markdown)
    cards=[]
    for video in videos:
        p=Path(video['file']).relative_to(out);kind='位移版本造成误伤' if video['kind']=='harm' else '位移版本救回基座失败' if video['kind']=='rescue' else video['kind']
        cards.append(f'<article><h3>{dict(strata)[video["stratum"]]} · {kind}</h3><p>种子{video["seed"]}，条件{video["condition"]}。先共同支持采集，再独立初态执行。</p><video controls preload="none" poster="{p.with_suffix(".png")}" src="{p}"></video></article>')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>P25小规模开发结果</title>
<style>body{{font:17px/1.8 system-ui,sans-serif;max-width:1380px;margin:35px auto;padding:0 28px;color:#193139;background:#f6f8f8}}h1{{font-size:30px}}h2{{margin-top:36px}}a{{color:#166c72}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:10px 18px;border-bottom:1px solid #dce4e4;text-align:right}}th:first-child{{text-align:left}}img,video{{display:block;width:100%}}article{{padding:18px;background:white;border-radius:8px;margin:25px 0}}.lead{{font-size:21px;font-weight:650}}</style>
<h1>P25 · 保护基座与阶段反馈的小规模尝试</h1><p class="lead">尺度档出现改进线索，但未超过基础控制；本轮不扩大、不晋升。</p>
<p>视觉前端完全不变。8个模型各训练6000步；每类12个新开发条件，两个种子共享条件。共1116次query、36条支持轨迹，训练和执行约7分钟。</p>
<p><a href="ROOT_REPORT_ZH.md">完整解释</a> · <a href="RESULTS.json">全部对照和种子</a> · <a href="ROOT_ACCEPTANCE.json">验收</a> · <a href="#videos">两个注释视频</a></p>
<table><thead><tr><th>方法</th><th>同分布</th><th>槽位置变化</th><th>整体尺度变化</th></tr></thead><tbody>{''.join(htmlrows)}</tbody></table>
<p>原结构也在相同的新阶段材料上重训；保护版本K0严格等于基座。以上保护／位移／History行均为读取三条反馈之后的表现。不同批次分母不同，不能将本批基座数字与P24连成学习曲线。</p>
<img src="success_rates.png" alt="开发成功率对比">
<h2>这轮定位了什么</h2><p>本轮反馈确实覆盖接触阶段，K0也确实保留基座行为，但更新后仍有误伤。位移预测与单步动作参考误差有所改善，完整执行成功率没有随之提高；下一步应定位接触及保持阶段的闭环问题。</p>
<p>位移版本相对基座：同分布0救回／6损害，位置1救回／4损害，尺度3救回／3损害（模型×条件执行）。相对History、交换z的稳定成功率优势尚未成立。同幅度真值参考说明当前修正范围内仍有恢复空间。</p>
<p><a href="AUDIT.json">1116条动作、latent和成功判据复算全部通过</a>。本批使用实际float32输入；P24旧例外保留。<a href="POSTHOC_COMMON_STATE_DIAGNOSTIC.json">单步改善与闭环差距的事后诊断</a>没有回流训练或运行时。</p>
<h2 id="videos">支持采集和独立执行的注释视频</h2><p>前段：32步共同支持采集、三个窗口各选择一条反馈。后段左起：冻结基座、原结构、保护基座、位移版本。query阶段z固定，几何和接触标注为事后评价。</p>{''.join(cards)}
<p>这是开发快筛，未启动正式最终测试；所有GPU作业已结束。Dashboard继续暂缓。</p></html>'''
    (out/'index.html').write_text(page)
    paths=[p for p in sorted(R.rglob('*')) if p.is_file() and p.name!='EVIDENCE_INDEX.json']
    save_json(R/'EVIDENCE_INDEX.json',{'status':status,'files':[dict(binding(p),bytes=p.stat().st_size) for p in paths],
        'count':len(paths),'large_binary_artifacts_in_git':False})
    print(json.dumps({'status':status,'indexed_files':len(paths),'videos':len(videos)}))


if __name__=='__main__':main()
