"""Publish the completed bounded pilot, including unsuccessful comparisons."""
import html
from datetime import datetime,timezone
from tools.fast_adapter.p33_common import *

def main():
    out=ROOT/'summary';summary=json.loads((out/'SUMMARY.json').read_text())
    audit=json.loads((ROOT/'AUDIT.json').read_text());assert audit['passed']
    videos=json.loads((out/'media_v2/VIDEO_INDEX.json').read_text())['videos']
    names={'base':'冻结基座','cross_fixed':'关系模型：固定z','cross_first3':'关系模型：前三次后固定',
           'cross_continuous':'关系模型：持续更新','within_fixed':'物体内注意力：固定z','within_continuous':'物体内注意力：持续更新'}
    lines=['| 方法 | ID成功数（种子11 / 22） | ID平均成功率 | OOD成功数（种子11 / 22） | OOD平均成功率 |',
           '|---|---|---:|---|---:|']
    tab=[]
    for arm in protocol()['evaluation']['arms']:
        row=[names[arm]]
        for split in ('train','geometry_ood'):
            r=summary['rates'][split][arm];c=r['successes']
            row.extend([f"{c['None']}/8（共用）" if arm=='base' else f"{c['11']}/8 / {c['22']}/8",f"{r['mean_rate']:.2f}%"])
        lines.append('| '+' | '.join(row)+' |');tab.append(row)
    table='\n'.join(lines)
    report=f'''# P33：关系动力学与持续梯度适配，小规模开发结果

完成状态：有局部适配收益，未达到可靠系统升级；不晋升、不扩量，下一批未启动。

## 结论

本轮的持续更新确实改变了动作，并在同一新控制器内补回少量失败；但两个训练种子的控制表现差距很大，平均成绩明显低于原冻结基座。新增适配救回均属于基座原本成功的条件。不能把减少新控制器的误伤写成基座之外的新能力，也没有证明跨物体注意力稳定优于同容量对照。

{table}

ID为训练范围内的新物体；OOD为锤柄更长、锤头更大的新物体。每档8个共同物体条件，两个模型种子共用这些条件，基座各档只运行8次。百分比为两个训练种子的平均，不把16次模型结果当成16个独立物体。共176个物理任务回合，准备失败为{summary['preparation_failures']}。

## 做了什么

- 复用P30-v2的128条已开放训练轨迹，来自64个训练范围内物体；48物体的96条记录用于训练，16物体的32条记录只用于预测验证。没有读取旧final来学习，没有新采示范。
- DINOv3、LiLa融合层与查询adapter、原小型视觉动作基座全部冻结。新模块读取LiLa融合后的4×4物体空间网格、查询向量的上下文、RGB图像位置及自身状态。LiLa的64个查询本身不是空间网格。
- 同容量比较跨物体注意力和物体内注意力后拼接。输入、参数数量、随机初始化和训练预算一致。
- 预测四步执行动作将带来的256维局部特征变化、8维RGB框变化和4维自身状态变化。所有目标是已经观察到的变化；专家示范的已执行动作可作为动力学输入，但不使用专家动作标签训练新模块。
- 每模型600次预测预训练、1000次外层元训练更新。完整200步顺序展开，每20步截断梯度而保留z数值；用过去完成的交互更新z，再预测尚未用于适配的后续变化。
- 推理时只有32维z更新。每执行一个控制动作接收反馈，四步转移完成后以最近8条转移更新一次。固定z为0次；前三次对照在执行步4、5、6后更新，再固定；持续组共197次更新。前三次并不表示三次已确认接触。
- 由同一个预测控制器，在冻结基座动作附近的27个有界候选中选择预测目标进展更大的动作，预测4步、只执行1步。任务方向来自训练RGB中木块运动方向，运行时不读取木块真值位置或成功标签。
- z更新使用预测损失的精确解析梯度，已与自动微分核对；网络权重测试期不变。它仍是梯度适配，不是历史均值或额外专家动作控制。

## 有哪些局部证据

持续更新相对同模型固定z：ID增加6.25个百分点，OOD增加12.5个百分点。OOD两个种子分别7/8→8/8和0/8→1/8；相对前三次停止，OOD平均增加18.75个百分点。

相对固定z新增的三条“模型—条件”成功为：ID种子22条件1，OOD种子11条件1，OOD种子22条件5。它们原冻结基座都成功，属于补回新控制器误伤。种子11相对基座新增的ID条件3和OOD条件4，在固定z的新控制器中已经成功，不能归给在线适配。

OOD持续更新对固定z的描述性95%条件簇bootstrap区间为[0,31.25]个百分点，样本不足以作稳定优势结论。统计以每档8个物体条件为簇，两个已训练种子固定，未作多重比较校正；不是正式确认性检验。

跨物体注意力持续组相对物体内注意力持续组：ID低12.5个百分点，OOD高6.25个百分点。OOD区间包含正负值；其独立贡献没有确证。

## 最明确的薄弱处

第一，新模型驱动的动作选择还不可靠。种子22在完全不更新z时，ID只有1/8，OOD为0/8；相同条件下原基座均为7/8。因此大量失败在适配之前就存在，不能全部归责于持续梯度更新。模型种子11和22虽然预测误差接近，控制表现却悬殊，说明应优先验证候选动作的效果排序，而不是仅继续降低平均预测损失。

第二，预期的物体交互预测改善没有出现。在相同新基座轨迹上，用已完成的过去转移适配，再预测后续四步真实反馈：ID综合归一化MSE在两个关系模型中分别0.6074→0.5805、0.5881→0.5733；OOD分别3.1640→3.1760、3.1136→3.1727，反而变差。

进一步按有效观测分组，ID的改善主要出现在自身状态预测；局部物体特征和木块运动的平均误差没有改善。OOD两个种子的物体特征、框和木块运动预测误差均增加，自身状态预测有所降低。这提示混合目标尚未把适配明确对准物体间作用，不能称为交互关系泛化成功。该诊断发生在共同基座记录上，不能直接解释每一条候选控制轨迹的成败。

## 已做的复核

5项针对性测试通过：精确z梯度、相同模型容量、缺失观测处理、有界动作、外层元梯度、因果窗口与前三次停止以及冻结权重。全部176回合的动作与z从缓存公开输入复算，最大误差均为0；35200个物理状态的成功判据和首次成功步全部复算一致。每个配置的初始物理状态和视觉输入完全配对。

同一RGB输入下，新提取器给原基座的输入与原前端误差为0；本轮从旧状态重绘的特征与历史缓存存在最大0.34072差异，所以以本轮新跑基座为参照，不能拿旧P30的绝对数字作单因比较。本轮未逐帧重新执行DINO编码审计，动作审计使用封存的视觉缓存。

原预测诊断中的附加box_mse与target_motion_mse没有屏蔽不可见物体帧，已保留原文件并增加FORECAST_MASKED_SUBGROUPS.json更正。这不改变原本已正确加mask的综合MSE、任何模型、动作、成功率或主比较。报告对分组预测的解释使用更正后的数据。

## 视频与材料

视频一：OOD种子11条件1，持续更新补回固定z控制器的失败，但原基座本来成功。视频二：OOD种子22条件0，原基座成功，多个新控制器均失败。二者都有执行阶段、更新次数、接触与成功注释，来自已封存状态重绘，不增加任务回合。旧首版两个正例的视频清单保留；展示改用一例救回、一例失败，避免只展示正例。

- [成功率与注释视频](index.html)
- [机器汇总](SUMMARY.json)、[全部配对结果](../development/RESULT.json)
- [动作与物理复核](../AUDIT.json)
- [共同记录预测诊断](../diagnostics/FORECAST.json)、[可见性分组更正](../diagnostics/FORECAST_MASKED_SUBGROUPS.json)
- [预定协议](../PROTOCOL.json)、[实施细节](../IMPLEMENTATION_NOTES.json)

## 范围与下一步

任务是固定色标、固定相机下的程序化同类工具水平推击，目标木块沿固定轨道移动；已持工具初始化使用独立准备流程，不含抓取学习，不是完整敲击、官方benchmark或真机结果。本轮没有History或其他物体反馈对照，不能宣称梯度算法普遍更优或环境专属性已经证明。

建议下一步先检查基座附近候选动作的真实效果排序，补充必要的局部动作覆盖；另以受控消融检验仅针对物体反馈的适配目标。它们是待验证的修订假设。本轮已经完成，不因这批结果自动追加训练或新的测试。
'''
    (out/'REPORT_ZH.md').write_text(report)
    body=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in tab)
    cards=[]
    for v in videos:
        caption='局部适配救回：原基座本来成功' if v['kind']=='adaptation_rescue' else '整体失败：原基座成功，新控制器失败'
        rel=str(Path(v['file']).relative_to(out))
        cards.append(f'<article><h3>{caption}</h3><video controls preload="metadata" poster="{rel[:-4]}.png" src="{rel}"></video><p>OOD · 模型种子{v["seed"]} · 条件{v["condition"]}。注释包含更新次数、交互阶段和物理结果。</p></article>')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>P33 关系动力学与持续适配</title>
<style>body{{margin:0;background:#101821;color:#e4edf3;font:17px/1.7 system-ui,sans-serif}}main{{max-width:1480px;margin:30px auto;padding:24px}}h1{{font-size:34px}}.lead{{background:#203442;border-left:5px solid #67cabb;padding:20px}}a{{color:#83d9f0}}table{{width:100%;border-collapse:collapse;margin:25px 0;background:#182431}}td,th{{padding:12px;border-bottom:1px solid #3a4957;text-align:left}}img,video{{width:100%;border-radius:8px}}article{{margin:28px 0;background:#182431;padding:18px}}.small{{color:#bed0dd}}</style>
<main><h1>P33：关系动力学与持续梯度适配</h1><div class="lead"><b>完成：有局部适配收益，尚未达到可靠升级。</b><br>持续更新补回少量新控制器误伤；平均成绩仍低于冻结基座，关系注意力的独立优势没有确认。</div>
<p>176回合 · ID/OOD各8个共同物体条件 · 两个模型种子 · 35200个物理状态复核通过。基座每档只执行一次8条件；其结果在两个模型种子间共用。</p>
<table><thead><tr><th>方法</th><th>ID成功数：种子11 / 22</th><th>ID平均</th><th>OOD成功数：种子11 / 22</th><th>OOD平均</th></tr></thead><tbody>{body}</tbody></table>
<img src="success_rates.svg" alt="六配置同分布与范围外成功率，黑点为两个训练种子">
<p><b>怎么理解：</b>OOD相对同模型固定z提高12.5个百分点，但新增救回条件原基座本来就能成功。模型种子差异很大；仅把改动后的系统与自身较弱版本比较，会高估升级程度。</p>
<p><b>表示与预测：</b>冻结DINOv3＋LiLa；新关系模型由真实视觉变化训练。持续更新并未降低OOD的综合预测误差，物体交互预测改善未成立。更详细的方法、终止指标、统计限制与后续建议见完整报告。</p>
{''.join(cards)}<p><a href="REPORT_ZH.md">完整中文报告</a> · <a href="SUMMARY.json">机器汇总</a> · <a href="SUCCESS_RATES.csv">成功率表</a> · <a href="../AUDIT.json">复核结果</a> · <a href="../../adapter_dashboard/">返回Adapter Dashboard</a></p>
<p class="small">仅开发验证，不晋升、不扩大实验。色标程序化工具与水平推击；不是完整敲击或真机结论。视频从已存状态重绘，注释真值不进入控制。</p></main></html>'''
    (out/'index.html').write_text(page)
    sources=[out/'SUMMARY.json',out/'REPORT_ZH.md',ROOT/'AUDIT.json',ROOT/'diagnostics/FORECAST.json',
             ROOT/'diagnostics/FORECAST_MASKED_SUBGROUPS.json',out/'media_v2/VIDEO_INDEX.json',ROOT/'PROTOCOL.json']
    dump(out/'ROOT_ACCEPTANCE.json',{'status':'completed_mixed_local_adaptation_no_system_upgrade','completed_at':datetime.now(timezone.utc).isoformat(),
         'all_prescribed_training_and_rollouts_completed':True,'episodes':176,'physical_states_recomputed':35200,
         'targeted_tests_passed':5,'promoted':False,'next_experiment_started':False,
         'runtime_gradient_changes_actions':True,'bounded_development_adaptation_rescues':3,
         'all_new_adaptation_rescues_succeed_under_original_base':True,
         'reliable_system_improvement_established':False,'relation_attention_advantage_established':False,
         'OOD_forecast_improvement_established':False,'history_superiority_tested':False,
         'visual_audit':'cached-input action replay plus same-image frontend compatibility; no full per-frame reencoding',
         'bindings':bindings(sources)})
    dump(ROOT/'RUN_STATUS.json',{'phase':'complete','report':str(out/'REPORT_ZH.md'),'promoted':False,'next_started':False})
    print('REPORT_COMPLETE',out/'index.html',flush=True)

if __name__=='__main__':main()
