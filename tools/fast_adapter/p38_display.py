"""Standalone scientific charts and desktop review page from sealed summaries."""
import json
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

R=Path('evaluations/fast_adapter_p38_causal_v01');OUT=R/'summary'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
plt.rcParams['font.family']=FontProperties(fname=FONT).get_name()
plt.rcParams['axes.unicode_minus']=False
COLORS=['#7b8794','#187f80','#dd8c2f','#3264ad','#a35c90','#7368b6']


def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    p=json.loads((OUT/'final_PHYSICAL_RESULTS.json').read_text());table=p['table']
    numeric=json.loads((R/'online_mapping/summary/CAUSAL_RESULTS.json').read_text())['summary']
    def value(split,kind,arm):return table[f'{split}/{kind}/{arm}']['success_rate']*100
    kinds=['references','motion','appearance','combined','history'];labels=['冻结基座','可见运动反馈','DINO特征反馈','两种反馈结合','History']
    fig,axes=plt.subplots(1,4,figsize=(16,5),sharey=True)
    for ax,split,title in zip(axes,['native','id','ood','contrast'],['原生阻力 · 32条件','范围内新条件 · 32条件','更大阻力 · 32条件','同外观阻力对照 · 32对']):
        vals=[value(split,k,'base' if k=='references' else 'continuous') for k in kinds]
        ax.bar(np.arange(5),vals,color=COLORS[:5],width=.72)
        for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.1f}',ha='center',fontsize=10)
        ax.set_title(title,fontsize=13);ax.set_xticks(range(5));ax.set_xticklabels(labels,rotation=40,ha='right',fontsize=10)
        ax.set_ylim(0,112);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('最终成功率（%）')
    fig.suptitle('短距离物体接触：只用当前图像和已执行动作的反馈',fontsize=17,y=1.02)
    finish(fig,'physical_success')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    names=['冻结基座','训练平均z','前三次后固定','交换反馈','持续当前反馈']
    vals=[value('contrast','references','base')]+[value('contrast','motion',a) for a in ['train_mean','first3','donor','continuous']]
    axes[0].bar(range(5),vals,color=['#7b8794']*4+['#187f80'])
    axes[0].set_xticks(range(5));axes[0].set_xticklabels(names,rotation=25,ha='right');axes[0].set_title('因果对照：当前物体的持续反馈带来增益')
    for i,v in enumerate(vals):axes[0].text(i,v+2,f'{v:.2f}%',ha='center',fontsize=10)
    names=['DINO特征反馈','可见运动反馈','结合表示'];vals=[value('contrast',k,'continuous') for k in ['appearance','motion','combined']]
    axes[1].bar(range(3),vals,color=[COLORS[2],COLORS[1],COLORS[3]])
    axes[1].set_xticks(range(3));axes[1].set_xticklabels(names);axes[1].set_title('本任务：可见运动比仅拟合特征变化更可靠')
    for i,v in enumerate(vals):axes[1].text(i,v+2,f'{v:.2f}%',ha='center')
    for ax in axes:ax.set_ylim(0,112);ax.set_ylabel('成功率（%）');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    finish(fig,'causal_and_representation')
    audits=[]
    for path in OUT.glob('PHYSICAL_AUDIT_shard*.json'):audits+=json.loads(path.read_text())['rows']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    settings=[('references','base','冻结基座',COLORS[0]),('motion','first3','前三次后固定','#a8b6bb'),
              ('appearance','continuous','DINO特征反馈',COLORS[2]),('motion','continuous','可见运动反馈',COLORS[1]),
              ('combined','continuous','结合表示',COLORS[3])]
    for kind,arm,label,color in settings:
        selected=[a for a in audits if '/push_contrast/' in a['file'] and f'/final/{kind}/' in a['file'] and a['file'].endswith('/'+arm+'_RESULT.json')]
        successes=np.mean([a['per_episode_success_curve'] for a in selected],axis=0)
        errors=np.mean([a['per_episode_error_curve'] for a in selected],axis=0)
        axes[0].plot(np.arange(31),successes.mean(0)*100,label=label,color=color,lw=2.3)
        axes[1].plot(np.arange(31),errors[1::2].mean(0)*1000,label=label,color=color,lw=2.3)
    axes[0].set_ylabel('当步达到位置与速度条件的比例（%）');axes[0].set_title('一个任务内部的适配过程，并非跨任务学习曲线')
    axes[1].set_ylabel('高阻力物块距目标的平均误差（mm）');axes[1].set_title('只做前三次更新仍不足以消除持续偏差')
    for ax in axes:
        ax.axvspan(0,3,color='#dddddd',alpha=.35);ax.set_xlabel('已完成物理动作数');ax.grid(alpha=.18);ax.legend(fontsize=10)
    finish(fig,'within_episode_curves')
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    names=['冻结z','6维梯度','32维梯度','History','最小二乘','解析Oracle']
    for ax,split,title in zip(axes,['native','id','ood'],['原生响应','同分布新响应','更大映射偏差']):
        vals=[numeric['effect/'+split]['frozen']['success_rate'],numeric['structured/'+split]['continuous']['success_rate'],
              numeric['effect/'+split]['continuous']['success_rate'],numeric['history/'+split]['continuous']['success_rate'],
              numeric['effect/'+split]['least_squares']['success_rate'],numeric['effect/'+split]['analytic_oracle']['success_rate']]
        vals=np.array(vals)*100;ax.bar(range(6),vals,color=COLORS)
        for i,v in enumerate(vals):ax.text(i,v+2,f'{v:.1f}',ha='center',fontsize=9)
        ax.set_xticks(range(6));ax.set_xticklabels(names,rotation=35,ha='right',fontsize=10);ax.set_title(title)
        ax.set_ylim(0,112);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('最终成功率（%）');fig.suptitle('数值机制：相同8次动作预算、持续更新；History同样重新训练',fontsize=16,y=1.03)
    finish(fig,'numeric_online')
    table_rows=''.join('<tr><td>'+s+'</td>'+''.join(f'<td>{value(s,k,"base" if k=="references" else "continuous"):.2f}%</td>' for k in kinds)+'</tr>' for s in ['native','id','ood','contrast'])
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P38 Fast Adapter 验证</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;background:#f4f7fb;color:#1b3044;margin:0}}main{{max-width:1320px;margin:32px auto;padding:0 30px 60px}}section{{background:white;border:1px solid #dce5eb;border-radius:12px;padding:24px;margin:24px 0}}h1{{font-size:32px}}h2{{font-size:23px}}img,video{{width:100%;display:block;border-radius:7px}}a{{color:#176f8b}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dde5eb}}.note{{color:#526678}}.lead{{font-size:21px;color:#116969}}.links a{{margin-right:24px}}</style>
<main><h1>P38：Fast Adapter 的因果机制与反馈表示</h1>
<p class="lead">受控条件下，持续识别当前物体的力—运动关系可以改善控制；运动反馈比仅拟合DINO特征变化更可靠。</p>
<p>全部独立批次完成：三档各32条件，另有32对同外观/不同阻力条件；两个固定物理模型。不是完整机器人、任意物体视觉或普遍优于History的结论。</p>
<p class="links"><a href="ROOT_REPORT_ZH.md">完整结论与边界</a><a href="ROOT_ACCEPTANCE.json">验收与完整性</a><a href="final_PHYSICAL_RESULTS.json">全部数字和配对区间</a><a href="../PROTOCOL.json">初始协议</a></p>
<section><h2>独立成功率</h2><table><tr><th>设置</th>{''.join('<th>'+x+'</th>' for x in labels)}</tr>{table_rows}</table><p class="note">native=原生阻力；id=范围内新条件；ood=更大阻力；contrast=同外观但阻力相反的配对条件。两模型结果在条件内平均，不增加独立环境数。</p><img src="physical_success.png"></section>
<section><h2>收益来自什么</h2><img src="causal_and_representation.png"><p>对比组运动反馈为64/64、64/64；固定训练平均z与前三次后固定均为32/64；错配反馈均为7/64。32对全通过不等于真实成功率必然100%。</p></section>
<section><h2>执行过程中是否持续改善</h2><img src="within_episode_curves.png"><p class="note">灰色区域是计入总预算的三次共同探测。每次完成动作后更新一次z；没有更新基座或视觉模型权重，也没有读取成功标签。</p></section>
<section><h2>注释对比：反馈表示</h2><video controls preload="metadata" poster="media_v03/representation_pair.jpg" src="media_v03/representation_pair.mp4"></video><p>绿色工具推动蓝色滑块，横向对齐红标记。两行的可见几何分别相同，但阻力不同。此视频特意选取能看到特征反馈失败的案例，完整成功率见上表。</p></section>
<section><h2>注释对比：正确反馈、错配反馈与固定经验</h2><video controls preload="metadata" poster="media_v03/correct_vs_wrong_feedback.jpg" src="media_v03/correct_vs_wrong_feedback.mp4"></video><p class="note">由已封存的真实物理轨迹重放；全部展示物理位置精确复现。一帧RGB哈希不同但检测位置相同，未用重渲染画面修改任何指标。</p></section>
<section><h2>数值机制与更强基线</h2><img src="numeric_online.png"><p>连续梯度适配优于自身冻结和错误反馈；数值同分布任务中History仍更强。更大偏差还受到冻结神经动作头外推能力限制，不能归咎于latent一项。</p><a href="../online_mapping/summary/CAUSAL_REPORT_ZH.md">数值实验完整结果</a></section>
<section><h2>当前能说与不能说</h2><p>能说：明确的响应结构、可见动作后果和匹配执行方式的训练，能让小latent适配改善后续行为。不能说：任何梯度更新都会改善；原自由高维视觉latent已全面修复；已建立通用机器人或RSI能力。</p><p class="note">该物理任务使用已知近静态响应形式、解析直接动作修正、固定颜色标记和相机标定。DINO分支也共用RGB目标关系控制，只比较适配反馈。原始负结果、训练平均经验、History、最小二乘和图像重放例外全部保留。</p></section></main></html>'''
    (OUT/'index.html').write_text(body)
    print('Four figures (PNG/PDF) and desktop report generated')


if __name__=='__main__':main()
