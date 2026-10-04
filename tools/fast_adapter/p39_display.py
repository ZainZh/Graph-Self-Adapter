"""P39 static scientific figures and review page, entirely post-seal."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tools.fast_adapter.p39_tokens import ROOT

OUT=ROOT/'summary'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
plt.rcParams['font.family']=FontProperties(fname=FONT).get_name();plt.rcParams['axes.unicode_minus']=False
KINDS=('references','global','local','relation','fused','history')
LABELS=('冻结基座','整图DINO','局部空间DINO','相对关系','局部＋关系','History')
COLORS=('#8b96a3','#d89732','#5175b8','#229582','#9555a2','#706c95')
SPLITS=('train','resistance','geometry','joint','paired')
TITLES=('范围内新条件','较高隐藏阻力','新尺寸与方向','几何与阻力变化','同外观／不同阻力')

def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def main():
    d=json.loads((OUT/'final_RESULTS.json').read_text());tab=d['table']
    def val(s,k,a=None):return tab[f'{s}/{k}/{a or ("base" if k=="references" else "continuous")}']['rate']*100
    fig,axes=plt.subplots(2,3,figsize=(17,10));axes=axes.ravel()
    for ax,s,title in zip(axes,SPLITS,TITLES):
        y=[val(s,k) for k in KINDS];ax.bar(range(6),y,color=COLORS)
        for i,v in enumerate(y):ax.text(i,v+1.8,f'{v:.1f}',ha='center',fontsize=11)
        ax.set_ylim(0,111);ax.set_xticks(range(6));ax.set_xticklabels(LABELS,rotation=30,ha='right');ax.set_ylabel('最终成功率（%）');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[-1].axis('off');axes[-1].text(0,.9,'32个独立条件／普通设置\n32对同外观条件／配对设置\n\n两个固定模型在条件内平均\n所有方法共享mask几何控制器\n40次动作含3次共同探测\n\n不是机械臂或官方benchmark',fontsize=15,va='top',linespacing=1.8)
    finish(fig,'success_rates')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,s,title in zip(axes,['joint','paired'],['几何与阻力同时变化','同外观、相反阻力反馈']):
        x=np.arange(4);width=.19
        for j,(arm,label) in enumerate(zip(('continuous','first3','mean','donor'),('持续更新','三次后固定','训练平均z','错配反馈'))):
            y=[val(s,k,arm) for k in KINDS[1:5]];ax.bar(x+(j-1.5)*width,y,width,label=label)
        ax.set_xticks(x);ax.set_xticklabels(LABELS[1:5]);ax.set_ylim(0,110);ax.set_title(title);ax.set_ylabel('最终成功率（%）');ax.legend(fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    finish(fig,'causal_controls')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,s,title in zip(axes,['joint','paired'],['几何与阻力同时变化','同外观低／高阻力混合']):
        for k,label,color in zip(KINDS,LABELS,COLORS):
            key=f'{s}/{k}/'+('base' if k=='references' else 'continuous')
            ax.plot(range(41),d['mask_error_curves_mm'][key],label=label,color=color,lw=2)
        ax.set_title(title);ax.set_xlabel('已执行动作数');ax.set_ylabel('mask测量的物块—目标平均距离（mm）');ax.axvspan(0,3,color='#aaa',alpha=.15);ax.grid(alpha=.2);ax.legend(fontsize=10)
    finish(fig,'within_episode')
    rows=''.join('<tr><td>'+t+'</td>'+''.join(f'<td>{val(s,k):.2f}%</td>' for k in KINDS)+'</tr>' for s,t in zip(SPLITS,TITLES))
    videos=''
    for name,title in [('representation_comparison','同一条件的六种方法'),('retained_failure','保留的失败案例'),('feedback_causality','固定经验、错配反馈与当前反馈')]:
        if (OUT/'media_v01'/(name+'.mp4')).exists():videos+=f'<section><h2>{title}</h2><video controls preload="metadata" poster="media_v01/{name}.jpg" src="media_v01/{name}.mp4"></video><p>示例由封存动作重放，全部帧有阶段与反馈注释；不替代完整统计。</p></section>'
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P39 物体关系Token与持续适配</title><style>
body{{font:17px/1.7 system-ui,sans-serif;color:#243b4b;background:#f3f6fa;margin:0}}main{{max-width:1400px;margin:30px auto;padding:0 25px 50px}}section{{background:white;border:1px solid #dce4eb;padding:23px;border-radius:12px;margin:23px 0}}h1{{font-size:32px}}h2{{font-size:24px}}img,video{{width:100%;display:block}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #dce4eb;text-align:left}}a{{color:#157b84}}.note{{color:#607183}}</style><main>
<h1>P39：较难物体交互中的Token与Fast Adapter</h1><p>二维近接触推移：尺寸、方向与隐藏物体阻力变化。冻结控制器，执行后持续更新四维响应latent。</p>
<p><a href="ROOT_REPORT_ZH.md">主Agent结论</a> · <a href="final_REPORT_ZH.md">全部成功率</a> · <a href="final_RESULTS.json">配对区间与机器数据</a> · <a href="final_AUDIT.json">复算验收</a> · <a href="../FINAL_FREEZE.json">结果前冻结</a></p>
<section><h2>全部独立结果</h2><table><tr><th>设置</th>{''.join('<th>'+x+'</th>' for x in LABELS)}</tr>{rows}</table><p class="note">四组各32个新条件；同外观组32对物块。两个固定训练种子在条件内平均，没有按结果挑选模型。置信区间和误差见机器结果。</p><img src="success_rates.png"></section>
<section><h2>在线更新的归因</h2><img src="causal_controls.png"></section><section><h2>单次执行过程</h2><img src="within_episode.png"><p class="note">展示mask测量误差；正式成功由封存后的真实末步位置与速度判定。前三次探测也计入动作预算。</p></section>{videos}
<section><h2>结论边界</h2><p>所有控制器都使用模拟器实例mask、相机标定与已知顶面高度推导几何。普通适配器不读取物体真实位姿、阻力、接触力或成功标签。圆柱只作二维平移，没有自由转动或机械臂。</p><p>局部与整图视觉均为冻结DINOv3／LiLa及训练PCA16，视觉后果头结构一致；几何关系分支改变了反馈目标，不能把它的优势都算作视觉注意力收益。融合必须与关系单独比较，才能判断DINO是否提供额外帮助。</p></section></main></html>'''
    (OUT/'index.html').write_text(html)

if __name__=='__main__':main()
