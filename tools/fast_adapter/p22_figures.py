"""Standalone scientific figures for accepted P22 v02 control results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

R=Path('evaluations/fast_adapter_p22_geometry_v02')

def main():
    result=json.loads((R/'summary/RESULTS.json').read_text())['strata']
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    strata=['id','position','scale'];labels=['同分布','槽位置变化','整体尺度变化']
    methods=[('base','视觉＋本体基座','#94a3b8'),('history','History','#67b7a5'),('u0','同模型 U0','#6c8acb'),('u3','同模型 U3','#e69350')]
    fig,ax=plt.subplots(figsize=(10.6,5.5));x=np.arange(3)
    for j,(arm,label,color) in enumerate(methods):
        values=[result[s]['rates_percent'][arm] for s in strata]
        bars=ax.bar(x+(j-1.5)*.18,values,width=.17,label=label,color=color)
        for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+1.2,f'{v:.1f}',ha='center',fontsize=10)
    ax.set_xticks(x,labels,fontproperties=font);ax.set_ylim(0,108);ax.set_ylabel('成功率（%）',fontproperties=font)
    ax.set_title('图像＋物块本体位置输入：新几何条件下的独立测试',fontproperties=font,fontsize=16)
    ax.legend(prop=font,loc='upper right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.1,.018,'每组32个独立条件；适配方法为3个模型种子的平均。真值参考三组均为100%，未画入主对照。',fontproperties=font,fontsize=10)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(R/'summary/success_rates.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    for i,s in enumerate(strata):
        q=result[s]['paired']['u3_minus_u0'];point=q['delta_pp'];lo,hi=q['ci95_pp']
        ax.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',capsize=5,color='#ba6c24',markersize=8)
    ax.axvline(0,color='#64748b',linestyle='--');ax.set_yticks(range(3),labels,fontproperties=font);ax.invert_yaxis()
    ax.set_xlabel('U3 − U0 成功率差（百分点），95%配对区间',fontproperties=font)
    ax.set_title('仅更新 latent 是否提供额外收益？',fontproperties=font,fontsize=15)
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(R/'summary/gradient_difference.png',dpi=180)
    print('FIGURES_COMPLETE')

if __name__=='__main__':main()
