"""Display only: plot already computed primary bootstrap intervals."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tools.fast_adapter.p27_common import R


def main():
    d=json.loads((R/'summary/RESULTS.json').read_text())['strata']
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),sharex=True,sharey=True)
    pairs=[('base','相对基座'),('refresh_matchedmean','相对固定平均z'),('refresh_donors','相对错配反馈')]
    for ax,(st,title) in zip(axes,[('id','同分布'),('position','槽位置变化'),('scale','整体尺度变化')]):
        ax.axvline(0,color='#66747b',linestyle='--',linewidth=1)
        for y,(key,label) in enumerate(pairs):
            r=d[st]['primary'][f'refresh_correct_minus_{key}'];x=r['delta_pp'];lo,hi=r['multiplicity_adjusted_ci'];a,b=r['paired_ci95'];color='#10796c' if r['conditional_positive_gate'] else '#8b8d96'
            ax.plot([lo,hi],[y,y],color=color,linewidth=1.6);ax.plot([a,b],[y,y],color=color,linewidth=5);ax.scatter([x],[y],color=color,s=45,zorder=3)
            ax.text(x,y-.16,f'{x:+.2f}',ha='center',fontsize=10,color=color)
        ax.set_title(title,fontproperties=font);ax.set_yticks(range(3),[l for _,l in pairs],fontproperties=font);ax.set_xlabel('成功率差（百分点）',fontproperties=font);ax.set_xlim(-15,32);ax.set_ylim(2.45,-.55);ax.grid(axis='x',alpha=.15)
    fig.suptitle('P27：正确反馈的额外收益｜粗线95%区间，细线为九项比较校正区间',fontproperties=font)
    fig.tight_layout();fig.savefig(R/'summary/attribution_intervals.png',dpi=175);plt.close(fig)

if __name__=='__main__':main()
