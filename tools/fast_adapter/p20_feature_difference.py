"""Posthoc temporal-feature diagnosis using already saved DINO tensors only."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tools.fast_adapter.p18_run import save_json,digest

R=Path('evaluations/fast_adapter_p20_insert3d_v01')
S=R/'diagnostics/dinov3_features_v01'
O=R/'diagnostics/dinov3_temporal_difference_v01'

def main():
    torch.set_num_threads(2);O.mkdir(parents=True,exist_ok=False)
    d=torch.load(S/'FEATURES.pt',map_location='cpu',weights_only=False)
    features=d['patch_features'];pool=d['pooled_features'];basis=d['display_pca_basis']
    pairs=[(0,1),(1,2)];maps=[];rows=[];times=[0,14,45]
    pcas=torch.load(R/'bank/PCA.pt',map_location='cpu',weights_only=False)
    visual=torch.stack([(pool[:,i]-p['mean'])@p['basis']/p['sd'] for i,p in enumerate(pcas)],1)
    for i,j in pairs:
        delta=features[j]-features[i]
        cosine=(1-F.cosine_similarity(features[i],features[j],dim=-1)).clamp_min(0)
        maps.append(cosine)
        rows.append({'from_step':times[i],'to_step':times[j],
            'full_patch_cosine_distance_mean':cosine.mean().item(),'full_patch_cosine_distance_max':cosine.max().item(),
            'fraction_delta_energy_in_display_pca3':float((delta@basis).square().sum()/delta.square().sum()),
            'pooled_A_B_cosine_distance':(1-F.cosine_similarity(pool[i],pool[j],dim=-1)).tolist(),
            'runtime_whitened_A_B_feature_change_rms':(visual[j]-visual[i]).square().mean(-1).sqrt().tolist()})
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':12})
    fig,axs=plt.subplots(2,2,figsize=(9.5,8.8),layout='constrained');vmax=max(float(x.max()) for x in maps)
    for row,(i,j) in enumerate(pairs):
        axs[row,0].imshow(d['rgb'][j].numpy());axs[row,0].set_title(f'当前图像：t={times[j]}')
        hm=axs[row,1].imshow(maps[row],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest');axs[row,1].set_title(f'全部1024维特征变化：t={times[i]} → {times[j]}')
        for ax in axs[row]:ax.set_xticks([]);ax.set_yticks([])
    fig.colorbar(hm,ax=axs[:,1].tolist(),shrink=.8,label='同一图像位置patch的余弦距离，越亮变化越大')
    fig.suptitle('动作前后，DINOv3特征在哪里发生变化？',fontsize=17)
    fig.supxlabel('两张热图共用色标，使用全部1024维；这是同位置比较，未做物体运动对齐，不是精确定位能力证明。',fontsize=10)
    fig.savefig(O/'temporal_feature_difference.png',dpi=160);plt.close(fig)
    result={'scope':'one already-opened trajectory, three saved frames, posthoc descriptive only','source':str(S/'FEATURES.pt'),'source_sha256':digest(S/'FEATURES.pt'),'display_pca_variance':json.loads((S/'MANIFEST.json').read_text())['display_pca_explained_variance'],'transitions':rows,'A_vs_B_raw_cosine_by_frame':F.cosine_similarity(pool[:,0],pool[:,1],dim=-1).tolist(),'no_new_rollout_or_training':True,'interpretation_limits':['Display PCA variance is across all shown spatial patches; temporal-delta energy is a different quantity.','High visual similarity does not imply identical full features.','Changes may reflect location, visibility and context, not accurate millimetre geometry or dynamic identifiability.'],'figure_sha256':digest(O/'temporal_feature_difference.png')}
    save_json(O/'DIAGNOSTIC.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
