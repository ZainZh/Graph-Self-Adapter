"""Posthoc DINOv3 feature figures from frozen P20 trajectory states, no rollouts."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import torch.nn.functional as F
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image
from prove.fast_adapter.insert3d import Insert3D
from prove.fast_adapter.visual_alignment import FrozenDinoROI,DEFAULT_MODEL,DINO_SHA256
from tools.fast_adapter.p18_run import digest,save_json

R=Path('evaluations/fast_adapter_p20_insert3d_v01')
O=R/'diagnostics/dinov3_features_v01'

def main():
    O.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    model_path=Path(DEFAULT_MODEL);weight=model_path/'model.safetensors'
    sha=digest(weight);assert sha==DINO_SHA256
    source=R/'final/train/seed11/hybrid_u3_RAW.pt';record=torch.load(source,weights_only=False,map_location='cpu')
    idx=7;times=[0,14,45];env=Insert3D();images=[];boxes=[];counts=[]
    for t in times:
        env.reset(record['positions'][idx,t]);env.data.qvel[:]=record['velocities'][idx,t];mujoco.mj_forward(env.model,env.data)
        rgb,box,count=env.observe();images.append(rgb);boxes.append(box);counts.append(count)
        Image.fromarray(rgb).save(O/f'rgb_t{t:02d}.png')
    env.close();rgb=np.stack(images);boxes=np.stack(boxes)
    enc=FrozenDinoROI(DEFAULT_MODEL,device='cuda',microbatch=8)
    with torch.inference_mode():
        pixels=torch.tensor(rgb,device='cuda').permute(0,3,1,2).float()/255
        pixels=((pixels-enc.mean)/enc.std).to(torch.bfloat16)
        hidden=enc.model(pixel_values=pixels).last_hidden_state
        patch=hidden[:,enc.prefix:].float().reshape(3,14,14,1024).cpu()
    patch=patch.clone()
    # The exact area-weighted bounding-box pooling used by the actual encoder.
    b=torch.tensor(boxes);lo=torch.arange(14)*16
    wx=(torch.minimum(b[:,:,2,None],lo+16)-torch.maximum(b[:,:,0,None],lo)).clamp(min=0)
    wy=(torch.minimum(b[:,:,3,None],lo+16)-torch.maximum(b[:,:,1,None],lo)).clamp(min=0)
    weights=wy[:,:,:,None]*wx[:,:,None,:]
    pooled=torch.einsum('bryx,byxc->brc',weights,patch)/weights.sum((-1,-2))[...,None]
    actual=enc.encode(rgb,boxes).cpu();maxerr=float((actual-pooled).abs().max())
    torch.testing.assert_close(actual,pooled,atol=2e-5,rtol=2e-5)
    # A shared PCA across these three frames is solely a display projection.
    flat=patch.flatten(0,2);center=flat.mean(0);u,sv,vh=torch.linalg.svd(flat-center,full_matrices=False)
    basis=vh[:3].T;projected=(flat-center)@basis
    lower=torch.quantile(projected,.01,dim=0);upper=torch.quantile(projected,.99,dim=0)
    color=((projected-lower)/(upper-lower).clamp_min(1e-8)).clamp(0,1).reshape(3,14,14,3).numpy()
    sim=torch.einsum('tyxc,rc->tryx',F.normalize(patch,dim=-1),F.normalize(pooled[0],dim=-1)).numpy()
    used=(weights>0).sum((-1,-2)).tolist();roi_area=weights.sum((-1,-2)).tolist()
    torch.save({'patch_features':patch,'pooled_features':pooled,'roi_boxes':b,'rgb':torch.tensor(rgb),'display_pca_mean':center,'display_pca_basis':basis,'display_pca_lower':lower,'display_pca_upper':upper,'cosine_to_first_frame_rois':torch.tensor(sim)},O/'FEATURES.pt')
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11})
    fig,axs=plt.subplots(3,4,figsize=(14.3,11.3),layout='constrained')
    titles=['输入RGB与实际ROI框','DINOv3 patch特征：PCA伪彩','与物块A首帧特征的相似度','与槽B首帧特征的相似度']
    stage=['初态 · t=0','对齐后 · t=14','插入完成 · t=45']
    for row,t in enumerate(times):
        axs[row,0].imshow(rgb[row]);axs[row,1].imshow(color[row],interpolation='nearest',extent=(0,224,224,0))
        for obj,c in [(0,'#ff513b'),(1,'#0be5e8')]:
            x0,y0,x1,y1=boxes[row,obj]
            axs[row,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=c,linewidth=1.6))
            axs[row,0].text(x0,max(y0-3,9),'A' if obj==0 else 'B',color=c,weight='bold',fontsize=11)
        for col in [2,3]:
            axs[row,col].imshow(sim[row,col-2],cmap='viridis',vmin=-1,vmax=1,interpolation='nearest',extent=(0,224,224,0))
        for col in range(4):
            ax=axs[row,col];ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=12,pad=12)
        axs[row,0].set_ylabel(stage[row]+'\nA可见像素：'+str(counts[row][0]),fontsize=11)
        axs[row,1].set_xlabel(f'A框汇聚 {used[row][0]} 个patch；A可见面积约 {counts[row][0]/256:.2f} 个patch',fontsize=9)
    sm=plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(-1,1));fig.colorbar(sm,ax=axs[:,2:].ravel().tolist(),shrink=.88,label='余弦相似度（不是注意力或成功概率）')
    fig.suptitle('实际冻结 DINOv3 ViT-L/16 特征｜单相机 224×224 → 14×14 patch，每patch 1024维',fontsize=16)
    fig.supxlabel('三帧共用一套显示PCA与色标；伪彩仅帮助观察特征差异，颜色不是物体标签。既有测试轨迹重绘，无新训练或物理执行。',fontsize=10)
    fig.savefig(O/'dinov3_feature_overview.png',dpi=170);plt.close(fig)
    # A larger single-frame artifact for inspecting coarse patches and ROI overlap.
    row=1;fig,axs=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
    axs[0].imshow(rgb[row]);axs[0].set_title('全局RGB：红框A，青框B')
    for obj,c in [(0,'#ff513b'),(1,'#0be5e8')]:
        x0,y0,x1,y1=boxes[row,obj];axs[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=c,linewidth=2))
    axs[1].imshow(color[row],interpolation='nearest',extent=(0,224,224,0));axs[1].set_title('实际14×14空间特征，PCA伪彩')
    axs[2].imshow(rgb[row]);axs[2].set_title('两个物体的实际汇聚权重')
    overlay=np.zeros((14,14,4),dtype=np.float32);wa=weights[row,0].numpy()/256;wb=weights[row,1].numpy()/256
    overlay[...,0]=wa;overlay[...,1]=wb;overlay[...,2]=wb;overlay[...,3]=np.maximum(wa,wb)*.65
    axs[2].imshow(overlay,interpolation='nearest',extent=(0,224,224,0))
    for ax in axs:
        ax.set_xticks(np.arange(0,225,16),minor=True);ax.set_yticks(np.arange(0,225,16),minor=True);ax.grid(which='minor',alpha=.18);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('对齐后的真实输入与局部特征提取方式',fontsize=15)
    fig.supxlabel('先编码整张图，再按GT框汇聚patch；框内含背景，两个框可能重叠。当前没有分别裁剪放大两个物体。',fontsize=11)
    fig.savefig(O/'dinov3_roi_detail.png',dpi=170);plt.close(fig)
    meta={'kind':'posthoc_feature_display_only','model_dir':str(model_path),'model_architecture':enc.model.__class__.__name__,'model_weights_sha256':sha,'weights_frozen':all(not p.requires_grad for p in enc.model.parameters()),'source_trajectory':str(source),'source_trajectory_sha256':digest(source),'domain_index':idx,'control_steps':times,'raw_shape':list(patch.shape),'roi_feature_shape':list(pooled.shape),'visible_pixels':counts,'roi_intersecting_patch_counts':used,'roi_area_pixels':roi_area,'actual_pooling_recomputation_max_error':maxerr,'display_pca_explained_variance':float(sv[:3].square().sum()/sv.square().sum()),'runtime_pca_distinct':'Runtime PCA32/object is fit on training pooled features. Display PCA3 is separately fit on all shown patch vectors only and never fed to the model.','similarity':'Each patch cosine to the area-weighted first-frame A/B pooled feature. Not attention or confidence.','runtime_input':'whole-image last-layer spatial tokens -> GT-box weighted average per object -> training-fit PCA32 per object ->64D visual representation','no_new_training_or_rollout':True,'artifacts':[{ 'path':str(p),'sha256':digest(p)} for p in sorted(O.glob('*.png'))]}
    save_json(O/'MANIFEST.json',meta)
    (O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P20 DINOv3实际特征</title><style>body{font:17px/1.7 sans-serif;max-width:1300px;margin:30px auto;padding:0 20px}img{width:100%}</style><h1>实际冻结DINOv3特征</h1><p>ViT-L/16，LVD-1689M预训练权重；224×224输入，14×14空间patch，每patch1024维。</p><img src="dinov3_roi_detail.png"><img src="dinov3_feature_overview.png"><p>伪彩来自显示用PCA；相似度不是注意力、分割标签或成功率。图为既有封存轨迹重绘，未新增训练或rollout。</p><a href="MANIFEST.json">权重、来源和可复核说明</a>')
    print(json.dumps(meta,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
