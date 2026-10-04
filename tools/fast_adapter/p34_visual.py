"""Frozen vision with role assignment. Camera + own EEF locate gripper, not object truth."""
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from tools.fast_adapter.p33_visual import RelationVisual
from tools.fast_adapter.p29_common import entities
from tools.fast_adapter.p34_common import ROOT

def calibration(env):
    env.rgb();cs=env.renderer.scene.camera
    return {k:np.mean([np.array(getattr(c,k)) for c in cs],axis=0).tolist()
            for k in ('pos','forward','up','frustum_bottom','frustum_top','frustum_center','frustum_near')}

def gripper_box(proprio,cal):
    p=np.array(cal['pos']);f=np.array(cal['forward']);up=np.array(cal['up']);right=np.cross(f,up)
    delta=np.asarray(proprio)[:3]-p;depth=float(delta@f)
    if depth<=.01:return None
    top=cal['frustum_top'];near=cal['frustum_near'];scale=112*near/top/depth
    center=np.array([112+scale*(delta@right),112-scale*(delta@up)])
    half=scale*.065 # fixed robot finger extent, not object-size supervision
    lo=np.maximum(np.floor(center-half),0).astype(int);hi=np.minimum(np.ceil(center+half),224).astype(int)
    if min(hi-lo)<4:return None
    return (*lo,*hi)

class RoleVisual(RelationVisual):
    def __init__(self,task,cal=None):
        super().__init__();self.task=task;self.cal=cal
        if task=='mug-handle-lift':
            from pathlib import Path
            p=Path('evaluations/fast_adapter_p29_object_transfer_v01/tasks')/task/'data/PCA.pt'
            self.pca={k:v.cuda() if torch.is_tensor(v) else v for k,v in torch.load(p,weights_only=True).items()}

    @torch.inference_mode()
    def encode(self,images,proprio=None):
        if self.task=='hammer-block':return super().encode(images)
        crops,meta,presence=entities(images,self.task);n=len(images)
        # New roles [gripper,cup]; frozen BASE keeps historical [cup,null].
        crops[:,1]=crops[:,0];meta[:,1]=meta[:,0];presence[:,1]=presence[:,0]
        rois=[]
        for j,(im,p) in enumerate(zip(images,proprio)):
            box=gripper_box(p,self.cal);presence[j,0]=box is not None
            if box is None:crops[j,0]=0;meta[j,0]=0;rois.append((0,0,224,224));continue
            x0,y0,x1,y1=box;rois.append(box)
            crops[j,0]=np.asarray(Image.fromarray(im).crop(box).resize((224,224),Image.Resampling.BILINEAR))
            meta[j,0]=[(x0+x1)/224-1,(y0+y1)/224-1,(x1-x0)/224,(y1-y0)/224,1.,1.]
        flat=crops.reshape(-1,224,224,3);ids=np.where(presence.reshape(-1)>0)[0]
        raw=torch.zeros(n*2,49152,device='cuda');tokens=torch.zeros(n*2,16,51,device='cuda')
        for start in range(0,len(ids),16):
            ii=ids[start:start+16];pix=torch.tensor(flat[ii].copy(),device='cuda').permute(0,3,1,2).float()/255
            inp=((pix-self.front.mean)/self.front.std).to(torch.bfloat16)
            o=self.front.model(pixel_values=inp,output_hidden_states=True,return_dict=True)
            fused=self.front.fusion([o.hidden_states[i].float() for i in (-12,-8,-4)])
            queries=self.front.adapter(fused);raw[ii]=queries.flatten(1)
            cup=(pix[:,0]>.25)&(pix[:,2]>.26)&(pix[:,0]>pix[:,1]*1.45)&(pix[:,2]>pix[:,1]*1.45)
            mask=torch.where(torch.tensor(ii%2==0,device='cuda')[:,None,None],torch.ones_like(cup),cup).float()[:,None]
            pm=F.adaptive_avg_pool2d(mask,(14,14));den=F.adaptive_avg_pool2d(pm,(4,4))
            patches=(fused[:,self.front.prefix:]@self.spatial_projection).transpose(1,2).reshape(-1,32,14,14)
            sp=(F.adaptive_avg_pool2d(patches*pm,(4,4))/den.clamp_min(1e-5)).flatten(2).transpose(1,2)
            valid=(den>.01).flatten(2).transpose(1,2).float()
            ctx=(queries.mean(1)@self.context_projection)[:,None].expand(-1,16,-1);xy=[]
            for idx in ii:
                if idx%2==0:x0,y0,x1,y1=rois[idx//2]
                else:
                    cx,cy,w,h=meta[idx//2,1,:4];x0=int(round(((cx+1)-w)*112));x1=int(round(((cx+1)+w)*112))
                    y0=int(round(((cy+1)-h)*112));y1=int(round(((cy+1)+h)*112));pad=max(6,int(max(x1-x0,y1-y0)*.25))
                    x0,y0=max(0,x0-pad),max(0,y0-pad);x1,y1=min(224,x1+pad),min(224,y1+pad)
                xy.append([[(x0+(x1-x0)*(x+.5)/4)/112-1,(y0+(y1-y0)*(y+.5)/4)/112-1]
                           for y in range(4) for x in range(4)])
            tokens[ii]=torch.cat((sp*valid,ctx*valid,torch.tensor(xy,device='cuda',dtype=torch.float32)*valid,valid),-1)
        raw=raw.reshape(n,2,-1);b=torch.tensor(meta,device='cuda');p=torch.tensor(presence,device='cuda')
        cupv=((raw[:,1]-self.pca['mean'])@self.pca['basis']/self.pca['sd'])*p[:,1,None]
        basecup=torch.cat((cupv,b[:,1],p[:,1,None]),-1)
        return {'tokens':tokens.reshape(n,2,16,51),'boxes':b[...,:4],'presence':p,
                'visual':torch.cat((basecup,torch.zeros_like(basecup)),-1)}
