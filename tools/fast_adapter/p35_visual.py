"""Native image crops and frozen LiLa features at 224 or 448; no image upsampling arm."""
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tools.fast_adapter.p33_visual import RelationVisual

def own_box(p,cal,size):
    pos=np.array(cal['pos']);forward=np.array(cal['forward']);up=np.array(cal['up'])
    right=np.cross(forward,up);delta=np.asarray(p)[:3]-pos;depth=float(delta@forward)
    if depth<=.01:return None
    scale=(size/2)*cal['frustum_near']/cal['frustum_top']/depth
    center=np.array([size/2+scale*(delta@right),size/2-scale*(delta@up)])
    half=scale*.065;lo=np.maximum(np.floor(center-half),0).astype(int);hi=np.minimum(np.ceil(center+half),size).astype(int)
    if min(hi-lo)<4*size/224:return None
    return (*lo,*hi)

def crops_for(images,proprio,calibrations,task):
    size=images.shape[1];crops=[];boxes=[];presence=[];rects=[];pixel_counts=[]
    for im,p,cal in zip(images,proprio,calibrations):
        f=im.astype(np.float32)/255
        purple=(f[...,0]>.25)&(f[...,2]>.26)&(f[...,0]>f[...,1]*1.45)&(f[...,2]>f[...,1]*1.45)
        green=(f[...,1]>.25)&(f[...,1]>f[...,0]*1.5)&(f[...,1]>f[...,2]*1.6)
        masks=[None,purple] if task=='mug-handle-lift' else [purple,green]
        rr=[];bb=[];pp=[];cc=[];pix=[]
        for mask in masks:
            if mask is None:
                box=own_box(p,cal,size);tight=box;count=0 if box is None else (box[2]-box[0])*(box[3]-box[1])
            else:
                yy,xx=np.where(mask);count=len(xx)
                if count<8*(size/224)**2:box=tight=None
                else:
                    x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1;tight=(x0,y0,x1,y1)
                    pad=max(round(6*size/224),int(max(x1-x0,y1-y0)*.25))
                    box=(max(0,x0-pad),max(0,y0-pad),min(size,x1+pad),min(size,y1+pad))
            if box is None:
                rr.append(np.zeros_like(im));bb.append([0.]*4);pp.append(0.);cc.append((0,0,size,size));pix.append(count);continue
            x0,y0,x1,y1=tight
            rr.append(np.asarray(Image.fromarray(im).crop(box).resize((size,size),Image.Resampling.BILINEAR)))
            bb.append([(x0+x1)/size-1,(y0+y1)/size-1,(x1-x0)/size,(y1-y0)/size]);pp.append(1.);cc.append(box);pix.append(count)
        crops.append(rr);boxes.append(bb);presence.append(pp);rects.append(cc);pixel_counts.append(pix)
    return np.array(crops),np.array(boxes,np.float32),np.array(presence,np.float32),np.array(rects),np.array(pixel_counts)

class NativeVisual(RelationVisual):
    @torch.inference_mode()
    def encode_view(self,images,proprio,calibrations,task):
        crops,boxes,presence,rects,pixels=crops_for(images,proprio,calibrations,task)
        size=images.shape[1];grid=size//16;n=len(images);flat=crops.reshape(-1,size,size,3)
        ids=np.where(presence.reshape(-1)>0)[0];tokens=torch.zeros(n*2,16,51,device='cuda')
        for start in range(0,len(ids),8):
            ii=ids[start:start+8];pix=torch.tensor(flat[ii].copy(),device='cuda').permute(0,3,1,2).float()/255
            inp=((pix-self.front.mean)/self.front.std).to(torch.bfloat16)
            o=self.front.model(pixel_values=inp,output_hidden_states=True,return_dict=True)
            fused=self.front.fusion([o.hidden_states[i].float() for i in (-12,-8,-4)])
            queries=self.front.adapter(fused)
            purple=(pix[:,0]>.25)&(pix[:,2]>.26)&(pix[:,0]>pix[:,1]*1.45)&(pix[:,2]>pix[:,1]*1.45)
            green=(pix[:,1]>.25)&(pix[:,1]>pix[:,0]*1.5)&(pix[:,1]>pix[:,2]*1.6)
            even=torch.tensor(ii%2==0,device='cuda')[:,None,None]
            mask=torch.where(even,torch.ones_like(purple),purple) if task=='mug-handle-lift' else torch.where(even,purple,green)
            pm=F.adaptive_avg_pool2d(mask.float()[:,None],(grid,grid));den=F.adaptive_avg_pool2d(pm,(4,4))
            patches=(fused[:,self.front.prefix:]@self.spatial_projection).transpose(1,2).reshape(-1,32,grid,grid)
            sp=(F.adaptive_avg_pool2d(patches*pm,(4,4))/den.clamp_min(1e-5)).flatten(2).transpose(1,2)
            valid=(den>.01).flatten(2).transpose(1,2).float();ctx=(queries.mean(1)@self.context_projection)[:,None].expand(-1,16,-1)
            xy=[]
            for idx in ii:
                x0,y0,x1,y1=rects.reshape(-1,4)[idx]
                xy.append([[(x0+(x1-x0)*(x+.5)/4)*2/size-1,(y0+(y1-y0)*(y+.5)/4)*2/size-1]
                           for y in range(4) for x in range(4)])
            tokens[ii]=torch.cat((sp*valid,ctx*valid,torch.tensor(xy,device='cuda',dtype=torch.float32)*valid,valid),-1)
        return {'tokens':tokens.reshape(n,2,16,51),'boxes':torch.tensor(boxes,device='cuda'),
                'presence':torch.tensor(presence,device='cuda'),'pixels':torch.tensor(pixels,device='cuda')}
