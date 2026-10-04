"""Read frozen LiLa spatial fusion AND query context; queries are not a grid."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from tools.fast_adapter.p29_common import entities
from tools.fast_adapter.p33_common import P29,DINO,TASK
from prove.fast_adapter.lila_frontend import ThreeFrontends

class RelationVisual:
    def __init__(self):
        self.front=ThreeFrontends(DINO,microbatch=16,selected='lila')
        self.pca={k:v.cuda() if torch.is_tensor(v) else v for k,v in
                  torch.load(P29/'data/PCA.pt',weights_only=True).items()}
        gen=torch.Generator().manual_seed(330013)
        self.spatial_projection=(torch.randn(1024,32,generator=gen)/math.sqrt(1024)).cuda()
        self.context_projection=(torch.randn(768,16,generator=gen)/math.sqrt(768)).cuda()

    @torch.inference_mode()
    def encode(self,images):
        crops,boxes,presence=entities(images,TASK);n=len(images)
        flat=crops.reshape(-1,224,224,3);present=presence.reshape(-1)>0
        raw=torch.zeros(n*2,49152,device='cuda');tokens=torch.zeros(n*2,16,51,device='cuda')
        ids=np.where(present)[0]
        for start in range(0,len(ids),16):
            ii=ids[start:start+16];rgb=flat[ii]
            pix=torch.as_tensor(rgb.copy(),device='cuda').permute(0,3,1,2).float()/255
            inp=((pix-self.front.mean)/self.front.std).to(torch.bfloat16)
            o=self.front.model(pixel_values=inp,output_hidden_states=True,return_dict=True)
            fused=self.front.fusion([o.hidden_states[i].float() for i in (-12,-8,-4)])
            queries=self.front.adapter(fused);raw[ii]=queries.flatten(1)
            actor=(pix[:,0]>.25)&(pix[:,2]>.26)&(pix[:,0]>pix[:,1]*1.45)&(pix[:,2]>pix[:,1]*1.45)
            target=(pix[:,1]>.25)&(pix[:,1]>pix[:,0]*1.5)&(pix[:,1]>pix[:,2]*1.6)
            mask=torch.where(torch.tensor(ii%2==0,device='cuda')[:,None,None],actor,target).float()[:,None]
            pm=F.adaptive_avg_pool2d(mask,(14,14))
            patches=(fused[:,self.front.prefix:]@self.spatial_projection).transpose(1,2).reshape(-1,32,14,14)
            denom=F.adaptive_avg_pool2d(pm,(4,4))
            pooled=F.adaptive_avg_pool2d(patches*pm,(4,4))/denom.clamp_min(1e-5)
            valid=(denom>.01).flatten(2).transpose(1,2).float()
            sp=pooled.flatten(2).transpose(1,2)
            ctx=(queries.mean(1)@self.context_projection)[:,None].expand(-1,16,-1)
            xy=[]
            for idx in ii:
                cx,cy,w,h=boxes.reshape(-1,6)[idx,:4]
                x0=int(round(((cx+1)-w)*112));x1=int(round(((cx+1)+w)*112))
                y0=int(round(((cy+1)-h)*112));y1=int(round(((cy+1)+h)*112))
                pad=max(6,int(max(x1-x0,y1-y0)*.25))
                lo=np.maximum([x0-pad,y0-pad],0);hi=np.minimum([x1+pad,y1+pad],224)
                xy.append([((lo+(hi-lo)*np.array([(x+.5)/4,(y+.5)/4]))/112-1).tolist()
                           for y in range(4) for x in range(4)])
            pos=torch.tensor(xy,device='cuda',dtype=torch.float32)
            tokens[ii]=torch.cat((sp*valid,ctx*valid,pos*valid,valid),-1)
        raw=raw.reshape(n,2,-1);b=torch.tensor(boxes,device='cuda');p=torch.tensor(presence,device='cuda')
        v=((raw-self.pca['mean'])@self.pca['basis']/self.pca['sd'])*p[...,None]
        return {'visual':torch.cat((v,b,p[...,None]),-1).flatten(1),
                'tokens':tokens.reshape(n,2,16,51),'boxes':b[...,:4],'presence':p}
