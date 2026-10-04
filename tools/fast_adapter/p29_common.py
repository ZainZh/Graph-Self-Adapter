"""P29 RGB-only entity extraction and frozen visual encoder; task-local IO."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from prove.fast_adapter.object_interaction_env import TASKS,ObjectEnv,object_config
R=Path('evaluations/fast_adapter_p29_object_transfer_v01')
HORIZON=200;SUPPORT=160;BLOCK=8;SEEDS=(11,22)
def save_json(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def weight_hash(net):
    h=hashlib.sha256()
    for k,v in sorted(net.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def task_root(task):assert task in TASKS;return R/'tasks'/task

def entities(images,task):
    """Fixed color-supported RGB segmentation, no masks/depth/poses from simulator.
    Appearance generalization beyond these color markers is explicitly not claimed.
    """
    imgs=np.asarray(images);f=imgs.astype(np.float32)/255
    masks=[(f[...,0]>.25)&(f[...,2]>.26)&(f[...,0]>f[...,1]*1.45)&(f[...,2]>f[...,1]*1.45)]
    masks.append((f[...,1]>.25)&(f[...,1]>f[...,0]*1.5)&(f[...,1]>f[...,2]*1.6))
    if task==TASKS[0]:masks[1]=np.zeros_like(masks[1])
    crops=[];meta=[];present=[]
    for im,a,b in zip(imgs,*masks):
        row=[];ms=[];ps=[]
        for mask in (a,b):
            yy,xx=np.where(mask)
            if len(xx)<8:
                row.append(np.zeros((224,224,3),np.uint8));ms.append(np.zeros(6,np.float32));ps.append(0.)
            else:
                x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1
                w,h=x1-x0,y1-y0;pad=max(6,int(max(w,h)*.25))
                box=(max(0,x0-pad),max(0,y0-pad),min(224,x1+pad),min(224,y1+pad))
                row.append(np.array(Image.fromarray(im).crop(box).resize((224,224),Image.Resampling.BILINEAR)))
                ms.append(np.array([(x0+x1)/224-1,(y0+y1)/224-1,w/224,h/224,len(xx)/(224*224),len(xx)/(w*h)],np.float32));ps.append(1.)
        crops.append(row);meta.append(ms);present.append(ps)
    return np.array(crops),np.array(meta),np.array(present,dtype=np.float32)

class Visual:
    def __init__(self,dino,task,pca=None):
        from tools.fast_adapter.p22_data import RGBEncoder
        self.net=RGBEncoder(dino);self.task=task
        self.pca=None if pca is None else {k:v.cuda() if torch.is_tensor(v) else v for k,v in pca.items()}
    def encode(self,images):
        crops,boxes,presence=entities(images,self.task);flat=crops.reshape(-1,224,224,3);valid=presence.reshape(-1)>0
        raw=torch.zeros(len(flat),49152,device='cuda')
        if valid.any():raw[torch.as_tensor(valid,device='cuda')]=self.net.encode(flat[valid])
        raw=raw.reshape(len(images),2,-1);box=torch.as_tensor(boxes,device='cuda');mask=torch.as_tensor(presence,device='cuda')
        if self.pca is None:return {'features':raw,'boxes':box,'presence':mask}
        val=((raw-self.pca['mean'])@self.pca['basis']/self.pca['sd'])*mask[...,None]
        return torch.cat((val,box,mask[...,None]),-1).flatten(1)

def load_base(task,round_name='base_final'):
    from prove.fast_adapter.object_interaction_adapter import ObjectSkill
    net=ObjectSkill().cuda();blob=torch.load(task_root(task)/'models'/f'{round_name}.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(blob['state']);return net.eval().requires_grad_(False)
def load_adapter(task,seed,variant,round_name='r1'):
    from prove.fast_adapter.object_interaction_adapter import ObjectAdapter
    blob=torch.load(task_root(task)/'models'/round_name/f'seed{seed}'/f'{variant}.pt',map_location='cpu',weights_only=True)
    net=ObjectAdapter(history=variant=='history',feedback_mode='robot' if variant=='robot' else 'object').cuda()
    net.load_state_dict(blob['state']);return net.eval().requires_grad_(False),blob
