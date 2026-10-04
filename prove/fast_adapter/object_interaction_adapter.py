"""P29 two-entity tokens and object-transition latent meta-adaptation.
No simulator/privileged labels are imported here. Absent slots are strictly masked.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

VISUAL_DIM=142
FEEDBACK_DIM=28

def mlp(n,out,width=256):
    return nn.Sequential(nn.Linear(n,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,out))

def mask_visual(v):
    s=v.reshape(*v.shape[:-1],2,71)
    return torch.cat((s[...,:70]*s[...,70:71],s[...,70:71]),-1).flatten(-2)

def inputs(v,p,prev):
    p=(p-p.new_tensor([0,.65,.16,.5]))/p.new_tensor([.2,.2,.2,.5])
    return torch.cat((mask_visual(v),p,prev),-1)

def feedback(v,p,nv,np_):
    s=v.reshape(*v.shape[:-1],2,71);ns=nv.reshape(*nv.shape[:-1],2,71)
    valid=s[...,70:71]*ns[...,70:71]
    # First 8 training-PCA feature changes plus RGB-derived box changes per role.
    visual=torch.cat((ns[...,:8]-s[...,:8],40*(ns[...,64:68]-s[...,64:68])),-1)
    visual=(visual*valid).flatten(-2)
    own=(np_-p)/p.new_tensor([.01,.01,.01,.1])
    target=torch.cat((visual,own*.2),-1)
    weights=torch.cat((valid.expand(*valid.shape[:-1],12).flatten(-2),torch.ones_like(own)), -1)
    return target,weights

class ObjectSkill(nn.Module):
    def __init__(self):super().__init__();self.net=mlp(150,4)
    def forward(self,v,p,prev):return self.net(inputs(v,p,prev)).clamp(-1,1)

class ObjectAdapter(nn.Module):
    def __init__(self,history=False,feedback_mode='object'):
        super().__init__();self.history=history;self.feedback_mode=feedback_mode
        self.encoder=nn.Sequential(mlp(150,128,128),nn.LayerNorm(128,elementwise_affine=False))
        self.actor=mlp(164,4,128);self.predictor=mlp(164,FEEDBACK_DIM,128)
        self.z0=nn.Parameter(torch.zeros(32));self.raw_alpha=nn.Parameter(torch.tensor(math.log(math.expm1(.05))))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        nn.init.normal_(self.predictor[-1].weight,std=.01);nn.init.zeros_(self.predictor[-1].bias)
        if history:
            self.token=nn.Sequential(nn.Linear(128+4+28,128),nn.SiLU());self.gru=nn.GRU(128,32,batch_first=True)
            self.raw_alpha.requires_grad_(False)
    @property
    def alpha(self):return F.softplus(self.raw_alpha)
    def encode(self,v,p,prev):return self.encoder(inputs(v,p,prev))
    def prediction(self,u,a,z):return self.predictor(torch.cat((u,a,z),-1))
    def action(self,v,p,prev,base,z,correction_scale=0.5):
        u=self.encode(v,p,prev)
        val=self.actor(torch.cat((u,base,z),-1));prior=self.actor(torch.cat((u,base,self.z0.expand_as(z)),-1))
        return (base+correction_scale*torch.tanh(val-prior)).clamp(-1,1)
    def adapt(self,s,steps=3,meta=False):
        with torch.set_grad_enabled(meta or self.history):
            u=self.encode(s['visual'],s['proprio'],s['previous']);a=s['actions']
            target,weights=feedback(s['visual'],s['proprio'],s['next_visual'],s['next_proprio'])
            if self.feedback_mode=='robot':weights=weights.clone();weights[...,:24]=0
            z=self.z0.unsqueeze(0).expand(len(u),-1)
        if self.history:
            _,h=self.gru(self.token(torch.cat((u,a,target*weights),-1)),z.unsqueeze(0).contiguous())
            return h[0],[]
        curve=[]
        for _ in range(steps):
            with torch.enable_grad():
                if not meta:z=z.detach().requires_grad_(True);u=u.detach()
                pred=self.prediction(u,a,z[:,None].expand(-1,u.shape[1],-1))
                loss=64*((pred-target.detach()).square()*weights.detach()).sum((1,2))/weights.sum((1,2)).clamp_min(1)
                grad,=torch.autograd.grad(loss.sum(),z,create_graph=meta)
                z=z-self.alpha*grad;curve.append(loss.detach())
            if not meta:z=z.detach()
        return z,curve
