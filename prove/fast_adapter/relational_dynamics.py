"""Action-conditioned visual effects with an episode-local, gradient-updated z.

No simulator or expert imports. All targets are fixed observed visual features.
Cross/within variants have EXACTLY the same parameters and public inputs.
"""
import math
import torch
from torch import nn

HORIZON=4
EFFECT_DIM=268

def mlp(n,m,width=128):
    return nn.Sequential(nn.Linear(n,width),nn.SiLU(),nn.Linear(width,m))

def observed_effect(tokens,boxes,proprio,next_tokens,next_boxes,next_proprio):
    a=tokens[...,:8].flatten(-3);b=next_tokens[...,:8].flatten(-3)
    y=torch.cat((b-a,(next_boxes-boxes).flatten(-2),next_proprio-proprio),-1)
    valid=tokens[...,-1]*next_tokens[...,-1]
    spatial=valid[...,None].expand(*valid.shape,8).flatten(-3)*(.25/256)
    present=(tokens[...,-1].sum(-1)>0)&(next_tokens[...,-1].sum(-1)>0)
    box=present[...,None].expand(*present.shape,4).flatten(-2).float()*(.5/8)
    own=torch.ones_like(proprio)*(.25/4)
    return y.detach(),torch.cat((spatial,box,own),-1).detach()

class RelationalDynamics(nn.Module):
    def __init__(self,family='cross'):
        super().__init__()
        if family not in ('cross','within'):raise ValueError(family)
        self.family=family
        self.token=nn.Linear(50,32);self.role=nn.Parameter(torch.randn(2,32)*.01)
        self.query=nn.Linear(32,32);self.key=nn.Linear(32,32);self.value=nn.Linear(32,32)
        self.post=nn.Sequential(nn.Linear(32,32),nn.SiLU(),nn.LayerNorm(32))
        self.encoder=mlp(88,64)
        self.core=mlp(80,128)
        self.mean=nn.Linear(128,EFFECT_DIM);self.gate=nn.Linear(128,32)
        self.basis=nn.Parameter(torch.randn(EFFECT_DIM,32)/math.sqrt(32))
        self.z0=nn.Parameter(torch.zeros(32))
        self.raw_alpha=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        self.register_buffer('token_mean',torch.zeros(50));self.register_buffer('token_sd',torch.ones(50))
        self.register_buffer('effect_sd',torch.ones(EFFECT_DIM))
        self.register_buffer('direction',torch.tensor([1.,0.]));self.register_buffer('progress_sd',torch.tensor(1.))

    def set_stats(self,s):
        for k in ('token_mean','token_sd','effect_sd','direction','progress_sd'):
            getattr(self,k).copy_(s[k])

    @property
    def alpha(self):return torch.nn.functional.softplus(self.raw_alpha)

    def encode(self,tokens,boxes,proprio,previous,box_velocity):
        shape=tokens.shape[:-3];valid=tokens[...,-1].reshape(-1,2,16)
        x=((tokens[...,:50]-self.token_mean)/self.token_sd).reshape(-1,2,16,50)
        x=self.token(x)+self.role[None,:,None]
        q=self.query(x);k=self.key(x);v=self.value(x)
        if self.family=='cross':k=k.flip(1);v=v.flip(1);kv_valid=valid.flip(1)
        else:kv_valid=valid
        scores=torch.einsum('beri,besi->bers',q,k)/math.sqrt(32)
        scores=scores.masked_fill(kv_valid[:,:,None,:]<=0,-1e4)
        context=torch.einsum('bers,besi->beri',scores.softmax(-1),v*kv_valid[...,None])
        h=self.post(x+context)*valid[...,None]
        pooled=h.sum(2)/valid.sum(2).clamp_min(1)[...,None]
        p=(proprio-proprio.new_tensor([0,.65,.16,.5]))/proprio.new_tensor([.2,.2,.2,.5])
        return self.encoder(torch.cat((pooled.reshape(*shape,64),boxes.flatten(-2),p,previous,
                                        box_velocity.flatten(-2)*20),-1))

    def coefficients(self,u,plans):
        c=self.core(torch.cat((u,plans.flatten(-2)),-1))
        return self.mean(c),2*torch.sigmoid(self.gate(c))

    def predict_coeff(self,mean,gate,z):
        return mean+(gate*z)@self.basis.T

    def loss(self,pred,y,w):
        return ((pred-y).square()*w).sum(-1)/w.sum(-1).clamp_min(1e-8)

    def update(self,z,mean,gate,y,w):
        """Exact gradient of 16 * weighted window MSE plus the prior penalty.

        Written explicitly to preserve efficient meta-gradients through z updates.
        A unit test checks this against autograd; no target enters prediction.
        """
        pred=self.predict_coeff(mean,gate,z[:,None])
        denom=w.sum((-1,-2)).clamp_min(1e-8)
        grad=32*((((pred-y)*w)@self.basis)*gate).sum(1)/denom[:,None]
        grad=grad+.002*(z-self.z0)
        norm=grad.norm(dim=-1,keepdim=True)
        step=grad*(5/norm.clamp_min(5))
        nz=z-self.alpha*step
        delta=nz-self.z0;radius=delta.norm(dim=-1,keepdim=True)
        nz=self.z0+delta*(5/radius.clamp_min(5))
        return nz,{'loss':(((pred-y).square()*w).sum((-1,-2))/denom).detach(),
                   'grad_norm':norm.detach().squeeze(-1),'delta_norm':(nz-z).detach().norm(dim=-1)}

def action_candidates(base):
    v=base.new_tensor([-.25,0.,.25])
    offsets=torch.cartesian_prod(v,v,v)
    a=base[:,None].expand(-1,len(offsets),-1).clone()
    a[...,:3]=(a[...,:3]+offsets).clamp(-1,1);a[...,3]=.9
    return a

def select_action(net,u,base,z):
    candidates=action_candidates(base)
    plans=candidates[:,:,None].expand(-1,-1,4,-1)
    mean,gate=net.coefficients(u[:,None].expand(-1,27,-1),plans)
    pred=net.predict_coeff(mean,gate,z[:,None])*net.effect_sd
    progress=(pred[...,260:262]*net.direction).sum(-1)/net.progress_sd
    penalty=.2*(candidates[...,:3]-base[:,None,:3]).square().sum(-1)
    scores=progress-penalty
    chosen=scores.argmax(-1);a=candidates[torch.arange(len(base),device=base.device),chosen]
    return a,chosen,scores
