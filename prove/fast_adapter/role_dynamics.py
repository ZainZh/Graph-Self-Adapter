"""P34 role-conditioned interaction states; runtime receives public observations only."""
import math
import torch
from torch import nn
from prove.fast_adapter.relational_dynamics import RelationalDynamics

FAMILIES=('flat','role_attention','relation_mlp','relation_attention')
EFFECT_DIM=134

def relative(boxes):
    return (boxes[...,1,:2]-boxes[...,0,:2])/boxes[...,1,2:].clamp_min(.03)

def observed_effect(tokens,boxes,proprio,next_tokens,next_boxes,next_proprio):
    # The actor's motion is input, never a prediction target.
    old=tokens[...,1,:,:8];new=next_tokens[...,1,:,:8]
    y=torch.cat(((new-old).flatten(-2),next_boxes[...,1,:]-boxes[...,1,:],
                 relative(next_boxes)-relative(boxes)),-1)
    mask=tokens[...,-1]*next_tokens[...,-1]
    sm=mask[...,1,:,None].expand(*mask.shape[:-2],16,8).flatten(-2)
    visible=mask.sum(-1)>0
    bm=visible[...,1,None].expand(*visible.shape[:-1],4).float()
    rm=visible.all(-1)[...,None].expand(*visible.shape[:-1],2).float()
    groups=[]
    for m,g in ((sm,.5),(bm,.25),(rm,.25)):
        groups.append(g*m/m.sum(-1,keepdim=True).clamp_min(1))
    w=torch.cat(groups,-1)
    return y.detach(),w.detach()

class Attention(nn.Module):
    def __init__(self,width,edge=False):
        super().__init__();self.edge=edge
        self.qkv=nn.Linear(256,768);self.out=nn.Linear(256,256)
        self.n1=nn.LayerNorm(256);self.n2=nn.LayerNorm(256)
        self.ff=nn.Sequential(nn.Linear(256,width),nn.GELU(),nn.Linear(width,256))
    def forward(self,x,available):
        b,l,_=x.shape
        q,k,v=self.qkv(self.n1(x)).reshape(b,l,3,4,64).permute(2,0,3,1,4)
        scores=q@k.transpose(-1,-2)/8
        permitted=available[:,None,None,:].expand(b,4,l,l).clone()
        if self.edge:
            topology=torch.eye(5,device=x.device,dtype=torch.bool)
            topology[3,[0,1]]=True;topology[4,[1,2]]=True
            permitted=permitted&topology[None,None]
        context=(scores.masked_fill(~permitted,-1e4).softmax(-1)@(v*available[:,None,:,None]))
        x=x+self.out(context.transpose(1,2).reshape(b,l,256))
        return (x+self.ff(self.n2(x)))*available[...,None]

class RoleDynamics(RelationalDynamics):
    def __init__(self,family='flat',task='hammer-block'):
        nn.Module.__init__(self)
        if family not in FAMILIES:raise ValueError(family)
        self.family=family;self.task=task
        # All arms receive the identical 16 spatial tokens, bbox and own state.
        self.role_project=nn.Linear(16*50+9,256)
        self.proprio_project=nn.Linear(8,256)
        self.goal_visual=nn.Linear(16*50+9,256)
        self.goal_semantic=nn.Parameter(torch.randn(256)*.01)
        self.roles=nn.Parameter(torch.randn(3,256)*.01)
        self.types=nn.Parameter(torch.randn(2,256)*.01)
        if family=='flat':
            self.mix=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,1024),nn.GELU(),nn.Linear(1024,256))
        elif family=='role_attention':self.mix=Attention(512)
        elif family=='relation_mlp':
            self.mix=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Linear(256,256))
            self.edge=nn.Sequential(nn.Linear(520,512),nn.GELU(),nn.Linear(512,256))
        else:
            self.edge=nn.Linear(520,256);self.mix=Attention(256,edge=True)
        self.readout=nn.Sequential(nn.Linear(792,256),nn.SiLU(),nn.LayerNorm(256))
        self.core=nn.Sequential(nn.Linear(272,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        self.mean=nn.Linear(128,EFFECT_DIM);self.gate=nn.Linear(128,32)
        self.basis=nn.Parameter(torch.randn(EFFECT_DIM,32)/math.sqrt(32))
        self.z0=nn.Parameter(torch.zeros(32));self.raw_alpha=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        self.register_buffer('token_mean',torch.zeros(50));self.register_buffer('token_sd',torch.ones(50))
        self.register_buffer('effect_sd',torch.ones(EFFECT_DIM));self.register_buffer('direction',torch.tensor([1.,0.]))
        self.register_buffer('progress_sd',torch.tensor(1.))

    def encode(self,tokens,boxes,proprio,previous,box_velocity,goal=None):
        shape=tokens.shape[:-3];n=math.prod(shape)
        mask=tokens[...,-1:];valid=mask.sum(-2).squeeze(-1)>0
        x=((tokens[...,:50]-self.token_mean)/self.token_sd)*mask
        inp=torch.cat((x.flatten(-2),boxes,box_velocity*20,valid[...,None]),-1)
        r=self.role_project(inp).reshape(n,2,256)
        p=(proprio-proprio.new_tensor([0,.65,.16,.5]))/proprio.new_tensor([.2,.2,.2,.5])
        own=torch.cat((p,previous),-1).reshape(n,8)
        r=torch.stack((r[:,0]+self.proprio_project(own),r[:,1]),1)
        # Semantic goal EXISTS without pixels. A hidden visual reference is distinct.
        if goal is None:
            g=self.goal_semantic[None].expand(n,-1)+self.types[0]
            gavail=torch.ones(n,device=tokens.device,dtype=torch.bool)
        else:
            gv=goal['visual'];gmask=gv[...,-1:]
            gx=((gv[...,:50]-self.token_mean)/self.token_sd)*gmask
            gp=goal['boxes'];availability=goal['visible'].bool()&goal['exists'].bool()
            gi=torch.cat((gx.flatten(-2),gp,torch.zeros_like(gp),availability[...,None]),-1)
            g=self.goal_visual(gi).reshape(n,256)+self.types[1]
            gavail=availability.reshape(n)
        r=torch.cat((r,g[:,None]),1)+self.roles
        # Actor own-state remains available even when its visual crop is hidden.
        available=torch.stack((torch.ones_like(gavail),valid[...,1].reshape(n),gavail),-1)
        geo=torch.cat((boxes.flatten(-2),box_velocity.flatten(-2)*20,p,previous),-1).reshape(n,24)
        edgegeo=torch.cat((relative(boxes),boxes[...,1,2:],valid.float(),
                           torch.ones_like(valid).float()),-1).reshape(n,8)
        goalgeo=torch.cat((boxes[...,1,:],valid.float(),
                           torch.ones_like(valid).float()),-1).reshape(n,8)
        # Both edge geometries are derivable from exactly the flat arm inputs.
        if self.family=='flat':h=(r+self.mix(r))*available[...,None]
        elif self.family=='role_attention':h=self.mix(r,available)
        elif self.family=='relation_mlp':
            r=(r+self.mix(r))*available[...,None]
            e1=self.edge(torch.cat((r[:,0],r[:,1],edgegeo),-1))
            e2=self.edge(torch.cat((r[:,1],r[:,2],goalgeo),-1))
            h=torch.stack((r[:,1],e1,e2),1)
        else:
            r=r*available[...,None]
            e1=self.edge(torch.cat((r[:,0],r[:,1],edgegeo),-1))
            e2=self.edge(torch.cat((r[:,1],r[:,2],goalgeo),-1))
            allr=torch.cat((r,e1[:,None],e2[:,None]),1)
            h=self.mix(allr,torch.cat((available,torch.ones(n,2,device=r.device,dtype=torch.bool)),1))[:,[1,3,4]]
        return self.readout(torch.cat((h.flatten(1),geo),-1)).reshape(*shape,256)

def action_candidates(base,task):
    offsets=base.new_zeros(9,4)
    for k in range(3):offsets[1+2*k,k]=.25;offsets[2+2*k,k]=-.25
    if task=='mug-handle-lift':offsets[7,3]=.35;offsets[8,3]=-.35
    else:offsets[7,:2]=.25;offsets[8,:2]=-.25
    a=(base[:,None]+offsets).clamp(-1,1)
    if task=='hammer-block':a[...,3]=.9
    return a

def select_action(net,u,base,z):
    a=action_candidates(base,net.task);plans=a[:,:,None].expand(-1,-1,4,-1)
    mean,gate=net.coefficients(u[:,None].expand(-1,9,-1),plans)
    pred=net.predict_coeff(mean,gate,z[:,None])*net.effect_sd
    progress=(pred[...,128:130]*net.direction).sum(-1)/net.progress_sd
    scores=progress-.2*(a-base[:,None]).square().sum(-1)
    idx=scores.argmax(-1)
    return a[torch.arange(len(base),device=base.device),idx],idx,scores
