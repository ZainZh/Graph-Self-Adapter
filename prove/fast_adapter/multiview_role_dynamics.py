"""Separate global/wrist slots with a common role interface and fixed target space."""
import math
import torch
from torch import nn
from prove.fast_adapter.role_dynamics import RoleDynamics,relative

class MultiViewRoleDynamics(RoleDynamics):
    def __init__(self,family='flat',task='hammer-block'):
        super().__init__(family,task)
        self.role_project=nn.Linear(2*(16*50+9),256)
        self.token_mean=torch.zeros(2,50);self.token_sd=torch.ones(2,50)

    def encode(self,tokens,boxes,proprio,previous,box_velocity):
        # [...,camera,entity,patch,feature]; coordinates remain in their own camera slot.
        shape=tokens.shape[:-4];n=math.prod(shape);mask=tokens[...,-1:]
        valid=mask.sum(-2).squeeze(-1)>0
        x=((tokens[...,:50]-self.token_mean[:,None,None,:])/self.token_sd[:,None,None,:])*mask
        inp=torch.cat((x.flatten(-2),boxes,box_velocity*20,valid[...,None]),-1)
        inp=inp.transpose(-3,-2).flatten(-2)
        r=self.role_project(inp).reshape(n,2,256)
        p=(proprio-proprio.new_tensor([0,.65,.16,.5]))/proprio.new_tensor([.2,.2,.2,.5])
        own=torch.cat((p,previous),-1).reshape(n,8)
        r=torch.stack((r[:,0]+self.proprio_project(own),r[:,1]),1)
        g=self.goal_semantic[None].expand(n,-1)+self.types[0]
        r=torch.cat((r,g[:,None]),1)+self.roles
        available=torch.stack((torch.ones(n,device=r.device,dtype=torch.bool),
              valid.any(-2)[...,1].reshape(n),torch.ones(n,device=r.device,dtype=torch.bool)),-1)
        b=boxes[...,0,:,:];bv=box_velocity[...,0,:,:];gv=valid[...,0,:]
        geo=torch.cat((b.flatten(-2),bv.flatten(-2)*20,p,previous),-1).reshape(n,24)
        edgegeo=torch.cat((relative(b),b[...,1,2:],gv.float(),torch.ones_like(gv).float()),-1).reshape(n,8)
        goalgeo=torch.cat((b[...,1,:],gv.float(),torch.ones_like(gv).float()),-1).reshape(n,8)
        if self.family=='flat':h=(r+self.mix(r))*available[...,None]
        elif self.family=='relation_attention':
            r=r*available[...,None]
            e1=self.edge(torch.cat((r[:,0],r[:,1],edgegeo),-1));e2=self.edge(torch.cat((r[:,1],r[:,2],goalgeo),-1))
            allr=torch.cat((r,e1[:,None],e2[:,None]),1)
            h=self.mix(allr,torch.cat((available,torch.ones(n,2,device=r.device,dtype=torch.bool)),1))[:,[1,3,4]]
        else:raise ValueError('P35 fixes the encoder comparison to flat versus relation_attention')
        return self.readout(torch.cat((h.flatten(1),geo),-1)).reshape(*shape,256)
