"""P20 learned visual interaction latent; no geometry or dynamics parameter input."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

PROBE_ACTIONS=((.65,0.,0.),(0.,.65,0.),(-.65,0.,0.),(0.,-.65,0.),(0.,0.,.65))

class InteractionAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.token=nn.Sequential(nn.Linear(131,128),nn.SiLU())
        self.history=nn.GRU(128,32,batch_first=True)
        self.actor=nn.Sequential(nn.Linear(99,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3))
        self.predictor=nn.Sequential(nn.Linear(99,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,64))
        self.raw_alpha=nn.Parameter(torch.tensor(math.log(math.expm1(.05))))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.predictor[-1].weight);nn.init.zeros_(self.predictor[-1].bias)

    @property
    def alpha(self):return F.softplus(self.raw_alpha)

    def initial(self,support,actions):
        # Only completed transitions enter the sequence; no current future frame.
        tokens=torch.cat((support[:,:-1],actions,support[:,1:]-support[:,:-1]),-1)
        return self.history(self.token(tokens))[1][-1]

    def action(self,visual,base,z):
        return (base+.35*torch.tanh(self.actor(torch.cat((visual,base,z),-1)))).clamp(-1,1)

    def predict(self,visual,action,z):
        return visual+self.predictor(torch.cat((visual,action,z),-1))

    def losses(self,support,actions,z,anchor):
        zz=z[:,None,:].expand(-1,actions.shape[1],-1)
        error=self.predict(support[:,:-1],actions,zz)-support[:,1:]
        return .5*error.square().mean(1).sum(-1)+.001*(z-anchor).square().sum(-1)

    def adapt(self,support,actions,steps=3,meta=False,initial=None):
        # A sum of per-domain losses makes updates independent of batch size.
        with torch.enable_grad():
            z=self.initial(support,actions) if initial is None else initial
            anchor=z
            if not meta:z=z.detach().requires_grad_(True);anchor=z.detach()
            curve=[]
            for _ in range(steps):
                losses=self.losses(support,actions,z,anchor);curve.append(losses.detach())
                grad,=torch.autograd.grad(losses.sum(),z,create_graph=meta)
                z=z-self.alpha*grad
                if not meta:z=z.detach().requires_grad_(True)
            curve.append(self.losses(support,actions,z,anchor).detach())
        return z if meta else z.detach(),torch.stack(curve,1)


def project(features,pcas):
    return torch.cat([(features[...,i,:].float()-p['mean'])@p['basis']/p['sd'] for i,p in enumerate(pcas)],-1)


def support_actions(n,device='cpu'):
    return torch.tensor(PROBE_ACTIONS,device=device).unsqueeze(0).expand(n,-1,-1)
