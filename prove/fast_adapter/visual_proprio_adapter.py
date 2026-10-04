"""P22 policies: RGB features + controlled-object proprioception only."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

def mlp(n, out, width=128):
    return nn.Sequential(nn.Linear(n,width), nn.SiLU(), nn.Linear(width,width), nn.SiLU(), nn.Linear(width,out))

class VisualBase(nn.Module):
    def __init__(self):
        super().__init__(); self.net = mlp(67,3,256)
    def forward(self, visual, proprio):
        return torch.tanh(self.net(torch.cat((visual,proprio/.1),-1)))

class ProprioAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(137,128),nn.SiLU())
        self.history = nn.GRU(128,32,batch_first=True)
        self.actor = mlp(102,3)
        self.predictor = mlp(102,67)
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(.05))))
        nn.init.zeros_(self.actor[-1].weight); nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.predictor[-1].weight); nn.init.zeros_(self.predictor[-1].bias)
    @property
    def alpha(self): return F.softplus(self.raw_alpha)
    def initial(self, visual, proprio, actions):
        u = torch.cat((visual[:,:-1],proprio[:,:-1]/.1,actions,
                       visual[:,1:]-visual[:,:-1],(proprio[:,1:]-proprio[:,:-1])/.008),-1)
        return self.history(self.token(u))[1][-1]
    def action(self, visual, proprio, base, z):
        residual = .35*torch.tanh(self.actor(torch.cat((visual,proprio/.1,base,z),-1)))
        return (base+residual).clamp(-1,1)
    def predict(self, visual, proprio, action, z):
        return self.predictor(torch.cat((visual,proprio/.1,action,z),-1))
    def loss(self, visual, proprio, actions, z, anchor):
        zz = z[:,None,:].expand(-1,actions.shape[1],-1)
        pred = self.predict(visual[:,:-1],proprio[:,:-1],actions,zz)
        dp = (proprio[:,1:]-proprio[:,:-1])/.008
        dv = visual[:,1:]-visual[:,:-1]
        return (.5*(pred[...,:3]-dp).square().sum(-1).mean(1)
                + .05*(pred[...,3:]-dv).square().mean((1,2))
                + .001*(z-anchor).square().sum(-1))
    def adapt(self, visual, proprio, actions, steps=3, meta=False):
        with torch.enable_grad():
            z = self.initial(visual,proprio,actions); anchor = z
            if not meta: z=z.detach().requires_grad_(True); anchor=z.detach()
            curve=[]
            for _ in range(steps):
                loss=self.loss(visual,proprio,actions,z,anchor); curve.append(loss.detach())
                grad,=torch.autograd.grad(loss.sum(),z,create_graph=meta)
                z=z-self.alpha*grad
                if not meta: z=z.detach().requires_grad_(True)
            curve.append(self.loss(visual,proprio,actions,z,anchor).detach())
        return z if meta else z.detach(),torch.stack(curve,1)

