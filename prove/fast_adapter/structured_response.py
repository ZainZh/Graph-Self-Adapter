"""P19 explicit affine fast state. Runtime estimators only consume observations.

The known affine model and analytic controller are declared inductive biases.
No estimator receives true dynamics, evaluator outcomes, or oracle actions.
"""
from __future__ import annotations
import torch
from torch import nn

SCALE = .25

def nominal_z(batch=1, *, device='cpu', dtype=torch.float32):
    return torch.tensor([[1.,0.,0.],[0.,1.,0.]],device=device,dtype=dtype).expand(batch,-1,-1).clone()

def design(actions):
    return torch.cat([actions / SCALE, torch.ones_like(actions[...,:1])],-1)

def prediction(z, actions):
    return torch.einsum('bij,sj->bsi',z,design(actions))*SCALE

def adapt(z0, displacement, actions, *, steps, rate=None, create_graph=False):
    """Full-prefix GD; normalize curvature using known action design only.

    Mean over support, sum over independent domains: no batch-size-dependent
    learning rate. This is gradient descent, never an LS solution warm start.
    """
    if steps == 0 or actions.shape[0] == 0:
        return z0.clone(), []
    x=design(actions)
    lipschitz=torch.linalg.eigvalsh(x.T@x/len(x))[-1].clamp_min(1e-6)
    rate=torch.ones_like(z0[0]) if rate is None else rate
    z=z0.clone().requires_grad_(True)
    trace=[]
    for _ in range(steps):
        error=(prediction(z,actions)-displacement)/SCALE
        per_domain=.5*error.square().sum(-1).mean(-1)
        grad=torch.autograd.grad(per_domain.sum(),z,create_graph=create_graph)[0]
        z=z-rate*grad/lipschitz
        if not create_graph:
            trace.append(float(per_domain.detach().mean()))
            z=z.detach().requires_grad_(True)
    return z,trace

def control(state, goal, z):
    """Shared GT-geometric controller; z, not simulator dynamics, sets response."""
    matrix=z[...,:2]
    invalid=(~torch.isfinite(z).all((-1,-2))) | (torch.linalg.det(matrix).abs()<.1)
    matrix=torch.where(invalid[:,None,None],torch.eye(2,device=z.device,dtype=z.dtype),matrix)
    bias=torch.where(invalid[:,None],torch.zeros_like(z[...,2]),z[...,2]*SCALE)
    base=.5*(goal-state)
    wanted=torch.linalg.solve(matrix,(base-bias).unsqueeze(-1)).squeeze(-1)
    return (base+(wanted-base).clamp(-.15,.15)).clamp(-1,1),invalid

def least_squares(displacement,actions):
    x=design(actions)
    return torch.linalg.lstsq(x.expand(len(displacement),-1,-1),displacement/SCALE).solution.transpose(1,2)

class MetaGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.z0=nn.Parameter(nominal_z()[0])
        self.rate_logits=nn.Parameter(torch.full((2,3),0.11778303565638346))
    @property
    def rate(self):
        return .1+1.7*torch.sigmoid(self.rate_logits)
    def forward(self,displacement,actions,*,steps=3,create_graph=False):
        return adapt(self.z0.expand(len(displacement),-1,-1),displacement,actions,steps=steps,rate=self.rate,create_graph=create_graph)[0]

class History(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(15,64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,6))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,displacement,actions,**kwargs):
        batch,k,_=displacement.shape
        tokens=displacement.new_zeros(batch,5,3)
        tokens[:,:k,:2]=displacement/SCALE
        tokens[:,:k,2]=1
        return nominal_z(batch,device=displacement.device)+self.net(tokens.flatten(1)).reshape(-1,2,3)
