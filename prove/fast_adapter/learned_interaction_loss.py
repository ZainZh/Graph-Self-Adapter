"""P32 bounded learned-loss experiment. Runtime consumes public transitions only.

The energy head receives observed feedback as input, unlike a forward predictor.
Its weights are trained by the OUTER objective, never by minimizing its own output.
All network weights freeze at runtime; only the episode-local latent can change.
"""
from collections import deque
import torch
from torch import nn
from prove.fast_adapter.continuous_adapter import ObjectAdapter, mlp, feedback


class LossAdapter(ObjectAdapter):
    def __init__(self, family):
        if family not in ('predictive', 'learned'):
            raise ValueError(family)
        super().__init__()
        self.family = family
        if family == 'predictive':
            # 81,823 loss/predictor parameters vs learned's 81,816. Both use
            # the original outer prediction auxiliary; common actor/encoder match.
            self.predictor = mlp(164, 28, 205)
            nn.init.normal_(self.predictor[-1].weight, std=.01)
            nn.init.zeros_(self.predictor[-1].bias)
        else:
            self.energy = mlp(220, 28, 112)
            nn.init.normal_(self.energy[-1].weight, std=.01)
            nn.init.zeros_(self.energy[-1].bias)

    def inner_loss(self, u, actions, y, weights, z):
        zs = z[:, None].expand(-1, u.shape[1], -1)
        if self.family == 'predictive':
            out = self.prediction(u, actions, zs)
        else:
            # Positive residual-shaped energy, not a calibrated predictor:
            # observed feedback is explicitly available to this head.
            out = self.energy(torch.cat((u, actions, zs, y*weights, weights), -1))
        return 64*((out-y).square()*weights).sum((1, 2))/weights.sum((1, 2)).clamp_min(1)


class LossState:
    def __init__(self, net, batch, *, mode='gradient', meta=False, window=8):
        if mode not in ('gradient', 'first3', 'fixed'):
            raise ValueError(mode)
        if not meta and any(p.requires_grad for p in net.parameters()):
            raise ValueError('Freeze network weights before runtime adaptation')
        self.net, self.mode, self.meta = net, mode, meta
        self.z = net.z0[None].expand(batch, -1)
        if not meta:
            self.z = self.z.detach().clone()
        self.counts = torch.zeros(batch, dtype=torch.long, device=self.z.device)
        self.buffer = deque(maxlen=window)
        self.last = None

    def action(self, v, p, previous, base, active):
        with torch.set_grad_enabled(self.meta):
            return torch.where(active[:, None], self.net.action(v,p,previous,base,self.z), base)

    def observe(self, v, p, previous, actual_action, nv, np_, active, *, history_active=None):
        valid = active if history_active is None else history_active
        self.buffer.append((tuple(x.detach() for x in (v,p,previous,actual_action,nv,np_)), valid.detach()))
        update = active & (self.counts < 3 if self.mode == 'first3' else True)
        self.last = {'updated': torch.zeros_like(update)}
        if self.mode == 'fixed' or not bool(update.any()):
            return
        vs,ps,prevs,actions,nvs,nps = [torch.stack(xs,1) for xs in zip(*(x[0] for x in self.buffer))]
        with torch.enable_grad():
            z = self.z if self.meta else self.z.detach().requires_grad_(True)
            u = self.net.encode(vs,ps,prevs)
            y,w = feedback(vs,ps,nvs,nps)
            w = w*torch.stack([x[1] for x in self.buffer],1)[...,None]
            loss = self.net.inner_loss(u,actions,y,w,z)
            grad, = torch.autograd.grad(loss.sum(),z,create_graph=self.meta)
            raw_norm = grad.norm(dim=-1,keepdim=True)
            grad = grad*(5/raw_norm.clamp_min(5))
            self.z = torch.where(update[:,None],z-self.net.alpha*grad,z)
            self.last = {'updated': update.detach(), 'inner_loss': loss.detach(),
                         'grad_norm': raw_norm.detach().squeeze(-1),
                         'valid_transitions': (w.sum(-1)>0).sum(-1).detach()}
        if not self.meta:
            self.z = self.z.detach()
        self.counts += update.long()

    def detach_segment(self):
        self.z = self.z.detach().requires_grad_(self.meta)
