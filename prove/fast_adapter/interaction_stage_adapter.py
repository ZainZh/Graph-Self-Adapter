"""P31 runtime: public-input stage observer and causally gated latent updates.

No simulator, privileged labels, object configurations, or success signals here.
Both the residual and the inner update pause outside the observed interaction.
"""
from collections import deque
import torch
from torch import nn
from prove.fast_adapter.continuous_adapter import inputs, mlp, feedback


class StageObserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(150))
        self.register_buffer('scale', torch.ones(150))
        self.net = mlp(150, 1, 64)

    def forward(self, visual, proprio, previous):
        x = (inputs(visual, proprio, previous) - self.mean) / self.scale
        return self.net(x.clamp(-10, 10)).squeeze(-1)

    @torch.no_grad()
    def active(self, visual, proprio, previous):
        return (self(visual, proprio, previous).sigmoid() >= .5) & (visual[..., 70] > 0)


class StageState:
    def __init__(self, net, batch, *, mode='gradient', stage=False, meta=False, window=8):
        if net.history or mode not in ('gradient', 'first3', 'fixed'):
            raise ValueError('P31 uses the gradient checkpoint only')
        if not meta and any(p.requires_grad for p in net.parameters()):
            raise ValueError('Runtime network weights must be frozen')
        self.net, self.mode, self.stage, self.meta = net, mode, stage, meta
        self.z = net.z0[None].expand(batch, -1)
        if not meta:
            self.z = self.z.detach().clone()
        self.counts = torch.zeros(batch, device=self.z.device, dtype=torch.long)
        self.buffer = deque(maxlen=window)

    def eligible(self, active):
        return active.bool() if self.stage else torch.ones_like(active, dtype=torch.bool)

    def action(self, v, p, previous, base, active):
        with torch.set_grad_enabled(self.meta):
            proposal = self.net.action(v, p, previous, base, self.z)
            return torch.where(self.eligible(active)[:, None], proposal, base)

    def observe(self, v, p, previous, actual_action, nv, np_, active):
        # Eligibility comes from the PRE-action observation. No future gating.
        eligible = self.eligible(active)
        self.buffer.append((tuple(x.detach() for x in (v, p, previous, actual_action, nv, np_)), eligible.detach()))
        update = eligible & (self.counts < 3 if self.mode == 'first3' else True)
        if self.mode == 'fixed' or not bool(update.any()):
            return
        vs, ps, prevs, actions, nvs, nps = [torch.stack(xs, 1) for xs in zip(*(x[0] for x in self.buffer))]
        valid = torch.stack([x[1] for x in self.buffer], 1)[..., None]
        with torch.enable_grad():
            z = self.z if self.meta else self.z.detach().requires_grad_(True)
            u = self.net.encode(vs, ps, prevs)
            y, w = feedback(vs, ps, nvs, nps)
            w = w * valid
            pred = self.net.prediction(u, actions, z[:, None].expand(-1, len(self.buffer), -1))
            loss = 64 * ((pred - y).square() * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1)
            grad, = torch.autograd.grad(loss.sum(), z, create_graph=self.meta)
            grad = grad * (5 / grad.norm(dim=-1, keepdim=True).clamp_min(5))
            self.z = torch.where(update[:, None], z - self.net.alpha * grad, z)
        if not self.meta:
            self.z = self.z.detach()
        self.counts += update.long()

    def detach_segment(self):
        self.z = self.z.detach().requires_grad_(self.meta)
