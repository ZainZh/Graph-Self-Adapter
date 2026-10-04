"""P30-v2: causal sliding-window state shared by training and closed-loop inference."""
from collections import deque
import torch
from prove.fast_adapter.continuous_adapter import feedback


class ContinuousState:
    def __init__(self, net, batch, *, mode='gradient', window=8, meta=False, fixed_z=None):
        if mode not in ('gradient', 'history', 'first3', 'fixed'):
            raise ValueError(mode)
        if bool(net.history) != (mode == 'history'):
            raise ValueError('Wrong adapter kind')
        if not meta and any(p.requires_grad for p in net.parameters()):
            raise ValueError('Inference weights must be frozen')
        self.net, self.mode, self.meta = net, mode, meta
        self.z = net.z0[None].expand(batch, -1)
        if fixed_z is not None:
            if mode != 'fixed': raise ValueError('Only a fixed control accepts external z')
            self.z = fixed_z.expand_as(self.z)
        if not meta: self.z = self.z.detach().clone()
        self.buffer = deque(maxlen=window)
        self.steps = self.updates = 0

    def action(self, v, p, previous, base):
        with torch.set_grad_enabled(self.meta):
            return self.net.action(v, p, previous, base, self.z)

    def observe(self, v, p, previous, actual_action, nv, np_):
        """Invoke exactly once AFTER actual_action executes; next action uses new z."""
        self.buffer.append(tuple(x.detach() for x in (v, p, previous, actual_action, nv, np_)))
        self.steps += 1
        if self.mode == 'fixed' or (self.mode == 'first3' and self.updates == 3):
            return
        if self.mode == 'history':
            with torch.set_grad_enabled(self.meta):
                u = self.net.encode(v, p, previous)
                y, w = feedback(v, p, nv, np_)
                token = self.net.token(torch.cat((u, actual_action, y*w), -1))
                _, h = self.net.gru(token[:, None], self.z[None].contiguous())
                self.z = h[0]
        else:
            sv, sp, spre, sa, snv, snp = [torch.stack(xs, 1) for xs in zip(*self.buffer)]
            with torch.enable_grad():
                z = self.z if self.meta else self.z.detach().requires_grad_(True)
                u = self.net.encode(sv, sp, spre)
                y, w = feedback(sv, sp, snv, snp)
                pred = self.net.prediction(u, sa, z[:, None].expand(-1, len(self.buffer), -1))
                loss = 64*((pred-y).square()*w).sum((1, 2))/w.sum((1, 2)).clamp_min(1)
                grad, = torch.autograd.grad(loss.sum(), z, create_graph=self.meta)
                grad = grad * (5 / grad.norm(dim=-1, keepdim=True).clamp_min(5))
                self.z = z - self.net.alpha*grad
        if not self.meta: self.z = self.z.detach()
        self.updates += 1

    def detach_segment(self):
        self.z = self.z.detach().requires_grad_(self.meta and self.mode != 'history')
