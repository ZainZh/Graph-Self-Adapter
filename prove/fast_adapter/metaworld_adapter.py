"""P28 RGB/proprio skills and shared-prior latent-only gradient adaptation.

This module has no simulator import or privileged task/dynamics inputs.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def mlp(n, out, width=256):
    return nn.Sequential(nn.Linear(n, width), nn.SiLU(), nn.Linear(width, width),
                         nn.SiLU(), nn.Linear(width, out))


def inputs(visual, proprio, previous):
    center = proprio.new_tensor([0., .7, .2, .5])
    scale = proprio.new_tensor([.15, .15, .15, .5])
    return torch.cat((visual, (proprio-center)/scale, previous), -1)


class RGBSkill(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(72, 4)

    def forward(self, visual, proprio, previous):
        return self.net(inputs(visual, proprio, previous)).clamp(-1, 1)


class RobotAdapter(nn.Module):
    def __init__(self, history=False):
        super().__init__()
        self.history = history
        self.encoder = nn.Sequential(mlp(72, 128, 128),
                                     nn.LayerNorm(128, elementwise_affine=False))
        self.actor = mlp(164, 4, 128)
        self.predictor = mlp(164, 3, 128)
        self.z0 = nn.Parameter(torch.zeros(32))
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(.05))))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.normal_(self.predictor[-1].weight, std=.01)
        nn.init.zeros_(self.predictor[-1].bias)
        if history:
            self.token = nn.Sequential(nn.Linear(135, 128), nn.SiLU())
            self.gru = nn.GRU(128, 32, batch_first=True)
            self.raw_alpha.requires_grad_(False)

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha)

    def encode(self, visual, proprio, previous):
        return self.encoder(inputs(visual, proprio, previous))

    def initial(self, n):
        return self.z0.unsqueeze(0).expand(n, -1)

    def prediction(self, u, action, z):
        return self.predictor(torch.cat((u, action, z), -1))

    def action(self, visual, proprio, previous, base, z):
        u = self.encode(visual, proprio, previous)
        value = self.actor(torch.cat((u, base, z), -1))
        prior = self.actor(torch.cat((u, base, self.z0.expand_as(z)), -1))
        return (base + .5*torch.tanh(value-prior)).clamp(-1, 1)

    def adapt(self, support, steps=3, meta=False):
        """80 actual transitions shared by all arms; few optimization steps.

        Targets are measured end-effector displacement, not object/goal truth.
        Full support is used per update, avoiding contact/noise-driven selection.
        """
        with torch.set_grad_enabled(meta or self.history):
            u = self.encode(support['visual'], support['proprio'], support['previous'])
            a = support['actions']
            target = (support['next_proprio'][..., :3]-support['proprio'][..., :3])/.01
            z = self.initial(len(u))
        if self.history:
            token = self.token(torch.cat((u, a, target), -1))
            _, hidden = self.gru(token, z.unsqueeze(0).contiguous())
            return hidden[0], []
        curve = []
        for _ in range(steps):
            with torch.enable_grad():
                if not meta:
                    z = z.detach().requires_grad_(True)
                    u = u.detach()
                pred = self.prediction(u, a, z[:, None].expand(-1, u.shape[1], -1))
                loss = 64*(pred-target.detach()).square().mean((1, 2))
                grad, = torch.autograd.grad(loss.sum(), z, create_graph=meta)
                z = z-self.alpha*grad
                curve.append(loss.detach())
            if not meta:
                z = z.detach()
        return z, curve
