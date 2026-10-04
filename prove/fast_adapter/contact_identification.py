"""P38 contact-response adaptation with a shared direct force correction.

Known quasi-static response is an explicit modelling assumption. Raw visual
effect, measured role motion, and their combination are competing inner losses.
The controller always receives the same RGB-derived goal relation.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

PRIOR = (4.7619047619, -.3333333333)


def response(z):
    prior = z.new_tensor(PRIOR)
    e = prior+z
    return torch.stack((e[..., 0].clamp(1., 12.), e[..., 1].clamp(-3., 0.)), -1)


def force(error, e):
    wanted = (4*error).clamp(0., .08)*20
    f = ((wanted-e[..., 1])/e[..., 0]).clamp(0., 1.)
    return torch.where(error < .0012, torch.zeros_like(f), f)


def effect(e, action):
    return (e[..., 0]*action+e[..., 1]).clamp_min(0.)


class VisualJacobian(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.SiLU(), nn.Linear(64, 64),
                                 nn.SiLU(), nn.Linear(64, 16))
    def forward(self, f):
        return self.net(f)


class ContactAdapter(nn.Module):
    def __init__(self, kind):
        super().__init__()
        assert kind in ('motion', 'appearance', 'combined', 'history')
        self.kind = kind
        # A meta-learned constant SPD preconditioner, not an online LS solve.
        self.log_diag = nn.Parameter(torch.tensor([.5, -.7]))
        self.off_diagonal = nn.Parameter(torch.tensor(-.5))
        if kind == 'history':
            self.token = nn.Sequential(nn.Linear(35, 64), nn.SiLU())
            self.gru = nn.GRUCell(64, 32)
            self.decoder = nn.Linear(32, 2, bias=False)
            nn.init.normal_(self.decoder.weight, std=.01)

    def initial(self, n, device):
        return torch.zeros(n, 32 if self.kind == 'history' else 2, device=device,
                           requires_grad=self.kind != 'history')

    def estimate(self, z):
        return response(self.decoder(z) if self.kind == 'history' else z)

    def preconditioner(self):
        d = self.log_diag.exp().clamp(.02, 8.)
        zero = torch.zeros_like(self.off_diagonal)
        l = torch.stack((d[0], zero, self.off_diagonal, d[1])).reshape(2, 2)
        return l@l.T

    def update(self, z, action, observed_motion, df, f, jacobian, meta=False):
        if self.kind == 'history':
            token = torch.cat((f[:, -1], df[:, -1], action[:, -1, None],
                               observed_motion[:, -1, None],
                               (action[:, -1]-action[:, 0])[:, None]), -1)
            return self.gru(self.token(token), z), None
        with torch.enable_grad():
            if not meta:
                z = z.detach().requires_grad_(True)
            estimated_motion = effect(self.estimate(z)[:, None], action)
            geometric = .5*(estimated_motion-observed_motion.detach()).square().mean(-1)
            predicted_feature = jacobian*estimated_motion[..., None]
            # Normalize by local visual sensitivity, not the number of channels.
            # Sensitivity is fitted on training images and frozen at runtime.
            visual = .5*(predicted_feature-df.detach()).square().sum(-1)
            visual = (visual/(jacobian.square().sum(-1).detach()+.05)).mean(-1)
            loss = geometric if self.kind == 'motion' else visual
            if self.kind == 'combined':
                loss = .5*(geometric+visual)
            grad, = torch.autograd.grad(loss.sum(), z, create_graph=meta)
            step = grad@self.preconditioner().T
            step = step/(1+step.norm(dim=-1, keepdim=True)/4.)
            out = z-step
            # Bounds are part of training and deployment, preserving z0 exactly.
            out = self.estimate(out)-out.new_tensor(PRIOR)
        return (out if meta else out.detach()), loss.detach()


def ridge_response(action, motion):
    x = torch.stack((action, torch.ones_like(action)), -1)
    eye = torch.eye(2, device=x.device)
    prior = x.new_tensor(PRIOR)
    # Fixed ridge prior. Includes stopped/sliding data without truth contact flags.
    e = torch.linalg.solve(x.transpose(1, 2)@x+.02*eye,
                           (x.transpose(1, 2)@motion[..., None]).squeeze(-1)+.02*prior)
    return response(e-prior)
