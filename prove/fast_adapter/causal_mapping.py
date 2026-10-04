"""P38 causal qualification. No simulator parameters enter an estimator.

The numerical task deliberately exposes measured position. Six response
coordinates connect identification to a shared, frozen learned action head.
This is a declared system-identification inductive bias, not generic vision.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def mlp(n, out, width=96):
    return nn.Sequential(nn.Linear(n, width), nn.SiLU(), nn.Linear(width, width),
                         nn.SiLU(), nn.Linear(width, out))


def nominal(n, device):
    return torch.tensor([1., 0., 0., 0., 1., 0.], device=device).expand(n, -1)


def design(a):
    return torch.cat((a, torch.ones_like(a[..., :1])), -1)


def effect(e, a):
    return torch.einsum('b...j,bij->b...i', design(a), e.reshape(-1, 2, 3))


def draw_domains(n, generator, device, split='id'):
    def unif(shape, lo, hi):
        return lo+(hi-lo)*torch.rand(shape, generator=generator, device=device)
    if split == 'native':
        return nominal(n, device).clone()
    if split == 'ood':
        angle = unif((n,), 40., 65.) * torch.where(unif((n,), 0., 1.) < .5, -1., 1.)
        gains, bias, coupling = (.45, 1.6), .10, .25
    else:
        angle = unif((n,), -30., 30.)
        gains, bias, coupling = (.65, 1.35), .05, .12
    angle = angle*math.pi/180
    c, s = angle.cos(), angle.sin()
    r = torch.stack((c, -s, s, c), -1).reshape(n, 2, 2)
    m = torch.diag_embed(unif((n, 2), *gains))
    m[:, 0, 1] = unif((n,), -coupling, coupling)
    m = r@m
    b = unif((n, 2, 1), -bias, bias)
    return torch.cat((m, b), -1).flatten(1)


def bounded_response(e):
    # Same bounded coordinates for every method; no hidden truth is inspected.
    n = nominal(len(e), e.device)
    bound = e.new_tensor([1.5, 1.5, .35, 1.5, 1.5, .35])
    return n+(e-n).clamp(-bound, bound)


class DirectController(nn.Module):
    """Frozen learned inverse response; zero adaptation is exactly base.

    Training may use simulator dynamics in its task loss. Runtime receives
    only estimated response e and the currently measured goal relation.
    There is no candidate enumeration, evaluator call, or action scorer.
    """
    def __init__(self):
        super().__init__()
        self.net = mlp(8, 2, 128)

    def forward(self, relation, estimated_response):
        base = (.6*relation).clamp(-1, 1)
        e = bounded_response(estimated_response)
        n = nominal(len(e), e.device)
        value = self.net(torch.cat((base, e-n), -1))
        prior = self.net(torch.cat((base, torch.zeros_like(e)), -1))
        return (base + 1.5*torch.tanh(value-prior)).clamp(-1, 1)


class Identification(nn.Module):
    """Four declared feedback models; all output the same six control factors.

    structured: z is the response itself; effect: 32D latent, learned linear
    decoder, bilinear action consequence; absolute/relation: free predictor
    given absolute or relative state plus action and the same 32D latent.
    """
    def __init__(self, kind):
        super().__init__()
        assert kind in ('structured', 'effect', 'absolute', 'relation', 'history')
        self.kind = kind
        self.zdim = 6 if kind == 'structured' else 32
        self.z0 = nn.Parameter(torch.zeros(self.zdim))
        self.raw_rate = nn.Parameter(torch.full((self.zdim,), 0.))
        if kind != 'structured':
            self.decode = nn.Linear(32, 6, bias=False)
            nn.init.orthogonal_(self.decode.weight)
        if kind in ('absolute', 'relation'):
            self.predictor = mlp((4 if kind == 'absolute' else 2)+2+32, 2)
        if kind == 'history':
            self.token = nn.Sequential(nn.Linear(6, 64), nn.SiLU())
            self.gru = nn.GRUCell(64, 32)

    @property
    def rate(self):
        return .02+1.78*torch.sigmoid(self.raw_rate)

    def initial(self, n):
        return self.z0.expand(n, -1)

    def response(self, z):
        dz = z-self.z0
        out = dz if self.kind == 'structured' else self.decode(dz)
        return nominal(len(z), z.device)+out

    def predict(self, state, goal, action, z):
        if self.kind in ('absolute', 'relation'):
            u = torch.cat((state, goal), -1) if self.kind == 'absolute' else goal-state
            zz = z[:, None].expand(-1, action.shape[1], -1)
            return self.predictor(torch.cat((u, action, zz), -1))
        return effect(self.response(z), action)

    def update(self, z, state, goal, action, observed_delta, meta=False):
        if self.kind == 'history':
            # Same measured action/consequence sequence; explicit position is
            # irrelevant to the affine environment. No privileged parameters.
            token = torch.cat((action[:, -1], observed_delta[:, -1],
                               goal[:, -1]-state[:, -1]), -1)
            return self.gru(self.token(token), z), None
        with torch.enable_grad():
            if not meta:
                z = z.detach().requires_grad_(True)
            error = self.predict(state, goal, action, z)-observed_delta.detach()
            loss = .5*error.square().sum(-1).mean(-1)
            grad, = torch.autograd.grad(loss.sum(), z, create_graph=meta)
            # Smooth norm cap, identical to training, protects long online use.
            grad = grad/(1+grad.norm(dim=-1, keepdim=True)/5.)
            out = z-self.rate*grad
        return (out if meta else out.detach()), loss.detach()

    def adapt(self, state, goal, action, delta, updates=3, meta=False):
        z = self.initial(len(state))
        losses = []
        if self.kind == 'history':
            for t in range(action.shape[1]):
                z, _ = self.update(z, state[:, :t+1], goal[:, :t+1],
                                   action[:, :t+1], delta[:, :t+1], meta)
        else:
            for _ in range(updates):
                z, loss = self.update(z, state, goal, action, delta, meta)
                losses.append(loss)
        return z, losses


def fit_ls(action, delta):
    x = design(action)
    eye = torch.eye(3, device=x.device)
    # Full-rank support used in the primary comparison. Tiny fixed ridge also
    # defines rank-deficient/no-excitation diagnostics without a failure drop.
    n = nominal(len(action), action.device).reshape(-1, 2, 3)
    return torch.linalg.solve(x.transpose(1, 2)@x+1e-5*eye,
                              x.transpose(1, 2)@delta+1e-5*n.transpose(1, 2)).transpose(1, 2).flatten(1)
