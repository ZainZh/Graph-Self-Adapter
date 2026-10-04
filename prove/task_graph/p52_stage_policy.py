"""P52 native Insert model. New trainable skill; no historical G-head reuse.

Inner losses accept public observations and executed actions only. Expert actions
are arguments exclusively to offline outer loss. Deployment freezes every weight.
Defaults are implementation proposals, not an experiment freeze.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    views: int = 2
    raw_dim: int = 3072
    appearance_dim: int = 1024
    proprio_dim: int = 18
    role_dim: int = 64
    hidden: int = 128
    latent: int = 16
    inner_lr: float = .1
    anchor: float = .0001
    history: int = 8


def mlp(n, h, out):
    return nn.Sequential(nn.Linear(n, h), nn.SiLU(), nn.Linear(h, out))


class StagePolicy(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        self.config = config
        c = config
        self.register_buffer("raw_mean", torch.zeros(c.views,3,c.raw_dim))
        self.register_buffer("raw_scale", torch.ones(c.views,3,c.raw_dim))
        self.register_buffer("box_mean", torch.zeros(c.views,3,4))
        self.register_buffer("box_scale", torch.ones(c.views,3,4))
        self.register_buffer("proprio_mean", torch.zeros(c.proprio_dim))
        self.register_buffer("proprio_scale", torch.ones(c.proprio_dim))
        self.role = mlp(c.raw_dim, c.role_dim, c.role_dim)
        # Explicit book-gripper and book-goal differences accompany role tokens.
        self.encoder = mlp(c.views*(5*c.role_dim+15)+c.proprio_dim, c.hidden, c.hidden)
        self.z0 = nn.Parameter(torch.zeros(c.latent))
        self.policy = nn.Sequential(nn.Linear(c.hidden+c.latent,c.hidden),nn.SiLU(),nn.Linear(c.hidden,c.hidden),nn.SiLU(),nn.Linear(c.hidden,6))
        self.predictor = nn.Sequential(nn.Linear(c.hidden+c.latent+6,c.hidden),nn.SiLU(),nn.Linear(c.hidden,c.hidden),nn.SiLU(),
                             nn.Linear(c.hidden,c.views*3*(c.appearance_dim+4)+c.proprio_dim))

    def encode(self, obs):
        c = self.config
        raw, visible = obs['raw'], obs['visible']
        if raw.shape[-3:] != (c.views, 3, c.raw_dim):
            raise ValueError('Expected [...,views,3,raw_dim] frozen role features')
        roles = self.role((raw-self.raw_mean)/self.raw_scale)*visible[..., None]
        rel = torch.cat((roles.flatten(-2), roles[..., 1, :]-roles[..., 0, :],
                         roles[..., 1, :]-roles[..., 2, :],
                         (((obs['boxes']-self.box_mean)/self.box_scale)*visible[...,None]).flatten(-2), visible), -1).flatten(-2)
        return self.encoder(torch.cat((rel, (obs['proprio']-self.proprio_mean)/self.proprio_scale), -1))

    def initial(self, obs):
        return self.z0.expand(*obs["proprio"].shape[:-1], self.config.latent)

    def action(self, obs, z):
        # LIBERO normalized OSC6; fixed closed finger command is attached below.
        return torch.tanh(self.policy(torch.cat((self.encode(obs), z), -1)))

    def command(self, obs, z):
        a = self.action(obs, z)
        return torch.cat((a, torch.ones_like(a[..., :1])), -1)

    def feedback_loss(self, obs, executed_action, next_obs, z, *, reduction="mean"):
        c = self.config
        if executed_action.shape[-1] != 6:
            raise ValueError('Predictor requires executed OSC6, not expert labels')
        pred = self.predictor(torch.cat((self.encode(obs), z, executed_action), -1))
        cut = c.views*3*c.appearance_dim
        appearance = pred[..., :cut].reshape(*pred.shape[:-1], c.views, 3, c.appearance_dim)
        boxes = pred[..., cut:cut+c.views*12].reshape(*pred.shape[:-1], c.views, 3, 4)
        proprio = pred[..., cut+c.views*12:]
        # Fixed pretrained targets, no learned target encoder/collapse shortcut.
        target = ((next_obs['raw'].detach()-self.raw_mean)/self.raw_scale)[..., :c.appearance_dim]
        mask = next_obs['visible'].detach()
        app_error = ((appearance-target)**2).mean(-1)
        box_error = ((boxes-(next_obs['boxes'].detach()-self.box_mean)/self.box_scale)**2).mean(-1)
        per_example = (((app_error+box_error)*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)
                + (proprio-(next_obs['proprio'].detach()-self.proprio_mean)/self.proprio_scale).square().mean(-1))
        return per_example.mean() if reduction == 'mean' else per_example

    def adapt_history(self, history, z, z0, *, create_graph=False):
        if not history:
            raise ValueError("Empty observed transition history")
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        loss = torch.stack([self.feedback_loss(o,a,n,z,reduction="none") for o,a,n in history[-self.config.history:]]).mean(0).sum()
        loss = loss+self.config.anchor*(z-z0).square().mean(-1).sum()
        grad, = torch.autograd.grad(loss, z, create_graph=create_graph)
        return z-self.config.inner_lr*grad, loss

    def adapt(self, obs, executed_action, next_obs, z, z0, *, create_graph=False):
        return self.adapt_history([(obs,executed_action,next_obs)],z,z0,create_graph=create_graph)

    def outer_loss(self, support, query_obs, expert_action, *, prediction_weight=.1):
        obs, executed, next_obs = support
        z0 = self.initial(obs)
        z, prediction = self.adapt(obs, executed, next_obs, z0, z0, create_graph=True)
        bc = (self.action(query_obs, z)-expert_action).square().mean()
        return bc+prediction_weight*prediction, dict(bc=bc, prediction=prediction)


class StageRuntime:
    """One stage lifecycle. Weights frozen; only current-stage z mutates."""
    def __init__(self, model, online=True):
        self.model = model.eval().requires_grad_(False)
        self.online = online
        self.z = self.z0 = None
        self.updates = 0
        self.history = []

    def enter(self, obs):
        with torch.no_grad():
            self.z0 = self.model.initial(obs).detach()
        self.z = self.z0.clone()
        self.updates = 0
        self.history = []

    def feedback(self, obs, executed_action, next_obs):
        if self.z is None:
            raise RuntimeError('Stage not entered')
        if self.online:
            self.history.append((obs,executed_action,next_obs))
            self.history=self.history[-self.model.config.history:]
            with torch.enable_grad():
                self.z, loss = self.model.adapt_history(self.history, self.z, self.z0)
            self.z = self.z.detach()
            self.updates += 1
            return float(loss.detach())
        return None

    def command(self, obs):
        if self.z is None:
            raise RuntimeError('Stage not entered')
        with torch.no_grad():
            return self.model.command(obs, self.z)

    def exit(self):
        self.z = self.z0 = None
        self.history = []
