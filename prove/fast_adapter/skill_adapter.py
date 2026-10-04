"""Shared-prior Fast Adapter: P(U), residual actor, next-U predictor, latent only TTT."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def mlp(ni, no, width=256):
    return nn.Sequential(nn.Linear(ni, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, no))


class SkillAdapter(nn.Module):
    def __init__(self, history=False):
        super().__init__()
        self.encoder = nn.Sequential(mlp(70, 256), nn.LayerNorm(256, elementwise_affine=False))
        self.actor = mlp(291, 3)
        self.predictor = mlp(291, 256)
        self.z0 = nn.Parameter(torch.zeros(32))
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(.05))))
        self.is_history = history
        if history:
            self.history_token = nn.Sequential(nn.Linear(515, 128), nn.SiLU())
            self.history_gru = nn.GRU(128, 32, batch_first=True)
            self.raw_alpha.requires_grad_(False)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.normal_(self.predictor[-1].weight, std=.01)
        nn.init.zeros_(self.predictor[-1].bias)

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha)

    def encode(self, visual, proprio, previous_action):
        return self.encoder(torch.cat((visual, proprio / .1, previous_action), -1))

    def initial(self, batch):
        return self.z0.unsqueeze(0).expand(batch, -1)

    def action_u(self, u, base, z):
        return (base + .35 * torch.tanh(self.actor(torch.cat((u, z, base), -1)))).clamp(-1, 1)

    def action(self, visual, proprio, previous_action, base, z):
        return self.action_u(self.encode(visual, proprio, previous_action), base, z)

    def predict(self, u, action, z):
        return u + self.predictor(torch.cat((u, action, z), -1))

    def transition_loss(self, u, action, target_u, z):
        # Fixed-scale LayerNorm U; next-observation target is stop-gradient.
        return .5 * (self.predict(u, action, z) - target_u.detach()).square().sum(-1)

    def update(self, u, action, target_u, z, meta=False):
        with torch.enable_grad():
            if not meta:
                u, action, target_u = u.detach(), action.detach(), target_u.detach()
                z = z.detach().requires_grad_(True)
            before = self.transition_loss(u, action, target_u, z)
            grad, = torch.autograd.grad(before.sum(), z, create_graph=meta)
            new_z = z - self.alpha * grad
            after = self.transition_loss(u, action, target_u, new_z)
        return (new_z if meta else new_z.detach(),
                torch.stack((before.detach(), after.detach()), -1))

    def adapt_path(self, visual, proprio, actions, meta=False):
        """One update per newly completed transition; never use future feedback."""
        assert visual.shape[1] == actions.shape[1] + 1
        assert proprio.shape[:2] == visual.shape[:2]
        previous = torch.cat((torch.zeros_like(proprio[:, :1]), actions), 1)
        with torch.set_grad_enabled(meta):
            u = self.encode(visual, proprio, previous)
            z = self.initial(len(visual))
        path = [z if meta else z.detach()]
        losses = []
        for i in range(actions.shape[1]):
            z, loss = self.update(u[:, i], actions[:, i], u[:, i+1], z, meta=meta)
            path.append(z)
            losses.append(loss)
        empty = visual.new_empty((len(visual), 0, 2))
        return path, torch.stack(losses, 1) if losses else empty

    def history_path(self, visual, proprio, actions):
        assert self.is_history
        previous = torch.cat((torch.zeros_like(proprio[:, :1]), actions), 1)
        u = self.encode(visual, proprio, previous)
        z = self.initial(len(visual))
        if actions.shape[1] == 0:
            return [z]
        tokens = self.history_token(torch.cat((u[:, :-1], actions, u[:, 1:]-u[:, :-1]), -1))
        states, _ = self.history_gru(tokens, z.unsqueeze(0).contiguous())
        return [z] + list(states.unbind(1))

    def latent(self, visual, proprio, actions, meta=False):
        if self.is_history:
            return self.history_path(visual, proprio, actions)[-1]
        return self.adapt_path(visual, proprio, actions, meta=meta)[0][-1]
