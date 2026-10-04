"""P25: identity-anchored correction and observable displacement adaptation."""
import torch
from torch import nn
from prove.fast_adapter.skill_adapter import SkillAdapter, mlp


class FeedbackAdapter(SkillAdapter):
    def __init__(self, variant):
        assert variant in ('original', 'anchored', 'physical', 'history')
        super().__init__(history=variant == 'history')
        self.variant = variant
        self.anchored = variant != 'original'
        self.physical = variant in ('physical', 'history')
        if self.physical:
            self.predictor = mlp(291, 3)
            nn.init.normal_(self.predictor[-1].weight, std=.01)
            nn.init.zeros_(self.predictor[-1].bias)
        if self.is_history:
            # History gets the same explicit measured displacement as the inner loss.
            self.history_token = nn.Sequential(nn.Linear(518, 128), nn.SiLU())

    def action_u(self, u, base, z):
        value = self.actor(torch.cat((u, z, base), -1))
        if self.anchored:
            prior = self.z0.expand_as(z)
            value = value - self.actor(torch.cat((u, prior, base), -1))
        return (base + .35 * torch.tanh(value)).clamp(-1, 1)

    def signal_prediction(self, u, action, z):
        value = self.predictor(torch.cat((u, action, z), -1))
        return value if self.physical else value + u

    def signal_target(self, next_u, proprio, next_proprio):
        value = (next_proprio - proprio) / .008 if self.physical else next_u
        return value.detach()

    def inner_loss(self, u, action, target, z):
        # Same per-coordinate scale: original feature loss equals .5*sum_256.
        return 128 * (self.signal_prediction(u, action, z)-target).square().mean(-1)

    def feedback_path(self, transitions, meta=False):
        v, nv, p, np_, previous, a = [transitions[k] for k in
            ('visual', 'next_visual', 'proprio', 'next_proprio', 'previous_action', 'actions')]
        with torch.set_grad_enabled(meta or self.is_history):
            u = self.encode(v, p, previous)
            nu = self.encode(nv, np_, a)
            target = self.signal_target(nu, p, np_)
            z = self.initial(len(v))
        if self.is_history:
            token = self.history_token(torch.cat((u, a, nu-u, (np_-p)/.008), -1))
            states, _ = self.history_gru(token, z.unsqueeze(0).contiguous())
            return [z] + list(states.unbind(1)), None
        paths = [z if meta else z.detach()]
        losses = []
        for t in range(v.shape[1]):
            with torch.enable_grad():
                if not meta:
                    z = z.detach().requires_grad_(True)
                cur_u = u[:, t] if meta else u[:, t].detach()
                before = self.inner_loss(cur_u, a[:, t], target[:, t], z)
                grad, = torch.autograd.grad(before.sum(), z, create_graph=meta)
                z = z-self.alpha*grad
                after = self.inner_loss(cur_u, a[:, t], target[:, t], z)
            if not meta:
                z = z.detach()
            paths.append(z)
            losses.append(torch.stack((before.detach(), after.detach()), -1))
        return paths, torch.stack(losses, 1)
