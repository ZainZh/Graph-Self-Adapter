"""Causal, execution-step adaptation using completed public transitions only.

P30 keeps the latent between actions. No simulator, evaluator, task labels, or
success signal enters this module. Old support/query adapters are unchanged.
"""
from collections import deque
import torch
from prove.fast_adapter.object_interaction_adapter import feedback


class OnlineLatent:
    MODES = {"gradient", "first3", "fixed", "history"}

    def __init__(self, adapter, batch_size, *, window=8, mode="gradient",
                 meta=False, initial_z=None, grad_clip=5.0, correction_scale=0.1, trust_radius=0.25, warmup_steps=0):
        if mode not in self.MODES or window < 1 or batch_size < 1 or grad_clip <= 0:
            raise ValueError("Invalid online adaptation configuration")
        if (mode == "history") != bool(adapter.history):
            raise ValueError("History requires its separately trained adapter")
        if not meta and any(p.requires_grad for p in adapter.parameters()):
            raise ValueError("Freeze all adapter parameters before inference")
        self.net, self.batch_size = adapter, batch_size
        self.window, self.mode, self.meta = window, mode, meta
        self.grad_clip = grad_clip
        if trust_radius <= 0:
            raise ValueError("trust_radius must be positive")
        self.correction_scale = correction_scale
        self.trust_radius = trust_radius
        self.warmup_steps = warmup_steps
        self.transitions = deque(maxlen=window)
        self.reset(initial_z)

    def reset(self, initial_z=None):
        """Independent objects/episodes start from the prior or a declared fixed z."""
        z = self.net.z0.unsqueeze(0).expand(self.batch_size, -1)
        if initial_z is not None:
            if self.mode != "fixed":
                raise ValueError("External latent is only allowed for fixed-z controls")
            z = initial_z.expand_as(z)
        self.z = z if self.meta else z.detach().clone()
        self.pending = None
        self.transitions.clear()
        self.completed_steps = self.updates = 0

    def act(self, visual, proprio, previous, base_action):
        """Choose an action BEFORE its outcome exists; observe() must follow."""
        if self.pending is not None:
            raise RuntimeError("The previous action still lacks its observed outcome")
        if len(visual) != self.batch_size:
            raise ValueError("Batch membership cannot change within an episode")
        self.pending = {
            "visual": visual.detach().clone(),
            "proprio": proprio.detach().clone(),
            "previous": previous.detach().clone(),
        }
        with torch.set_grad_enabled(self.meta):
            return self.net.action(visual, proprio, previous, base_action, self.z, self.correction_scale)

    def observe(self, next_visual, next_proprio, executed_action):
        """Consume one completed transition and update for the NEXT action.

        executed_action must include any clipping or fixed-gripper intervention;
        predictions must use the action that actually reached the environment.
        """
        if self.pending is None:
            raise RuntimeError("No preceding action to associate with this feedback")
        row = dict(self.pending)
        row.update(next_visual=next_visual.detach().clone(),
                   next_proprio=next_proprio.detach().clone(),
                   actions=executed_action.detach().clone())
        if any(not bool(torch.isfinite(v).all()) for v in row.values()):
            raise ValueError("Non-finite feedback cannot update a latent")
        self.pending = None
        self.transitions.append(row)
        self.completed_steps += 1
        info = {"completed_step": self.completed_steps,
                "window_steps": len(self.transitions), "updated": False}
        if self.mode == "fixed" or (self.mode == "first3" and self.updates >= 3) or self.completed_steps < self.warmup_steps:
            return info
        before = self.z.detach().clone()
        if self.mode == "history":
            # Consume each transition ONCE; carry GRU state, without replaying an
            # overlapping window into the recurrent hidden state multiple times.
            with torch.set_grad_enabled(self.meta):
                u = self.net.encode(row["visual"], row["proprio"], row["previous"])
                target, weight = feedback(row["visual"], row["proprio"],
                                          row["next_visual"], row["next_proprio"])
                token = self.net.token(torch.cat((u, row["actions"], target * weight), -1))
                _, hidden = self.net.gru(token[:, None], self.z[None].contiguous())
                self.z = hidden[0]
        else:
            s = {k: torch.stack([r[k] for r in self.transitions], 1) for k in row}
            with torch.enable_grad():
                z = self.z if self.meta else self.z.detach().requires_grad_(True)
                u = self.net.encode(s["visual"], s["proprio"], s["previous"])
                target, weight = feedback(s["visual"], s["proprio"],
                                          s["next_visual"], s["next_proprio"])
                if self.net.feedback_mode == "robot":
                    weight = weight.clone()
                    weight[..., :24] = 0
                pred = self.net.prediction(u, s["actions"], z[:, None].expand(-1, u.shape[1], -1))
                loss = 64 * ((pred - target.detach()).square() * weight.detach()).sum((1, 2))
                loss = loss / weight.sum((1, 2)).clamp_min(1)
                grad, = torch.autograd.grad(loss.sum(), z, create_graph=self.meta)
                norms = grad.norm(dim=-1, keepdim=True)
                grad = grad * (self.grad_clip / norms.clamp_min(self.grad_clip))
                candidate = z - self.net.alpha * grad
                delta = candidate - z
                dnorm = delta.norm(dim=-1, keepdim=True)
                candidate = z + delta * (self.trust_radius / dnorm.clamp_min(self.trust_radius))
                pred_after = self.net.prediction(u, s["actions"], candidate[:, None].expand(-1, u.shape[1], -1))
                loss_after = 64 * ((pred_after - target.detach()).square() * weight.detach()).sum((1, 2))
                loss_after = loss_after / weight.sum((1, 2)).clamp_min(1)
                accepted = loss_after < loss.detach()
                self.z = torch.where(accepted[:, None], candidate, z)
                info["loss_before_update"] = loss.detach().cpu().tolist()
                info["loss_after_update"] = loss_after.detach().cpu().tolist()
                info["accepted"] = accepted.detach().cpu().tolist()
                info["gradient_norm_before_clip"] = norms.detach().flatten().cpu().tolist()
        if not bool(torch.isfinite(self.z).all()):
            raise FloatingPointError("Non-finite online latent")
        if not self.meta:
            self.z = self.z.detach()
        self.updates += 1
        info.update(updated=True, updates=self.updates,
                    z_delta_rms=float((self.z.detach() - before).square().mean().sqrt()))
        return info

    def detach_segment(self):
        """Truncate training graphs without resetting the learned latent value."""
        if self.pending is not None:
            raise RuntimeError("Finish the physical transition before truncation")
        self.z = self.z.detach().requires_grad_(self.meta and self.mode != "history")
