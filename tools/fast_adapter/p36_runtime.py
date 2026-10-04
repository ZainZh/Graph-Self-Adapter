"""P36 causal runtime state: past-only z updates with frozen weights."""
from collections import deque

import torch

from prove.fast_adapter.geometry_role_dynamics import observed_effect, select_action


class GeometryRoleState:
    def __init__(self, net, batch, mode="continuous"):
        if mode not in ("fixed", "first3", "continuous"):
            raise ValueError(mode)
        if any(p.requires_grad for p in net.parameters()):
            raise ValueError("Freeze runtime network weights")
        self.net = net
        self.mode = mode
        self.z = net.z0[None].expand(batch, -1).detach().clone()
        self.obs = []
        self.actions = []
        self.contexts = []
        self.buffer = deque(maxlen=8)
        self.updates = 0

    @torch.no_grad()
    def act(self, ob, base):
        self.obs.append({k: v.detach() for k, v in ob.items()})
        previous = self.actions[-1] if self.actions else torch.zeros_like(base)
        velocity = ob["boxes"] - self.obs[-2]["boxes"] if len(self.obs) > 1 else torch.zeros_like(ob["boxes"])
        u = self.net.encode(ob["tokens"], ob["boxes"], ob["proprio"], previous, velocity)
        self.contexts.append(u)
        return select_action(self.net, u, base, self.z)

    @torch.no_grad()
    def observe(self, actual_action, next_ob):
        self.actions.append(actual_action.detach())
        t = len(self.actions)
        diag = {"updated": False, "source_start": None, "source_end": t, "count": self.updates}
        if t >= 4:
            start = t - 4
            old = self.obs[start]
            plan = torch.stack(self.actions[start:t], 1)
            mean, gate = self.net.coefficients(self.contexts[start], plan)
            y, w = observed_effect(
                old["tokens"],
                old["boxes"],
                old["proprio"],
                next_ob["tokens"],
                next_ob["boxes"],
                next_ob["proprio"],
            )
            self.buffer.append((mean, gate, y / self.net.effect_sd, w))
            diag["source_start"] = start
            if self.mode == "continuous" or (self.mode == "first3" and self.updates < 3):
                mean_w, gate_w, y_w, weight_w = [torch.stack(x, 1) for x in zip(*self.buffer)]
                next_z, update_diag = self.net.update(self.z, mean_w, gate_w, y_w, weight_w)
                self.z = next_z.detach()
                self.updates += 1
                diag.update(update_diag, updated=True, count=self.updates)
        return diag
