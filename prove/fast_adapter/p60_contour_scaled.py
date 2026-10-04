"""P60-B2 scale-matched optimizer for the frozen P60 contour candidate.

Only the numerical units of the Gauss-Newton solve change.  Public inputs,
residuals, local nearest-segment assignments, persistent z, frozen P45 model,
history, three inner steps, step multiplier and norm limit are inherited byte
for byte from :mod:`prove.fast_adapter.p60_contour`.
"""
from __future__ import annotations

import torch

from prove.fast_adapter.p60_contour import P60ContourLatent


class P60ContourScaledLatent(P60ContourLatent):
    """Normalize H and g by mean latent curvature before fixed damping."""

    def _adapt(self, reference, offset, start_z, support, steps):
        z = start_z.detach()
        records = []
        latent_dim = int(z.shape[-1])
        identity = torch.eye(latent_dim, dtype=z.dtype, device=z.device)
        for _ in range(steps):
            z = z.detach().requires_grad_(True)
            residual = self._residual(reference, offset, z, support)
            loss = .5 * residual.square().mean()
            gradient, = torch.autograd.grad(loss, z, retain_graph=False)

            def residual_at(active):
                return self._residual(reference, offset, active, support)

            jacobian = torch.func.jacfwd(residual_at)(z).squeeze(-2)
            metric = jacobian.T @ jacobian / max(1, residual.numel())
            curvature_scale = (torch.trace(metric) / latent_dim).clamp_min(1e-12)
            normalized_metric = metric / curvature_scale
            normalized_gradient = gradient / curvature_scale
            damped_metric = normalized_metric + identity * self.damping
            change = torch.linalg.solve(
                damped_metric, normalized_gradient.reshape(-1, 1)).reshape_as(z)
            norm = change.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            change = change * torch.clamp(self.step_clip / norm, max=1.0)
            next_z = (z - self.step_scale * change).detach()
            records.append({
                "loss": float(loss.detach()),
                "gradient": float(gradient.norm().detach()),
                "delta": float((next_z - z.detach()).norm()),
                "curvature_scale": float(curvature_scale.detach()),
                "normalized_damping": float(self.damping),
                "local_correspondence": "nearest cyclic segment",
            })
            z = next_z
        return z, records
