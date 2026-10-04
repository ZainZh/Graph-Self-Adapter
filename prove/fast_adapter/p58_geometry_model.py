"""Shared P58 low-rank geometry adapter for task-local semantic points."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GeometryAdapterConfig:
    point_count: int
    latent_dim: int = 8
    inner_steps: int = 3
    inner_lr: float = .65
    damping: float = 1e-4
    step_cap: float = 1.
    geometry_scale: float = .01


class LowRankRayAdapter(nn.Module):
    """Decode ``g + scale*Bz`` and fit z to calibrated point rays.

    ``g`` is ``[batch, points, 3]``. Ray projectors/origins are
    ``[batch, support, points, 3, 3]`` and ``[..., 3]``; validity is
    ``[batch, support, points]``. Loss normalization is over every valid ray
    within each batch element, matching the P57 global-valid convention.
    """
    def __init__(self, basis, config: GeometryAdapterConfig):
        super().__init__()
        if config.point_count < 1 or config.latent_dim < 1:
            raise ValueError("P58 point and latent dimensions must be positive")
        basis = torch.as_tensor(basis, dtype=torch.float32)
        if basis.shape != (config.point_count * 3, config.latent_dim):
            raise ValueError("Basis must be [3*point_count, latent_dim]")
        self.config = config
        self.B = nn.Parameter(basis.clone())
        self.z0 = nn.Parameter(torch.zeros(config.latent_dim))

    def decode(self, g, z):
        if g.shape[-2:] != (self.config.point_count, 3) or z.shape[-1] != self.config.latent_dim:
            raise ValueError("Invalid P58 geometry or latent shape")
        delta = torch.einsum("dk,...k->...d", self.B, z)
        return g + self.config.geometry_scale * delta.reshape(*z.shape[:-1], self.config.point_count, 3)

    @staticmethod
    def _weights(A, valid):
        if valid is None:
            return torch.ones(A.shape[:-2], device=A.device, dtype=A.dtype)
        if valid.shape != A.shape[:-2]:
            raise ValueError("Ray validity shape differs from projectors")
        return valid.to(A.dtype)

    def ray_loss(self, g, z, A, C, valid=None):
        points = self.decode(g, z)
        residual = (points[:, None] - C) / self.config.geometry_scale
        error = torch.einsum("bspi,bspij,bspj->bsp", residual, A, residual)
        weight = self._weights(A, valid)
        return .5 * (error * weight).sum((1, 2)) / weight.sum((1, 2)).clamp_min(1)

    def _metric(self, A, valid=None):
        weight = self._weights(A, valid)
        denominator = weight.sum((1, 2)).clamp_min(1)
        H = (A * weight[..., None, None]).sum(1) / denominator[:, None, None, None]
        batch = H.shape[0]
        blocks = A.new_zeros((batch, self.config.point_count*3, self.config.point_count*3))
        for point in range(self.config.point_count):
            sl = slice(point*3, point*3+3)
            blocks[:, sl, sl] = H[:, point]
        basis = self.B.expand(batch, -1, -1)
        eye = torch.eye(self.config.latent_dim, device=A.device, dtype=A.dtype).expand(batch, -1, -1)
        return (basis.transpose(1, 2) @ blocks @ basis + self.config.damping * eye).detach()

    def adapt(self, g, z, A, C, valid=None, *, create_graph=False):
        metric = self._metric(A, valid)
        losses = []
        for _ in range(self.config.inner_steps):
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            per_item = self.ray_loss(g, z, A, C, valid)
            gradient, = torch.autograd.grad(per_item.sum(), z, create_graph=create_graph)
            direction = torch.linalg.solve(metric, gradient.unsqueeze(-1)).squeeze(-1)
            norm = direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            direction = direction * torch.clamp(self.config.step_cap / norm, max=1.)
            z = z - self.config.inner_lr * direction
            losses.append(per_item)
            if not create_graph:
                z = z.detach().requires_grad_(True)
        return z, torch.stack(losses, 1)

    def direct(self, g, z_prior, A, C, valid=None):
        """Regularized minimum-change solution in the identical B subspace."""
        weight = self._weights(A, valid)
        denominator = weight.sum((1, 2)).clamp_min(1)
        residual = (g[:, None] - C) / self.config.geometry_scale
        projected = torch.einsum("bspij,bspj->bspi", A, residual) * weight[..., None]
        gradient_points = projected.sum(1) / denominator[:, None, None]
        rhs = -torch.einsum("dk,bd->bk", self.B, gradient_points.reshape(len(g), -1))
        rhs = rhs + self.config.damping * z_prior
        return torch.linalg.solve(self._metric(A, valid), rhs.unsqueeze(-1)).squeeze(-1)
