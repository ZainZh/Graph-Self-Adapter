"""P61 canonical geometry correction and fully differentiable contour updates.

The only persistent test-time state is z8. Targets are observed outer contours;
no corner detector, PnP estimate, learned self-target, or evaluator label enters
this module. ``create_graph=True`` retains all three GN steps for meta-training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from prove.fast_adapter.p60_contour import ContourMeasurement, _measurement_residual
from prove.task_graph.p59.core import pack_geometry, so3_exp_map


@dataclass(frozen=True)
class P61InnerConfig:
    steps: int = 3
    damping: float = 1e-4
    step_scale: float = .65
    step_clip: float = 1.0
    samples_per_edge: int = 8


def corrected_components(geometry8: torch.Tensor, z: torch.Tensor):
    """Center, rotation and positive half sizes in the fixed episode anchor."""
    if geometry8.shape[-1] != 8 or z.shape[-1] != 8:
        raise ValueError("P61 geometry and persistent latent must both be 8D")
    scales = geometry8.new_tensor([.02] * 3 + [.25] * 3 + [.25] * 2)
    correction = z * scales
    center = geometry8[..., :3] + correction[..., :3]
    rotation = so3_exp_map(geometry8[..., 3:6]) @ so3_exp_map(correction[..., 3:6])
    half_size = (geometry8[..., 6:8] + correction[..., 6:8]).exp()
    return center, rotation, half_size


def world_geometry(geometry8: torch.Tensor, z: torch.Tensor,
                   T_world_anchor: torch.Tensor, held_length: torch.Tensor):
    center, rotation, half_size = corrected_components(geometry8, z)
    anchor_rotation = T_world_anchor[..., :3, :3]
    center = (anchor_rotation @ center[..., None])[..., 0] + T_world_anchor[..., :3, 3]
    rotation = anchor_rotation @ rotation
    return pack_geometry(center, rotation, half_size, held_length)


def world_corners(geometry8: torch.Tensor, z: torch.Tensor,
                  T_world_anchor: torch.Tensor):
    center, rotation, half_size = corrected_components(geometry8, z)
    signs = geometry8.new_tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    # Publicly declared frame construction prior, shared with the cache:
    # control geometry is the opening; the measured contour is its outer rim.
    half_size = half_size + .006
    center = center + rotation[..., :, 2] * .0006
    xy = signs * half_size[..., None, :]
    local = torch.cat((xy, torch.zeros_like(xy[..., :1])), dim=-1)
    anchored = local @ rotation.transpose(-1, -2) + center[..., None, :]
    return anchored @ T_world_anchor[..., :3, :3].transpose(-1, -2) + T_world_anchor[..., None, :3, 3]


def contour_residual(geometry8, z, T_world_anchor, measurements,
                     config=P61InnerConfig()):
    points = world_corners(geometry8, z, T_world_anchor)
    if points.ndim != 2:
        raise ValueError("one episode must be adapted at a time")
    return _measurement_residual(points, measurements, config.samples_per_edge)


def adapt_z(geometry8: torch.Tensor, start_z: torch.Tensor,
            T_world_anchor: torch.Tensor, measurements: Sequence[ContourMeasurement],
            *, config=P61InnerConfig(), create_graph: bool = False):
    """Exactly three normalized damped GN steps, with optional full outer graph.

    Only discrete nearest-segment assignments are stopped inside the residual.
    Jacobians, curvature normalization, solve and clipping remain differentiable
    during training. Runtime detaches between steps and freezes model weights.
    """
    if config.steps != 3:
        raise ValueError("P61 candidate fixes exactly three inner steps")
    if not measurements:
        return start_z, []
    z = start_z if create_graph else start_z.detach()
    identity = torch.eye(8, device=z.device, dtype=z.dtype)
    records = []
    for _ in range(config.steps):
        if not z.requires_grad:
            z = z.requires_grad_(True)
        residual_at = lambda active: contour_residual(geometry8, active, T_world_anchor,
                                                     measurements, config)
        residual = residual_at(z)
        # Forward AD is only 8 tangent directions and retains reverse-mode
        # dependencies on geometry and z, avoiding one backward per pixel.
        jacobian = torch.func.jacfwd(residual_at)(z)
        count = max(1, residual.numel())
        metric = jacobian.transpose(-1, -2) @ jacobian / count
        gradient = jacobian.transpose(-1, -2) @ residual / count
        curvature = (metric.trace() / 8).clamp_min(1e-12)
        change = torch.linalg.solve(metric / curvature + config.damping * identity,
                                    gradient / curvature)
        norm = change.square().sum().clamp_min(1e-24).sqrt()
        change = change * (config.step_clip / norm).clamp(max=1.0)
        next_z = z - config.step_scale * change
        records.append({"loss": float((.5 * residual.square().mean()).detach()),
                        "gradient_norm": float(gradient.norm().detach()),
                        "curvature_scale": float(curvature.detach()),
                        "delta_norm": float((next_z - z).norm().detach())})
        z = next_z if create_graph else next_z.detach()
    return z, records
