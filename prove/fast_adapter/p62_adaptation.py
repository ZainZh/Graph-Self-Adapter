"""P62 bounded latent-subspace updates over the frozen P61 geometry model.

This module changes only which components of the persistent z8 may respond to
the same public contour residual.  Encoder, geometry decoder, projection,
support history, P50 controller and three-step update budget remain unchanged.
The spectral arm changes only the linear solve: it retains local contour
Jacobian eigendirections whose eigenvalues reach one percent of the largest
eigenvalue. This is a numerical local-sensitivity filter, not proof that the
retained directions are physically identifiable.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
from torch import Tensor

from prove.fast_adapter.p60_contour import ContourMeasurement
from prove.fast_adapter.p61_adaptation import contour_residual


FIXED = "fixed"
# Import compatibility for early P62 evaluator drafts; the canonical ID is
# strictly ``fixed`` and only that value appears in CANDIDATES/contracts.
FIXED_ZERO = FIXED
LEGACY_FULL8 = "legacy_full8"
TRANSLATION3 = "translation3"
TRANSLATION_SIZE5 = "translation_size5"
FULL8_SPECTRAL = "full8_spectral"
CANDIDATES = (FIXED, LEGACY_FULL8, TRANSLATION3,
              TRANSLATION_SIZE5, FULL8_SPECTRAL)
ACTIVE_INDICES = {
    FIXED: (),
    LEGACY_FULL8: tuple(range(8)),
    TRANSLATION3: (0, 1, 2),
    TRANSLATION_SIZE5: (0, 1, 2, 6, 7),
    FULL8_SPECTRAL: tuple(range(8)),
}


def candidate_configuration(candidate: str,
                            config: "P62InnerConfig" | None = None) -> dict[str, object]:
    """Serializable, result-free identity of one P62 candidate."""
    config = P62InnerConfig() if config is None else config
    indices = _validate(candidate, config)
    return {
        "candidate": candidate,
        "active_indices": list(indices),
        "active_dim": len(indices),
        "steps": config.steps,
        "damping": config.damping,
        "step_scale": config.step_scale,
        "step_clip": config.step_clip,
        "samples_per_edge": config.samples_per_edge,
        "curvature_scale": f"trace(H)/{len(indices)}" if indices else None,
        "spectral_relative_threshold": (config.spectral_relative_threshold
                                        if candidate == FULL8_SPECTRAL else None),
        "spectral_zero_threshold": (config.spectral_zero_threshold
                                    if candidate == FULL8_SPECTRAL else None),
        "spectral_interpretation": ("local contour sensitivity; not proven identifiability"
                                    if candidate == FULL8_SPECTRAL else None),
    }


@dataclass(frozen=True)
class P62InnerConfig:
    steps: int = 3
    damping: float = 1e-4
    step_scale: float = 0.65
    step_clip: float = 1.0
    samples_per_edge: int = 8
    spectral_relative_threshold: float = 0.01
    spectral_zero_threshold: float = 1e-12


def _validate(candidate: str, config: P62InnerConfig) -> tuple[int, ...]:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown P62 candidate {candidate!r}; expected {CANDIDATES}")
    if config.steps != 3:
        raise ValueError("P62 fixes exactly three update steps")
    values = (config.damping, config.step_scale, config.step_clip,
              config.spectral_relative_threshold, config.spectral_zero_threshold)
    if (not all(math.isfinite(value) for value in values)
            or config.damping <= 0 or config.step_scale <= 0 or config.step_clip <= 0
            or not 0 < config.spectral_relative_threshold <= 1
            or config.spectral_zero_threshold <= 0):
        raise ValueError("P62 update constants must be finite and positive")
    return ACTIVE_INDICES[candidate]


def _expand_active(active: Tensor, indices: tuple[int, ...]) -> Tensor:
    """Pad an active latent into z8; inactive coordinates are structural zero."""
    if active.shape[-1] != len(indices):
        raise ValueError("Active latent width does not match candidate subspace")
    full = active.new_zeros(*active.shape[:-1], 8)
    if indices:
        index = torch.tensor(indices, dtype=torch.long, device=active.device)
        full = full.scatter(-1, index.expand(*active.shape[:-1], len(indices)), active)
    return full


def _block_norms(value: Tensor, indices: tuple[int, ...]) -> dict[str, float]:
    full = _expand_active(value, indices)
    return {"translation": float(full[:3].norm().detach()),
            "rotation": float(full[3:6].norm().detach()),
            "log_size": float(full[6:8].norm().detach())}


def _adapt_residual(
    residual_from_z: Callable[[Tensor], Tensor], start_z: Tensor, *,
    candidate: str, config: P62InnerConfig, create_graph: bool,
) -> tuple[Tensor, list[dict[str, float | int | list[int] | str]]]:
    """Core active-block solver, separated for numerical contract tests."""
    indices = _validate(candidate, config)
    if start_z.shape != (8,):
        raise ValueError("P62 adapts one episode z8 at a time")
    if not torch.isfinite(start_z).all():
        raise ValueError("P62 start_z must be finite")
    if not indices:
        return torch.zeros_like(start_z), []
    index = torch.tensor(indices, dtype=torch.long, device=start_z.device)
    source = start_z if create_graph else start_z.detach()
    active = source.index_select(0, index)
    identity = torch.eye(len(indices), dtype=start_z.dtype, device=start_z.device)
    records: list[dict[str, float | int | list[int] | str]] = []
    for _ in range(config.steps):
        if not active.requires_grad:
            active = active.detach().requires_grad_(True)

        def residual_at(value: Tensor) -> Tensor:
            return residual_from_z(_expand_active(value, indices))

        residual = residual_at(active)
        if residual.numel() == 0 or not torch.isfinite(residual).all():
            raise FloatingPointError("P62 public contour residual is empty or non-finite")
        jacobian = torch.func.jacfwd(residual_at)(active).to(dtype=residual.dtype)
        count = residual.numel()
        metric = jacobian.transpose(-1, -2) @ jacobian / count
        gradient = jacobian.transpose(-1, -2) @ residual / count
        curvature = (metric.trace() / len(indices)).clamp_min(1e-12)
        eigenvalues = torch.linalg.eigvalsh(metric)
        spectral_rank = len(indices)
        if candidate == FULL8_SPECTRAL:
            eigenvalues, eigenvectors = torch.linalg.eigh(metric)
            maximum = eigenvalues[-1]
            if bool((maximum.detach() <= config.spectral_zero_threshold)):
                change = torch.zeros_like(active)
                spectral_rank = 0
            else:
                keep = eigenvalues >= config.spectral_relative_threshold * maximum
                spectral_rank = int(keep.detach().sum())
                basis = eigenvectors[:, keep]
                inverse = 1.0 / (eigenvalues[keep] + config.damping * curvature)
                change = basis @ (inverse * (basis.transpose(-1, -2) @ gradient))
        else:
            system = metric / curvature + config.damping * identity
            change = torch.linalg.solve(system, gradient / curvature)
        norm = change.square().sum().clamp_min(1e-24).sqrt()
        change = change * (config.step_clip / norm).clamp(max=1.0)
        next_active = active - config.step_scale * change
        next_z = _expand_active(next_active, indices)
        records.append({
            "loss": float((0.5 * residual.square().mean()).detach()),
            "gradient_norm": float(gradient.norm().detach()),
            "curvature_scale": float(curvature.detach()),
            "delta_norm": float((next_active - active).norm().detach()),
            "active_dim": len(indices),
            "active_indices": list(indices),
            "eigenvalues": [float(value) for value in eigenvalues.detach().cpu()],
            "retained_rank": spectral_rank,
            "gradient_block_norms": _block_norms(gradient, indices),
            "delta_block_norms": _block_norms(next_active - active, indices),
            "z_block_norms": _block_norms(next_active, indices),
            "spectral_interpretation": ("local contour sensitivity; not proven identifiability"
                                        if candidate == FULL8_SPECTRAL else "not filtered"),
        })
        active = next_active if create_graph else next_active.detach()
    return _expand_active(active, indices), records


def adapt_z(
    geometry8: Tensor,
    start_z: Tensor,
    T_world_anchor: Tensor,
    measurements: Sequence[ContourMeasurement],
    *,
    candidate: str,
    config: P62InnerConfig = P62InnerConfig(),
    create_graph: bool = False,
) -> tuple[Tensor, list[dict[str, float | int | list[int] | str]]]:
    """Apply one fixed three-step P62 update from public contour history."""
    _validate(candidate, config)
    if geometry8.shape != (8,) or T_world_anchor.shape != (4, 4):
        raise ValueError("P62 expects one geometry8 and one 4x4 fixed anchor")
    if not measurements:
        if candidate == FIXED:
            return torch.zeros_like(start_z), []
        indices = ACTIVE_INDICES[candidate]
        active = start_z.index_select(0, torch.tensor(indices, device=start_z.device))
        return _expand_active(active, indices), []

    def residual_from_z(z: Tensor) -> Tensor:
        return contour_residual(geometry8, z, T_world_anchor, measurements, config)

    return _adapt_residual(residual_from_z, start_z, candidate=candidate,
                           config=config, create_graph=create_graph)
