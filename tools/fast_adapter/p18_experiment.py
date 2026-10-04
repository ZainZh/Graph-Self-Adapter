
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Protocol, Sequence

import torch
from torch import Tensor, nn

P18_ARMS = ("base", "fixed_z", "gradient_z", "history", "hybrid", "oracle")
K_VALUES = (0, 1, 2, 3, 5)
PRIMARY_SUPPORT_PROTOCOL = (
    "shared_fixed_calibration_actions",
    "actions are independent of state and hidden M,b; K uses frozen prefixes 0/1/2/3/5",
    "query uses normal base+adapter on fresh positions with z/context fixed",
)


@dataclass(frozen=True)
class P18ModelConfig:
    obj_dino_dim: int = 1024
    target_dino_dim: int = 1024
    proprio_dim: int = 0
    roi_dim: int = 0
    action_dim: int = 2
    u_dim: int = 256
    z_dim: int = 32
    residual_width: int = 256
    history_width: int = 256
    max_residual: float = 0.15
    inner_alpha_init: float = 0.5
    lambda_z: float = 1.0e-4


@dataclass(frozen=True)
class P18PhysicsConfig:
    bias_abs: float = 0.04
    state_abs: float = 0.6
    target_abs: float = 0.6
    nominal_gain: float = 0.5
    action_limit: float = 1.0
    residual_limit: float = 0.15
    query_horizon: int = 12
    success_threshold: float = 0.05


@dataclass(frozen=True)
class P18HiddenShift:
    M: Tensor
    b: Tensor


@dataclass(frozen=True)
class P18FeatureBatch:
    """Runtime-visible visual features for one vectorized time step.

    ``obj_dino`` and ``target_dino`` must come from the frozen DINO/ROI path.
    Hidden action-response parameters, success labels, and oracle actions are
    deliberately absent.
    """

    obj_dino: Tensor
    target_dino: Tensor
    proprio: Tensor
    prev_action: Tensor
    roi_geometry: Tensor
    base_action: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.base_action.shape[0])


@dataclass(frozen=True)
class P18TransitionRecord:
    u: Tensor
    action: Tensor
    u_next: Tensor
    physical_fingerprint: Optional[str] = None


class P18VisualEnvProtocol(Protocol):
    """Adapter boundary for the visual alignment environment supplied by records.

    The experiment runner should implement this protocol around the actual RGB
    renderer and frozen DINO ROI extractor. Runtime methods must not expose the
    hidden ``M,b`` action-response parameters to model decisions.
    """

    def reset(self, batch_size: int, seed: int, *, same_hidden_as: object | None = None) -> tuple[P18FeatureBatch, object]: ...

    def step(self, state: object, action: Tensor) -> tuple[P18FeatureBatch, object]: ...

    def oracle_action(self, state: object) -> Tensor: ...

    def goal_error(self, state: object) -> Tensor: ...

    def success(self, state: object) -> Tensor: ...


class VisualInteractionProjector(nn.Module):
    def __init__(self, cfg: P18ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.obj_dino_dim + cfg.target_dino_dim + cfg.proprio_dim + cfg.action_dim + cfg.roi_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, cfg.u_dim),
            nn.LayerNorm(cfg.u_dim),
        )

    def forward(self, batch: P18FeatureBatch) -> Tensor:
        return self.net(torch.cat([batch.obj_dino, batch.target_dino, batch.proprio, batch.prev_action, batch.roi_geometry], dim=-1))


class ResidualActor(nn.Module):
    def __init__(self, cfg: P18ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("max_residual", torch.full((cfg.action_dim,), float(cfg.max_residual)))
        self.net = nn.Sequential(
            nn.Linear(cfg.u_dim + cfg.z_dim + cfg.action_dim, cfg.residual_width),
            nn.GELU(),
            nn.Linear(cfg.residual_width, cfg.residual_width),
            nn.GELU(),
            nn.Linear(cfg.residual_width, cfg.action_dim),
        )

    def forward(self, u: Tensor, z: Tensor, base_action: Tensor) -> Tensor:
        raw = self.net(torch.cat([u, z, base_action], dim=-1))
        return self.max_residual.to(raw.device) * torch.tanh(raw)


class DinoFuturePredictor(nn.Module):
    def __init__(self, cfg: P18ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.u_dim + cfg.action_dim + cfg.z_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, cfg.u_dim),
        )

    def forward(self, u: Tensor, action: Tensor, z: Tensor) -> Tensor:
        return u + self.net(torch.cat([u, action, z], dim=-1))


class HistoryInitializer(nn.Module):
    def __init__(self, cfg: P18ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.transition_proj = nn.Sequential(
            nn.Linear(cfg.u_dim + cfg.action_dim + cfg.u_dim, cfg.history_width),
            nn.GELU(),
            nn.Linear(cfg.history_width, cfg.z_dim),
        )
        self.rnn = nn.GRUCell(cfg.z_dim, cfg.z_dim)

    def initial_context(self, batch_size: int, *, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.cfg.z_dim, device=device)

    def update(self, context: Tensor, record: P18TransitionRecord) -> Tensor:
        transition = torch.cat([record.u, record.action.detach(), record.u_next - record.u], dim=-1)
        return self.rnn(self.transition_proj(transition), context)

    def forward(self, records: Sequence[P18TransitionRecord], *, batch_size: int, device: torch.device) -> Tensor:
        context = self.initial_context(batch_size, device=device)
        for record in records:
            context = self.update(context, record)
        return context


class P18FastSystem(nn.Module):
    def __init__(self, cfg: P18ModelConfig, *, use_history: bool, use_gradient: bool):
        super().__init__()
        self.cfg = cfg
        self.use_history = bool(use_history)
        self.use_gradient = bool(use_gradient)
        self.projector = VisualInteractionProjector(cfg)
        self.actor = ResidualActor(cfg)
        self.dynamics = DinoFuturePredictor(cfg) if use_gradient else None
        self.history = HistoryInitializer(cfg) if use_history else None
        self.z0 = nn.Parameter(torch.zeros(cfg.z_dim))
        self.log_inner_alpha = nn.Parameter(torch.log(torch.tensor(float(cfg.inner_alpha_init)))) if use_gradient else None

    @property
    def inner_alpha(self) -> Tensor:
        if self.log_inner_alpha is None:
            return torch.tensor(0.0, device=self.z0.device)
        return self.log_inner_alpha.exp()

    def project(self, batch: P18FeatureBatch) -> Tensor:
        return self.projector(batch)

    def initial_latent(self, records: Sequence[P18TransitionRecord], *, batch_size: int, device: torch.device) -> Tensor:
        if self.history is None:
            return self.z0.unsqueeze(0).expand(batch_size, -1)
        return self.history(records, batch_size=batch_size, device=device)

    def act_from_u(self, u: Tensor, z: Tensor, base_action: Tensor) -> tuple[Tensor, Tensor]:
        delta = self.actor(u, z, base_action)
        return base_action + delta, delta

    def act(self, batch: P18FeatureBatch, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        u = self.project(batch)
        action, delta = self.act_from_u(u, z, batch.base_action)
        return action, delta, u

    def freeze_for_eval(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class OracleShiftActor(nn.Module):
    """Upper-bound interface: consumes explicit shift embedding outside runtime arms."""

    def __init__(self, cfg: P18ModelConfig, shift_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.u_dim + cfg.action_dim + shift_dim, cfg.residual_width),
            nn.GELU(),
            nn.Linear(cfg.residual_width, cfg.action_dim),
        )
        self.register_buffer("max_residual", torch.full((cfg.action_dim,), float(cfg.max_residual)))

    def forward(self, u: Tensor, base_action: Tensor, shift_embedding: Tensor) -> Tensor:
        raw = self.net(torch.cat([u, base_action, shift_embedding], dim=-1))
        return base_action + self.max_residual.to(raw.device) * torch.tanh(raw)


def build_gradient_system(cfg: P18ModelConfig | None = None) -> P18FastSystem:
    return P18FastSystem(cfg or P18ModelConfig(), use_history=False, use_gradient=True)


def build_history_system(cfg: P18ModelConfig | None = None) -> P18FastSystem:
    return P18FastSystem(cfg or P18ModelConfig(), use_history=True, use_gradient=False)


def build_hybrid_system(cfg: P18ModelConfig | None = None) -> P18FastSystem:
    return P18FastSystem(cfg or P18ModelConfig(), use_history=True, use_gradient=True)


def encode_transition_records(
    system: P18FastSystem,
    current: Sequence[P18FeatureBatch],
    actions: Sequence[Tensor],
    future: Sequence[P18FeatureBatch],
    *,
    physical_fingerprints: Optional[Sequence[str]] = None,
) -> List[P18TransitionRecord]:
    if not (len(current) == len(actions) == len(future)):
        raise ValueError("current/actions/future lengths must match")
    records: List[P18TransitionRecord] = []
    for index, (cur, action, fut) in enumerate(zip(current, actions, future)):
        records.append(P18TransitionRecord(
            u=system.project(cur),
            action=action.detach(),
            u_next=system.project(fut),
            physical_fingerprint=None if physical_fingerprints is None else physical_fingerprints[index],
        ))
    return records


def prediction_terms(system: P18FastSystem, records: Sequence[P18TransitionRecord], z: Tensor, z_reference: Tensor, cfg: P18ModelConfig) -> Dict[str, Tensor]:
    if system.dynamics is None:
        raise ValueError("gradient adaptation requires a dynamics predictor")
    if not records:
        zeros = torch.zeros(z.shape[0], device=z.device)
        return {"mse": zeros, "z_reg": zeros, "total": zeros}
    mse_sum = torch.zeros(z.shape[0], device=z.device)
    for record in records:
        pred = system.dynamics(record.u, record.action.detach(), z)
        mse_sum = mse_sum + (pred - record.u_next.detach()).square().mean(dim=-1)
    mse = mse_sum / float(len(records))
    z_reg = (z - z_reference.detach()).square().mean(dim=-1)
    return {"mse": mse, "z_reg": z_reg, "total": mse + cfg.lambda_z * z_reg}


def adapt_latent(
    system: P18FastSystem,
    records: Sequence[P18TransitionRecord],
    z_init: Tensor,
    *,
    steps: int,
    cfg: P18ModelConfig | None = None,
    create_graph: bool,
    detach_initial: bool = False,
) -> tuple[Tensor, List[Dict[str, float]]]:
    """Sequential P18 test-time adaptation over the frozen K-prefix.

    Each support transition contributes exactly one observed prediction loss and
    one z update. This matches the primary protocol for K=0/1/2/3/5 fixed probe
    prefixes and avoids repeatedly optimizing on the same batch unless a later
    protocol explicitly asks for that diagnostic.
    """
    cfg = cfg or system.cfg
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps > len(records):
        raise ValueError(f"requested {steps} updates but only {len(records)} records are available")
    z = z_init.detach().clone().requires_grad_(True) if detach_initial else z_init.clone().requires_grad_(True)
    metrics: List[Dict[str, float]] = []
    for step, record in enumerate(records[:steps]):
        terms = prediction_terms(system, [record], z, z_init, cfg)
        grad_z = torch.autograd.grad(terms["total"].sum(), z, create_graph=create_graph, retain_graph=create_graph, only_inputs=True)[0]
        z = z - system.inner_alpha * grad_z
        post = prediction_terms(system, [record], z, z_init, cfg)
        metrics.append({
            "step": step,
            "support_record_index": step,
            "pre_mse": float(terms["mse"].detach().mean().cpu()),
            "post_mse": float(post["mse"].detach().mean().cpu()),
            "grad_z_norm": float(torch.linalg.norm(grad_z.detach(), dim=-1).mean().cpu()),
        })
    return z, metrics


def prepare_latent_for_arm(
    system: P18FastSystem,
    arm: Literal["fixed_z", "gradient_z", "history", "hybrid"],
    records: Sequence[P18TransitionRecord],
    *,
    batch_size: int,
    k: int,
    create_graph: bool,
    detach_initial: bool = False,
) -> tuple[Tensor, List[Dict[str, float]]]:
    device = system.z0.device
    if arm in {"history", "hybrid"}:
        z_init = system.initial_latent(records[:k], batch_size=batch_size, device=device)
    else:
        z_init = system.z0.unsqueeze(0).expand(batch_size, -1)
    if arm in {"gradient_z", "hybrid"} and k > 0:
        return adapt_latent(system, records[:k], z_init, steps=k, create_graph=create_graph, detach_initial=detach_initial)
    return (z_init.detach().clone() if detach_initial else z_init), []


def action_imitation_loss(system: P18FastSystem, query: P18FeatureBatch, z: Tensor, oracle_action: Tensor, *, residual_reg: float = 1.0e-3) -> tuple[Tensor, Dict[str, float]]:
    action, delta, _u = system.act(query, z)
    loss = (action - oracle_action.detach()).square().mean() + residual_reg * delta.square().mean()
    return loss, {
        "action_mse": float((action.detach() - oracle_action.detach()).square().mean().cpu()),
        "residual_norm": float(torch.linalg.norm(delta.detach(), dim=-1).mean().cpu()),
    }


def swapped_latent(z_init_current: Tensor, z_adapted: Tensor, source_perm: Tensor) -> Tensor:
    return z_init_current + (z_adapted - z_init_current)[source_perm]


def bounded_axis_probe_actions(*, amplitude: float = 0.25, diagonal: float = 0.18, action_dim: int = 2, device: torch.device | str = "cpu") -> Tensor:
    """Frozen primary shared-support action design for P18.

    The order is part of the protocol: K prefixes use the first 0/1/2/3/5
    rows. These actions are fixed calibration probes and must not depend on
    state, hidden physics, model arm, or previous visual outcomes.
    """
    if action_dim != 2:
        raise ValueError("P18 first validation assumes 2D actions")
    a = float(amplitude)
    d = float(diagonal)
    return torch.tensor([[a, 0.0], [0.0, a], [-a, 0.0], [0.0, -a], [d, d]], device=device)


def observable_action_design_rank(actions: Tensor) -> Dict[str, object]:
    if actions.ndim != 2:
        raise ValueError("actions must have shape [K, action_dim]")
    design = torch.cat([actions.detach().double(), torch.ones(actions.shape[0], 1, device=actions.device, dtype=torch.float64)], dim=-1)
    singular = torch.linalg.svdvals(design)
    tol = torch.finfo(torch.float64).eps * max(design.shape) * singular[0].clamp_min(torch.finfo(torch.float64).tiny)
    rank = int((singular > tol).sum().item())
    condition = float((singular[0] / singular[-1].clamp_min(torch.finfo(torch.float64).tiny)).cpu())
    return {
        "rank": rank,
        "required_rank_for_full_2x2M_plus_b": 3,
        "identifiable_full_affine_response": rank >= 3,
        "condition": condition,
        "singular_values": [float(x) for x in singular.cpu()],
    }


def ablate_dino(batch: P18FeatureBatch, mode: Literal["zero", "constant"]) -> P18FeatureBatch:
    if mode == "zero":
        obj = torch.zeros_like(batch.obj_dino)
        target = torch.zeros_like(batch.target_dino)
    elif mode == "constant":
        obj = torch.ones_like(batch.obj_dino) * batch.obj_dino.detach().mean()
        target = torch.ones_like(batch.target_dino) * batch.target_dino.detach().mean()
    else:
        raise ValueError(mode)
    return P18FeatureBatch(obj_dino=obj, target_dino=target, proprio=batch.proprio, prev_action=batch.prev_action, roi_geometry=batch.roi_geometry, base_action=batch.base_action)


def physical_support_fingerprint(*, rgb: Tensor, action: Tensor, next_rgb: Tensor, roi_geometry: Tensor, proprio: Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in [rgb, action, next_rgb, roi_geometry, proprio]:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _uniform(shape: tuple[int, ...], low: float, high: float, *, generator: torch.Generator, device: torch.device) -> Tensor:
    return torch.empty(shape, device=device).uniform_(low, high, generator=generator)


def _signed_band(shape: tuple[int, ...], low: float, high: float, *, generator: torch.Generator, device: torch.device) -> Tensor:
    mag = _uniform(shape, low, high, generator=generator, device=device)
    sign = torch.where(torch.randint(0, 2, shape, generator=generator, device=device, dtype=torch.bool), torch.ones(shape, device=device), -torch.ones(shape, device=device))
    return sign * mag


def sample_p18_shifts(num_domains: int, seed: int, stratum: Literal["train", "id", "mild", "strong"], *, device: torch.device | str = "cpu", cfg: P18PhysicsConfig | None = None) -> P18HiddenShift:
    cfg = cfg or P18PhysicsConfig()
    device = torch.device(device)
    g = torch.Generator(device=device).manual_seed(seed)
    if stratum in {"train", "id"}:
        theta = _uniform((num_domains,), -10.0, 10.0, generator=g, device=device) * math.pi / 180.0
        sx = _uniform((num_domains,), 0.8, 1.2, generator=g, device=device)
        sy = _uniform((num_domains,), 0.8, 1.2, generator=g, device=device)
        cxy = _uniform((num_domains,), -0.05, 0.05, generator=g, device=device)
        cyx = _uniform((num_domains,), -0.05, 0.05, generator=g, device=device)
    elif stratum == "mild":
        theta = _signed_band((num_domains,), 10.0, 20.0, generator=g, device=device) * math.pi / 180.0
        low_s = _uniform((num_domains, 2), 0.7, 0.8, generator=g, device=device)
        high_s = _uniform((num_domains, 2), 1.2, 1.3, generator=g, device=device)
        choose_high = torch.randint(0, 2, (num_domains, 2), generator=g, device=device, dtype=torch.bool)
        scale = torch.where(choose_high, high_s, low_s)
        sx, sy = scale[:, 0], scale[:, 1]
        cxy = _uniform((num_domains,), -0.1, 0.1, generator=g, device=device)
        cyx = _uniform((num_domains,), -0.1, 0.1, generator=g, device=device)
    elif stratum == "strong":
        theta = _signed_band((num_domains,), 20.0, 35.0, generator=g, device=device) * math.pi / 180.0
        low_s = _uniform((num_domains, 2), 0.7, 0.8, generator=g, device=device)
        high_s = _uniform((num_domains, 2), 1.2, 1.3, generator=g, device=device)
        choose_high = torch.randint(0, 2, (num_domains, 2), generator=g, device=device, dtype=torch.bool)
        scale = torch.where(choose_high, high_s, low_s)
        sx, sy = scale[:, 0], scale[:, 1]
        cxy = _signed_band((num_domains,), 0.1, 0.2, generator=g, device=device)
        cyx = _signed_band((num_domains,), 0.1, 0.2, generator=g, device=device)
    else:
        raise ValueError(stratum)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([cos_t, -sin_t], dim=-1), torch.stack([sin_t, cos_t], dim=-1)], dim=-2)
    D = torch.zeros(num_domains, 2, 2, device=device)
    D[:, 0, 0] = sx
    D[:, 1, 1] = sy
    C = torch.zeros(num_domains, 2, 2, device=device)
    C[:, 0, 0] = 1.0
    C[:, 1, 1] = 1.0
    C[:, 0, 1] = cxy
    C[:, 1, 0] = cyx
    M = R @ D @ C
    b = _uniform((num_domains, 2), -cfg.bias_abs, cfg.bias_abs, generator=g, device=device)
    return P18HiddenShift(M=M, b=b)


def sample_p18_positions(batch_size: int, seed: int, *, device: torch.device | str = "cpu", cfg: P18PhysicsConfig | None = None) -> tuple[Tensor, Tensor]:
    cfg = cfg or P18PhysicsConfig()
    device = torch.device(device)
    g = torch.Generator(device=device).manual_seed(seed)
    state = _uniform((batch_size, 2), -cfg.state_abs, cfg.state_abs, generator=g, device=device)
    target = _uniform((batch_size, 2), -cfg.target_abs, cfg.target_abs, generator=g, device=device)
    return state, target


def nominal_base_action_from_roi(state_center: Tensor, target_center: Tensor, *, cfg: P18PhysicsConfig | None = None) -> Tensor:
    cfg = cfg or P18PhysicsConfig()
    desired = cfg.nominal_gain * (target_center - state_center)
    return torch.clamp(desired, -cfg.action_limit, cfg.action_limit)


def step_p18_shift(state: Tensor, shift: P18HiddenShift, action: Tensor, *, cfg: P18PhysicsConfig | None = None) -> Tensor:
    cfg = cfg or P18PhysicsConfig()
    clipped = torch.clamp(action, -cfg.action_limit, cfg.action_limit)
    return state + torch.bmm(shift.M, clipped.unsqueeze(-1)).squeeze(-1) + shift.b


def projected_oracle_action(state_center: Tensor, target_center: Tensor, shift: P18HiddenShift, *, cfg: P18PhysicsConfig | None = None) -> Tensor:
    cfg = cfg or P18PhysicsConfig()
    base = nominal_base_action_from_roi(state_center, target_center, cfg=cfg)
    desired = cfg.nominal_gain * (target_center - state_center)
    unconstrained = torch.linalg.solve(shift.M, (desired - shift.b).unsqueeze(-1)).squeeze(-1)
    residual = torch.clamp(unconstrained - base, -cfg.residual_limit, cfg.residual_limit)
    return torch.clamp(base + residual, -cfg.action_limit, cfg.action_limit)


def trainable_parameter_groups(system: P18FastSystem) -> Dict[str, int]:
    groups = {
        "projector": sum(p.numel() for p in system.projector.parameters() if p.requires_grad),
        "actor": sum(p.numel() for p in system.actor.parameters() if p.requires_grad),
        "dynamics": 0 if system.dynamics is None else sum(p.numel() for p in system.dynamics.parameters() if p.requires_grad),
        "history": 0 if system.history is None else sum(p.numel() for p in system.history.parameters() if p.requires_grad),
        "z0": int(system.z0.numel()) if system.z0.requires_grad else 0,
        "log_inner_alpha": 0 if system.log_inner_alpha is None or not system.log_inner_alpha.requires_grad else int(system.log_inner_alpha.numel()),
    }
    groups["total"] = sum(groups.values())
    return groups
