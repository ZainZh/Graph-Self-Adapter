from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class AdaptationLossConfig:
    lambda_cos: float = 0.1
    lambda_z: float = 1.0e-4


@dataclass(frozen=True)
class AdapterObservation:
    """Runtime observation exposed to the adapter.

    In the toy task, ``obj_feat`` and ``tar_feat`` are small state/target
    features rather than DINO outputs. Hidden domain gain/bias are deliberately
    absent from this structure.
    """

    obj_feat: Tensor
    tar_feat: Tensor
    proprio: Tensor
    prev_action: Tensor


@dataclass(frozen=True)
class InnerUpdateResult:
    z_next: Tensor
    loss: Tensor
    terms: Mapping[str, Tensor]
    grad_z: Tensor


class FrozenVisualEncoder(nn.Module):
    """Placeholder for a frozen dense visual encoder.

    P1 toy experiments do not use RGB or DINOv3. This class preserves the
    interface and enforces the freeze contract for later P2 wiring.
    """

    def __init__(self, in_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, feature_dim)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        return super().train(False)

    def forward(self, image_like: Tensor) -> Tensor:
        with torch.no_grad():
            return self.proj(image_like)


class InteractionEncoder(nn.Module):
    def __init__(
        self,
        obj_dim: int,
        tar_dim: int,
        proprio_dim: int,
        action_dim: int,
        dino_proj_dim: int = 128,
        u_dim: int = 256,
    ):
        super().__init__()
        self.u_dim = u_dim
        self.obj_proj = nn.Linear(obj_dim, dino_proj_dim)
        self.tar_proj = nn.Linear(tar_dim, dino_proj_dim)
        input_dim = dino_proj_dim * 2 + proprio_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, u_dim),
        )

    def forward(
        self,
        obj_feat: Tensor,
        tar_feat: Tensor,
        proprio: Tensor,
        prev_action: Tensor,
    ) -> Tensor:
        x = torch.cat(
            [
                self.obj_proj(obj_feat),
                self.tar_proj(tar_feat),
                proprio,
                prev_action,
            ],
            dim=-1,
        )
        return self.net(x)


class BaseSkillPolicy(nn.Module):
    def forward(self, obs: AdapterObservation, u: Tensor) -> Tensor:
        raise NotImplementedError


class ScriptedNominalPolicy2D(BaseSkillPolicy):
    """Frozen nominal policy: a = 0.5 * (target - state), clipped to [-1, 1]."""

    def __init__(self, action_limit: float = 1.0):
        super().__init__()
        self.action_limit = float(action_limit)

    def forward(self, obs: AdapterObservation, u: Tensor) -> Tensor:
        del u
        with torch.no_grad():
            action = 0.5 * (obs.tar_feat[..., :2] - obs.obj_feat[..., :2])
            return torch.clamp(action, -self.action_limit, self.action_limit)


class FastAdapter(nn.Module):
    def __init__(
        self,
        u_dim: int = 256,
        z_dim: int = 32,
        action_dim: int = 2,
        max_residual: float | Tensor = 0.15,
    ):
        super().__init__()
        max_residual_tensor = torch.as_tensor(max_residual, dtype=torch.float32)
        if max_residual_tensor.ndim == 0:
            max_residual_tensor = max_residual_tensor.repeat(action_dim)
        if tuple(max_residual_tensor.shape) != (action_dim,):
            raise ValueError("max_residual must be scalar or [action_dim]")
        self.register_buffer("max_residual", max_residual_tensor)
        self.net = nn.Sequential(
            nn.Linear(u_dim + z_dim + action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, u: Tensor, z: Tensor, a_base: Tensor) -> Tensor:
        raw = self.net(torch.cat([u, z, a_base], dim=-1))
        return self.max_residual.to(raw.device) * torch.tanh(raw)


class DynamicsPredictor(nn.Module):
    def __init__(self, u_dim: int = 256, z_dim: int = 32, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(u_dim + action_dim + z_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, u_dim),
        )

    def forward(self, u: Tensor, action: Tensor, z: Tensor) -> Tensor:
        delta_u = self.net(torch.cat([u, action, z], dim=-1))
        return u + delta_u


class FastAdapterSystem(nn.Module):
    def __init__(
        self,
        encoder: InteractionEncoder,
        base_policy: BaseSkillPolicy,
        adapter: FastAdapter,
        dynamics: DynamicsPredictor,
        z_dim: int = 32,
        learn_inner_lr: bool = False,
        inner_lr: float = 0.05,
    ):
        super().__init__()
        self.encoder = encoder
        self.base_policy = base_policy
        self.adapter = adapter
        self.dynamics = dynamics
        self.z0 = nn.Parameter(torch.zeros(z_dim))
        if learn_inner_lr:
            self.log_inner_lr = nn.Parameter(torch.log(torch.tensor(float(inner_lr))))
        else:
            self.register_buffer("fixed_inner_lr", torch.tensor(float(inner_lr)))
            self.log_inner_lr = None
        self.freeze_base_policy()

    @property
    def inner_lr(self) -> Tensor:
        if self.log_inner_lr is not None:
            return self.log_inner_lr.exp()
        return self.fixed_inner_lr

    def freeze_base_policy(self) -> None:
        self.base_policy.eval()
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)

    def encode(self, obs: AdapterObservation) -> Tensor:
        return self.encoder(obs.obj_feat, obs.tar_feat, obs.proprio, obs.prev_action)

    def base_action(self, obs: AdapterObservation, u: Tensor) -> Tensor:
        with torch.no_grad():
            return self.base_policy(obs, u)

    def act_from_u(self, u: Tensor, z: Tensor, a_base: Tensor) -> tuple[Tensor, Tensor]:
        delta = self.adapter(u, z, a_base)
        return a_base + delta, delta

    def act(self, obs: AdapterObservation, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        u = self.encode(obs)
        a_base = self.base_action(obs, u)
        action, delta = self.act_from_u(u, z, a_base)
        return action, delta, a_base, u


def expand_latent(z0: Tensor, batch_size: int) -> Tensor:
    return z0.unsqueeze(0).expand(batch_size, -1).clone().requires_grad_(True)


def adaptation_loss(
    u_hat_next: Tensor,
    u_next: Tensor,
    z: Tensor,
    z0: Tensor,
    config: AdaptationLossConfig,
) -> tuple[Tensor, Dict[str, Tensor]]:
    target = u_next.detach()
    mse_per_sample = (u_hat_next - target).square().mean(dim=-1)
    cos_per_sample = 1.0 - F.cosine_similarity(u_hat_next, target, dim=-1)
    z_ref = z0.unsqueeze(0).expand_as(z)
    z_reg_per_sample = (z - z_ref).square().mean(dim=-1)
    per_sample = mse_per_sample + config.lambda_cos * cos_per_sample + config.lambda_z * z_reg_per_sample
    loss = per_sample.sum()
    return loss, {
        "mse": mse_per_sample.mean(),
        "cos": cos_per_sample.mean(),
        "z_reg": z_reg_per_sample.mean(),
        "per_sample": per_sample.detach(),
    }


def inner_update_z(
    *,
    system: FastAdapterSystem,
    u: Tensor,
    action_observed: Tensor,
    z: Tensor,
    u_next: Tensor,
    loss_config: AdaptationLossConfig,
    create_graph: bool = True,
    detach_executed_action: bool = True,
) -> InnerUpdateResult:
    """Apply one z-only inner update.

    ``detach_executed_action=True`` records the safety correction requested for
    the toy protocol: the action already executed in the environment is treated
    as an observed variable for system-identification loss. This prevents the
    inner loss from improving by changing the explanation of a past action.
    """

    action_for_dynamics = action_observed.detach() if detach_executed_action else action_observed
    u_hat_next = system.dynamics(u, action_for_dynamics, z)
    loss, terms = adaptation_loss(u_hat_next, u_next, z, system.z0, loss_config)
    grad_z = torch.autograd.grad(
        loss,
        z,
        create_graph=create_graph,
        retain_graph=create_graph,
        only_inputs=True,
    )[0]
    z_next = z - system.inner_lr * grad_z
    z_next.retain_grad()
    return InnerUpdateResult(z_next=z_next, loss=loss, terms=terms, grad_z=grad_z)


def named_grad_flags(parameters: Iterable[tuple[str, nn.Parameter]]) -> Dict[str, bool]:
    return {name: parameter.grad is not None for name, parameter in parameters}
