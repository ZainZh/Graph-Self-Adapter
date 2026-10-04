from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

import torch
from torch import Tensor

from .core import AdapterObservation


@dataclass(frozen=True)
class Toy2DConfig:
    gain_min: float = 0.75
    gain_max: float = 1.25
    bias_min: float = -0.075
    bias_max: float = 0.075
    state_min: float = -1.0
    state_max: float = 1.0
    target_min: float = -1.0
    target_max: float = 1.0
    action_limit: float = 1.0
    nominal_gain: float = 0.5
    success_threshold: float = 0.05

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Toy2DDomain:
    gain: Tensor
    bias: Tensor
    domain_id: Tensor


@dataclass(frozen=True)
class Toy2DObservation:
    state: Tensor
    target: Tensor
    prev_action: Tensor

    def adapter_observation(self) -> AdapterObservation:
        zeros = torch.zeros_like(self.state)
        return AdapterObservation(
            obj_feat=self.state,
            tar_feat=self.target,
            proprio=zeros,
            prev_action=self.prev_action,
        )


def _rand_uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    return torch.empty(shape, device=device).uniform_(low, high, generator=generator)


def sample_toy2d_domains(
    num_domains: int,
    seed: int,
    config: Toy2DConfig,
    *,
    device: torch.device | str = "cpu",
) -> Toy2DDomain:
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    gain = _rand_uniform((num_domains, 2), config.gain_min, config.gain_max, generator=generator, device=device)
    bias = _rand_uniform((num_domains, 2), config.bias_min, config.bias_max, generator=generator, device=device)
    domain_id = torch.arange(num_domains, device=device, dtype=torch.long)
    return Toy2DDomain(gain=gain, bias=bias, domain_id=domain_id)


class Toy2DVectorEnv:
    """Vectorized 2D numerical environment.

    The hidden per-axis gain and bias are used only by ``step`` and
    ``expert_action``. They are never included in runtime adapter observations.
    The transition detaches the executed action before updating state, matching
    the non-differentiable robot/simulator protocol.
    """

    def __init__(self, config: Toy2DConfig, device: torch.device | str = "cpu"):
        self.config = config
        self.device = torch.device(device)

    def reset(
        self,
        batch_size: int,
        seed: int,
        *,
        domain: Optional[Toy2DDomain] = None,
    ) -> Toy2DObservation:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        state = _rand_uniform(
            (batch_size, 2),
            self.config.state_min,
            self.config.state_max,
            generator=generator,
            device=self.device,
        )
        target = _rand_uniform(
            (batch_size, 2),
            self.config.target_min,
            self.config.target_max,
            generator=generator,
            device=self.device,
        )
        prev_action = torch.zeros(batch_size, 2, device=self.device)
        del domain
        return Toy2DObservation(state=state, target=target, prev_action=prev_action)

    def step(self, obs: Toy2DObservation, domain: Toy2DDomain, action: Tensor) -> Toy2DObservation:
        clipped_action = torch.clamp(action, -self.config.action_limit, self.config.action_limit)
        next_state = obs.state + domain.gain * clipped_action.detach() + domain.bias
        return Toy2DObservation(state=next_state, target=obs.target, prev_action=clipped_action.detach())

    def nominal_action(self, obs: Toy2DObservation) -> Tensor:
        action = self.config.nominal_gain * (obs.target - obs.state)
        return torch.clamp(action, -self.config.action_limit, self.config.action_limit)

    def expert_action(self, obs: Toy2DObservation, domain: Toy2DDomain) -> Tensor:
        desired_delta = self.config.nominal_gain * (obs.target - obs.state)
        action = (desired_delta - domain.bias) / torch.clamp(domain.gain, min=1.0e-6)
        return torch.clamp(action, -self.config.action_limit, self.config.action_limit)

    def goal_error(self, obs: Toy2DObservation) -> Tensor:
        return torch.linalg.norm(obs.state - obs.target, dim=-1)

    def success(self, obs: Toy2DObservation) -> Tensor:
        return self.goal_error(obs) <= self.config.success_threshold
