"""Tensor-level contracts for P59.

The data exporter owns RGB, masks, camera calibration, and feature-cache
schemas.  This module starts after those public inputs have been encoded as
``u``.  It deliberately keeps ``u_dim`` configurable while fixing the
control-facing geometry consumed by P50.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class P59Contract:
    u_dim: int
    latent_dim: int = 16
    active_latent_dim: int = 9
    geometry_dim: int = 12
    q_dim: int = 9
    action_dim: int = 6
    history: int = 8
    inner_steps: int = 3
    inner_lr: float = 0.1
    anchor: float = 1e-4

    def __post_init__(self) -> None:
        if self.u_dim <= 0:
            raise ValueError("u_dim must be positive")
        if (self.latent_dim, self.active_latent_dim) != (16, 9):
            raise ValueError("P59 v1 uses a 16D container with nine identifiable control directions")
        if (self.geometry_dim, self.q_dim, self.action_dim) != (12, 9, 6):
            raise ValueError("P59 v1 control geometry/q/action dimensions changed")
        if (self.history, self.inner_steps) != (8, 3):
            raise ValueError("P59 v1 requires a last-eight window and three inner steps")


@dataclass(frozen=True)
class PublicTransition:
    """One actual, consecutive transition available to runtime adaptation.

    ``executed_action`` is the action that really went to the controller.  It
    must never be replaced by an offline Oracle label.  Sparse cached frames
    do not form a transition: callers must retain the source action index and
    only construct this object for index ``t -> t+1``.
    """

    u: torch.Tensor
    q: torch.Tensor
    executed_action: torch.Tensor
    next_u: torch.Tensor
    next_q: torch.Tensor
    phase: torch.Tensor
    next_phase: torch.Tensor
    step_index: int
    next_step_index: int
    episode_id: str

    def validate(self, contract: P59Contract) -> None:
        expected = {
            "u": (contract.u_dim,),
            "q": (contract.q_dim,),
            "executed_action": (contract.action_dim,),
            "next_u": (contract.u_dim,),
            "next_q": (contract.q_dim,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape[-1:] != shape or not torch.isfinite(value).all():
                raise ValueError(f"{name} must end in finite shape {shape}")
        for name in ("phase", "next_phase"):
            value = getattr(self, name)
            if value.numel() < 1 or value.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must contain integer phase indices")
            if torch.any((value < 0) | (value >= 4)):
                raise ValueError(f"{name} outside P50's four phases")
        if self.next_step_index != self.step_index + 1:
            raise ValueError("Sparse/nonconsecutive cached frames are not one real transition")
        if not self.episode_id:
            raise ValueError("episode_id is required to prevent cross-episode support")


def validate_history(history: list[PublicTransition], contract: P59Contract) -> None:
    if not history:
        raise ValueError("empty support history")
    episode = history[0].episode_id
    previous = None
    for transition in history:
        transition.validate(contract)
        if transition.episode_id != episode:
            raise ValueError("support history crosses episode boundary")
        if previous is not None and transition.step_index != previous.next_step_index:
            raise ValueError("support history is not causally consecutive")
        previous = transition
