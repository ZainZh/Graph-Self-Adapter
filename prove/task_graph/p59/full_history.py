"""Full-prefix recurrent public-history reference for P59.

This sidecar is independent of the frozen P59 candidate and its last-eight
History baseline.  It consumes every arrived true transition in order and
emits the same canonical nine-DOF geometry correction used by the core.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .contracts import PublicTransition, validate_history
from .core import P59AdaptationCore, geometry_loss, world_pose_loss


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, output_dim))


class FullHistoryReference(nn.Module):
    """Persistent GRU state over all public transitions in a stage."""

    token_dim = 21
    active_dim = 9

    def __init__(self, u_dim: int, hidden_dim: int = 128):
        super().__init__()
        if u_dim <= 0 or hidden_dim <= 0:
            raise ValueError("FullHistory dimensions must be positive")
        self.u_dim = u_dim
        self.hidden_dim = hidden_dim
        self.initializer = _mlp(u_dim, hidden_dim, hidden_dim)
        self.recurrent = nn.GRUCell(self.token_dim, hidden_dim)
        self.decoder = _mlp(hidden_dim, hidden_dim, self.active_dim)

    def initial_hidden(self, core: P59AdaptationCore, first_u: torch.Tensor) -> torch.Tensor:
        if first_u.shape[-1] != self.u_dim or core.contract.u_dim != self.u_dim:
            raise ValueError("FullHistory U dimension differs from the frozen core")
        normalized = ((first_u - core.u_mean) / core.u_scale).detach()
        return self.initializer(normalized)

    def token(self, core: P59AdaptationCore, transition: PublicTransition) -> torch.Tensor:
        """[current relation6 + public size/length3, action6, next relation6]."""
        transition.validate(core.contract)
        with torch.no_grad():
            current_geometry = core.public_geometry(transition.u).detach()
            next_geometry = core.public_geometry(transition.next_u).detach()
            current = core.p50.features(current_geometry, transition.q.detach(),
                                         transition.phase).detach()[..., :9]
            nxt = core.p50.features(next_geometry, transition.next_q.detach(),
                                     transition.next_phase).detach()[..., :6]
        value = torch.cat((current, transition.executed_action.detach(), nxt), -1)
        if value.shape[-1] != self.token_dim:
            raise RuntimeError("FullHistory token dimension drift")
        return value

    def advance(self, core: P59AdaptationCore, hidden: torch.Tensor,
                transition: PublicTransition) -> torch.Tensor:
        return self.recurrent(self.token(core, transition), hidden)

    def correction(self, hidden: torch.Tensor) -> torch.Tensor:
        active = self.decoder(hidden)
        reserved = torch.zeros(*active.shape[:-1], 7, dtype=active.dtype, device=active.device)
        return torch.cat((active, reserved), -1)

    def replay(self, core: P59AdaptationCore, first_u: torch.Tensor,
               support: list[PublicTransition]) -> tuple[torch.Tensor, torch.Tensor]:
        if support:
            validate_history(support, core.contract)
        hidden = self.initial_hidden(core, first_u)
        for transition in support:
            hidden = self.advance(core, hidden, transition)
        return hidden, self.correction(hidden)


def full_history_outer_loss(core: P59AdaptationCore, reference: FullHistoryReference,
                            first_u: torch.Tensor, support: list[PublicTransition],
                            query_u: torch.Tensor, query_q: torch.Tensor,
                            query_phase: torch.Tensor, oracle_action: torch.Tensor,
                            oracle_geometry: torch.Tensor, *, geometry_weight: float = .25):
    """World-command BC plus the same control-geometry supervision as P59."""
    if not core._runtime_frozen:
        raise ValueError("FullHistory requires the completely frozen P59 core")
    hidden, z = reference.replay(core, first_u, support)
    with torch.no_grad():
        public_geometry = core.public_geometry(query_u).detach()
    predicted_geometry = core.geometry_from_public(public_geometry, z)
    predicted_action = core.p50(predicted_geometry, query_q, query_phase)
    predicted_pose = core.p50.decode_world_pose(predicted_action, predicted_geometry, query_q)
    oracle_pose = core.p50.decode_world_pose(oracle_action, oracle_geometry, query_q)
    action_loss = world_pose_loss(predicted_pose, oracle_pose)
    geom_loss = geometry_loss(predicted_geometry, oracle_geometry)
    return action_loss + geometry_weight * geom_loss, {
        "world_action": action_loss, "geometry": geom_loss, "hidden": hidden,
        "z": z, "action": predicted_action,
    }


class FullHistoryRuntime:
    """Deployment-order state machine; only the GRU hidden state changes."""

    def __init__(self, core: P59AdaptationCore, reference: FullHistoryReference):
        if not core._runtime_frozen or any(parameter.requires_grad for parameter in reference.parameters()):
            raise ValueError("freeze the core and FullHistory reference before runtime")
        self.core, self.reference = core, reference
        self.hidden = None
        self.episode_id = None
        self.next_step = None

    def enter(self, first_u: torch.Tensor, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode identity is required")
        with torch.no_grad():
            self.hidden = self.reference.initial_hidden(self.core, first_u)
        self.episode_id, self.next_step = episode_id, 0

    def command(self, u: torch.Tensor, q: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if self.hidden is None:
            raise RuntimeError("FullHistory stage not entered")
        with torch.no_grad():
            return self.core.action(u, q, phase, self.reference.correction(self.hidden))

    def observe_transition(self, transition: PublicTransition) -> None:
        if self.hidden is None:
            raise RuntimeError("FullHistory stage not entered")
        if transition.episode_id != self.episode_id or transition.step_index != self.next_step:
            raise ValueError("FullHistory transition is not the next actual episode step")
        with torch.no_grad():
            self.hidden = self.reference.advance(self.core, self.hidden, transition)
        self.next_step = transition.next_step_index

    def exit(self) -> None:
        self.hidden = self.episode_id = self.next_step = None


def load_full_history_checkpoint(path: str | Path, main_checkpoint: str | Path,
                                 *, device="cpu") -> FullHistoryReference:
    """Load a sidecar only when it is bound to the supplied main candidate."""
    path, main_checkpoint = Path(path), Path(main_checkpoint)
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != "prove.p59-full-history-checkpoint.v1":
        raise ValueError("unknown FullHistory checkpoint schema")
    main_digest = hashlib.sha256(main_checkpoint.read_bytes()).hexdigest()
    if payload.get("main_checkpoint_sha256") != main_digest:
        raise ValueError("FullHistory sidecar belongs to a different main checkpoint")
    if (payload.get("u_dim"), payload.get("hidden_dim")) != (244, 128):
        raise ValueError("FullHistory architecture differs from its preregistered contract")
    reference = FullHistoryReference(244, 128).to(device)
    reference.load_state_dict(payload["state_dict"], strict=True)
    return reference.eval().requires_grad_(False)
