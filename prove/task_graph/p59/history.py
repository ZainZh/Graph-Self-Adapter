"""Matched non-gradient History baseline for P59."""
from __future__ import annotations

import torch
from torch import nn

from .contracts import PublicTransition, validate_history
from .core import P59AdaptationCore, geometry_loss, world_pose_loss


class HistoryCorrection(nn.Module):
    """Last-eight public relation history to the same canonical 9D correction."""

    token_dim = 18

    def __init__(self, history: int = 8, hidden: int = 128):
        super().__init__()
        if history != 8:
            raise ValueError("P59 matched History uses the same last-eight window")
        self.history = history
        self.network = nn.Sequential(
            nn.Linear(history * self.token_dim + history, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 9))

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2:] != (self.history, self.token_dim):
            raise ValueError("History tokens must end in [8,18]")
        if valid.shape != tokens.shape[:-1]:
            raise ValueError("History validity mask shape mismatch")
        active = self.network(torch.cat((tokens.flatten(-2), valid.to(tokens.dtype)), -1))
        return torch.cat((active, torch.zeros(*active.shape[:-1], 7,
                                              dtype=active.dtype, device=active.device)), -1)


def history_tokens(core: P59AdaptationCore,
                   support: list[PublicTransition]) -> tuple[torch.Tensor, torch.Tensor]:
    validate_history(support, core.contract)
    selected = support[-core.contract.history:]
    rows = []
    with torch.no_grad():
        for item in selected:
            current = core.p50.features(core.public_geometry(item.u), item.q, item.phase)[..., :6]
            nxt = core.p50.features(core.public_geometry(item.next_u), item.next_q, item.next_phase)[..., :6]
            rows.append(torch.cat((current, item.executed_action, nxt), -1).detach())
    example = rows[0]
    padding = core.contract.history - len(rows)
    zeros = torch.zeros(*example.shape[:-1], padding, HistoryCorrection.token_dim,
                        dtype=example.dtype, device=example.device)
    tokens = torch.cat((zeros, torch.stack(rows, -2)), -2)
    valid = torch.cat((torch.zeros(*example.shape[:-1], padding, dtype=torch.bool, device=example.device),
                       torch.ones(*example.shape[:-1], len(rows), dtype=torch.bool, device=example.device)), -1)
    return tokens, valid


def history_outer_loss(core: P59AdaptationCore, history_model: HistoryCorrection,
                       support: list[PublicTransition], query_u: torch.Tensor,
                       query_q: torch.Tensor, query_phase: torch.Tensor,
                       oracle_action: torch.Tensor, oracle_geometry: torch.Tensor,
                       *, query_step_index: int, query_episode_id: str,
                       geometry_weight: float = .25) -> tuple[torch.Tensor, dict]:
    validate_history(support, core.contract)
    if not core._readout_frozen:
        raise ValueError("public readout must be frozen for matched History")
    if query_episode_id != support[0].episode_id or query_step_index < support[-1].next_step_index:
        raise ValueError("History query must be a future frame in the same episode")
    tokens, valid = history_tokens(core, support)
    z = history_model(tokens, valid)
    with torch.no_grad():
        public_geometry = core.public_geometry(query_u).detach()
    predicted_geometry = core.geometry_from_public(public_geometry, z)
    predicted_action = core.p50(predicted_geometry, query_q, query_phase)
    predicted_pose = core.p50.decode_world_pose(predicted_action, predicted_geometry, query_q)
    oracle_pose = core.p50.decode_world_pose(oracle_action, oracle_geometry, query_q)
    action_loss = world_pose_loss(predicted_pose, oracle_pose)
    geom_loss = geometry_loss(predicted_geometry, oracle_geometry)
    return action_loss + geometry_weight * geom_loss, {
        "world_action": action_loss, "geometry": geom_loss, "z": z,
        "action": predicted_action, "valid_history": valid.sum(-1),
    }


def cyclic_wrong_history_map(episode_ids: list[str], strata: list[str]) -> dict[str, str]:
    """Result-independent +1 cyclic shift within each declared stratum."""
    if len(episode_ids) != len(strata) or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("wrong-history identities/strata invalid")
    mapping = {}
    for stratum in sorted(set(strata)):
        members = sorted(identity for identity, group in zip(episode_ids, strata) if group == stratum)
        if len(members) < 2:
            raise ValueError("each wrong-history stratum needs at least two episodes")
        mapping.update({identity: members[(index + 1) % len(members)] for index, identity in enumerate(members)})
    return mapping
