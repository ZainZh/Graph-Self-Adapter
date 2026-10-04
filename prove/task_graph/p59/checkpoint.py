"""Strict runtime loader for the frozen P59 core and matched History model."""
from __future__ import annotations

from pathlib import Path

import torch

from .contracts import P59Contract
from .core import P59AdaptationCore
from .history import HistoryCorrection


def load_frozen_checkpoint(path: str | Path, p50_checkpoint: str | Path,
                           *, device: str = "cpu") -> tuple[P59AdaptationCore, HistoryCorrection, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("schema") != "prove.p59-control-latent-checkpoint.v1":
        raise ValueError("unknown P59 checkpoint schema")
    contract = payload.get("contract", {})
    if contract.get("schema") != "prove.p59-model-contract.v1":
        raise ValueError("checkpoint MODEL_CONTRACT missing or changed")
    u_dim = contract["public_representation"]["dimension"]
    core = P59AdaptationCore(P59Contract(u_dim=u_dim), p50_checkpoint).to(device)
    core.load_state_dict(payload["state_dict"], strict=True)
    core.freeze_public_readout().freeze_runtime()
    history = HistoryCorrection(history=core.contract.history).to(device)
    history.load_state_dict(payload["history_state_dict"], strict=True)
    history.eval().requires_grad_(False)
    return core, history, contract
