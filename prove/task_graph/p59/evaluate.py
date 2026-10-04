"""Sealed, zero-physics offline evaluation for P59.

The evaluator replays only the verified t->t+1 prefix.  Retained t27/t29
frames are future queries under the state obtained at t24; they never create
adaptation transitions.  All four P50 phases are evaluated at every query.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import torch

from .contracts import PublicTransition
from .checkpoint import load_frozen_checkpoint
from .core import rotation_6d_to_matrix, so3_log_map, split_geometry
from .history import cyclic_wrong_history_map, history_tokens
from .train import load_dataset


METHODS = ("current_only", "fixed_init", "online", "wrong_online",
           "history", "wrong_history", "oracle")
FUTURE_STEPS = (27, 29)
SENSITIVITY_STEPS = (0, 24, 29)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sealed_candidate(checkpoint_path, training_seal_path, training_freeze_path,
                          p50_checkpoint, *, device="cpu"):
    """Load the single sealed final candidate and freeze every parameter."""
    checkpoint_path, training_seal_path, training_freeze_path = map(
        Path, (checkpoint_path, training_seal_path, training_freeze_path))
    seal = json.loads(training_seal_path.read_text())
    freeze = json.loads(training_freeze_path.read_text())
    if seal.get("schema") != "prove.p59-training-seal.v1":
        raise ValueError("unknown P59 training seal")
    if freeze.get("schema") != "prove.p59-training-freeze.v1":
        raise ValueError("unknown P59 training freeze")
    if seal.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("P59 checkpoint differs from the training seal")
    if seal.get("developer_split_opened_for_selection") is not False:
        raise ValueError("developer-selected checkpoints are forbidden")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "prove.p59-control-latent-checkpoint.v1":
        raise ValueError("unknown P59 checkpoint schema")
    if payload.get("training_freeze_sha256") != sha256(training_freeze_path):
        raise ValueError("checkpoint is not bound to the supplied training freeze")
    model, history_model, _ = load_frozen_checkpoint(
        checkpoint_path, p50_checkpoint, device=device)
    dataset_digest = freeze.get("dataset_sha256")
    if not isinstance(dataset_digest, str) or len(dataset_digest) != 64:
        raise ValueError("training freeze lacks its materialized dataset hash")
    return model, history_model, {"checkpoint_sha256": sha256(checkpoint_path),
                                  "training_freeze_sha256": sha256(training_freeze_path),
                                  "training_dataset_sha256": dataset_digest}


def _to(value: torch.Tensor, device) -> torch.Tensor:
    return value.to(device=device, dtype=torch.float32)


def _transition(episode: dict, step: int, device) -> PublicTransition:
    phase = torch.zeros((), dtype=torch.long, device=device)
    return PublicTransition(
        _to(episode["u"][step], device), _to(episode["q"][step], device),
        _to(episode["executed_world_delta6"][step], device),
        _to(episode["u"][step + 1], device), _to(episode["q"][step + 1], device),
        phase, phase.clone(), step, step + 1, episode["episode_id"])


def _history_z(model, history_model, support: list[PublicTransition], fallback: torch.Tensor):
    if not support:
        return fallback
    tokens, valid = history_tokens(model, support)
    with torch.no_grad():
        return history_model(tokens, valid).detach()


def _latent_action_sensitivity(model, u, q, z) -> tuple[float, float]:
    """Normalized-action and scaled decoded-world Jacobian norms."""
    active = z[..., :model.contract.active_latent_dim].detach().requires_grad_(True)

    def outputs(active_value):
        full = torch.cat((active_value, torch.zeros(7, device=active_value.device)), -1)
        geometry = model.geometry(u, full)
        geometry4 = geometry.expand(4, 12)
        q4 = q.expand(4, 9)
        action = model.p50(geometry4, q4, torch.arange(4, device=q.device))
        position, rotation = model.p50.decode_world_pose(action, geometry4, q4)
        return action, position, rotation

    with torch.no_grad():
        _, _, reference_rotation = outputs(active.detach())
    def normalized(active_value):
        return outputs(active_value)[0]
    def world(active_value):
        _, position, rotation = outputs(active_value)
        tangent = so3_log_map(reference_rotation.transpose(-1, -2) @ rotation)
        return torch.cat((position / .2, tangent / .25), -1)
    with torch.enable_grad():
        normalized_jacobian = torch.autograd.functional.jacobian(normalized, active)
        world_jacobian = torch.autograd.functional.jacobian(world, active)
    return (float(torch.linalg.vector_norm(normalized_jacobian).detach().cpu()),
            float(torch.linalg.vector_norm(world_jacobian).detach().cpu()))


def _geometry_metrics(predicted, target, q) -> dict:
    pp, pr, ps, pl = split_geometry(predicted)
    tp, tr, ts, tl = split_geometry(target)
    relation_p = (pr.transpose(-1, -2) @ (q[:3] - pp)[..., None])[..., 0]
    target_relation_p = (tr.transpose(-1, -2) @ (q[:3] - tp)[..., None])[..., 0]
    qr = rotation_6d_to_matrix(q[3:])
    relation_r = pr.transpose(-1, -2) @ qr
    target_relation_r = tr.transpose(-1, -2) @ qr
    return {
        "geometry_position_mm": float(torch.linalg.vector_norm(pp - tp) * 1000),
        "geometry_rotation_deg": float(torch.linalg.vector_norm(so3_log_map(tr.T @ pr)) * 180 / math.pi),
        "geometry_half_size_mae_mm": float((ps - ts).abs().mean() * 1000),
        "geometry_held_length_mm": float((pl - tl).abs().mean() * 1000),
        "relation_position_mm": float(torch.linalg.vector_norm(relation_p - target_relation_p) * 1000),
        "relation_rotation_deg": float(torch.linalg.vector_norm(
            so3_log_map(target_relation_r.transpose(-1, -2) @ relation_r)) * 180 / math.pi),
    }


def _query_record(model, episode: dict, frame_row: int, step: int, scope: str,
                  method: str, geometry, latent, z0, fixed_geometry, fixed_actions,
                  *, inner_loss=None, compute_sensitivity=True) -> dict:
    device = geometry.device
    q = _to(episode["readout_q"][frame_row], device)
    target_geometry = _to(episode["readout_oracle_geometry"][frame_row], device)
    target_actions = _to(episode["readout_oracle_p50_action6"][frame_row], device)
    phases = torch.arange(4, device=device)
    geometry4 = geometry.expand(4, 12)
    q4 = q.expand(4, 9)
    actions = model.p50(geometry4, q4, phases)
    predicted_pose = model.p50.decode_world_pose(actions, geometry4, q4)
    target_pose = model.p50.decode_world_pose(target_actions, target_geometry.expand(4, 12), q4)
    position = torch.linalg.vector_norm(predicted_pose[0] - target_pose[0], dim=-1) * 1000
    rotation = torch.linalg.vector_norm(
        so3_log_map(target_pose[1].transpose(-1, -2) @ predicted_pose[1]), dim=-1) * 180 / math.pi
    fixed_pose = model.p50.decode_world_pose(fixed_actions, fixed_geometry.expand(4, 12), q4)
    delta_position = torch.linalg.vector_norm(predicted_pose[0] - fixed_pose[0], dim=-1) * 1000
    delta_rotation = torch.linalg.vector_norm(
        so3_log_map(fixed_pose[1].transpose(-1, -2) @ predicted_pose[1]), dim=-1) * 180 / math.pi
    record = {
        "episode_id": episode["episode_id"], "distribution": episode["distribution"],
        "method": method, "query_scope": scope, "step_index": step,
        **_geometry_metrics(geometry, target_geometry, q),
        "world_position_mm": float(position.mean()),
        "world_rotation_deg": float(rotation.mean()),
        "world_position_mm_by_phase": position.detach().cpu().tolist(),
        "world_rotation_deg_by_phase": rotation.detach().cpu().tolist(),
        "delta_action_normalized_rms": float((actions - fixed_actions).square().mean().sqrt()),
        "delta_world_position_mm": float(delta_position.mean()),
        "delta_world_rotation_deg": float(delta_rotation.mean()),
        "inner_loss": None if inner_loss is None else float(inner_loss),
        "delta_z_l2": None if latent is None else float(torch.linalg.vector_norm(latent - z0).detach().cpu()),
        "action_dz_frobenius": None,
        "world_action_dz_frobenius": None,
    }
    if compute_sensitivity and step in SENSITIVITY_STEPS and latent is not None and method != "oracle":
        u = _to(episode["readout_u"][frame_row], device)
        normalized, world = _latent_action_sensitivity(model, u, q, latent)
        record["action_dz_frobenius"] = normalized
        record["world_action_dz_frobenius"] = world
    return record


def _aggregate(records: list[dict]) -> list[dict]:
    keys = ("geometry_position_mm", "geometry_rotation_deg", "geometry_half_size_mae_mm",
            "geometry_held_length_mm", "relation_position_mm", "relation_rotation_deg",
            "world_position_mm", "world_rotation_deg", "delta_action_normalized_rms",
            "delta_world_position_mm", "delta_world_rotation_deg", "inner_loss",
            "delta_z_l2", "action_dz_frobenius", "world_action_dz_frobenius")
    groups = {}
    for row in records:
        identity = (row["split"], row["distribution"], row["query_scope"], row["method"])
        groups.setdefault(identity, []).append(row)
    result = []
    for identity, rows in sorted(groups.items()):
        item = dict(zip(("split", "distribution", "query_scope", "method"), identity))
        item["queries"] = len(rows)
        for key in keys:
            values = sorted(float(row[key]) for row in rows if row[key] is not None)
            if values:
                item[key] = {"mean": sum(values) / len(values),
                             "median": values[len(values) // 2],
                             "p90": values[min(len(values) - 1, int(.9 * len(values)))]}
        result.append(item)
    return result


def _validate_evaluation_fields(dataset: dict) -> None:
    required = {
        "readout_u": (64, dataset["u_dim"]), "readout_q": (64, 9),
        "readout_step_index": (64,), "readout_oracle_geometry": (64, 12),
        "readout_oracle_p50_action6": (64, 4, 6),
    }
    for split, episodes in dataset["splits"].items():
        for episode in episodes:
            if episode.get("distribution") not in ("id", "geometry"):
                raise ValueError(f"{split} episode lacks a fixed ID/OOD stratum")
            for name, shape in required.items():
                value = episode.get(name)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                    raise ValueError(f"{split}/{episode.get('episode_id')}/{name} must be {shape}")
                if not torch.isfinite(value).all():
                    raise ValueError(f"{split}/{episode.get('episode_id')}/{name} is nonfinite")
            steps = episode["readout_step_index"].tolist()
            if steps[:25] != list(range(25)) or any(step not in steps for step in FUTURE_STEPS):
                raise ValueError("evaluation frames must preserve t0..24 and future t27/t29 identities")


def evaluate_dataset(model, history_model, dataset: dict, *, compute_sensitivity=True) -> dict:
    """Evaluate train and opened developer ID/OOD without any model update."""
    if dataset.get("schema") != "prove.p59-training-dataset.v1":
        raise ValueError("unknown P59 evaluation dataset")
    _validate_evaluation_fields(dataset)
    device = next(model.parameters()).device
    records = []
    for split, episodes in dataset["splits"].items():
        ids = [episode["episode_id"] for episode in episodes]
        strata = [episode["distribution"] for episode in episodes]
        wrong_map = cyclic_wrong_history_map(ids, strata)
        by_id = {episode["episode_id"]: episode for episode in episodes}
        for episode in episodes:
            donor = by_id[wrong_map[episode["episode_id"]]]
            own_support, donor_support = [], []
            u0 = _to(episode["u"][0], device)
            with torch.no_grad():
                z0 = model.initial(u0).detach()
            online_z = z0.clone()
            wrong_online_z = z0.clone()
            for step in range(25):
                inner = wrong_inner = None
                if step:
                    own_support.append(_transition(episode, step - 1, device))
                    donor_support.append(_transition(donor, step - 1, device))
                    with torch.enable_grad():
                        online_z, losses = model.adapt_history(
                            own_support[-model.contract.history:], online_z, z0, create_graph=False)
                        wrong_online_z, wrong_losses = model.adapt_history(
                            donor_support[-model.contract.history:], wrong_online_z, z0, create_graph=False)
                    online_z = online_z.detach()
                    wrong_online_z = wrong_online_z.detach()
                    inner = float(losses.detach().mean().cpu())
                    wrong_inner = float(wrong_losses.detach().mean().cpu())
                with torch.no_grad():
                    history_z = _history_z(model, history_model, own_support, z0)
                    wrong_z = _history_z(model, history_model, donor_support, z0)
                frame_row = step
                records.extend(_evaluate_query_methods(
                    model, episode, frame_row, step, "continuous", z0, online_z,
                    wrong_online_z, history_z, wrong_z, inner, wrong_inner,
                    compute_sensitivity))
            # State is frozen at t24.  t27/t29 supply fresh public U/q only.
            steps = episode["readout_step_index"].tolist()
            for future_step in FUTURE_STEPS:
                if future_step not in steps:
                    raise ValueError(f"episode lacks future diagnostic t{future_step}")
                frame_row = steps.index(future_step)
                records.extend(_evaluate_query_methods(
                    model, episode, frame_row, future_step, "future_no_adaptation", z0,
                    online_z, wrong_online_z, history_z, wrong_z, None, None,
                    compute_sensitivity))
            for row in records[-(25 + len(FUTURE_STEPS)) * len(METHODS):]:
                row.setdefault("split", split)
    return {
        "schema": "prove.p59-offline-evaluation.v1",
        "physics_rollouts": 0,
        "developer_split_role": "opened evaluation only; never checkpoint selection",
        "future_query_contract": "t27/t29 use the t24 state and do not create sparse transitions",
        "empty_history_contract": "at t0 matched and wrong History use fixed-init; learned History begins after t0->t1",
        "sensitivity_definition": {
            "steps": list(SENSITIVITY_STEPS),
            "normalized": "Frobenius norm of d(all four normalized P50 actions)/d(active z9)",
            "world": "Frobenius norm of d([world_position/0.2, local world rotation tangent/0.25] for four phases)/d(active z9)",
        },
        "records": records,
        "aggregate": _aggregate(records),
    }


def _evaluate_query_methods(model, episode, frame_row, step, scope, z0, online_z,
                            wrong_online_z, history_z, wrong_z, inner, wrong_inner,
                            compute_sensitivity):
    device = z0.device
    u = _to(episode["readout_u"][frame_row], device)
    q = _to(episode["readout_q"][frame_row], device)
    oracle_geometry = _to(episode["readout_oracle_geometry"][frame_row], device)
    zero = torch.zeros_like(z0)
    latents = {"current_only": zero, "fixed_init": z0, "online": online_z,
               "wrong_online": wrong_online_z,
               "history": history_z, "wrong_history": wrong_z}
    with torch.no_grad():
        geometries = {name: model.geometry(u, z) for name, z in latents.items()}
        geometries["oracle"] = oracle_geometry
        phases = torch.arange(4, device=device)
        fixed_actions = model.p50(geometries["fixed_init"].expand(4, 12), q.expand(4, 9), phases)
    result = []
    for method in METHODS:
        latent = latents.get(method)
        result.append(_query_record(
            model, episode, frame_row, step, scope, method, geometries[method], latent, z0,
            geometries["fixed_init"], fixed_actions,
            inner_loss=inner if method == "online" else wrong_inner if method == "wrong_online" else None,
            compute_sensitivity=compute_sensitivity))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-seal", type=Path, required=True)
    parser.add_argument("--training-freeze", type=Path, required=True)
    parser.add_argument("--p50-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    model, history_model, provenance = load_sealed_candidate(
        args.checkpoint, args.training_seal, args.training_freeze,
        args.p50_checkpoint, device=args.device)
    dataset = load_dataset(args.dataset, model.contract.u_dim)
    if sha256(args.dataset) != provenance["training_dataset_sha256"]:
        raise ValueError("evaluation dataset differs from the dataset sealed before training")
    result = evaluate_dataset(model, history_model, dataset,
                              compute_sensitivity=not args.skip_sensitivity)
    result["provenance"] = {**provenance, "dataset_sha256": sha256(args.dataset)}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
