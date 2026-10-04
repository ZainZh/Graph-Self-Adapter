"""P59 derived-data loader with an explicit public/private boundary.

The main v1 representation is ``fusion_u244``: the 196D mask engineering
reference plus the frozen P45 role-box-relative PCA48 projection.  P45's
absolute and relative PCA projections are alternatives over the same raw
DINOv3/LiLa moments; they are never concatenated with each other.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from prove.fast_adapter.p59_measurement import rotation_6d


REPRESENTATIONS = ("mask_u196", "dino_lila_relative_pca48", "fusion_u244")
PUBLIC_FILE = "public.npz"
TRANSITION_FILE = "transitions.npz"
PRIVATE_FILE = "private_labels.npz"


def _arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def relative_pca48(public, pca_path, device="cpu"):
    """Apply the frozen P45 relative projection to exported raw role moments."""
    import torch
    from prove.fast_adapter.p45_visual import project_features, validate_projection

    raw = np.asarray(public["dino_lila_role_moments"], dtype=np.float32)
    token = np.asarray(public["mask_tokens"], dtype=np.float32)
    visible = np.asarray(public["mask_valid"])
    if raw.ndim != 3 or raw.shape[1:] != (3, 3072):
        raise ValueError("Expected exported DINO/LiLa moments [T,3,3072]")
    if token.shape != (len(raw), 3, 55) or visible.shape != (len(raw), 3):
        raise ValueError("P59 mask fields do not align with visual moments")
    if visible.dtype != np.bool_:
        raise ValueError("mask_valid must remain boolean")
    package = torch.load(Path(pca_path), map_location="cpu", weights_only=True)
    validate_projection(package, "relative_fusion", verify_training_file=False)
    boxes_px = token[..., :4] * 448.
    with torch.inference_mode():
        result = project_features(raw, visible, boxes_px, package, device=device)
    value = result.detach().cpu().numpy().astype(np.float32)
    if value.shape != (len(raw), 48) or not np.isfinite(value).all():
        raise RuntimeError("P59 relative PCA projection dimension drift")
    return value


def public_representation(public, representation, pca_path=None, device="cpu"):
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown P59 representation: {representation}")
    mask = np.asarray(public["mask_u"], dtype=np.float32)
    if mask.ndim != 2 or mask.shape[1] != 196 or not np.isfinite(mask).all():
        raise ValueError("Expected finite mask_u [T,196]")
    if representation == "mask_u196":
        return mask
    if pca_path is None:
        raise ValueError("Relative DINO/LiLa representations require frozen PCA path")
    visual = relative_pca48(public, pca_path, device=device)
    return visual if representation == "dino_lila_relative_pca48" else np.concatenate((mask, visual), -1)


def q9(public):
    """Return EEF position plus the same x/y-axis rotation6D as geometry12."""
    position = np.asarray(public["eef_position"], dtype=np.float32)
    rotation = np.asarray(public["eef_rotation"], dtype=np.float32)
    if position.ndim != 2 or position.shape[1] != 3 or rotation.shape != (len(position), 3, 3):
        raise ValueError("Invalid exported EEF fields")
    axes = np.stack([rotation_6d(value) for value in rotation])
    return np.concatenate((position, axes), -1).astype(np.float32)


def load_public_episode(path, representation="fusion_u244", pca_path=None, device="cpu"):
    """Load public readout frames and strict t->t+1 transition arrays only."""
    path = Path(path)
    public = _arrays(path / PUBLIC_FILE)
    transitions = _arrays(path / TRANSITION_FILE)
    u = public_representation(public, representation, pca_path, device)
    q = q9(public)
    rows = np.asarray(transitions["one_step_interval_row"], dtype=np.int64)
    start = np.asarray(transitions["interval_start"], dtype=np.int64)[rows]
    stop = np.asarray(transitions["interval_stop"], dtype=np.int64)[rows]
    step = np.asarray(public["frame_index"], dtype=np.int64)
    action = np.asarray(transitions["executed_world_delta6"], dtype=np.float32)
    if (len(rows) != 24 or action.shape != (len(rows), 6)
            or not np.array_equal(step[stop], step[start] + 1)):
        raise ValueError("Episode lacks the fixed 24 strict one-step transitions")
    seed = int(path.name)
    episode_id = np.full(len(rows), seed, dtype=np.int64)
    phase = np.full(len(rows), "observe", dtype="U7")
    return dict(
        readout_frames=dict(u=u, q=q, step_index=step),
        continuous=dict(u=u[start], q=q[start], executed_action6=action,
                        next_u=u[stop], next_q=q[stop], phase=phase,
                        next_phase=phase.copy(), step_index=step[start],
                        next_step_index=step[stop], episode_id=episode_id),
    )


def load_private_labels(path, *, allow_private=False):
    """Load offline outer/evaluation labels only after an explicit opt-in."""
    if allow_private is not True:
        raise PermissionError("Private P59 labels require allow_private=True")
    value = _arrays(Path(path) / PRIVATE_FILE)
    if value.get("control_geometry", np.empty(0)).shape != (64, 12):
        raise ValueError("P59 private geometry12 shape drift")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _oracle_actions(p50, geometry: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Generate all four private, offline-only P50 query targets."""
    count = geometry.shape[0]
    geometry4 = geometry[:, None].expand(count, 4, 12).reshape(-1, 12)
    q4 = q[:, None].expand(count, 4, 9).reshape(-1, 9)
    phase = torch.arange(4, device=geometry.device).repeat(count)
    with torch.inference_mode():
        return p50(geometry4, q4, phase).reshape(count, 4, 6).cpu()


def load_training_dataset(dataset_root, manifest, pca_path, p50_checkpoint,
                          *, allow_private=False, device="cpu"):
    """Materialize the frozen P59 training/evaluation tensor contract.

    This is a zero-physics derivation.  Private geometry and Oracle actions
    are included only after explicit opt-in and are never inputs to runtime
    adaptation.  Sparse retained frames remain readout/query material; only
    original indices 0..24 enter the continuous transition arrays.
    """
    if allow_private is not True:
        raise PermissionError("P59 materialization requires explicit offline private-label access")
    dataset_root, manifest, pca_path = map(Path, (dataset_root, manifest, pca_path))
    p50_checkpoint = Path(p50_checkpoint)
    descriptor = json.loads(manifest.read_text())
    if descriptor.get("schema") != "prove.p59-derived-dataset.v1":
        raise ValueError("unknown P59 derived dataset manifest")
    if descriptor.get("physics_integration_steps") != 0:
        raise ValueError("P59 materialization accepts zero-physics derivations only")
    if descriptor.get("split_counts") != {"train": 96, "dev": 12}:
        raise ValueError("P59 manifest split counts differ from the frozen contract")

    from prove.task_graph.p59.core import FrozenP50Action
    p50 = FrozenP50Action(p50_checkpoint).to(device)
    splits = {"train": [], "developer": []}
    for record in descriptor.get("records", []):
        source_split = record["split"]
        if source_split not in ("train", "dev"):
            raise ValueError("unexpected P59 source split")
        episode_path = dataset_root / "episodes" / source_split / str(record["seed"])
        episode_manifest = episode_path / "MANIFEST.json"
        if _sha256(episode_manifest) != record["manifest_sha256"]:
            raise ValueError(f"episode manifest hash mismatch: {record['seed']}")
        public = load_public_episode(episode_path, "fusion_u244", pca_path, device)
        private = load_private_labels(episode_path, allow_private=True)
        readout = public["readout_frames"]
        continuous = public["continuous"]
        u = torch.from_numpy(readout["u"]).float()
        q = torch.from_numpy(readout["q"]).float()
        step = torch.from_numpy(readout["step_index"]).long()
        geometry = torch.from_numpy(np.asarray(private["control_geometry"], dtype=np.float32))
        if not torch.equal(step[:25], torch.arange(25)):
            raise ValueError("continuous P59 prefix must be original frames 0..24")
        actions = _oracle_actions(p50, geometry.to(device), q.to(device))
        episode = {
            "episode_id": str(record["seed"]),
            "source_split": source_split,
            "distribution": record["distribution"],
            "episode_manifest_sha256": record["manifest_sha256"],
            "readout_u": u,
            "readout_q": q,
            "readout_step_index": step,
            "readout_oracle_geometry": geometry,
            "readout_oracle_p50_action6": actions,
            "u": u[:25],
            "q": q[:25],
            "executed_world_delta6": torch.from_numpy(continuous["executed_action6"]).float(),
            "oracle_geometry": geometry[:25],
            "oracle_p50_action6": actions[:25],
            "step_index": step[:25],
        }
        if episode["executed_world_delta6"].shape != (24, 6):
            raise ValueError("P59 episode does not contain 24 true one-step actions")
        splits["train" if source_split == "train" else "developer"].append(episode)

    for values in splits.values():
        values.sort(key=lambda item: item["episode_id"])
    if (len(splits["train"]), len(splits["developer"])) != (96, 12):
        raise ValueError("materialized P59 split counts drifted")
    return {
        "schema": "prove.p59-training-dataset.v1",
        "feature_variant": "fusion_mask196_dino_pca48",
        "u_dim": 244,
        "feature_provenance": {
            "dataset_manifest_sha256": _sha256(manifest),
            "relative_fusion_pca_sha256": _sha256(pca_path),
            "p50_checkpoint_sha256": _sha256(p50_checkpoint),
            "physics_integration_steps": 0,
        },
        "splits": splits,
    }
