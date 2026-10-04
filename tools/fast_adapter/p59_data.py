#!/usr/bin/env python3
"""Inventory/export P45 insert bytes into the isolated P59 data contract.

The exporter performs no physics integration.  It copies public observations,
derives mask tokens from already-rendered masks, uses recorded issued commands,
and writes private functional labels to a separate file.  Sparse observed
intervals retain their actual delta and complete recorded command segment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prove.fast_adapter.geometry3d_env import configuration
from prove.fast_adapter.p59_measurement import private_labels
from prove.fast_adapter.p59_observation import (
    executed_world_delta6, mask_representation, observed_intervals, public_frame,
    unpack_role_masks, validate_dino_lila_cache,
)


SCHEMA = "prove.p59-insert-data.v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_trace(path):
    raw = np.load(path, allow_pickle=True)
    if raw.shape != (127,) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("Expected the retained 127-step P45 collector trace")
    fields = ("t", "command_p", "command_R", "closed", "qpos", "qvel")
    if any(not set(fields).issubset(row) for row in raw):
        raise ValueError("P45 collector trace lacks actual transition fields")
    if [int(row["t"]) for row in raw] != list(range(127)):
        raise ValueError("P45 action indices are not the full 0..126 sequence")
    return raw


def _camera_pose(position, rotation):
    result = np.broadcast_to(np.eye(4, dtype=np.float32), (len(position), 4, 4)).copy()
    result[:, :3, :3] = rotation
    result[:, :3, 3] = position
    return result


def export_episode(source, output):
    source, output = Path(source), Path(output)
    receipt = json.loads((source / "RECEIPT.json").read_text())
    pre = json.loads((source / "PRE_RUN.json").read_text())
    if receipt.get("task") != "insert" or not receipt.get("collection_quality_qualified"):
        raise ValueError("P59 accepts only qualified P45 insert collection episodes")
    if receipt["seed"] != pre["seed"] or receipt["split"] != pre["split"]:
        raise ValueError("P45 receipt/pre-run identity mismatch")
    for filename, key in (("public_observations.npz", "public_sha256"),
                          ("visible_images.npz", "images_sha256")):
        if sha(source / filename) != receipt[key]:
            raise ValueError(f"P45 source digest mismatch: {filename}")
    with np.load(source / "public_observations.npz", allow_pickle=False) as arrays:
        public = {key: arrays[key].copy() for key in arrays.files}
    with np.load(source / "visible_images.npz", allow_pickle=False) as arrays:
        rgb = arrays["rgb"].copy()
        masks = np.stack([unpack_role_masks(row) for row in arrays["role_masks_packed"]])
    encoded = json.loads((source / "ENCODE.json").read_text())
    if (sha(source / "raw_visual.npy") != encoded.get("raw_sha256")
            or encoded.get("shape") != [64, 3, 3072]
            or encoded.get("encoding_freeze", {}).get("sha256")
            != "c2ee5879fd470e3a61250b5f183830f3a9a422c9887493b014239992461994dd"):
        raise ValueError("P45 raw visual cache lacks its frozen ENCODE binding")
    cache = np.load(source / "raw_visual.npy", allow_pickle=False)
    if cache.shape != (64, 3, 3072):
        raise ValueError("P45 DINO/LiLa cache shape differs from its frozen contract")
    trace = _load_trace(source / "COLLECTOR_TRACE.npy")
    frame_indices = public["t"].astype(np.int64)
    if frame_indices.tolist() != pre["times"] or len(frame_indices) != 64:
        raise ValueError("P45 observed frame schedule mismatch")
    camera_pose = _camera_pose(public["camera_position"], public["camera_rotation"])
    mask_u, tokens, valid = [], [], []
    for index in range(64):
        frame = public_frame(rgb[index], masks[index], public["K"][index],
                             public["camera_position"][index], public["camera_rotation"][index],
                             public["position"][index], public["rotation"][index],
                             public["previous_action"][index], frame_indices[index])
        mask_u.append(mask_representation(frame))
        tokens.append(frame["mask_tokens"])
        valid.append(frame["mask_valid"])
        validate_dino_lila_cache(cache[index])
    interval = observed_intervals(frame_indices)
    action_offsets = [0]
    action_p, action_R, action_closed = [], [], []
    for start, stop in zip(frame_indices[:-1], frame_indices[1:]):
        segment = trace[int(start):int(stop)]
        action_p.extend(np.asarray(row["command_p"], np.float32) for row in segment)
        action_R.extend(np.asarray(row["command_R"], np.float32) for row in segment)
        action_closed.extend(bool(row["closed"]) for row in segment)
        action_offsets.append(len(action_p))
    # Only observed t->t+1 pairs have the actual pre-action Cartesian EEF in
    # this cache.  Sparse intervals retain their raw command segments and are
    # deliberately excluded from this one-step feedback array.
    one_step_rows = np.flatnonzero(interval["one_step"])
    executed_delta6 = np.stack([
        executed_world_delta6(public["position"][row], public["rotation"][row],
                              trace[int(frame_indices[row])]["command_p"],
                              trace[int(frame_indices[row])]["command_R"])
        for row in one_step_rows
    ])
    cfg = configuration("insert", int(receipt["seed"]), receipt["distribution"])
    label_path = Path(receipt["labels"]["path"])
    if not label_path.is_absolute():
        # Receipt paths are repository-relative; infer the repository from the
        # .../evaluations/<study>/data_raw/... source ancestry.
        repo = next(parent for parent in source.parents if parent.name == "evaluations").parent
        label_path = repo / label_path
    if sha(label_path) != receipt["labels"]["sha256"]:
        raise ValueError("P45 private label digest mismatch")
    with np.load(label_path, allow_pickle=False) as labels:
        center, rotation, size = (labels[key].copy() for key in ("center", "rotation", "size"))
    private = [private_labels(center[i], rotation[i], size[i],
                              public["position"][i], public["rotation"][i], cfg["length"],
                              public["K"][i], camera_pose[i], public["role_visible"][i])
               for i in range(64)]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "public.npz",
                        rgb=rgb, role_masks_packed=np.packbits(masks, axis=-1),
                        mask_tokens=np.asarray(tokens), mask_valid=np.asarray(valid),
                        mask_u=np.asarray(mask_u), dino_lila_role_moments=cache,
                        camera_K=public["K"].astype(np.float32), camera_pose=camera_pose,
                        eef_position=public["position"].astype(np.float32),
                        eef_rotation=public["rotation"].astype(np.float32),
                        previous_action=public["previous_action"].astype(np.float32),
                        frame_index=frame_indices)
    np.savez_compressed(output / "transitions.npz",
                        interval_start=interval["start"], interval_stop=interval["stop"],
                        delta_steps=interval["delta_steps"], one_step=interval["one_step"],
                        action_offsets=np.asarray(action_offsets, dtype=np.int64),
                        issued_command_p=np.asarray(action_p), issued_command_R=np.asarray(action_R),
                        issued_closed=np.asarray(action_closed, dtype=bool),
                        one_step_interval_row=one_step_rows.astype(np.int64),
                        executed_world_delta6=executed_delta6)
    np.savez_compressed(output / "private_labels.npz",
                        **{key: np.asarray([row[key] for row in private]) for key in private[0]})
    manifest = dict(schema=SCHEMA, seed=int(receipt["seed"]), split=receipt["split"],
                    distribution=receipt["distribution"], frames=64,
                    one_step_transitions=int(interval["one_step"].sum()),
                    sparse_intervals=int((~interval["one_step"]).sum()),
                    source={name: dict(path=str(source / name), sha256=sha(source / name))
                            for name in ("PRE_RUN.json", "RECEIPT.json", "public_observations.npz",
                                         "visible_images.npz", "raw_visual.npy", "ENCODE.json",
                                         "COLLECTOR_TRACE.npy")},
                    visual_encoding_freeze=encoded["encoding_freeze"],
                    visual_source_sha256=encoded.get("vision_sources", {}),
                    private_source=dict(path=str(label_path), sha256=sha(label_path)),
                    output={name: sha(output / name) for name in
                            ("public.npz", "transitions.npz", "private_labels.npz")},
                    public_private_separate=True, physics_integration_steps=0,
                    dino_lila_cache="available: frozen DINOv3 layers fused by official LiLa fusion weights; 3x3072 role moments",
                    mask_u="196D mask/calibration/EEF engineering reference; not DINO")
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def inventory(root):
    root = Path(root)
    rows = []
    for split in ("train", "dev"):
        for source in sorted((root / "data_raw" / "insert" / split).glob("*")):
            if not (source / "RECEIPT.json").exists():
                continue
            receipt = json.loads((source / "RECEIPT.json").read_text())
            pre = json.loads((source / "PRE_RUN.json").read_text())
            interval = observed_intervals(np.asarray(pre["times"], dtype=np.int64))
            rows.append(dict(seed=receipt["seed"], split=split,
                             distribution=receipt["distribution"], frames=receipt["frames"],
                             qualified=receipt["collection_quality_qualified"],
                             one_step_transitions=int(interval["one_step"].sum()),
                             sparse_intervals=int((~interval["one_step"]).sum()),
                             rgb_masks=(source / "visible_images.npz").exists(),
                             dino_lila_cache=(source / "raw_visual.npy").exists(),
                             actual_action_trace=(source / "COLLECTOR_TRACE.npy").exists()))
    return rows


def export_dataset(root, output):
    """Export the fixed 96/12 P45 cohort and bind every derived episode."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("P59 dataset output already exists; never overwrite an attempt")
    expected = {"train": 96, "dev": 12}
    records = []
    for split, count in expected.items():
        sources = sorted((root / "data_raw" / "insert" / split).glob("*"))
        if len(sources) != count:
            raise ValueError(f"P45 {split} episode count differs from {count}")
        for source in sources:
            destination = output / "episodes" / split / source.name
            manifest = export_episode(source, destination)
            records.append(dict(seed=manifest["seed"], split=manifest["split"],
                                distribution=manifest["distribution"],
                                manifest_path=str(destination / "MANIFEST.json"),
                                manifest_sha256=sha(destination / "MANIFEST.json")))
    split_counts = {split: sum(row["split"] == split for row in records) for split in expected}
    distribution_counts = {
        split: {distribution: sum(row["split"] == split and row["distribution"] == distribution
                                 for row in records)
                for distribution in ("id", "geometry")}
        for split in expected
    }
    if split_counts != expected or distribution_counts != {
            "train": {"id": 96, "geometry": 0},
            "dev": {"id": 6, "geometry": 6}}:
        raise RuntimeError("P59 cohort identity drift")
    value = dict(schema="prove.p59-derived-dataset.v1", source_study="P45",
                 split_counts=split_counts, distribution_counts=distribution_counts,
                 frames={"train": 6144, "dev": 768},
                 strict_one_step_transitions={"train": 2304, "dev": 288},
                 development_identity="P45 dev is opened developer material, not fresh held-out test",
                 public_private_separate=True, physics_integration_steps=0,
                 records=records)
    (output / "DATASET_MANIFEST.json").write_text(json.dumps(value, indent=2) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    one = sub.add_parser("export-episode")
    one.add_argument("--source", required=True, type=Path)
    one.add_argument("--output", required=True, type=Path)
    inv = sub.add_parser("inventory")
    inv.add_argument("--root", required=True, type=Path)
    inv.add_argument("--output", type=Path)
    dataset = sub.add_parser("export-dataset")
    dataset.add_argument("--root", required=True, type=Path)
    dataset.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "export-episode":
        value = export_episode(args.source, args.output)
    elif args.operation == "export-dataset":
        value = export_dataset(args.root, args.output)
    else:
        rows = inventory(args.root)
        value = dict(schema="prove.p59-source-inventory.v1", episodes=len(rows), rows=rows,
                     counts={split: sum(row["split"] == split for row in rows)
                             for split in ("train", "dev")},
                     one_step_transitions=sum(row["one_step_transitions"] for row in rows))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
