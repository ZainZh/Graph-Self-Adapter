#!/usr/bin/env python3
"""Finite, zero-physics replay for the P60 public-contour candidate.

The command is deliberately split into ``prepare``, ``predict`` and ``audit``.
``predict`` only opens public observation/image archives.  ``audit`` refuses to
open a private label pack until the public prediction file and its seal agree.
Sparse t27/t29 frames are queries: they re-encode the current public frame with
the latent frozen at t24 and never enter the contour history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from prove.fast_adapter.geometry3d_adapter import pose
from prove.fast_adapter.p45_neural import load_model
from prove.fast_adapter.p60_contour_scaled import P60ContourScaledLatent


SCHEMA = "prove.p60-contour-scaled-replay.v1"
P45_EPISODES = (("train", 456000, "id"), ("train", 456001, "id"),
                ("dev", 456400, "id"), ("dev", 456406, "geometry"))
UPDATE_TIMES = tuple(range(25))
FUTURE_TIMES = (27, 29)
ARMS = ("fixed", "continuous")
DEFAULTS = dict(boundary_samples=32, samples_per_edge=8, damping=1e-4,
                step_scale=.65, step_clip=1.0, inner_steps=3,
                batch_fit_steps=15)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _source_binding(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def prepare_contract(out: Path, checkpoint: Path, sources: list[dict]) -> dict:
    """Write the result-free candidate contract; never replace an attempt."""
    out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__)
    core = Path("prove/fast_adapter/p60_contour_scaled.py")
    base_core = Path("prove/fast_adapter/p60_contour.py")
    amendment = Path("evaluations/fast_adapter_p60_observation_adaptation_v01/AMENDMENT_B2_CURVATURE_SCALE.md")
    contract = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "candidate": {
            "initialization": "P45 semantic-corner IPPE/PnP",
            "current_encoder": "frozen P45 insert/geometry/seed11; retains aligned corners",
            "update_measurement": "public slot-mask outer convex hull; no semantic correspondence",
            "shape_prior": "planar rectangular outer frame decoded by frozen P45 rank-8 basis",
            "latent": "persistent z8; weights frozen",
            "arms": list(ARMS),
            "parameters": DEFAULTS,
            "optimizer_only_change": "s=max(trace(H)/8,1e-12); solve(H/s+1e-4I,g/s)",
            "held_length": "unchanged P50 public calibration; not estimated by this replay",
        },
        "causal_schedule": {
            "update_times": list(UPDATE_TIMES),
            "future_query_times": list(FUTURE_TIMES),
            "future_queries_update_history": False,
            "sparse_gap_is_transition": False,
        },
        "cohort": sources,
        "development_identity": "opened developer material; never an independent test",
        "selection_rule": "no parameter or checkpoint selection from these four episodes",
        "truth_access": "predict is public-only; private GT may be opened only by audit after PREDICTION_SEAL",
        "physics_integration_steps": 0,
        "checkpoint": _source_binding(checkpoint),
        "sources": {str(p): sha256(p) for p in (script, core, base_core, amendment)},
    }
    write_json(out / "CANDIDATE_CONTRACT.json", contract)
    return contract


def _unpack_p45_episode(root: Path, split: str, seed: int) -> tuple[list[dict], np.ndarray, dict]:
    episode = root / "data_raw" / "insert" / split / str(seed)
    public_path, image_path = episode / "public_observations.npz", episode / "visible_images.npz"
    receipt_path = episode / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("seed") != seed or receipt.get("split") != split:
        raise ValueError(f"P45 episode identity mismatch: {episode}")
    if receipt.get("public_sha256") != sha256(public_path) or receipt.get("images_sha256") != sha256(image_path):
        raise ValueError(f"P45 episode source hash mismatch: {episode}")
    with np.load(public_path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    with np.load(image_path, allow_pickle=False) as archive:
        masks = np.unpackbits(archive["role_masks_packed"], axis=-1)[..., :448]
    times = data["t"].astype(int)
    required = UPDATE_TIMES + FUTURE_TIMES
    rows = []
    for t in required:
        where = np.flatnonzero(times == t)
        if len(where) != 1:
            raise ValueError(f"Expected exactly one public frame at t={t}: {episode}")
        i = int(where[0])
        if "corners_valid" in data and not bool(data["corners_valid"][i]):
            raise ValueError(f"P60 frozen smoke frame has invalid P45 initialization corners: {episode}/t{t}")
        obs = {key: data[key][i] for key in (
            "corners", "K", "camera_position", "camera_rotation", "position", "rotation",
            "role_bbox", "role_visible", "previous_action")}
        obs["local"] = np.zeros(48, np.float32)
        rows.append({"t": t, "obs": obs, "slot_mask": masks[i, 2].astype(np.uint8)})
    meta = {"kind": "p45_raw", "split": split, "seed": seed,
            "distribution": receipt["distribution"],
            "public": _source_binding(public_path), "images": _source_binding(image_path),
            "receipt": _source_binding(receipt_path)}
    return rows, times, meta


def _unpack_p50_cell(cell: Path, identity: str, distribution: str) -> tuple[list[dict], np.ndarray, dict]:
    obs_path, image_path = cell / "OBS.npy", cell / "PUBLIC_PIXELS.npz"
    raw = np.load(obs_path, allow_pickle=True)
    with np.load(image_path, allow_pickle=False) as archive:
        masks = archive["visible_masks"].copy()
    times = np.asarray([int(row["t"]) for row in raw])
    rows = []
    for t in UPDATE_TIMES + FUTURE_TIMES:
        where = np.flatnonzero(times == t)
        if len(where) != 1:
            raise ValueError(f"Expected one P50 public frame at t={t}: {cell}")
        i = int(where[0]); obs = dict(raw[i])
        obs["local"] = np.zeros(48, np.float32)
        rows.append({"t": t, "obs": obs, "slot_mask": masks[i, 2].astype(np.uint8)})
    meta = {"kind": "p50_opened_cell", "identity": identity, "distribution": distribution,
            "cell": str(cell), "public": _source_binding(obs_path), "images": _source_binding(image_path)}
    return rows, times, meta


def load_public_episodes(args) -> list[tuple[list[dict], dict]]:
    episodes = []
    if args.source_kind == "p45":
        for split, seed, expected_distribution in P45_EPISODES:
            rows, _, meta = _unpack_p45_episode(args.data_root, split, seed)
            if meta["distribution"] != expected_distribution:
                raise ValueError(f"P45 split definition drift for {seed}")
            episodes.append((rows, meta))
    else:
        specs = (("00_id_505100_neural_fixed", "id505100_fixed", "id"),
                 ("01_id_505100_neural_continuous", "id505100_continuous", "id"),
                 ("16_geometry_505200_neural_fixed", "geometry505200_fixed", "geometry"),
                 ("17_geometry_505200_neural_continuous", "geometry505200_continuous", "geometry"))
        for directory, identity, distribution in specs:
            rows, _, meta = _unpack_p50_cell(args.data_root / directory, identity, distribution)
            episodes.append((rows, meta))
    return episodes


def _public_source_specs(args) -> list[dict]:
    specs = []
    if args.source_kind == "p45":
        for split, seed, distribution in P45_EPISODES:
            specs.append({"kind": "p45_raw", "split": split, "seed": seed,
                          "distribution": distribution})
    else:
        specs = [{"kind": "p50_opened_cell", "identity": x, "distribution": d}
                 for x, d in (("id505100_fixed", "id"), ("id505100_continuous", "id"),
                              ("geometry505200_fixed", "geometry"),
                              ("geometry505200_continuous", "geometry"))]
    return specs


def _future_prediction(adapter, obs: dict) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read current public features while freezing z/history at the last support frame."""
    if adapter.z is None:
        raise RuntimeError("Future query requires an initialized latent")
    _, _, reference, _, offset, _ = adapter._encode_current(obs, reset=False)
    points = adapter._decode(reference, offset, adapter.z).detach().cpu().numpy()[0]
    adapter.current_offset, adapter.points = offset.detach(), points
    return points, pose(points)


def run_public_prediction(out: Path, checkpoint: Path, episodes, device: str) -> dict:
    contract_path = out / "CANDIDATE_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    if contract["schema"] != SCHEMA or contract["checkpoint"]["sha256"] != sha256(checkpoint):
        raise ValueError("Candidate contract/checkpoint binding mismatch")
    values = []
    for rows, meta in episodes:
        episode = dict(meta)
        episode["arms"] = {}
        for arm in ARMS:
            model, package = load_model(checkpoint, device=device)
            adapter = P60ContourScaledLatent(model, arm=arm, device=device, **DEFAULTS)
            if package.get("model_config", {}).get("representation") != "geometry":
                raise ValueError("P60 v1 is bound to the P45 geometry representation")
            records = []
            for row in rows:
                t, obs, slot = row["t"], row["obs"], row["slot_mask"]
                if t in UPDATE_TIMES:
                    admitted = adapter.update(obs, t, slot)
                    if not admitted:
                        records.append({"t": t, "role": "continuous_support", "admitted": False})
                        continue
                    record = dict(adapter.records[-1])
                    record.update(role="continuous_support", admitted=True, history_updated=True)
                else:
                    points, estimate = _future_prediction(adapter, obs)
                    center, rotation, size = estimate
                    record = {"t": t, "role": "future_query", "admitted": True,
                              "history_updated": False, "points": points.tolist(),
                              "z": adapter.z.cpu().numpy()[0].tolist(),
                              "center": center.tolist(), "rotation": rotation.tolist(),
                              "half_size": size.tolist()}
                records.append(record)
            episode["arms"][arm] = records
        values.append(episode)
    prediction = {"schema": "prove.p60-public-predictions.v1", "contract_sha256": sha256(contract_path),
                  "physics_integration_steps": 0, "episodes": values}
    prediction_path = out / "PUBLIC_PREDICTIONS.json"
    write_json(prediction_path, prediction)
    seal = {"schema": "prove.p60-public-prediction-seal.v1",
            "contract": _source_binding(contract_path), "predictions": _source_binding(prediction_path),
            "private_truth_opened": False, "sealed_unix": time.time()}
    write_json(out / "PREDICTION_SEAL.json", seal)
    return prediction


def _geodesic_degrees(a, b) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(np.asarray(a) @ np.asarray(b).T).as_rotvec())))


def audit_p45(out: Path, data_root: Path) -> dict:
    seal_path, prediction_path = out / "PREDICTION_SEAL.json", out / "PUBLIC_PREDICTIONS.json"
    seal = json.loads(seal_path.read_text())
    if seal.get("private_truth_opened") is not False or seal["predictions"]["sha256"] != sha256(prediction_path):
        raise ValueError("Public prediction seal is absent or invalid")
    prediction = json.loads(prediction_path.read_text())
    audits = []
    for episode in prediction["episodes"]:
        if episode.get("kind") != "p45_raw":
            raise ValueError("Private corner audit is defined only for the sealed P45 collection")
        label_path = data_root / "data_labels" / "insert" / episode["split"] / f'{episode["seed"]}.npz'
        with np.load(label_path, allow_pickle=False) as archive:
            truth = archive["corners"].copy()
        public_path = Path(episode["public"]["path"])
        with np.load(public_path, allow_pickle=False) as archive:
            times = archive["t"].astype(int)
        rows = []
        for arm, records in episode["arms"].items():
            for record in records:
                if not record.get("admitted"):
                    continue
                i = int(np.flatnonzero(times == record["t"])[0])
                predicted = np.asarray(record["points"])
                target = truth[i]
                pc, pr, ps = pose(predicted); tc, tr, ts = pose(target)
                rows.append({"arm": arm, "t": record["t"], "role": record["role"],
                             "corner_rmse_mm": float(np.sqrt(np.mean((predicted-target)**2))*1000),
                             "center_error_mm": float(np.linalg.norm(pc-tc)*1000),
                             "rotation_error_deg": _geodesic_degrees(pr, tr),
                             "half_size_error_mm": (np.abs(ps-ts)*1000).tolist()})
        audits.append({"split": episode["split"], "seed": episode["seed"],
                       "distribution": episode["distribution"], "private_labels": _source_binding(label_path),
                       "rows": rows})
    result = {"schema": "prove.p60-private-postseal-audit.v1", "prediction_seal": _source_binding(seal_path),
              "opened_development_not_test": True, "physical_success_claim": False, "episodes": audits}
    write_json(out / "PRIVATE_GEOMETRY_AUDIT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=("prepare", "predict", "audit"))
    p.add_argument("--source-kind", choices=("p45", "p50"), default="p45")
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, default=Path(
        "evaluations/fast_adapter_p50_precision_policy_v01/input_cache/p45_insert_geometry_seed11.pt"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.operation == "prepare":
        value = prepare_contract(args.out, args.checkpoint, _public_source_specs(args))
    elif args.operation == "predict":
        value = run_public_prediction(args.out, args.checkpoint, load_public_episodes(args), args.device)
    else:
        if args.source_kind != "p45":
            raise ValueError("P50 local smoke has no private P45 label audit")
        value = audit_p45(args.out, args.data_root)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
