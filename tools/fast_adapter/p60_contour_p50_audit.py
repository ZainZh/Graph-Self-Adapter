#!/usr/bin/env python3
"""Post-seal evaluator-only audit for the finite P60/P50 public replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from prove.fast_adapter.geometry3d_adapter import pose
from prove.task_graph.precision_action_policy import (
    PHASES, decode_action, encode_state, load_frozen_head,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def rotation_error_degrees(a, b):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(np.asarray(a) @ np.asarray(b).T).as_rotvec())))


def true_outer_geometry(cfg):
    center = np.asarray(cfg["center"], dtype=np.float64)
    rotation = Rotation.from_euler("xyz", cfg["angles"], degrees=True).as_matrix()
    half = np.asarray(cfg["half"]) + np.asarray(cfg["clearance"]) + .006
    a, b = half
    local = np.asarray([[-a, b, 0], [a, b, 0], [a, -b, 0], [-a, -b, 0]])
    points = local @ rotation.T + center + rotation[:, 2] * .0006
    return points, pose(points)


def world_command(model, estimate, proprio, length, phase):
    feature = encode_state(estimate, proprio, length, phase)
    with torch.no_grad():
        action = model(torch.as_tensor(feature)[None])[0].numpy()
    return decode_action(action, estimate[1], proprio)


def direction_metrics(fixed, online, oracle):
    change = np.asarray(online) - np.asarray(fixed)
    desirable = np.asarray(oracle) - np.asarray(fixed)
    cn, dn = np.linalg.norm(change), np.linalg.norm(desirable)
    return {"delta_norm": float(cn), "oracle_direction_norm": float(dn),
            "cosine_toward_oracle": None if cn < 1e-12 or dn < 1e-12 else float(change @ desirable / (cn*dn)),
            "signed_projection_toward_oracle": None if dn < 1e-12 else float(change @ desirable / dn)}


def audit(replay_dir: Path, freeze_path: Path, head_path: Path, output: Path):
    seal_path = replay_dir / "PREDICTION_SEAL.json"
    prediction_path = replay_dir / "PUBLIC_PREDICTIONS.json"
    seal = json.loads(seal_path.read_text())
    if (seal.get("schema") != "prove.p60-public-prediction-seal.v1"
            or seal.get("private_truth_opened") is not False
            or seal["predictions"]["sha256"] != sha(prediction_path)):
        raise ValueError("Public predictions must be sealed before evaluator truth is opened")
    predictions = json.loads(prediction_path.read_text())
    frozen = json.loads(freeze_path.read_text())
    configurations = {x["condition_seed"]: x["evaluator_configuration"] for x in frozen["conditions"]}
    model = load_frozen_head(head_path, device="cpu")
    rows = []
    for episode in predictions["episodes"]:
        cell = Path(episode["cell"])
        result_path, obs_path = cell / "RESULT.json", cell / "OBS.npy"
        result = json.loads(result_path.read_text())
        seed = int(result["cell"]["condition_seed"])
        cfg = configurations[seed]
        true_points, truth = true_outer_geometry(cfg)
        length = float(result["tool_calibration"]["length"])
        observations = np.load(obs_path, allow_pickle=True)
        by_t = {int(x["t"]): x for x in observations}
        arm_records = {arm: {r["t"]: r for r in records if r.get("admitted")}
                       for arm, records in episode["arms"].items()}
        metrics = []
        for arm, records in arm_records.items():
            for t, record in records.items():
                predicted_points = np.asarray(record["points"])
                estimate = pose(predicted_points)
                qrow = by_t[t]
                proprio = (np.asarray(qrow["position"]), np.asarray(qrow["rotation"]))
                actions = {}
                for phase in PHASES:
                    pp, pr = world_command(model, estimate, proprio, length, phase)
                    tp, tr = world_command(model, truth, proprio, length, phase)
                    actions[phase] = {"position_error_mm": float(np.linalg.norm(pp-tp)*1000),
                                      "rotation_error_deg": rotation_error_degrees(pr, tr),
                                      "position": pp.tolist(), "oracle_conditioned_position": tp.tolist()}
                pc, pr, ps = estimate; tc, tr, ts = truth
                metrics.append({"arm": arm, "t": t, "role": record["role"],
                                "corner_rmse_mm": float(np.sqrt(np.mean((predicted_points-true_points)**2))*1000),
                                "center_error_mm": float(np.linalg.norm(pc-tc)*1000),
                                "rotation_error_deg": rotation_error_degrees(pr, tr),
                                "half_size_abs_error_mm": (np.abs(ps-ts)*1000).tolist(),
                                "actions": actions})
        directions = []
        for t in (24, 27, 29):
            fixed = arm_records["fixed"][t]
            online = arm_records["continuous"][t]
            qrow = by_t[t]; proprio = (np.asarray(qrow["position"]), np.asarray(qrow["rotation"]))
            for phase in PHASES:
                fp, _ = world_command(model, pose(np.asarray(fixed["points"])), proprio, length, phase)
                op, _ = world_command(model, pose(np.asarray(online["points"])), proprio, length, phase)
                tp, _ = world_command(model, truth, proprio, length, phase)
                directions.append({"t": t, "phase": phase, **direction_metrics(fp, op, tp),
                                   "fixed_error_mm": float(np.linalg.norm(fp-tp)*1000),
                                   "online_error_mm": float(np.linalg.norm(op-tp)*1000)})
        rows.append({"condition_seed": seed, "distribution": episode["distribution"],
                     "historical_trajectory": episode["identity"],
                     "private_sources": {"physical_freeze": binding(freeze_path), "result": binding(result_path)},
                     "public_length_calibration_m": length, "metrics": metrics,
                     "fixed_to_continuous_direction": directions})
    compact = []
    for row in rows:
        arm_summary = {}
        for arm in ("fixed", "continuous", "current_only", "batch_fit"):
            vals = [x for x in row["metrics"] if x["arm"] == arm and x["t"] in (24, 27, 29)]
            arm_summary[arm] = {key: float(np.mean([x[key] for x in vals])) for key in
                                ("center_error_mm", "rotation_error_deg", "corner_rmse_mm")}
            arm_summary[arm]["world_position_error_mm_all4phases"] = float(np.mean([
                action["position_error_mm"] for x in vals for action in x["actions"].values()]))
        directions = row["fixed_to_continuous_direction"]
        compact.append({"condition_seed": row["condition_seed"], "distribution": row["distribution"],
                        "historical_trajectory": row["historical_trajectory"], "t24_27_29": arm_summary,
                        "online_better_world_position_cells": sum(x["online_error_mm"] < x["fixed_error_mm"] for x in directions),
                        "direction_cells": len(directions),
                        "positive_projection_cells": sum((x["signed_projection_toward_oracle"] or 0) > 0 for x in directions)})
    value = {"schema": "prove.p60-p50-postseal-audit.v1", "prediction_seal": binding(seal_path),
             "audit_source": binding(Path(__file__)), "frozen_action_head": binding(head_path),
             "truth_opened_only_after_prediction_seal": True, "underlying_object_conditions": 2,
             "historical_trajectories": 4, "independent_sample_claim": False,
             "geometry_truth_role": "evaluator-only audit; never supplied to public replay",
             "physical_success_claim": False, "physics_integration_steps": 0,
             "summary": compact, "episodes": rows}
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay-dir", required=True, type=Path)
    p.add_argument("--physical-freeze", required=True, type=Path)
    p.add_argument("--action-head", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.replay_dir, args.physical_freeze, args.action_head, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
