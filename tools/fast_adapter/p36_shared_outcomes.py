"""Score P36 sealed predictions against root-run shared P35/P36 outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.fast_adapter.p36_common import MODES, ROOT, TASKS, bindings, digest, dump


def key(item):
    return (item["task"], item["role"], int(item["object_id"]), int(item["t"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-outcomes", required=True)
    parser.add_argument("--shared-seal", default="evaluations/fast_adapter_p35_resolution_views_v01/branches/SHARED_BRANCH_SEAL.json")
    args = parser.parse_args()
    out = ROOT / "branches"
    predictions = torch.load(out / "PREDICTIONS.pt", weights_only=True)
    shared_outcomes = Path(args.shared_outcomes)
    outcomes_raw = torch.load(shared_outcomes, weights_only=True)["outcomes"]
    outcome_by_key = {key(item): item for item in outcomes_raw}
    ordered = []
    key_rows = []
    for snap in predictions["snapshots"]:
        k = key(snap)
        if k not in outcome_by_key:
            raise KeyError(k)
        ordered.append(outcome_by_key[k])
        key_rows.append({"task": k[0], "role": k[1], "object_id": k[2], "t": k[3]})
    if len(ordered) != 96:
        raise AssertionError(len(ordered))
    rows = []
    gates = {}
    for task in TASKS:
        task_outcomes = [item for item in ordered if item["task"] == task]
        progress = np.array([item["progress"] for item in task_outcomes])
        informative = np.ptp(progress, axis=1) > 0.002
        best = progress.max(-1)
        base = progress[:, 0]
        for row in predictions["rows"]:
            if row["task"] != task:
                continue
            chosen = row["choice"].numpy()
            actual = progress[np.arange(len(chosen)), chosen]
            regret = best - actual
            rows.append({k: row[k] for k in ("task", "seed", "geometry_mode", "adaptation")} | {
                "mean_progress_m": float(actual.mean()),
                "mean_regret_m": float(regret.mean()),
                "per_snapshot_progress_m": actual.tolist(),
                "per_snapshot_regret_m": regret.tolist(),
                "beats_base": int((actual > base + 1e-8).sum()),
                "harms_base": int((actual < base - 1e-8).sum()),
                "choices": chosen.tolist(),
            })
        checks = []
        for seed in (11, 22):
            explicit = next(r for r in rows if r["task"] == task and r["seed"] == seed and r["geometry_mode"] == "p36_explicit_relation_geometry" and r["adaptation"] == "continuous")
            existing = next(r for r in rows if r["task"] == task and r["seed"] == seed and r["geometry_mode"] == "p34_existing_bbox_geometry" and r["adaptation"] == "continuous")
            zero = next(r for r in rows if r["task"] == task and r["seed"] == seed and r["geometry_mode"] == "no_explicit_bbox_geometry" and r["adaptation"] == "continuous")
            checks.append({
                "seed": seed,
                "explicit_beats_existing_regret": explicit["mean_regret_m"] < existing["mean_regret_m"],
                "explicit_beats_zero_regret": explicit["mean_regret_m"] < zero["mean_regret_m"],
                "explicit_at_least_base_progress": explicit["mean_progress_m"] >= float(base.mean()),
            })
        gates[task] = {
            "informative_snapshots": int(informative.sum()),
            "snapshots": len(task_outcomes),
            "base_mean_progress_m": float(base.mean()),
            "base_mean_regret_m": float((best - base).mean()),
            "checks": checks,
            "pass": bool(informative.sum() >= 5 and all(all(v for k, v in c.items() if k != "seed") for c in checks)),
        }
    shared_seal = Path(args.shared_seal)
    receipt = {
        "shared_outcomes_path": str(shared_outcomes),
        "shared_seal_path": str(shared_seal),
        "shared_outcomes_sha256": digest(shared_outcomes),
        "shared_seal_sha256": digest(shared_seal) if shared_seal.exists() else None,
        "p36_prediction_seal": str(out / "PREDICTION_SEAL.json"),
        "key_order": key_rows,
        "key_count": len(key_rows),
        "bindings": bindings([out / "PREDICTIONS.pt", out / "PREDICTION_SEAL.json", shared_outcomes] + ([shared_seal] if shared_seal.exists() else [])),
    }
    dump(out / "SHARED_OUTCOME_RECEIPT.json", receipt)
    dump(out / "RESULTS.json", {
        "rows": rows,
        "gates": gates,
        "shared_outcomes_used": True,
        "shared_outcomes_path": str(shared_outcomes),
        "not_full_task_success": True,
        "bindings": bindings([out / "PREDICTIONS.pt", out / "PREDICTION_SEAL.json", shared_outcomes, out / "SHARED_OUTCOME_RECEIPT.json"]),
    })
    print("P36_SHARED_RESULTS", gates, flush=True)


if __name__ == "__main__":
    main()
