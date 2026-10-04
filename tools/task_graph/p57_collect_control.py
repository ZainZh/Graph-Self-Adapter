#!/usr/bin/env python3
"""Read-only P57 B closeout from all 20 sealed local development cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ARMS = ("strong_bc", "geometry_prior", "geometry_fixed", "geometry_online", "geometry_direct")
ENTRIES = 4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _mean(values):
    return sum(values) / len(values) if values else None


def telemetry_stats(rows, arm):
    if arm == "strong_bc":
        return None
    all_rows = [row["telemetry"] for row in rows]
    for i, item in enumerate(all_rows):
        require(item["t"] == i and item["arm"] == arm, "Telemetry time/arm differs")
    measured = all_rows[1:]  # Includes terminal readout; t=0 has no update.
    executed_measured = [row["telemetry"] for row in rows[1:] if row["executed"]]
    output = {}
    for name, field in (("same_observation_action_delta_norm", "action_delta_norm"),
                        ("same_observation_geometry_delta_m", "geometry_delta_m"),
                        ("ray_loss_before", "ray_loss_before"),
                        ("ray_loss_after", "ray_loss_after")):
        values = [float(item[field]) for item in measured]
        require(all(math.isfinite(v) and v >= 0 for v in values), "Invalid geometry telemetry")
        output[name] = dict(mean=_mean(values), max=max(values) if values else None)
    for item in all_rows:
        action = np.asarray(item["action"], dtype=float)
        fixed = np.asarray(item["fixed_z_same_obs_action"], dtype=float)
        decoded = np.asarray(item["decoded_g"], dtype=float)
        fixed_g = np.asarray(item["fixed_z_decoded_g"], dtype=float)
        require(action.shape == fixed.shape == (7,) and decoded.shape == fixed_g.shape == (6,),
                "Unexpected same-observation diagnostic shape")
        require(np.all(np.isfinite(action)) and np.all(np.isfinite(decoded)), "Nonfinite telemetry vector")
        require(abs(np.linalg.norm(action[:6] - fixed[:6]) - item["action_delta_norm"]) < 1e-5,
                "Action delta does not match sealed same-observation readout")
        require(abs(np.linalg.norm(decoded - fixed_g) - item["geometry_delta_m"]) < 1e-5,
                "Geometry delta does not match sealed same-observation readout")
    output["readout_steps_after_entry"] = len(measured)
    output["physically_executed_readout_steps_after_entry"] = len(executed_measured)
    output["action_changed_steps"] = sum(item["action_delta_norm"] > 1e-6 for item in measured)
    output["action_changed_executed_steps"] = sum(item["action_delta_norm"] > 1e-6
                                                   for item in executed_measured)
    output["geometry_changed_steps"] = sum(item["geometry_delta_m"] > 1e-8 for item in measured)
    output["loss_decreased_steps"] = sum(item["ray_loss_after"] < item["ray_loss_before"] - 1e-9
                                         for item in measured)
    output["loss_increased_steps"] = sum(item["ray_loss_after"] > item["ray_loss_before"] + 1e-9
                                         for item in measured)
    output["mean_loss_change_after_minus_before"] = _mean(
        [item["ray_loss_after"] - item["ray_loss_before"] for item in measured])
    output["inner_update_steps"] = sum(item["inner_steps"] > 0 for item in measured)
    if arm == "geometry_online":
        require(all(item["inner_steps"] == 3 for item in measured), "Online update count differs")
    else:
        require(all(item["inner_steps"] == 0 for item in measured), "Non-online arm updated z")
    return output


def summarize_cells(cells):
    indexed = {(row["entry"], row["arm"]): row for row in cells}
    require(len(cells) == len(indexed) == ENTRIES * len(ARMS), "Incomplete/duplicate 20-cell scope")
    require(set(indexed) == {(i, arm) for i in range(ENTRIES) for arm in ARMS}, "Missing arm/entry")
    by_arm = {}
    for arm in ARMS:
        rows = [indexed[i, arm] for i in range(ENTRIES)]
        by_arm[arm] = dict(denominator=ENTRIES,
                           strict_success=sum(row["strict_success"] for row in rows),
                           official_success_at_stop=sum(row["official_success_at_stop"] for row in rows),
                           official_instant_ever=sum(row["official_instant_ever"] for row in rows),
                           official_hold3_ever=sum(row["official_hold3_ever"] for row in rows),
                           action_steps=[row["actions"] for row in rows],
                           mean_action_steps=_mean([row["actions"] for row in rows]))
        telemetry = [row.get("telemetry") for row in rows]
        if all(item is not None for item in telemetry):
            by_arm[arm]["same_observation_readout"] = dict(
                scope="per-trajectory mean of t>=1 readouts, including terminal unexecuted proposal; not physical counterfactual",
                readout_steps=sum(item["readout_steps_after_entry"] for item in telemetry),
                executed_readout_steps=sum(item["physically_executed_readout_steps_after_entry"]
                                           for item in telemetry),
                action_changed_steps=sum(item["action_changed_steps"] for item in telemetry),
                action_changed_executed_steps=sum(item["action_changed_executed_steps"]
                                                  for item in telemetry),
                geometry_changed_steps=sum(item["geometry_changed_steps"] for item in telemetry),
                loss_decreased_steps=sum(item["loss_decreased_steps"] for item in telemetry),
                loss_increased_steps=sum(item["loss_increased_steps"] for item in telemetry),
                mean_action_delta_norm=_mean([item["same_observation_action_delta_norm"]["mean"]
                                              for item in telemetry if item["same_observation_action_delta_norm"]["mean"] is not None]),
                mean_geometry_delta_m=_mean([item["same_observation_geometry_delta_m"]["mean"]
                                             for item in telemetry if item["same_observation_geometry_delta_m"]["mean"] is not None]),
                mean_ray_loss_before=_mean([item["ray_loss_before"]["mean"] for item in telemetry
                                            if item["ray_loss_before"]["mean"] is not None]),
                mean_ray_loss_after=_mean([item["ray_loss_after"]["mean"] for item in telemetry
                                           if item["ray_loss_after"]["mean"] is not None]))
    def paired(reference, candidate):
        rescue = [i for i in range(ENTRIES) if not indexed[i, reference]["strict_success"]
                  and indexed[i, candidate]["strict_success"]]
        harm = [i for i in range(ENTRIES) if indexed[i, reference]["strict_success"]
                and not indexed[i, candidate]["strict_success"]]
        return dict(reference=reference, candidate=candidate, rescue_entries=rescue,
                    harm_entries=harm, rescue=len(rescue), harm=len(harm),
                    same_result=ENTRIES-len(rescue)-len(harm))
    return dict(by_arm=by_arm,
                paired_strict=dict(fixed_to_online=paired("geometry_fixed", "geometry_online"),
                                   fixed_to_direct=paired("geometry_fixed", "geometry_direct"),
                                   online_to_direct=paired("geometry_online", "geometry_direct"),
                                   strong_bc_to_fixed=paired("strong_bc", "geometry_fixed")))


def collect(queue_root):
    queue_root = Path(queue_root).resolve()
    queue = read(queue_root / "QUEUE.json")
    plan = queue["manifest"]
    result = read(queue_root / "QUEUE_RESULT.json")
    require(plan["schema"] == "p57.geometry-control-queue.v1", "Wrong queue schema")
    require(result["schema"] == "p57.geometry-control-queue-result.v1", "Wrong queue result schema")
    require(result["planned_jobs"] == result["attempted_jobs"] == result["completed_engineering_ok"] == 20
            and result["all_batches_completed"] is True and result["automatic_retries"] == 0,
            "Queue incomplete or engineering abort present")
    manifest_bytes = (json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(queue["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(),
            "Embedded queue manifest differs from frozen manifest hash")
    binding_path = Path(plan["binding"]["path"])
    require(sha(binding_path) == plan["binding"]["sha256"], "Evaluation binding changed")
    binding = read(binding_path)
    require(binding["schema"] == "p57.geometry-control-eval.v1" and binding["arms"] == list(ARMS)
            and binding["max_actions"] == 24 and binding["automatic_retries"] == 0,
            "Changed B evaluation scope")
    expected = [dict(index=i*len(ARMS)+j, entry=i, arm=arm)
                for i in range(ENTRIES) for j, arm in enumerate(ARMS)]
    require(binding["cells"] == expected and len(binding["entries"]) == ENTRIES,
            "Changed 20-cell pairing")
    for name in ("collection_binding", "train_manifest", "geometry_manifest", "a2_analysis",
                 "plan", "training_freeze", "a2_head", "action_checkpoint", "strong_bc"):
        ref = binding[name]
        require(sha(ref["path"]) == ref["sha256"], "Frozen binding reference changed: " + name)
    for source, expected_hash in binding["sources"].items():
        require(sha(Path(__file__).resolve().parents[2] / source) == expected_hash,
                "Frozen evaluator source changed: " + source)
    require(len(plan["jobs"]) == 20 and [dict(index=j["index"], entry=j["entry"], arm=j["arm"])
            for j in plan["jobs"]] == expected, "Queue jobs differ from binding")
    require(len(result["receipt_sha256"]) == 20, "Missing receipts")
    cells = []
    for cell in expected:
        i = cell["index"]
        folder = queue_root / f"cell_{i:03d}"
        episode = folder / "episode"
        receipt = read(folder / "RECEIPT.json")
        require(sha(folder / "RECEIPT.json") == result["receipt_sha256"][i], "Receipt hash differs")
        require(receipt["job"]["index"] == i and receipt["engineering_ok"] is True
                and receipt["returncode"] == 0 and receipt["attempt"] == 1
                and receipt["automatic_retry"] is False
                and receipt["manifest_sha256"] == queue["manifest_sha256"],
                "Cell execution receipt invalid")
        require(receipt["artifacts"]["RESULT.json"] == sha(episode / "RESULT.json")
                and receipt["artifacts"]["SEAL.json"] == sha(episode / "SEAL.json"),
                "Receipt artifact hash differs")
        require(not (episode / "ABORT.json").exists(), "Cell has ABORT")
        cell_result, seal = read(episode / "RESULT.json"), read(episode / "SEAL.json")
        require(cell_result["schema"] == "p57.geometry-control-result.v1"
                and seal["schema"] == "p57.geometry-control-seal.v1", "Wrong cell schema")
        require(cell_result["cell"] == seal["cell"] == read(episode / "CELL.json") == cell,
                "Cell identity differs")
        require(read(episode / "BINDING.json") == binding and
                seal["binding_sha256"] == sha(episode / "BINDING.json"), "Cell binding differs")
        require(cell_result["seal_sha256"] == sha(episode / "SEAL.json"), "Result seal hash differs")
        require({"public/ACTIONS.jsonl", "private/JUDGMENTS.jsonl"}.issubset(seal["files"]),
                "Sealed action/judgment logs missing")
        for relative, expected_hash in seal["files"].items():
            path = (episode / relative).resolve()
            require(path.is_relative_to(episode) and sha(path) == expected_hash,
                    "Sealed payload changed: " + relative)
        require(cell_result["near_entry_qualified"] is True and cell_result["micro_executed"] is True
                and cell_result["automatic_retries"] == 0
                and cell_result["full_task_success_evidence"] is False, "Result scope differs")
        actions = cell_result["executed_physical_steps"]
        require(type(actions) is int and 0 <= actions <= 24 and seal["executed_physical_steps"] == actions,
                "Physical step count differs")
        public = lines(episode / "public/ACTIONS.jsonl")
        private = lines(episode / "private/JUDGMENTS.jsonl")
        require(len(public) == len(private) == actions+1 and
                [row["t"] for row in public] == list(range(actions+1)) and
                sum(row["executed"] is True for row in public) == actions and
                public[-1]["executed"] is False, "Action/judgment count differs")
        require(cell_result["system_success"] is cell_result["micro_success"] is
                private[-1]["complete"] and cell_result["official_success"] is private[-1]["official_success"],
                "Final strict/official judgment differs")
        for row in public:
            require(row["executed_physical_steps"] == row["t"] and
                    row["proposed_action"][-1] == 1, "Action step/closed gripper differs")
            if cell["arm"] != "strong_bc":
                require(np.allclose(row["proposed_action"], row["telemetry"]["action"],
                                    atol=1e-8, rtol=0), "Telemetry action differs from proposed command")
            frame = episode / "public" / row["observation"]
            features = episode / "public" / f"features_{row['t']:04d}.pt"
            require(row["observation"] == f"frame_{row['t']:04d}.npz" and
                    sha(frame) == row["observation_sha256"] and
                    sha(features) == row["features_sha256"], "Per-step public source hash differs")
        entry = cell["entry"]
        require(cell_result["source_cell_index"] == binding["entries"][entry]["source_cell_index"],
                "Source entry differs")
        cells.append(dict(index=i, entry=entry, arm=cell["arm"],
                          condition_id=cell_result["condition_id"],
                          strict_success=cell_result["system_success"],
                          official_success_at_stop=cell_result["official_success"],
                          official_instant_ever=cell_result["official_instant_ever"],
                          official_hold3_ever=cell_result["official_only_hold3"],
                          actions=actions, reason=cell_result["reason"],
                          telemetry=telemetry_stats(public, cell["arm"]),
                          result_sha256=sha(episode / "RESULT.json"), seal_sha256=sha(episode / "SEAL.json")))
    for entry in range(ENTRIES):
        require(len({cell["condition_id"] for cell in cells if cell["entry"] == entry}) == 1,
                "Paired condition IDs differ")
    summary = summarize_cells(cells)
    return dict(schema="p57.geometry-control-closeout.v1", scope="20 opened P56 validation near-entry local s9 cells; no full-task or held-out claim",
                queue_result_sha256=sha(queue_root / "QUEUE_RESULT.json"),
                binding_sha256=sha(binding_path), collector_sha256=sha(__file__),
                cells=cells, **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    output = collect(args.queue_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
