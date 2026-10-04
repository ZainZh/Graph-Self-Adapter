#!/usr/bin/env python3
"""Independent, local-only audit of returned P57 B control artifacts.

Never imports the original collector, launches a simulator, or contacts the
server. Remote absolute source references were verified by the remote closeout;
this audit independently verifies the returned bytes and recomputes statistics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np

EXPECTED_TAR_SHA256 = "beb5e9c23df6abef2292d39208be364c598b1c6b52343f7d435bf744b2fa4b55"
ARMS = ("strong_bc", "geometry_prior", "geometry_fixed", "geometry_online", "geometry_direct")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, path="root"):
    """Recursive equality with a tiny allowance for independent float sums."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        require(set(actual) == set(expected), f"{path} keys differ")
        for key in actual:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list) and isinstance(expected, list):
        require(len(actual) == len(expected), f"{path} length differs")
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{i}]")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10),
                f"{path}: {actual} != {expected}")
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def extract_verified(archive, destination):
    archive = Path(archive).resolve()
    destination = Path(destination).resolve()
    require(digest(archive) == EXPECTED_TAR_SHA256, "P57 B return tar SHA differs")
    require(not destination.exists(), "Fresh audit extraction directory required")
    with tarfile.open(archive, "r:") as source:
        names = set()
        for member in source.getmembers():
            relative = Path(member.name)
            require(not relative.is_absolute() and ".." not in relative.parts,
                    "Unsafe archive member path")
            require(not member.issym() and not member.islnk() and
                    (member.isfile() or member.isdir()), "Archive link or special file")
            require(member.name not in names, "Duplicate archive member")
            names.add(member.name)
        destination.mkdir(parents=True)
        source.extractall(destination)
    queues = list(destination.rglob("QUEUE.json"))
    require(len(queues) == 1, "Returned archive must contain one control queue")
    return queues[0].parent, len(names)


def _mean(values):
    return sum(values)/len(values) if values else None


def telemetry_stats(actions, arm):
    if arm == "strong_bc":
        return None
    telemetry = [row["telemetry"] for row in actions]
    for t, item in enumerate(telemetry):
        require(item["t"] == t and item["arm"] == arm, "Telemetry time/arm differs")
        for key, shape in (("g", (6,)), ("z", (6,)), ("decoded_g", (6,)),
                           ("fixed_z_decoded_g", (6,)), ("action", (7,)),
                           ("fixed_z_same_obs_action", (7,)), ("pred_uv", (2,2,2)),
                           ("ray_A", (2,2,3,3)), ("ray_C", (2,2,3)),
                           ("ray_valid", (2,2))):
            value = np.asarray(item[key])
            require(value.shape == shape and np.isfinite(value).all(),
                    f"Invalid public telemetry {key}")
        require(abs(np.linalg.norm(np.asarray(item["action"])[:6] -
                                   np.asarray(item["fixed_z_same_obs_action"])[:6]) -
                    item["action_delta_norm"]) < 1e-5, "Action delta differs")
        require(abs(np.linalg.norm(np.asarray(item["decoded_g"]) -
                                   np.asarray(item["fixed_z_decoded_g"])) -
                    item["geometry_delta_m"]) < 1e-5, "Geometry delta differs")
        require(item["valid_rays"] == int(np.asarray(item["ray_valid"]).sum()),
                "Ray validity count differs")
    after = telemetry[1:]
    executed = [row["telemetry"] for row in actions[1:] if row["executed"]]
    stats = {}
    for name, key in (("same_observation_action_delta_norm", "action_delta_norm"),
                      ("same_observation_geometry_delta_m", "geometry_delta_m"),
                      ("ray_loss_before", "ray_loss_before"),
                      ("ray_loss_after", "ray_loss_after")):
        values = [float(item[key]) for item in after]
        require(all(math.isfinite(value) and value >= 0 for value in values),
                "Nonfinite/negative geometry telemetry")
        stats[name] = dict(mean=_mean(values), max=max(values) if values else None)
    stats.update(readout_steps_after_entry=len(after),
                 physically_executed_readout_steps_after_entry=len(executed),
                 action_changed_steps=sum(item["action_delta_norm"] > 1e-6 for item in after),
                 action_changed_executed_steps=sum(item["action_delta_norm"] > 1e-6 for item in executed),
                 geometry_changed_steps=sum(item["geometry_delta_m"] > 1e-8 for item in after),
                 loss_decreased_steps=sum(item["ray_loss_after"] < item["ray_loss_before"]-1e-9 for item in after),
                 loss_increased_steps=sum(item["ray_loss_after"] > item["ray_loss_before"]+1e-9 for item in after),
                 mean_loss_change_after_minus_before=_mean([
                     item["ray_loss_after"]-item["ray_loss_before"] for item in after]),
                 inner_update_steps=sum(item["inner_steps"] > 0 for item in after))
    expected_inner = 3 if arm == "geometry_online" else 0
    require(all(item["inner_steps"] == expected_inner for item in after),
            "Unexpected latent update count")
    return stats


def audit_cell(root, index, binding, queue_hash, queue_result):
    arm = ARMS[index % 5]
    entry = index // 5
    cell = dict(index=index, entry=entry, arm=arm)
    folder = root/f"cell_{index:03d}"
    episode = folder/"episode"
    receipt = read(folder/"RECEIPT.json")
    attempt = read(folder/"ATTEMPT.json")
    require(digest(folder/"RECEIPT.json") == queue_result["receipt_sha256"][index],
            "Queue receipt SHA differs")
    require(receipt["job"]["index"] == index and receipt["job"]["arm"] == arm and
            receipt["job"]["entry"] == entry and receipt["job"]["gpu"] == index%4,
            "Queue job differs")
    require(receipt["manifest_sha256"] == queue_hash and
            receipt["attempt"] == attempt["attempt"] == 1 and
            receipt["automatic_retry"] is False and attempt["automatic_retry"] is False and
            receipt["engineering_ok"] is True and receipt["returncode"] == 0,
            "Queue execution receipt differs")
    require(digest(folder/"process.log") == receipt["log_sha256"], "Process log SHA differs")
    require(not (episode/"ABORT.json").exists(), "Cell has engineering ABORT")
    result = read(episode/"RESULT.json")
    seal = read(episode/"SEAL.json")
    require(receipt["artifacts"] == {
        "RESULT.json": digest(episode/"RESULT.json"),
        "SEAL.json": digest(episode/"SEAL.json")}, "Receipt artifact hashes differ")
    require(result["schema"] == "p57.geometry-control-result.v1" and
            seal["schema"] == "p57.geometry-control-seal.v1", "Wrong result/seal schema")
    require(result["cell"] == seal["cell"] == read(episode/"CELL.json") == cell,
            "Cell identity differs")
    require(read(episode/"BINDING.json") == binding and
            seal["binding_sha256"] == digest(episode/"BINDING.json") and
            result["seal_sha256"] == digest(episode/"SEAL.json"),
            "Binding/result/seal chain differs")
    require(result["near_entry_qualified"] is True and result["micro_executed"] is True and
            result["automatic_retries"] == seal["automatic_retries"] == 0 and
            result["full_task_success_evidence"] is False, "Local skill scope differs")
    for relative, expected in seal["files"].items():
        path = (episode/relative).resolve()
        require(path.is_relative_to(episode) and path.is_file() and digest(path) == expected,
                "Unsealed payload: " + relative)
    for subtree in ("public", "private"):
        actual = {str(path.relative_to(episode)) for path in (episode/subtree).rglob("*") if path.is_file()}
        require(actual.issubset(seal["files"]), "Unlisted public/private output file")
    steps = result["executed_physical_steps"]
    require(type(steps) is int and 0 <= steps <= 24 and seal["executed_physical_steps"] == steps,
            "H24 physical step count differs")
    public = [json.loads(line) for line in (episode/"public/ACTIONS.jsonl").read_text().splitlines()]
    private = [json.loads(line) for line in (episode/"private/JUDGMENTS.jsonl").read_text().splitlines()]
    require(len(public) == len(private) == steps+1 and
            [row["t"] for row in public] == list(range(steps+1)) and
            [row["t"] for row in private] == list(range(steps+1)) and
            sum(row["executed"] is True for row in public) == steps and
            public[-1]["executed"] is False, "Executed action/terminal readout count differs")
    for t, row in enumerate(public):
        require(row["executed_physical_steps"] == t and row["proposed_action"][-1] == 1,
                "Action count or closed gripper differs")
        require(row["observation"] == f"frame_{t:04d}.npz" and
                digest(episode/"public"/row["observation"]) == row["observation_sha256"] and
                digest(episode/"public"/f"features_{t:04d}.pt") == row["features_sha256"],
                "Public observation/feature SHA differs")
        if arm != "strong_bc":
            require(np.allclose(row["proposed_action"], row["telemetry"]["action"],
                                atol=1e-8, rtol=0), "Telemetry action differs")
    require(result["system_success"] is result["micro_success"] is private[-1]["complete"] and
            result["official_success"] is private[-1]["official_success"] and
            result["source_cell_index"] == binding["entries"][entry]["source_cell_index"],
            "Strict/official result or source entry differs")
    return dict(index=index, entry=entry, arm=arm, condition_id=result["condition_id"],
        strict_success=result["system_success"],
        official_success_at_stop=result["official_success"],
        official_instant_ever=result["official_instant_ever"],
        official_hold3_ever=result["official_only_hold3"], actions=steps, reason=result["reason"],
        telemetry=telemetry_stats(public, arm),
        result_sha256=digest(episode/"RESULT.json"), seal_sha256=digest(episode/"SEAL.json"))


def aggregate(cells):
    indexed = {(row["entry"], row["arm"]): row for row in cells}
    require(len(indexed) == len(cells) == 20, "Incomplete 20-cell results")
    by_arm = {}
    for arm in ARMS:
        rows = [indexed[entry,arm] for entry in range(4)]
        values = [row["actions"] for row in rows]
        summary = dict(denominator=4,
            strict_success=sum(row["strict_success"] for row in rows),
            official_success_at_stop=sum(row["official_success_at_stop"] for row in rows),
            official_instant_ever=sum(row["official_instant_ever"] for row in rows),
            official_hold3_ever=sum(row["official_hold3_ever"] for row in rows),
            action_steps=values, mean_action_steps=_mean(values))
        if arm != "strong_bc":
            stats = [row["telemetry"] for row in rows]
            summary["same_observation_readout"] = dict(
                scope="per-trajectory mean of t>=1 readouts, including terminal unexecuted proposal; not physical counterfactual",
                readout_steps=sum(x["readout_steps_after_entry"] for x in stats),
                executed_readout_steps=sum(x["physically_executed_readout_steps_after_entry"] for x in stats),
                action_changed_steps=sum(x["action_changed_steps"] for x in stats),
                action_changed_executed_steps=sum(x["action_changed_executed_steps"] for x in stats),
                geometry_changed_steps=sum(x["geometry_changed_steps"] for x in stats),
                loss_decreased_steps=sum(x["loss_decreased_steps"] for x in stats),
                loss_increased_steps=sum(x["loss_increased_steps"] for x in stats),
                mean_action_delta_norm=_mean([x["same_observation_action_delta_norm"]["mean"]
                                              for x in stats if x["same_observation_action_delta_norm"]["mean"] is not None]),
                mean_geometry_delta_m=_mean([x["same_observation_geometry_delta_m"]["mean"]
                                             for x in stats if x["same_observation_geometry_delta_m"]["mean"] is not None]),
                mean_ray_loss_before=_mean([x["ray_loss_before"]["mean"] for x in stats
                                            if x["ray_loss_before"]["mean"] is not None]),
                mean_ray_loss_after=_mean([x["ray_loss_after"]["mean"] for x in stats
                                           if x["ray_loss_after"]["mean"] is not None]))
        by_arm[arm] = summary
    def pair(reference, candidate):
        rescued = [entry for entry in range(4) if not indexed[entry,reference]["strict_success"]
                   and indexed[entry,candidate]["strict_success"]]
        harmed = [entry for entry in range(4) if indexed[entry,reference]["strict_success"]
                  and not indexed[entry,candidate]["strict_success"]]
        return dict(reference=reference, candidate=candidate,
            rescue_entries=rescued, harm_entries=harmed, rescue=len(rescued), harm=len(harmed),
            same_result=4-len(rescued)-len(harmed))
    pairs = dict(fixed_to_online=pair("geometry_fixed","geometry_online"),
                 fixed_to_direct=pair("geometry_fixed","geometry_direct"),
                 online_to_direct=pair("geometry_online","geometry_direct"),
                 strong_bc_to_fixed=pair("strong_bc","geometry_fixed"))
    return by_arm, pairs


def audit(archive, extract_to, summary_path, binding_path, queue_manifest_path):
    queue_root, members = extract_verified(archive, extract_to)
    summary = read(summary_path)
    binding = read(binding_path)
    plan = read(queue_manifest_path)
    queue = read(queue_root/"QUEUE.json")
    result = read(queue_root/"QUEUE_RESULT.json")
    require(queue["manifest"] == plan and queue["manifest_sha256"] == digest(queue_manifest_path),
            "Queue manifest bytes differ from frozen local copy")
    require(plan["binding"]["sha256"] == digest(binding_path),
            "Frozen evaluation binding SHA differs")
    require(plan["schema"] == "p57.geometry-control-queue.v1" and
            result["schema"] == "p57.geometry-control-queue-result.v1" and
            result["planned_jobs"] == result["attempted_jobs"] ==
            result["completed_engineering_ok"] == 20 and
            result["all_batches_completed"] is True and result["automatic_retries"] == 0,
            "Queue result incomplete or changed")
    expected = [dict(index=i, entry=i//5, arm=ARMS[i%5]) for i in range(20)]
    require(binding["schema"] == "p57.geometry-control-eval.v1" and
            binding["cells"] == expected and binding["max_actions"] == 24 and
            binding["automatic_retries"] == 0, "Frozen B evaluation scope differs")
    require(plan["jobs"] == [dict(index=i, entry=i//5, arm=ARMS[i%5], gpu=i%4,
                                  timeout_seconds=1800) for i in range(20)] and
            plan["batches"] == [list(range(i,i+4)) for i in range(0,20,4)],
            "Queue jobs or GPU allocation differ")
    require(summary["binding_sha256"] == digest(binding_path) and
            summary["queue_result_sha256"] == digest(queue_root/"QUEUE_RESULT.json"),
            "Remote summary binding or queue result SHA differs")
    for batch in range(5):
        record = read(queue_root/f"BATCH_{batch:03d}.json")
        require(record["batch"] == batch and record["indices"] == list(range(batch*4,batch*4+4))
                and record["engineering_ok"] is True, "Batch receipt differs")
        for row in record["receipts"]:
            require(digest(queue_root/f"cell_{row['index']:03d}"/"RECEIPT.json") ==
                    row["receipt_sha256"], "Batch receipt hash differs")
    cells = [audit_cell(queue_root, index, binding, queue["manifest_sha256"], result)
             for index in range(20)]
    for entry in range(4):
        require(len({row["condition_id"] for row in cells if row["entry"] == entry}) == 1,
                "Paired condition identity differs")
    by_arm, pairs = aggregate(cells)
    compare(cells, summary["cells"], "summary.cells")
    compare(by_arm, summary["by_arm"], "summary.by_arm")
    compare(pairs, summary["paired_strict"], "summary.paired_strict")
    return dict(schema="p57.local-independent-audit.v1", archive_sha256=digest(archive),
                extracted_members=members, queue_result_sha256=digest(queue_root/"QUEUE_RESULT.json"),
                binding_sha256=digest(binding_path), summary_sha256=digest(summary_path),
                verified_cells=20, verified_sealed_files=sum(len(read(queue_root/f"cell_{i:03d}"/
                    "episode/SEAL.json")["files"]) for i in range(20)),
                by_arm=by_arm, paired_strict=pairs,
                remote_absolute_source_refs="verified by original remote collector; not revalidated locally",
                physics_reexecuted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-to", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--queue-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh audit output required")
    report = audit(args.archive, args.extract_to, args.summary,
                   args.binding, args.queue_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(dict(verified_cells=report["verified_cells"],
                          archive_sha256=report["archive_sha256"],
                          strict_success={name: value["strict_success"] for name,value in report["by_arm"].items()}),
                     sort_keys=True))


if __name__ == "__main__":
    main()
