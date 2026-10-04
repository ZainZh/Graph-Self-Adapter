"""P36 source, geometry-mask and causal prediction audit."""
import json
from pathlib import Path

import torch

from prove.fast_adapter.geometry_role_dynamics import GEOMETRY_MODES, GeometryRoleDynamics, geometry_features
from tools.fast_adapter.p36_common import MODES, ROOT, TASKS, bindings, digest, dump, load_data, model, state_hash
from tools.fast_adapter.p36_runtime import GeometryRoleState


def audit_geometry_masks():
    boxes = torch.tensor([[[0.2, 0.3, 0.1, 0.2], [0.7, 0.8, 0.2, 0.1]]])
    velocity = torch.ones_like(boxes) * 0.01
    none_visible = torch.tensor([[False, False]])
    actor_only = torch.tensor([[True, False]])
    checks = {}
    for mode in GEOMETRY_MODES:
        hidden = geometry_features(boxes, velocity, none_visible, mode)
        actor = geometry_features(boxes, velocity, actor_only, mode)
        checks[mode] = {
            "all_hidden_zero": bool(hidden.abs().max() == 0),
            "pair_relation_zero_when_object_hidden": bool(actor[..., 16:20].abs().max() == 0 and actor[..., 22:].abs().max() == 0),
            "visibility_flags_may_remain": bool(actor[..., 20:22].abs().max() >= 0),
        }
        assert checks[mode]["all_hidden_zero"]
        assert checks[mode]["pair_relation_zero_when_object_hidden"]
    counts = {mode: sum(param.numel() for param in GeometryRoleDynamics(mode).parameters()) for mode in GEOMETRY_MODES}
    assert len(set(counts.values())) == 1, counts
    return {"mask_checks": checks, "parameter_counts": counts}


def audit_predictions():
    pred_path = ROOT / "branches/PREDICTIONS.pt"
    if not pred_path.exists():
        return {"prediction_seal_present": False}
    seal = json.loads((ROOT / "branches/PREDICTION_SEAL.json").read_text())
    source_hash_mismatches = []
    non_branch_mismatches = []
    for item in seal["bindings"]:
        current = digest(item["path"])
        if current != item["sha256"]:
            row = {"path": item["path"], "expected": item["sha256"], "current": current}
            source_hash_mismatches.append(row)
            if not item["path"].endswith("tools/fast_adapter/p36_branches.py"):
                non_branch_mismatches.append(row)
    assert not non_branch_mismatches, non_branch_mismatches
    predictions = torch.load(pred_path, weights_only=True)
    checks = []
    for task in TASKS:
        bank = load_data(task)
        snaps = [snap for snap in predictions["snapshots"] if snap["task"] == task]
        base = torch.stack([snap["candidates"][0] for snap in snaps]).cuda()
        for seed in (11, 22):
            for mode in MODES:
                row = next(
                    row
                    for row in predictions["rows"]
                    if row["task"] == task and row["seed"] == seed and row["geometry_mode"] == mode and row["adaptation"] == "continuous"
                )
                net = model(task, seed, mode)
                before = state_hash(net)
                ids = row["episode_indices"].cuda()
                times = row["times"].cuda()
                runtime = GeometryRoleState(net, len(ids), "continuous")
                zerr = 0.0
                scoreerr = 0.0
                choices_ok = True
                for t in range(int(times.max()) + 1):
                    ob = {key: bank[key][ids, t] for key in ("tokens", "boxes", "proprio")}
                    _, choice, scores = runtime.act(ob, base)
                    matches = times == t
                    if bool(matches.any()):
                        mask = matches.cpu()
                        zerr = max(zerr, float((runtime.z[matches].cpu() - row["z"][mask]).abs().max()))
                        scoreerr = max(scoreerr, float((scores[matches].cpu() - row["scores"][mask]).abs().max()))
                        choices_ok &= torch.equal(choice[matches].cpu(), row["choice"][mask])
                    runtime.observe(bank["actions"][ids, t], {key: bank[key][ids, t + 1] for key in ("tokens", "boxes", "proprio")})
                assert zerr < 2e-5 and scoreerr < 2e-5 and choices_ok, (task, seed, mode, zerr, scoreerr)
                assert state_hash(net) == before
                checks.append(
                    {
                        "task": task,
                        "seed": seed,
                        "geometry_mode": mode,
                        "max_z_error": zerr,
                        "max_score_error": scoreerr,
                        "choices_match": choices_ok,
                        "weights_frozen": True,
                    }
                )
                print("P36_CAUSAL_AUDIT", task, seed, mode, zerr, scoreerr, flush=True)
    return {"prediction_seal_present": True, "causal_replay": checks, "source_hash_mismatches": source_hash_mismatches, "non_branch_source_hash_mismatches": non_branch_mismatches}


def audit_physical_if_present():
    out = ROOT / "branches"
    outcomes_path = out / "OUTCOMES.pt"
    if not outcomes_path.exists() and (out / "RESULTS.json").exists():
        results = json.loads((out / "RESULTS.json").read_text())
        if results.get("shared_outcomes_path"):
            outcomes_path = Path(results["shared_outcomes_path"])
    if not outcomes_path.exists():
        return {"physical_outcomes_present": False}
    from prove.fast_adapter.held_hammer_env import HeldHammerEnv
    from prove.fast_adapter.object_interaction_env import ObjectEnv, mujoco
    from tools.fast_adapter.p36_branches import physical_value

    outcomes = torch.load(outcomes_path, weights_only=True)["outcomes"]
    maxerr = 0.0
    count = 0
    for item in outcomes:
        cls = HeldHammerEnv if item["task"] == "hammer-block" else ObjectEnv
        env = cls(item["config"], render=False)
        try:
            env.reset(item["reset_seed"])
            for j, trace in enumerate(item["traces"]):
                values = []
                for t in (0, 4):
                    for key, value in trace["states"].items():
                        getattr(env.env.data, key)[:] = value[t].numpy()
                    mujoco.mj_forward(env.env.model, env.env.data)
                    values.append(physical_value(env, item["task"]))
                maxerr = max(maxerr, abs((values[1] - values[0]) - item["progress"][j]))
                count += 1
        finally:
            env.close()
    assert maxerr < 1e-10
    return {"physical_outcomes_present": True, "physical_branches_regraded": count, "physical_max_error": maxerr, "outcomes_path": str(outcomes_path)}


def main():
    torch.set_num_threads(2)
    source_checks = []
    for task in TASKS:
        bank = load_data(task)
        assert not any(key in bank for key in ("teacher", "metrics", "success", "states"))
        fit = set(bank["object_ids"][bank["object_ids"] < 24].tolist())
        val = set(bank["object_ids"][(bank["object_ids"] >= 48) & (bank["object_ids"] < 56)].tolist())
        assert fit == set(range(24)) and val == set(range(48, 56)) and not fit & val
        for path in ROOT.glob(f"{task}/models/seed*/SOURCE_FREEZE.json"):
            for item in json.loads(path.read_text())["bindings"]:
                assert digest(item["path"]) == item["sha256"], item["path"]
        source_checks.append({"task": task, "fit_ids": sorted(fit), "validation_ids": sorted(val), "no_privileged_bank_fields": True})
    geometry = audit_geometry_masks()
    predictions = audit_predictions()
    physical = audit_physical_if_present()
    prediction_audit_passed = bool(predictions.get("prediction_seal_present") and not predictions.get("non_branch_source_hash_mismatches"))
    functional_audit_passed = bool(prediction_audit_passed and physical.get("physical_outcomes_present"))
    overall_passed = bool(functional_audit_passed and not predictions.get("source_hash_mismatches"))
    payload = {
        "passed": overall_passed,
        "prediction_audit_passed": prediction_audit_passed,
        "functional_audit_passed": functional_audit_passed,
        "geometry_mask_and_capacity": geometry,
        "source_and_split": source_checks,
        "predictions": predictions,
        "physical": physical,
        "no_full_rollouts": True,
        "not_task_success": True,
        "bindings": bindings(
            [
                ROOT / "validation/PREDICTION_RESULTS.json",
                ROOT / "branches/PREDICTION_SEAL.json",
                Path(__file__),
                Path("tools/fast_adapter/p36_runtime.py"),
                Path("prove/fast_adapter/geometry_role_dynamics.py"),
            ]
        ),
    }
    dump(ROOT / "AUDIT.json", payload)
    print("P36_AUDIT_COMPLETE", payload["passed"], flush=True)


if __name__ == "__main__":
    main()
