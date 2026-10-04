"""Seal P36 branch predictions before any shared physical outcomes are read."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from prove.fast_adapter.geometry_role_dynamics import action_candidates, select_action
from tools.fast_adapter.p36_common import MODES, ROOT, TASKS, base_model, bindings, digest, dump, load_data, model
from tools.fast_adapter.p36_train import encode, plans, targets

TIMES = (20, 40, 60)


@torch.no_grad()
def predict():
    torch.set_num_threads(2)
    out = ROOT / "branches"
    out.mkdir(exist_ok=False)
    all_rows = []
    snapshots = []
    for task in TASKS:
        bank = load_data(task)
        base = base_model(task)
        ids = torch.where((bank["object_ids"] >= 48) & (bank["object_ids"] < 56))[0]
        for t in TIMES:
            ts = torch.full((len(ids),), t, device="cuda", dtype=torch.long)
            prev = bank["actions"][ids, ts - 1]
            base_actions = base(bank["visual"][ids, ts], bank["proprio"][ids, ts], prev)
            if task == "hammer-block":
                base_actions[:, 3] = 0.9
            candidates = action_candidates(base_actions, task)
            for j, idx in enumerate(ids.tolist()):
                snapshots.append(
                    {
                        "task": task,
                        "episode_index": idx,
                        "object_id": int(bank["object_ids"][idx]),
                        "role": "base" if int(bank["role"][idx]) else "expert",
                        "t": t,
                        "candidates": candidates[j].cpu(),
                    }
                )
        yy, weights = targets(bank)
        aa = plans(bank)
        all_times = torch.arange(197, device="cuda")[None].expand(len(ids), -1)
        flat_snapshot_times = torch.tensor([t for t in TIMES for _ in range(len(ids))], device="cuda")
        flat_ids = ids.repeat(len(TIMES))
        base_for_flat = torch.stack([snap["candidates"][0] for snap in snapshots if snap["task"] == task]).cuda()
        for seed in (11, 22):
            for mode in MODES:
                net = model(task, seed, mode)
                u = encode(net, bank, ids[:, None], all_times)
                mean, gate = net.coefficients(u, aa[ids])
                y = yy[ids] / net.effect_sd
                w = weights[ids]
                for adapt in ("fixed", "first3", "continuous"):
                    z_values = []
                    counts = []
                    for row_id, t in enumerate(flat_snapshot_times.tolist()):
                        src = row_id % len(ids)
                        z = net.z0[None].clone()
                        count = 0
                        for k in range(4, t + 1):
                            if adapt == "continuous" or (adapt == "first3" and count < 3):
                                lo = max(0, k - 11)
                                hi = k - 3
                                z, _ = net.update(z, mean[src : src + 1, lo:hi], gate[src : src + 1, lo:hi], y[src : src + 1, lo:hi], w[src : src + 1, lo:hi])
                                count += 1
                        z_values.append(z[0])
                        counts.append(count)
                    z = torch.stack(z_values)
                    current = encode(net, bank, flat_ids, flat_snapshot_times)
                    act, choice, scores = select_action(net, current, base_for_flat, z)
                    all_rows.append(
                        {
                            "task": task,
                            "seed": seed,
                            "geometry_mode": mode,
                            "adaptation": adapt,
                            "episode_indices": flat_ids.cpu(),
                            "times": flat_snapshot_times.cpu(),
                            "z": z.cpu(),
                            "updates": counts,
                            "scores": scores.cpu(),
                            "choice": choice.cpu(),
                            "chosen_action": act.cpu(),
                        }
                    )
                print("P36_BRANCH_PREDICTIONS", task, seed, mode, flush=True)
    torch.save({"snapshots": snapshots, "rows": all_rows}, out / "PREDICTIONS.pt")
    dump(
        out / "PREDICTION_SEAL.json",
        {
            "before_physical_branch_outcomes": True,
            "snapshots_per_task": 48,
            "total_snapshots": 96,
            "candidate_branches": 864,
            "fixed_times": list(TIMES),
            "past_only": True,
            "shared_physical_branch_intent": True,
            "do_not_read_p35_predictions_or_outcomes": True,
            "bindings": bindings(
                [out / "PREDICTIONS.pt", Path(__file__), ROOT / "validation/SELECTION_SEAL.json"]
                + [ROOT / t / "models" / f"seed{s}" / f"{m}.pt" for t in TASKS for s in (11, 22) for m in MODES]
            ),
        },
    )


def raw_source(task, role, obj):
    offset = (obj // 8) * 8
    if task == "hammer-block":
        path = Path("evaluations/fast_adapter_p30_online_v02/hammer-block/data") / f"meta_{role}_{offset:03d}.pt"
        meta = path.with_suffix(".json")
    else:
        path = Path("evaluations/fast_adapter_p29_object_transfer_v01/tasks") / task / "data/meta" / f"{role}_{offset:03d}.pt"
        meta = path.with_name(path.stem + "_EVALUATOR.json")
    raw = torch.load(path, weights_only=True)
    cfg = json.loads(meta.read_text())["configs"][obj % 8]
    return raw, cfg, obj % 8, path


def physical_value(env, task):
    if task == "hammer-block":
        return float(env.env.data.joint("target_slide").qpos[0])
    left, right, *_ = env.env.contacts()
    return max(0.0, float(env.env.data.body("actor").xpos[2]) - env.env.start_z) * float(left and right)


def run():
    from prove.fast_adapter.held_hammer_env import HeldHammerEnv
    from prove.fast_adapter.object_interaction_env import ObjectEnv, mujoco

    torch.set_num_threads(2)
    out = ROOT / "branches"
    seal = json.loads((out / "PREDICTION_SEAL.json").read_text())
    for item in seal["bindings"]:
        assert digest(item["path"]) == item["sha256"], item["path"]
    if (out / "OUTCOMES.pt").exists():
        raise FileExistsError("Physical branch results already exist")
    predictions = torch.load(out / "PREDICTIONS.pt", weights_only=True)
    outcomes = []
    started = time.time()
    for snap in predictions["snapshots"]:
        task = snap["task"]
        raw, cfg, source_index, path = raw_source(task, snap["role"], snap["object_id"])
        env = (HeldHammerEnv if task == "hammer-block" else ObjectEnv)(cfg)
        traces = []
        progress = []
        starts = []
        frames = []
        try:
            for action in snap["candidates"].numpy():
                env.reset(raw["reset_seeds"][source_index])
                for key, value in raw["states"].items():
                    getattr(env.env.data, key)[:] = value[source_index, snap["t"]].numpy()
                mujoco.mj_forward(env.env.model, env.env.data)
                start = physical_value(env, task)
                states = [env.state()]
                rgb = [env.rgb()]
                proprio = []
                values = [start]
                for _ in range(4):
                    proprio.append(env.step(action))
                    states.append(env.state())
                    rgb.append(env.rgb())
                    values.append(physical_value(env, task))
                progress.append(values[-1] - start)
                starts.append(start)
                frames.append(np.stack(rgb))
                traces.append(
                    {
                        "states": {k: torch.tensor(np.stack([s[k] for s in states])) for k in states[0]},
                        "proprio": torch.tensor(np.stack(proprio)),
                        "values": values,
                    }
                )
        finally:
            env.close()
        outcomes.append(
            {
                **{k: v for k, v in snap.items() if k != "candidates"},
                "progress": progress,
                "start_values": starts,
                "traces": traces,
                "rgb": torch.tensor(np.stack(frames)),
                "source": str(path),
                "config": cfg,
                "reset_seed": int(raw["reset_seeds"][source_index]),
            }
        )
        assert np.ptp(starts) < 1e-8
        print("P36_BRANCHES", task, snap["role"], snap["object_id"], snap["t"], "range_m", round(float(np.ptp(progress)), 6), flush=True)
    torch.save({"outcomes": outcomes}, out / "OUTCOMES.pt")
    summarize(seconds=time.time() - started)


def summarize(seconds=None):
    out = ROOT / "branches"
    predictions = torch.load(out / "PREDICTIONS.pt", weights_only=True)
    outcomes = torch.load(out / "OUTCOMES.pt", weights_only=True)["outcomes"]
    rows = []
    gates = {}
    for task in TASKS:
        task_outcomes = [x for x in outcomes if x["task"] == task]
        progress = np.array([x["progress"] for x in task_outcomes])
        informative = np.ptp(progress, axis=1) > 0.002
        best = progress.max(-1)
        base = progress[:, 0]
        for row in predictions["rows"]:
            if row["task"] != task:
                continue
            chosen = row["choice"].numpy()
            actual = progress[np.arange(len(chosen)), chosen]
            regret = best - actual
            rows.append(
                {k: row[k] for k in ("task", "seed", "geometry_mode", "adaptation")}
                | {
                    "mean_progress_m": float(actual.mean()),
                    "mean_regret_m": float(regret.mean()),
                    "per_snapshot_progress_m": actual.tolist(),
                    "per_snapshot_regret_m": regret.tolist(),
                    "beats_base": int((actual > base + 1e-8).sum()),
                    "harms_base": int((actual < base - 1e-8).sum()),
                    "choices": chosen.tolist(),
                }
            )
        checks = []
        for seed in (11, 22):
            explicit = next(r for r in rows if r["task"] == task and r["seed"] == seed and r["geometry_mode"] == "p36_explicit_relation_geometry" and r["adaptation"] == "continuous")
            existing = next(r for r in rows if r["task"] == task and r["seed"] == seed and r["geometry_mode"] == "p34_existing_bbox_geometry" and r["adaptation"] == "continuous")
            zero = next(r for r in rows if r["task"] == task and r["seed"] == seed and r["geometry_mode"] == "no_explicit_bbox_geometry" and r["adaptation"] == "continuous")
            checks.append(
                {
                    "seed": seed,
                    "explicit_beats_existing_regret": explicit["mean_regret_m"] < existing["mean_regret_m"],
                    "explicit_beats_zero_regret": explicit["mean_regret_m"] < zero["mean_regret_m"],
                    "explicit_at_least_base_progress": explicit["mean_progress_m"] >= float(base.mean()),
                }
            )
        gates[task] = {
            "informative_snapshots": int(informative.sum()),
            "snapshots": len(task_outcomes),
            "base_mean_progress_m": float(base.mean()),
            "base_mean_regret_m": float((best - base).mean()),
            "checks": checks,
            "pass": bool(informative.sum() >= 5 and all(all(v for k, v in c.items() if k != "seed") for c in checks)),
        }
    dump(
        out / "RESULTS.json",
        {
            "rows": rows,
            "gates": gates,
            "seconds": seconds,
            "not_full_task_success": True,
            "bindings": bindings([out / "PREDICTIONS.pt", out / "OUTCOMES.pt", out / "PREDICTION_SEAL.json"]),
        },
    )
    print("P36_BRANCH_GATE", gates, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["predict", "run", "summarize"])
    args = parser.parse_args()
    if args.mode == "predict":
        predict()
    elif args.mode == "run":
        run()
    else:
        summarize()
