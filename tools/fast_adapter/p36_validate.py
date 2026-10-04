"""Offline prediction validation on object ids 48..55."""
import numpy as np
import torch

from tools.fast_adapter.p36_common import MODES, ROOT, TASKS, bindings, dump, load_data, model
from tools.fast_adapter.p36_train import encode, plans, targets


@torch.no_grad()
def score(bank, net):
    count = len(bank["actions"])
    ts = torch.arange(197, device="cuda")[None].expand(count, -1)
    ids = torch.arange(count, device="cuda")[:, None]
    u = encode(net, bank, ids, ts)
    aa = plans(bank)
    yy, weights = targets(bank)
    yy = yy / net.effect_sd
    mean, gate = net.coefficients(u, aa)
    out = {}
    for mode in ("fixed", "first3", "continuous"):
        z = net.z0[None].expand(count, -1)
        errs = []
        updates = 0
        for t in range(197):
            if t >= 4 and (mode == "continuous" or (mode == "first3" and updates < 3)):
                lo = max(0, t - 11)
                hi = t - 3
                z, _ = net.update(z, mean[:, lo:hi], gate[:, lo:hi], yy[:, lo:hi], weights[:, lo:hi])
                updates += 1
            pred = net.predict_coeff(mean[:, t], gate[:, t], z)
            errs.append((pred - yy[:, t]).square())
        err = torch.stack(errs, 1)
        values = {}
        for key, sl in (
            ("all", slice(None)),
            ("features", slice(0, 128)),
            ("boxes", slice(128, 132)),
            ("relation", slice(132, 134)),
        ):
            wt = weights[..., sl]
            valid = wt.sum(-1) > 0
            per = (err[..., sl] * wt).sum(-1) / wt.sum(-1).clamp_min(1e-8)
            per_episode = (per * valid).sum(-1) / valid.sum(-1).clamp_min(1)
            values[key] = {"mean": float(per_episode.mean()), "per_episode": per_episode.tolist()}
        values["updates_before_last_query"] = updates
        out[mode] = values
    return out


def main():
    torch.set_num_threads(2)
    out = ROOT / "validation"
    out.mkdir(exist_ok=False)
    rows = []
    selected = {}
    for task in TASKS:
        bank = load_data(task)
        keep = (bank["object_ids"] >= 48) & (bank["object_ids"] < 56)
        bank = {k: v[keep] for k, v in bank.items() if torch.is_tensor(v)}
        for seed in (11, 22):
            for mode in MODES:
                net = model(task, seed, mode)
                metrics = score(bank, net)
                rows.append({"task": task, "seed": seed, "geometry_mode": mode, "metrics": metrics})
                print("P36_VALIDATE", task, seed, mode, {m: round(metrics[m]["all"]["mean"], 5) for m in metrics}, flush=True)
        means = {
            mode: np.mean(
                [r["metrics"]["continuous"]["all"]["mean"] for r in rows if r["task"] == task and r["geometry_mode"] == mode]
            )
            for mode in MODES
        }
        selected[task] = min(MODES, key=lambda mode: means[mode])
        print("P36_SELECTED", task, selected[task], means, flush=True)
    dump(
        out / "PREDICTION_RESULTS.json",
        {
            "rows": rows,
            "selection": selected,
            "fit_ids": list(range(24)),
            "validation_ids": list(range(48, 56)),
            "geometry_axis_only": True,
            "readout": "training-normalized future observed-effect MSE, not task success",
        },
    )
    dump(
        out / "SELECTION_SEAL.json",
        {
            "selected": selected,
            "rule": "lowest continuous mean validation MSE across both seeds; all modes retained for branch prediction",
            "not_a_physical_success_claim": True,
            "bindings": bindings([out / "PREDICTION_RESULTS.json"]),
        },
    )


if __name__ == "__main__":
    main()
