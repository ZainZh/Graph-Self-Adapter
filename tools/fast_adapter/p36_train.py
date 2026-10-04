"""P36 training: same capacity, only the public RGB-geometry slot changes."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from prove.fast_adapter.geometry_role_dynamics import GeometryRoleDynamics, observed_effect
from tools.fast_adapter.p36_common import H, MODES, ROOT, TASKS, bank_path, bindings, dump, load_data, source_paths, stats_path


def encode(net, bank, ids, times):
    prev = (times - 1).clamp_min(0)
    previous = bank["actions"][ids, prev] * (times > 0)[..., None]
    velocity = (bank["boxes"][ids, times] - bank["boxes"][ids, prev]) * (times > 0)[..., None, None]
    return net.encode(bank["tokens"][ids, times], bank["boxes"][ids, times], bank["proprio"][ids, times], previous, velocity)


def targets(bank):
    return observed_effect(
        bank["tokens"][:, :-H],
        bank["boxes"][:, :-H],
        bank["proprio"][:, :-H],
        bank["tokens"][:, H:],
        bank["boxes"][:, H:],
        bank["proprio"][:, H:],
    )


def plans(bank):
    return bank["actions"].unfold(1, H, 1).transpose(-1, -2).contiguous()



def ensure_stats(task):
    path = stats_path(task)
    if path.exists():
        return
    bank = load_data(task, "cpu")
    train_mask = bank["object_ids"] < 24
    tokens = bank["tokens"][train_mask]
    mask = tokens[..., -1:]
    mean = (tokens[..., :50] * mask).sum((0, 1, 2, 3)) / mask.sum().clamp_min(1)
    sd = (((tokens[..., :50] - mean).square() * mask).sum((0, 1, 2, 3)) / mask.sum().clamp_min(1)).sqrt().clamp_min(0.02)
    y, w = targets(bank)
    y = y[train_mask]
    w = w[train_mask]
    available = (w > 0).float()
    effect_sd = ((y.square() * available).sum((0, 1)) / available.sum((0, 1)).clamp_min(1)).sqrt().clamp_min(0.001)
    motion = y[..., 128:130]
    ok = available[..., 128:130]
    direction = (motion * ok).sum((0, 1))
    direction = direction / direction.norm().clamp_min(1e-8)
    progress_sd = ((motion @ direction).square() * ok[..., 0]).sum() / ok[..., 0].sum().clamp_min(1)
    stats = {
        "token_mean": mean,
        "token_sd": sd,
        "effect_sd": effect_sd,
        "direction": direction,
        "progress_sd": progress_sd.sqrt().clamp_min(0.0001),
        "fit_ids": torch.arange(24),
        "validation_ids": torch.arange(48, 56),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, path)
    dump(
        path.with_suffix(".json"),
        {
            "fit_object_ids": list(range(24)),
            "validation_object_ids": list(range(48, 56)),
            "direction_from_fit_rgb_object_box_motion": direction.tolist(),
            "progress_sd": float(stats["progress_sd"]),
            "bindings": bindings([bank_path(task), path]),
        },
    )

def train(task, seed):
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled = False
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())["learning"]
    out = ROOT / task / "models" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=False)
    ensure_stats(task)
    bank = load_data(task)
    stats = torch.load(stats_path(task), weights_only=True)
    yy, ww = targets(bank)
    aa = plans(bank)
    yy = yy / stats["effect_sd"].cuda()
    candidates = torch.where(bank["object_ids"] < 24)[0]
    nets = {}
    opts = {}
    for mode in MODES:
        torch.manual_seed(seed)
        net = GeometryRoleDynamics(mode, task).cuda()
        net.set_stats(stats)
        nets[mode] = net
        opts[mode] = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
    counts = {mode: sum(p.numel() for p in net.parameters()) for mode, net in nets.items()}
    assert len(set(counts.values())) == 1, counts
    base_state = nets[MODES[0]].state_dict()
    for mode in MODES[1:]:
        nets[mode].load_state_dict(base_state, strict=True)
    dump(
        out / "SOURCE_FREEZE.json",
        {
            "before_training": True,
            "parameter_counts": counts,
            "train_object_ids": list(range(24)),
            "validation_object_ids": list(range(48, 56)),
            "geometry_axis_only": True,
            "runtime_forbidden": ["object_pose", "contact", "success", "simulator_truth", "future_frames"],
            "bindings": bindings(source_paths(task) + [Path(__file__)]),
        },
    )
    gen = torch.Generator(device="cuda").manual_seed(seed + 3600)
    started = time.time()
    log = (out / "TRAIN.jsonl").open("x")

    def report(row):
        row.update(seconds=time.time() - started)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print("P36_TRAIN", task, seed, row, flush=True)

    for update in range(cfg["warmup_updates"]):
        ids = candidates[torch.randint(len(candidates), (128,), device="cuda", generator=gen)]
        ts = torch.randint(201 - H, (128,), device="cuda", generator=gen)
        for mode, net in nets.items():
            u = encode(net, bank, ids, ts)
            mean, gate = net.coefficients(u, aa[ids, ts])
            pred = net.predict_coeff(mean, gate, net.z0)
            loss = net.loss(pred, yy[ids, ts], ww[ids, ts]).mean()
            opts[mode].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            opts[mode].step()
        if update == 0 or (update + 1) % 200 == 0:
            report({"phase": "warmup", "updates": update + 1, "last_loss": float(loss.detach())})

    for iteration in range(cfg["meta_trajectory_batches"]):
        ids = candidates[torch.randint(len(candidates), (cfg["batch_episodes"],), device="cuda", generator=gen)]
        z_states = {mode: net.z0[None].expand(len(ids), -1) for mode, net in nets.items()}
        losses = {mode: [] for mode in MODES}
        for start in range(0, 200, cfg["segment_steps"]):
            lo = max(0, start - H - 7)
            hi = min(start + cfg["segment_steps"], 201 - H)
            times = torch.arange(lo, hi, device="cuda")[None].expand(len(ids), -1)
            for mode, net in nets.items():
                u = encode(net, bank, ids[:, None], times)
                mean, gate = net.coefficients(u, aa[ids[:, None], times])
                z = z_states[mode]
                loss = 0.0
                number = 0
                for t in range(start, start + cfg["segment_steps"]):
                    if t >= H:
                        a = max(0, t - H - 7)
                        e = t - H + 1
                        sl = slice(a - lo, e - lo)
                        z, _ = net.update(z, mean[:, sl], gate[:, sl], yy[ids, a:e], ww[ids, a:e])
                    if t < 201 - H:
                        pred = net.predict_coeff(mean[:, t - lo], gate[:, t - lo], z)
                        loss = loss + net.loss(pred, yy[ids, t], ww[ids, t]).mean()
                        number += 1
                loss = loss / number
                opts[mode].zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("nonfinite meta gradient")
                opts[mode].step()
                z_states[mode] = z.detach().requires_grad_(True)
                losses[mode].append(float(loss.detach()))
        if iteration == 0 or (iteration + 1) % 10 == 0:
            report(
                {
                    "phase": "meta",
                    "batches": iteration + 1,
                    "loss": {mode: float(np.mean(values)) for mode, values in losses.items()},
                    "alpha": {mode: float(net.alpha.detach()) for mode, net in nets.items()},
                    "z_delta_rms": {
                        mode: float((z - nets[mode].z0).detach().square().mean().sqrt())
                        for mode, z in z_states.items()
                    },
                }
            )
    for mode, net in nets.items():
        torch.save(
            {
                "state": net.cpu().state_dict(),
                "mode": mode,
                "task": task,
                "seed": seed,
                "warmup": cfg["warmup_updates"],
                "meta_outer_updates": cfg["outer_updates"],
            },
            out / f"{mode}.pt",
        )
    log.close()
    dump(
        out / "COMPLETE.json",
        {
            "seconds": time.time() - started,
            "parameter_counts": counts,
            "fit_ids": list(range(24)),
            "no_expert_action_targets": True,
            "bindings": bindings([out / f"{mode}.pt" for mode in MODES]),
        },
    )
    print("P36_TRAIN_COMPLETE", task, seed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    args = parser.parse_args()
    train(args.task, args.seed)
