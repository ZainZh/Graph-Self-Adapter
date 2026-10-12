#!/usr/bin/env python3
"""Offline torch trainer that exports an exact frozen NumPy local skill."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


SCHEMA = "prove.twin-local-skill.numpy-mlp.v1"


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        if not {"features", "labels", "metadata_json"} <= set(data.files):
            raise ValueError("dataset requires features, labels, metadata_json")
        x = np.asarray(data["features"], dtype=np.float32)
        y = np.asarray(data["labels"], dtype=np.float32)
        metadata = json.loads(str(data["metadata_json"].item()))
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 32:
        raise ValueError(f"invalid dataset dimensions: {x.shape}, {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("dataset contains non-finite values")
    if metadata.get("teacher") != "actual_twin_jacobian_damped_least_squares":
        raise ValueError("training requires actual twin Jacobian teacher labels")
    return x, y, metadata


def train(dataset: Path, output: Path, *, steps: int, batch_size: int,
          seed: int, learning_rate: float, action_limit: float) -> dict:
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise RuntimeError("run with the declared torch training interpreter") from exc
    x, y, source = load_dataset(dataset)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    split = max(1, int(0.9 * len(x)))
    train_i, validation_i = order[:split], order[split:]
    if not len(validation_i):
        validation_i = train_i[-1:]
    input_mean = x[train_i].mean(0)
    input_scale = np.maximum(x[train_i].std(0), 1e-5)
    output_mean = y[train_i].mean(0)
    output_scale = np.maximum(y[train_i].std(0), 1e-5)
    xn = (x - input_mean) / input_scale
    yn = (y - output_mean) / output_scale
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(),
                          nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, y.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    tx, ty = torch.from_numpy(xn).to(device), torch.from_numpy(yn).to(device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses = []
    started = time.time()
    model.train()
    for step in range(steps):
        pick = train_i[torch.randint(len(train_i), (min(batch_size, len(train_i)),), generator=gen).numpy()]
        prediction = model(tx[pick])
        loss = torch.nn.functional.smooth_l1_loss(prediction, ty[pick], beta=0.25)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if step % max(1, steps // 20) == 0 or step + 1 == steps:
            losses.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        train_pred = model(tx[train_i]).cpu().numpy() * output_scale + output_mean
        val_pred = model(tx[validation_i]).cpu().numpy() * output_scale + output_mean
    train_rmse = float(np.sqrt(np.mean(np.square(train_pred - y[train_i]))))
    val_rmse = float(np.sqrt(np.mean(np.square(val_pred - y[validation_i]))))
    linears = [m for m in model if isinstance(m, nn.Linear)]
    metadata = {
        "schema": SCHEMA,
        "model": "mlp_relu_64x128",
        "input_dim": int(x.shape[1]), "output_dim": int(y.shape[1]),
        "z_feature_slice": source["z_feature_slice"],
        "correction_feature_slices": source["correction_feature_slices"],
        "action_limit_rad": float(action_limit),
        "seed": seed, "steps": steps, "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_examples": int(len(train_i)), "validation_examples": int(len(validation_i)),
        "train_rmse_rad": train_rmse, "validation_rmse_rad": val_rmse,
        "loss_history": losses,
        "teacher": source["teacher"],
        "dataset_metadata": source,
        "torch_version": torch.__version__, "device": str(device),
        "elapsed_sec": time.time() - started,
    }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "input_mean": input_mean, "input_scale": input_scale,
        "output_mean": output_mean, "output_scale": output_scale,
    }
    for i, layer in enumerate(linears):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy()
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=51005)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--action-limit", type=float, default=0.04)
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.batch_size <= 0 or args.action_limit <= 0:
        parser.error("steps, batch-size and action-limit must be positive")
    report = train(args.dataset, args.output, steps=args.steps, batch_size=args.batch_size,
                   seed=args.seed, learning_rate=args.learning_rate, action_limit=args.action_limit)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
