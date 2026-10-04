#!/usr/bin/env python3
"""P57 A1b: isolate mask/aux normalization-floor change; offline only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from prove.task_graph.p57_measurement_head import GeometryMeasurementHead, MeasurementConfig, INPUT_KEYS, MODES
from tools.task_graph import p57_train_measurement as base


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    Path(__file__).resolve(),
    ROOT / "tools/task_graph/p57_train_measurement.py",
    ROOT / "prove/task_graph/p57_measurement_head.py",
    ROOT / "prove/task_graph/p57_public_geometry.py",
)
SEEDS = (11, 22)
TRAINING = dict(updates=1500, batch=64, optimizer="Adam", learning_rate=3e-4,
                sampling="uniform train trajectory then uniform actual frame",
                normalization="train trajectories only; raw std floor 0.001; mask and aux std floor 0.01",
                selection="final checkpoint only")


def fit_normalization(model, episodes, train_indices):
    """A1 computation/order, with the only intervention on mask and aux floors."""
    with torch.no_grad():
        rows = [base.public_input(episodes[i]) for i in train_indices]
        raw = torch.cat([r["raw"] for r in rows], 0)
        mask = torch.cat([r["mask_tokens"] for r in rows], 0)
        aux = torch.cat([model._auxiliary(r) for r in rows], 0)
        for name, x in (("raw", raw), ("mask", mask), ("aux", aux)):
            mean = x.mean(0)
            scale = x.std(0, unbiased=False).clamp_min(1e-3 if name == "raw" else 1e-2)
            getattr(model, name + "_mean").copy_(mean)
            getattr(model, name + "_scale").copy_(scale)


def build_freeze(manifest_path, plan_path, output):
    manifest_path, plan_path, output = (Path(p).resolve() for p in (manifest_path, plan_path, output))
    if output.exists():
        raise FileExistsError(output)
    if plan_path != ROOT / "evaluations/fast_adapter_p57_geometry_tokens_v01/PLAN_A1B.md":
        raise ValueError("Require exact A1b plan")
    a1_path = output.parent.parent / "TRAINING_FREEZE_v01.json"
    a1 = json.loads(a1_path.read_text())
    if a1["schema"] != "p57.geometry-training-freeze.v1" or a1["physics_steps"] != 0:
        raise ValueError("Invalid A1 parent freeze")
    if base.sha256(manifest_path) != a1["pack_manifest"]["sha256"]:
        raise ValueError("A1b pack differs from A1")
    for path, digest in a1["sources"].items():
        if base.sha256(path) != digest:
            raise ValueError(f"A1 frozen source changed: {path}")
    arms = []
    for arm in a1["arms"]:
        expected = asdict(MeasurementConfig(mode=arm["mode"]))
        count = GeometryMeasurementHead(MeasurementConfig(mode=arm["mode"])).parameter_count()
        if arm["config"] != expected or arm["parameter_count"] != count:
            raise ValueError("A1 model contract differs")
        arms.append(dict(mode=arm["mode"], seed=arm["seed"], config=expected,
                         parameter_count=count, output_relative=arm["output_relative"]))
    freeze = dict(schema="p57.a1b-normalization-freeze.v1", scope="offline A1 single-factor diagnosis",
                  parent_a1_freeze=dict(path=str(a1_path), sha256=base.sha256(a1_path)),
                  pack_manifest=a1["pack_manifest"], splits=a1["splits"],
                  plan=dict(path=str(plan_path), sha256=base.sha256(plan_path)),
                  sources={str(path): base.sha256(path) for path in SOURCES},
                  intervention=dict(raw_std_floor=0.001, mask_std_floor=0.01, aux_std_floor=0.01),
                  training=TRAINING, arms=arms, physics_steps=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    return freeze


def verify(freeze_path, manifest_path, mode, seed, output):
    freeze_path, manifest_path, output = (Path(p).resolve() for p in (freeze_path, manifest_path, output))
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema") != "p57.a1b-normalization-freeze.v1" or freeze.get("physics_steps") != 0:
        raise ValueError("Invalid A1b freeze")
    parent = freeze["parent_a1_freeze"]
    if base.sha256(parent["path"]) != parent["sha256"]:
        raise ValueError("A1 parent freeze changed")
    if base.sha256(manifest_path) != freeze["pack_manifest"]["sha256"]:
        raise ValueError("A1b pack changed")
    if base.sha256(freeze["plan"]["path"]) != freeze["plan"]["sha256"]:
        raise ValueError("A1b plan changed")
    for path, digest in freeze["sources"].items():
        if base.sha256(path) != digest:
            raise ValueError(f"A1b source changed: {path}")
    if freeze["training"] != TRAINING or freeze["intervention"] != dict(
            raw_std_floor=0.001, mask_std_floor=0.01, aux_std_floor=0.01):
        raise ValueError("A1b training contract changed")
    arm = next((a for a in freeze["arms"] if (a["mode"], a["seed"]) == (mode, seed)), None)
    if arm is None or output != freeze_path.parent / arm["output_relative"]:
        raise ValueError("A1b output outside frozen six-arm queue")
    return freeze


def train(manifest_path, output, mode, seed, *, freeze_path, device="cpu"):
    freeze = verify(freeze_path, manifest_path, mode, seed, output)
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    prediction_path, report_path = output.with_name("PREDICTIONS.pt"), output.with_name("REPORT.json")
    for path in (output, prediction_path, report_path):
        if path.exists():
            raise FileExistsError(path)
    bundle = base.load_bundle(manifest_path)
    train_indices, dev_indices = base.validate_bundle(bundle)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    config = MeasurementConfig(mode=mode)
    model = GeometryMeasurementHead(config)
    fit_normalization(model, bundle["episodes"], train_indices)
    bbox_prior = base.fit_bbox_relative_mean(bundle["episodes"], train_indices, config.image_size)
    pose_prior = base.fit_pose_prior(bundle["episodes"], train_indices)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    episodes, loss_trace = bundle["episodes"], []
    for step in range(1500):
        chosen = [train_indices[int(torch.randint(len(train_indices), (), generator=rng))] for _ in range(64)]
        pairs = [(i, int(torch.randint(len(episodes[i]["input"]["features"]["proprio"]), (), generator=rng)))
                 for i in chosen]
        samples = [base.public_input(episodes[i], t) for i, t in pairs]
        obs = {key: torch.stack([sample[key] for sample in samples]).to(device) for key in INPUT_KEYS}
        labels = [base.targets(episodes[i], t) for i, t in pairs]
        uv = torch.stack([item[0] for item in labels]).to(device)
        valid = torch.stack([item[1] for item in labels]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = base.supervised_loss(model, obs, uv, valid)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            entry = dict(update=step + 1, loss=float(loss.detach()))
            loss_trace.append(entry)
            print(json.dumps(entry), flush=True)
    model.to("cpu")
    report = base.evaluate(model, episodes, dev_indices, bbox_prior, pose_prior)
    public_readout = base.public_predictions(model, episodes, dev_indices)
    sources = freeze["sources"]
    payload = dict(schema="p57.geometry-measurement-checkpoint.v1", mode=mode, seed=seed,
                   config=asdict(config), state_dict=model.state_dict(), parameter_count=model.parameter_count(),
                   pack_manifest_sha256=base.sha256(manifest_path), source_sha256=sources,
                   training=TRAINING, training_loss_trace=loss_trace,
                   baseline_bbox_relative_mean=bbox_prior,
                   baseline_pose_prior={key: torch.from_numpy(value) for key, value in pose_prior.items()},
                   validation_per_episode=report, validation_macro=base.macro_report(report),
                   public_predictions_path=str(prediction_path),
                   claim="offline A1b normalization diagnosis only; no physics or adapter success")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public_readout, prediction_path)
    payload["public_predictions_sha256"] = base.sha256(prediction_path)
    torch.save(payload, output)
    report_path.write_text(json.dumps(dict(schema="p57.geometry-measurement-report.v1", mode=mode, seed=seed,
                                           checkpoint=dict(path=str(output), sha256=base.sha256(output)),
                                           public_predictions=dict(path=str(prediction_path), sha256=payload["public_predictions_sha256"]),
                                           freeze_sha256=base.sha256(freeze_path), pack_manifest_sha256=base.sha256(manifest_path),
                                           source_sha256=sources, config=asdict(config), parameters=payload["parameter_count"],
                                           training=TRAINING, training_loss_trace=loss_trace,
                                           validation_macro=payload["validation_macro"], validation_per_episode=report,
                                           limitation="Opened development trajectories; no physics or Fast Adapter comparison"),
                                      indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(output=str(output), mode=mode, seed=seed, validation_episodes=len(report))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("freeze")
    make.add_argument("--manifest", type=Path, required=True)
    make.add_argument("--plan", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("train")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--freeze", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "freeze":
        build_freeze(args.manifest, args.plan, args.output)
    else:
        torch.set_num_threads(1)
        train(args.manifest, args.output, args.mode, args.seed, freeze_path=args.freeze, device=args.device)


if __name__ == "__main__":
    main()
