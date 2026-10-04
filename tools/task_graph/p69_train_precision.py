#!/usr/bin/env python3
"""P69 seed-explicit tiny precision-MLP training entrypoint.

This is a local copy of the necessary P51 one-shot OSC imitation routine with
an explicit seed and update budget.  It trains a new checkpoint when run; it
never relabels an older P51/P53/P55/P64 checkpoint as a P69 training result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from prove.task_graph.libero_precision_p51 import PrecisionMLP, PRECISE, features, project_progress
from tools.task_graph.collect_libero_demo_gt import dump, sha

SCHEMA = "prove.p69-precision-training-config.v1"
MANIFEST_SCHEMA = "prove.p69-precision-training-manifest.v1"
ALLOWED_SEGMENTS = {"Grasp": "s3", "Align": "s6", "Insert": "s7", "s3": "s3", "s6": "s6", "s7": "s7"}


def read(path: Path):
    return json.loads(Path(path).read_text())


def validate_config(config: dict) -> dict:
    if config.get("schema") != SCHEMA:
        raise ValueError("P69 precision training config schema required")
    required = {"round_id", "seed", "source_hdf5", "capture_dir", "graph", "output", "warmstart", "training_role", "allowed_demo_segments", "updates"}
    if not required <= set(config):
        raise ValueError("P69 training config missing required fields")
    if not isinstance(config["seed"], int) or config["seed"] < 69010:
        raise ValueError("P69 training seed must be explicit, e.g. 69010+r")
    if not isinstance(config["updates"], int) or not 1 <= config["updates"] <= 2000:
        raise ValueError("P69 training updates must be in 1..2000")
    stages = [ALLOWED_SEGMENTS[x] for x in config["allowed_demo_segments"]]
    if not stages or any(s not in PRECISE for s in stages):
        raise ValueError("allowed_demo_segments must name Grasp/Align/Insert or s3/s6/s7")
    if config["warmstart"].get("mode") not in ("none", "checkpoint"):
        raise ValueError("warmstart.mode must be none or checkpoint")
    if config["warmstart"]["mode"] == "checkpoint":
        item = config["warmstart"].get("checkpoint")
        if not item or sha(item["path"]) != item["sha256"]:
            raise ValueError("warmstart checkpoint hash mismatch")
    if config["training_role"] not in ("bootstrap_small_mlp", "round_revision_small_mlp"):
        raise ValueError("unknown P69 training role")
    return config


def source_manifest(config_path: Path) -> dict:
    config_path = Path(config_path).resolve()
    config = validate_config(read(config_path))
    files = {
        "config": {"path": str(config_path), "sha256": sha(config_path)},
        "source_hdf5": {"path": str(Path(config["source_hdf5"]).resolve()), "sha256": sha(config["source_hdf5"])},
        "graph": {"path": str(Path(config["graph"]).resolve()), "sha256": sha(config["graph"])},
    }
    capture = Path(config["capture_dir"]).resolve()
    for name in ("BINDING.json", "FRAMES.jsonl", "SUPERVISOR_NAMES.json"):
        files["capture_" + name] = {"path": str(capture / name), "sha256": sha(capture / name)}
    for relative in ("tools/task_graph/p69_train_precision.py", "prove/task_graph/libero_precision_p51.py"):
        files["source_" + relative] = {"path": str((ROOT / relative).resolve()), "sha256": sha(ROOT / relative)}
    return dict(schema=MANIFEST_SCHEMA, config=config, files=files, simulator_rollouts=0, max_updates=2000,
                interpretation="P69 seed-explicit small MLP training manifest; no task success evidence")


def freeze(config_path: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = source_manifest(config_path)
    dump(output / "TRAINING_MANIFEST.json", manifest)
    return manifest


def _reference_transform(capture: Path, names: dict, reference: Path | None):
    with np.load(capture / "frame_0000.npz") as z:
        k = names["site"].index("desk_caddy_1_back_contain_region")
        transform = np.eye(4)
        transform[:3, 3] = z["supervisor_site_xpos"][k]
        transform[:3, :3] = z["supervisor_site_xmat"][k].reshape(3, 3)
    if reference is not None:
        transform = np.asarray(read(reference)["nominal_reference"], float)
    return transform


def _load_training_rows(config: dict):
    source = Path(config["source_hdf5"])
    capture = Path(config["capture_dir"])
    graph = read(config["graph"])
    names = read(capture / "SUPERVISOR_NAMES.json")
    transform = _reference_transform(capture, names, Path(config["reference"]) if config.get("reference") else None)
    teacher_binding = read(capture / "BINDING.json")
    if teacher_binding["source_sha256"] != sha(source):
        raise ValueError("teacher capture/source mismatch")
    with h5py.File(source) as f:
        actions = f["data/demo_0/actions"][()]
    frame_rows = [json.loads(x) for x in (capture / "FRAMES.jsonl").read_text().splitlines()]
    allowed = {ALLOWED_SEGMENTS[x] for x in config["allowed_demo_segments"]}
    raw, paths, source_frames = [], {}, {}
    for stage in graph["nodes"]:
        sid = stage["id"]
        if sid not in PRECISE or sid not in allowed:
            continue
        start, end = stage["evidence"]["start_step"], stage["evidence"]["end_step"]
        positions, rotations, grippers = [], [], []
        for t in range(start, end):
            p = capture / f"frame_{t:04d}.npz"
            if sha(p) != frame_rows[t]["sha256"]:
                raise ValueError("teacher frame hash mismatch")
            source_frames[p.name] = sha(p)
            with np.load(p) as z:
                if not np.array_equal(z["action_next"], actions[t]):
                    raise ValueError("teacher frame/action alignment mismatch")
                positions.append(z["observation_robot0_eef_pos"])
                rotations.append(Rotation.from_quat(z["observation_robot0_eef_quat"]).as_matrix())
                grippers.append(z["observation_robot0_gripper_qpos"])
        positions = np.asarray(positions)
        rotations = np.asarray(rotations)
        paths[sid] = dict(positions=positions.tolist(), rotations=rotations.tolist())
        for i, t in enumerate(range(start, end)):
            denom = max(end - start - 1, 1)
            raw.append((sid, t, positions[i], rotations[i], grippers[i], positions, rotations, i / denom))
    if not raw:
        raise ValueError("P69 training config selected no demo rows")
    return transform, actions, raw, paths, source_frames


def train_from_manifest(config_path: Path, manifest_path: Path):
    manifest = read(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("P69 training manifest required")
    current = source_manifest(config_path)
    if current != manifest:
        raise ValueError("P69 training manifest drift")
    config = current["config"]
    output = Path(config["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    transform, actions, raw, paths, source_frames = _load_training_rows(config)
    X, Y, cleanX, cleanY, stage_ids = [], [], [], [], []
    for sid, t, p, R, grip, positions, rotations, progress in raw:
        local = np.r_[transform[:3, :3].T @ actions[t, :3], transform[:3, :3].T @ actions[t, 3:6]]
        x0 = features(p, R, grip, transform, progress, sid)
        cleanX.append(x0)
        cleanY.append(local)
        stage_ids.append(sid)
        for sample in range(33):
            dp = np.zeros(3) if sample == 0 else rng.uniform(-.012, .012, 3)
            dr = np.zeros(3) if sample == 0 else rng.uniform(-.07, .07, 3)
            noisyR = Rotation.from_rotvec(dr).as_matrix() @ R
            prog = progress if sample == 0 else project_progress(p + dp, noisyR, positions, rotations)
            correction = np.r_[transform[:3, :3].T @ (-dp / .05), transform[:3, :3].T @ (-dr / .5)]
            correction = np.clip(correction, -.35, .35)
            X.append(features(p + dp, noisyR, grip, transform, prog, sid))
            Y.append(local + correction)
    x = torch.tensor(np.asarray(X), dtype=torch.float32)
    y = torch.tensor(np.asarray(Y), dtype=torch.float32)
    model = PrecisionMLP()
    if config["warmstart"]["mode"] == "checkpoint":
        package = torch.load(config["warmstart"]["checkpoint"]["path"], map_location="cpu", weights_only=False)
        model.load_state_dict(package["state_dict"], strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", .002)))
    protocol = dict(schema="prove.p69-seed-explicit-osc-training.v1", seed=config["seed"], updates=config["updates"],
                    batch_size=int(config.get("batch_size", 256)), learning_rate=float(config.get("learning_rate", .002)),
                    source_hdf5=current["files"]["source_hdf5"], graph=current["files"]["graph"], source_frame_hashes=source_frames,
                    allowed_demo_segments=config["allowed_demo_segments"], raw_pairs=len(raw), augmented_pairs=len(X),
                    warmstart=config["warmstart"], simulator_rollouts=0,
                    nominal_reference=transform.tolist(), runtime_inputs=["public EEF pose", "public gripper qpos", "reference transform", "stage", "monotone nearest-path progress"])
    dump(output / "TRAINING_FREEZE.json", protocol)
    batch = protocol["batch_size"]
    for step in range(config["updates"]):
        idx = torch.randint(len(x), (batch,))
        loss = (model(x[idx]) - y[idx]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        clean = torch.tensor(np.asarray(cleanX), dtype=torch.float32)
        pred = model(clean).numpy()
        aug_mse = float((model(x) - y).square().mean())
    stage_ids = np.asarray(stage_ids)
    cleanY_arr = np.asarray(cleanY)
    report = dict(schema="prove.p69-seed-explicit-osc-fit.v1", original_pairs=len(raw), augmented_pairs=len(X),
                  augmented_mse=aug_mse, original_mse=float(np.mean((pred - cleanY_arr) ** 2)),
                  stage_mse={s: float(np.mean((pred[stage_ids == s] - cleanY_arr[stage_ids == s]) ** 2)) for s in sorted(set(stage_ids))},
                  seed=config["seed"], updates=config["updates"], interpretation="fit diagnostic only; no rollout evidence")
    torch.save(dict(state_dict=model.state_dict(), nominal_reference=transform.tolist(), paths=paths,
                    schema="prove.p69-precision-action-head.v1", seed=config["seed"], updates=config["updates"],
                    training_freeze_sha256=sha(output / "TRAINING_FREEZE.json")), output / "action_head.pt")
    report["checkpoint_sha256"] = sha(output / "action_head.pt")
    dump(output / "FIT_REPORT.json", report)
    receipt = copy.deepcopy(manifest)
    receipt.update(schema="prove.p69-precision-training-receipt.v1", output=str(output),
                   checkpoint={"path": str(output / "action_head.pt"), "sha256": sha(output / "action_head.pt")},
                   fit_report={"path": str(output / "FIT_REPORT.json"), "sha256": sha(output / "FIT_REPORT.json")},
                   training_freeze={"path": str(output / "TRAINING_FREEZE.json"), "sha256": sha(output / "TRAINING_FREEZE.json")})
    dump(output / "P69_TRAINING_RECEIPT.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    value = freeze(args.config, args.output) if args.command == "freeze" else train_from_manifest(args.config, args.manifest)
    if args.command == "freeze":
        summary = {"schema": value["schema"], "manifest": str((args.output / "TRAINING_MANIFEST.json").resolve()),
                   "manifest_sha256": sha(args.output / "TRAINING_MANIFEST.json"),
                   "seed": value["config"]["seed"], "updates": value["config"]["updates"]}
    else:
        summary = {"schema": value["schema"], "output": value["output"],
                   "checkpoint_sha256": value["checkpoint"]["sha256"],
                   "fit_report_sha256": value["fit_report"]["sha256"]}
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
