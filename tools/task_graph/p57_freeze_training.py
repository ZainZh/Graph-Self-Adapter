#!/usr/bin/env python3
"""Create and verify the six-head P57 Stage A result-free training freeze."""
import argparse
import hashlib
import json
from pathlib import Path

from prove.task_graph.p57_measurement_head import GeometryMeasurementHead, MeasurementConfig, MODES


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (ROOT / "prove/task_graph/p57_measurement_head.py",
           ROOT / "prove/task_graph/p57_public_geometry.py",
           ROOT / "tools/task_graph/p57_train_measurement.py",
           ROOT / "tools/task_graph/p57_freeze_training.py")
SEEDS = (11, 22)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build(manifest_path, plan_path, output):
    manifest_path = Path(manifest_path).resolve()
    plan_path = Path(plan_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "p57.geometry-pack-manifest.v1" or manifest.get("cohort_counts") != dict(
            train=34, validation=8, transitions=573, frames=615) or len(manifest["episodes"]) != 42:
        raise ValueError("Require exact P57 geometry pack")
    split = {}
    for name in ("train", "validation"):
        rows = [dict(cell_index=row["cell_index"], condition_id=row["condition_id"],
                     source_policy=row["source_policy"], episode_sha256=row["bundle"]["sha256"],
                     source_seal_sha256=row["source_seal"]["sha256"])
                for row in manifest["episodes"] if row["split"] == name]
        if len(rows) != (34 if name == "train" else 8):
            raise ValueError("P57 pack split count differs")
        split[name] = dict(episodes=len(rows), sha256=_digest(rows))
    arms = [dict(mode=mode, seed=seed,
                 config=vars(MeasurementConfig(mode=mode)),
                 parameter_count=GeometryMeasurementHead(MeasurementConfig(mode=mode)).parameter_count(),
                 output_relative=f"runs/{mode}_seed{seed}/HEAD.pt")
            for mode in MODES for seed in SEEDS]
    freeze = dict(schema="p57.geometry-training-freeze.v1", scope="offline Stage A virtual-anchor measurement only",
                  pack_manifest=dict(path=str(manifest_path), sha256=sha256(manifest_path)),
                  plan=dict(path=str(plan_path), sha256=sha256(plan_path)),
                  sources={str(path): sha256(path) for path in SOURCES}, splits=split,
                  training=dict(updates=1500, batch=64, optimizer="Adam", learning_rate=3e-4,
                                sampling="uniform train trajectory then uniform actual frame",
                                normalization="train trajectories only", selection="final only"),
                  arms=arms, physics_steps=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n")
    return freeze


def verify(freeze_path, manifest_path, mode, seed, output):
    freeze_path = Path(freeze_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("schema") != "p57.geometry-training-freeze.v1" or freeze.get("physics_steps") != 0:
        raise ValueError("Invalid P57 training freeze")
    if sha256(manifest_path) != freeze["pack_manifest"]["sha256"]:
        raise ValueError("P57 frozen pack manifest changed")
    plan = freeze["plan"]
    if sha256(plan["path"]) != plan["sha256"]:
        raise ValueError("P57 plan changed after training freeze")
    for source in SOURCES:
        if freeze["sources"].get(str(source)) != sha256(source):
            raise ValueError("P57 source changed after training freeze")
    if freeze["training"] != dict(updates=1500, batch=64, optimizer="Adam", learning_rate=3e-4,
                                  sampling="uniform train trajectory then uniform actual frame",
                                  normalization="train trajectories only", selection="final only"):
        raise ValueError("P57 training budget changed")
    arm = next((arm for arm in freeze["arms"] if arm["mode"] == mode and arm["seed"] == seed), None)
    if arm is None or output != (freeze_path.parent / arm["output_relative"]).resolve():
        raise ValueError("P57 head/output is outside frozen six-arm queue")
    if arm["config"] != vars(MeasurementConfig(mode=mode)) or arm["parameter_count"] != GeometryMeasurementHead(
            MeasurementConfig(mode=mode)).parameter_count():
        raise ValueError("P57 frozen model configuration changed")
    return freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.manifest, args.plan, args.output)
    print(json.dumps(dict(output=str(args.output), arms=len(result["arms"]), splits=result["splits"])))


if __name__ == "__main__":
    main()
