#!/usr/bin/env python3
"""Public-first offline evaluator for the bounded P61 2x2 development study.

``predict`` opens only public episode packs, writes all four arms for all twelve
prespecified development conditions, and seals them. ``score`` first verifies
that complete seal and only then opens private geometry labels. This is an
offline geometric/action-error study; it makes no physical-success claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from prove.fast_adapter.p60_contour import model_state_sha256
from prove.fast_adapter.p61_adaptation import adapt_z, corrected_components
from prove.fast_adapter.p61_encoder import P61RelationEncoder
from prove.task_graph.p59.core import (
    FrozenP50Action, pack_geometry, so3_log_map, split_geometry,
)
from tools.fast_adapter.p61_train import contour_measurements, encode_frames


PREDICTION_SCHEMA = "prove.p61-public-dev-predictions.v1"
SEAL_SCHEMA = "prove.p61-public-dev-prediction-seal.v1"
REPORT_SCHEMA = "prove.p61-private-postseal-dev-report.v1"
CHECKPOINT_SCHEMA = "prove.p61-relation-checkpoint.v1"
FRAME_INDICES = (0, 4, 8, 12, 16, 20, 24, 29)
SUPPORT_TIMES = (0, 8, 16, 24)
MODEL_ARMS = ("frozen_lila", "task_trained_lila")
ADAPTATION_ARMS = ("fixed", "online")
PHASES = (0, 1, 2, 3)
SOURCE_CLOSURE = (
    "prove/fast_adapter/p61_encoder.py",
    "prove/fast_adapter/p61_adaptation.py",
    "tools/fast_adapter/p61_train.py",
    "tools/fast_adapter/p61_evaluate.py",
    "prove/fast_adapter/p60_contour.py",
    "prove/task_graph/p59/core.py",
)
REQUIRED_TRAINING_SOURCES = (
    "prove/fast_adapter/p61_encoder.py",
    "prove/fast_adapter/p61_adaptation.py",
    "tools/fast_adapter/p61_train.py",
    "prove/fast_adapter/p60_contour.py",
    "prove/task_graph/p59/core.py",
)
DEFAULT_TRAINING_FREEZE = Path(
    "evaluations/fast_adapter_p61_task_visual_v01/TRAINING_FREEZE.json")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _tensor_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_final_checkpoint(path: Path, expected_arm: str, device: str):
    if "engineering" in path.resolve().parts:
        raise ValueError("Engineering/smoke checkpoints are forbidden in P61 evaluation")
    checkpoint_hash = sha256(path)
    package = torch.load(path, map_location="cpu", weights_only=True)
    if package.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unknown P61 checkpoint schema: {path}")
    training = package.get("training", {})
    expected_tuned = expected_arm == "task_trained_lila"
    if (training.get("arm") != expected_arm
            or training.get("supervised_steps") != 1000
            or training.get("meta_steps") != 300
            or training.get("batch_size") != 2
            or training.get("selection") != "final fixed budget"):
        raise ValueError(f"P61 evaluator accepts only final-budget {expected_arm}: {path}")
    constructor = dict(package["constructor"])
    if bool(constructor.get("finetune_lila")) != expected_tuned:
        raise ValueError("Checkpoint arm and LiLa trainability metadata disagree")
    model = P61RelationEncoder(**constructor)
    model.load_state_dict(package["state_dict"], strict=True)
    if model.lila_binding != package.get("lila_binding"):
        raise ValueError("Official LiLa source/checkpoint binding changed")
    model.to(device).eval().requires_grad_(False)
    return model, {
        "checkpoint": binding(path),
        "model_state_sha256": model_state_sha256(model),
        "tensor_state_sha256": _tensor_hash(model.state_dict()),
        "training": training,
    }, checkpoint_hash


def _dev_records(dataset: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = dataset / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "prove.p61-native-dino-dataset.v1":
        raise ValueError("Unknown P61 dataset manifest")
    records = [record for record in manifest.get("records", []) if record.get("split") == "dev"]
    if len(records) != 12:
        raise ValueError(f"P61 requires exactly 12 dev conditions, got {len(records)}")
    identities = [(int(record["seed"]), record["distribution"]) for record in records]
    if len(set(identities)) != 12:
        raise ValueError("Duplicate P61 dev condition identity")
    counts = {name: sum(record["distribution"] == name for record in records)
              for name in ("id", "geometry")}
    if counts != {"id": 6, "geometry": 6}:
        raise ValueError(f"P61 dev strata changed: {counts}")
    for record in records:
        public = Path(record["path"]) / "public.pt"
        if not public.is_file() or sha256(public) != record["public_sha256"]:
            raise ValueError(f"Public episode binding changed: {public}")
    return {"manifest": binding(manifest_path), "records": records}, records


def _verify_training_freeze(training_freeze: Path, dataset: Path) -> dict[str, Any]:
    frozen = json.loads(training_freeze.read_text())
    if (frozen.get("schema") != "prove.p61-task-visual-training-freeze.v1"
            or frozen.get("arms") != list(MODEL_ARMS)
            or frozen.get("steps") != {"supervised": 1000, "meta": 300}
            or frozen.get("batch_configurations") != 2):
        raise ValueError("P61 training freeze scope changed")
    bindings = frozen.get("source_bindings", {})
    manifest_path = (dataset / "MANIFEST.json").resolve()
    manifest_keys = [name for name in bindings
                     if (Path(name) if Path(name).is_absolute() else Path.cwd() / name).resolve()
                     == manifest_path]
    if len(manifest_keys) != 1:
        raise ValueError("P61 training freeze must bind exactly this dataset manifest")
    expected_manifest = manifest_keys[0]
    if bindings.get(expected_manifest) != sha256(dataset / "MANIFEST.json"):
        raise ValueError("P61 training freeze does not bind this exact dataset manifest")
    for name in REQUIRED_TRAINING_SOURCES:
        if bindings.get(name) != sha256(name):
            raise ValueError(f"P61 frozen training source changed: {name}")
    return {"training_freeze": binding(training_freeze),
            "verified_training_sources": {name: bindings[name]
                                           for name in REQUIRED_TRAINING_SOURCES},
            "dataset_manifest_sha256": bindings[expected_manifest]}


def _world_opening(geometry8: torch.Tensor, z: torch.Tensor,
                   T_world_anchor: torch.Tensor) -> dict[str, Any]:
    center, rotation, half_size = corrected_components(geometry8, z)
    anchor_rotation = T_world_anchor[:3, :3]
    center = anchor_rotation @ center + T_world_anchor[:3, 3]
    rotation = anchor_rotation @ rotation
    return {"center_world": center.detach().cpu().tolist(),
            "rotation_world": rotation.detach().cpu().tolist(),
            "opening_half_size": half_size.detach().cpu().tolist()}


def _predict_episode(model: P61RelationEncoder, public: dict[str, Any], device: str):
    frames = tuple(int(value) for value in public["frame_index"].tolist())
    if frames != FRAME_INDICES:
        raise ValueError(f"P61 dev frame schedule changed: {frames}")
    lookup = {frame: index for index, frame in enumerate(frames)}
    if not all(bool(public["boundary_valid"][lookup[t]]) for t in SUPPORT_TIMES):
        raise ValueError("A required P61 public support contour is invalid")
    # Model weights and DINO caches do not require gradients during evaluation;
    # the later z-only GN block deliberately runs under ordinary grad mode.
    with torch.no_grad():
        encoded = encode_frames(model, public, [lookup[16], lookup[24], lookup[29]], device)
    if encoded.shape != (3, 8) or not torch.isfinite(encoded).all():
        raise ValueError("P61 encoder did not return three finite geometry8 values")
    T_world_anchor = public["T_world_anchor"].to(device).float()
    z = torch.zeros(8, device=device)
    updates = []
    for geometry_index, times in enumerate(((0, 8, 16), SUPPORT_TIMES)):
        measurements = contour_measurements(public, [lookup[t] for t in times])
        if tuple(int(measurement.t) for measurement in measurements) != tuple(times):
            raise ValueError("P61 causal support measurement set is incomplete")
        with torch.enable_grad():
            z, records = adapt_z(encoded[geometry_index], z, T_world_anchor,
                                 measurements, create_graph=False)
        if len(records) != 3 or not torch.isfinite(z).all():
            raise FloatingPointError("P61 runtime adaptation is incomplete or non-finite")
        updates.append({"at_t": times[-1], "support_times": list(times),
                        "steps": records, "z_after": z.detach().cpu().tolist()})
    query = encoded[2]
    zero = torch.zeros_like(z)
    return {
        "query_t": 29,
        "query_freshly_encoded": True,
        "query_measurement_excluded": True,
        "encoded_current_times": [16, 24],
        "support_measurement_times": list(SUPPORT_TIMES),
        "updates": updates,
        "query_geometry8": query.detach().cpu().tolist(),
        "fixed": {"z": zero.cpu().tolist(),
                  **_world_opening(query, zero, T_world_anchor)},
        "online": {"z": z.detach().cpu().tolist(),
                   **_world_opening(query, z, T_world_anchor)},
    }


def predict(dataset: Path, frozen_checkpoint: Path, task_checkpoint: Path,
            output: Path, device: str,
            training_freeze: Path = DEFAULT_TRAINING_FREEZE) -> dict[str, Any]:
    """Write and seal all public predictions. This function never opens private.pt."""
    output.mkdir(parents=True, exist_ok=False)
    freeze_admission = _verify_training_freeze(training_freeze, dataset)
    dataset_info, records = _dev_records(dataset)
    checkpoints = {"frozen_lila": frozen_checkpoint, "task_trained_lila": task_checkpoint}
    model_records: dict[str, Any] = {}
    predictions: dict[tuple[int, str], dict[str, Any]] = {
        (int(record["seed"]), record["distribution"]): {
            "seed": int(record["seed"]), "distribution": record["distribution"],
            "public": binding(Path(record["path"]) / "public.pt"), "models": {}}
        for record in records}
    for arm in MODEL_ARMS:
        model, metadata, checkpoint_hash = _load_final_checkpoint(
            Path(checkpoints[arm]), arm, device)
        before = model_state_sha256(model)
        model_records[arm] = metadata
        for record in records:
            public = torch.load(Path(record["path"]) / "public.pt",
                                map_location="cpu", weights_only=True)
            result = _predict_episode(model, public, device)
            predictions[(int(record["seed"]), record["distribution"])]["models"][arm] = result
        after = model_state_sha256(model)
        if before != after or checkpoint_hash != sha256(checkpoints[arm]):
            raise RuntimeError(f"Frozen P61 checkpoint/model changed during prediction: {arm}")
        model_records[arm]["model_state_sha256_after"] = after
    episodes = [predictions[(int(record["seed"]), record["distribution"])] for record in records]
    if len(episodes) != 12 or any(set(row["models"]) != set(MODEL_ARMS) for row in episodes):
        raise RuntimeError("P61 public prediction cohort is incomplete")
    value = {
        "schema": PREDICTION_SCHEMA, "plan": binding(
            Path("evaluations/fast_adapter_p61_task_visual_v01/PLAN.md")),
        "dataset": dataset_info, "training_admission": freeze_admission,
        "models": model_records,
        "source_closure": {name: binding(Path(name)) for name in SOURCE_CLOSURE},
        "arms": [f"{model}/{adaptation}" for model in MODEL_ARMS
                 for adaptation in ADAPTATION_ARMS],
        "episodes": episodes, "development_data_opened_previously": True,
        "private_truth_opened": False, "physics_integration_steps": 0,
        "physical_success_claim": False,
    }
    prediction_path = output / "PUBLIC_PREDICTIONS.json"
    write_new(prediction_path, value)
    seal = {
        "schema": SEAL_SCHEMA, "status": "all_public_predictions_sealed_before_private",
        "predictions": binding(prediction_path),
        "dataset_manifest": dataset_info["manifest"],
        "training_freeze": freeze_admission["training_freeze"],
        "checkpoints": {arm: binding(path) for arm, path in checkpoints.items()},
        "evaluator_source": binding(Path(__file__)),
        "source_closure": {name: binding(Path(name)) for name in SOURCE_CLOSURE},
        "private_truth_opened": False,
        "episodes": 12, "prediction_arms": 48, "sealed_unix": time.time(),
    }
    write_new(output / "PREDICTION_SEAL.json", seal)
    return value


def _rotation_error_degrees(predicted: torch.Tensor, target: torch.Tensor) -> float:
    radians = so3_log_map(target.transpose(-1, -2) @ predicted).norm()
    return float(torch.rad2deg(radians))


def _finite_row(row: Mapping[str, Any]) -> None:
    def visit(value: Any):
        if isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        elif isinstance(value, float) and not np.isfinite(value):
            raise FloatingPointError("Non-finite P61 evaluation metric")
    visit(row)


def _metrics(prediction: Mapping[str, Any], target_geometry: torch.Tensor,
             q: torch.Tensor, p50: FrozenP50Action) -> dict[str, Any]:
    center = torch.tensor(prediction["center_world"], dtype=torch.float32)
    rotation = torch.tensor(prediction["rotation_world"], dtype=torch.float32)
    size = torch.tensor(prediction["opening_half_size"], dtype=torch.float32)
    target_center, target_rotation, target_size, held_length = split_geometry(target_geometry)
    predicted_geometry = pack_geometry(center, rotation, size, held_length)
    actions = []
    for phase in PHASES:
        phase_tensor = torch.tensor(phase, dtype=torch.long)
        with torch.no_grad():
            predicted_action = p50(predicted_geometry, q, phase_tensor)
            target_action = p50(target_geometry, q, phase_tensor)
            predicted_pose = p50.decode_world_pose(predicted_action, predicted_geometry, q)
            target_pose = p50.decode_world_pose(target_action, target_geometry, q)
        actions.append({
            "phase": phase,
            "position_error_mm": float((predicted_pose[0] - target_pose[0]).norm() * 1000),
            "rotation_error_deg": _rotation_error_degrees(predicted_pose[1], target_pose[1]),
        })
    result = {
        "center_error_mm": float((center - target_center).norm() * 1000),
        "rotation_error_deg": _rotation_error_degrees(rotation, target_rotation),
        "half_size_abs_error_mm": ((size - target_size).abs() * 1000).tolist(),
        "half_size_mean_abs_error_mm": float((size - target_size).abs().mean() * 1000),
        "world_action_position_error_mm_all4phases": float(np.mean(
            [row["position_error_mm"] for row in actions])),
        "world_action_rotation_error_deg_all4phases": float(np.mean(
            [row["rotation_error_deg"] for row in actions])),
        "actions": actions,
    }
    _finite_row(result)
    return result


METRIC_KEYS = (
    "center_error_mm", "rotation_error_deg", "half_size_mean_abs_error_mm",
    "world_action_position_error_mm_all4phases",
    "world_action_rotation_error_deg_all4phases",
)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = []
    for model_arm in MODEL_ARMS:
        for adaptation in ADAPTATION_ARMS:
            for distribution in ("id", "geometry"):
                selected = [row["arms"][f"{model_arm}/{adaptation}"] for row in rows
                            if row["distribution"] == distribution]
                if len(selected) != 6:
                    raise ValueError("P61 aggregate requires six conditions per stratum")
                groups.append({
                    "model_arm": model_arm, "adaptation": adaptation,
                    "distribution": distribution, "n": 6,
                    "metrics": {key: {"mean": float(np.mean([row[key] for row in selected])),
                                      "median": float(np.median([row[key] for row in selected]))}
                                for key in METRIC_KEYS},
                })
    comparisons = []
    definitions = []
    for model_arm in MODEL_ARMS:
        definitions.append((f"{model_arm}:online-vs-fixed",
                            f"{model_arm}/online", f"{model_arm}/fixed"))
    for adaptation in ADAPTATION_ARMS:
        definitions.append((f"task-vs-frozen:{adaptation}",
                            f"task_trained_lila/{adaptation}",
                            f"frozen_lila/{adaptation}"))
    for name, candidate, reference in definitions:
        for distribution in ("id", "geometry"):
            selected = [row for row in rows if row["distribution"] == distribution]
            metrics = {}
            for key in METRIC_KEYS:
                delta = [row["arms"][candidate][key] - row["arms"][reference][key]
                         for row in selected]
                metrics[key] = {
                    "wins": sum(value < -1e-9 for value in delta),
                    "losses": sum(value > 1e-9 for value in delta),
                    "ties": sum(abs(value) <= 1e-9 for value in delta),
                    "mean_candidate_minus_reference": float(np.mean(delta)),
                    "median_candidate_minus_reference": float(np.median(delta)),
                }
            comparisons.append({"comparison": name, "candidate": candidate,
                                "reference": reference, "distribution": distribution,
                                "n": 6, "metrics": metrics})
    return {"groups": groups, "paired_comparisons": comparisons}


def _brief(report: Mapping[str, Any]) -> str:
    lines = [
        "# P61 随任务训练 LiLa／ROI 离线开发评估",
        "",
        "这是12个已开放开发配置上的几何与冻结P50动作误差评估，不是物理成功率，也不是独立测试。",
        "全部四臂公开预测先写入并封存，之后才读取私有几何标签。query t29使用新编码，且不进入反馈历史。",
        "动作诊断使用相同的教师物长，只比较离线条件化的冻结P50世界动作；它不是实际闭环动作或任务成功率。",
        "",
        "| 模型 | z | 分布 | n | 中心mm mean/median | 旋转° mean/median | 半尺寸mm mean/median | 动作位置mm mean/median | 动作旋转° mean/median |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["aggregate"]["groups"]:
        values = [group["metrics"][key] for key in METRIC_KEYS]
        fields = [f"{value['mean']:.4f}/{value['median']:.4f}" for value in values]
        lines.append("| " + " | ".join((group["model_arm"], group["adaptation"],
                                          group["distribution"], str(group["n"]), *fields)) + " |")
    lines += ["", "逐条件结果与配对胜／负／平计数见 `REPORT.json`。所有无效或缺失输入都会使整批失败，不静默删行。", ""]
    return "\n".join(lines)


def score(output: Path, dataset: Path, p50_checkpoint: Path) -> dict[str, Any]:
    """Verify the complete public seal, then and only then open private labels."""
    seal_path, prediction_path = output / "PREDICTION_SEAL.json", output / "PUBLIC_PREDICTIONS.json"
    seal = json.loads(seal_path.read_text())
    if (seal.get("schema") != SEAL_SCHEMA or seal.get("private_truth_opened") is not False
            or seal.get("episodes") != 12 or seal.get("prediction_arms") != 48
            or seal.get("predictions", {}).get("sha256") != sha256(prediction_path)
            or seal.get("dataset_manifest", {}).get("sha256") != sha256(dataset / "MANIFEST.json")):
        raise ValueError("Complete P61 public prediction seal is absent or invalid")
    for arm in MODEL_ARMS:
        item = seal.get("checkpoints", {}).get(arm, {})
        if item.get("sha256") != sha256(item.get("path", "")):
            raise ValueError(f"Sealed P61 checkpoint changed: {arm}")
    frozen_item = seal.get("training_freeze", {})
    if frozen_item.get("sha256") != sha256(frozen_item.get("path", "")):
        raise ValueError("Sealed P61 training freeze changed")
    closure = seal.get("source_closure", {})
    if set(closure) != set(SOURCE_CLOSURE):
        raise ValueError("P61 evaluator source closure is incomplete")
    for name, item in closure.items():
        if item.get("sha256") != sha256(item.get("path", "")):
            raise ValueError(f"P61 sealed source changed: {name}")
    predictions = json.loads(prediction_path.read_text())
    if (predictions.get("schema") != PREDICTION_SCHEMA
            or predictions.get("private_truth_opened") is not False
            or len(predictions.get("episodes", [])) != 12):
        raise ValueError("P61 public prediction artifact is incomplete")
    _, records = _dev_records(dataset)
    expected = {(int(record["seed"]), record["distribution"]): record for record in records}
    actual = {(int(row["seed"]), row["distribution"]): row
              for row in predictions["episodes"]}
    if set(actual) != set(expected):
        raise ValueError("P61 sealed condition identities differ from the dataset")
    p50 = FrozenP50Action(p50_checkpoint).eval().requires_grad_(False)
    p50_before = model_state_sha256(p50)
    rows = []
    for identity, record in expected.items():
        episode = actual[identity]
        if set(episode.get("models", {})) != set(MODEL_ARMS):
            raise ValueError("P61 sealed model arms are incomplete")
        public_path = Path(record["path"]) / "public.pt"
        private_path = Path(record["path"]) / "private.pt"
        if sha256(private_path) != record["private_sha256"]:
            raise ValueError(f"Private episode binding changed: {private_path}")
        public = torch.load(public_path, map_location="cpu", weights_only=True)
        private = torch.load(private_path, map_location="cpu", weights_only=True)
        frames = [int(value) for value in public["frame_index"].tolist()]
        if frames != list(FRAME_INDICES):
            raise ValueError("P61 score frame schedule changed")
        query_index = frames.index(29)
        target = private["geometry12_world"][query_index].float()
        q = public["q"][query_index].float()
        arms = {}
        for model_arm in MODEL_ARMS:
            model_prediction = episode["models"][model_arm]
            if (model_prediction.get("query_t") != 29
                    or model_prediction.get("query_freshly_encoded") is not True
                    or model_prediction.get("query_measurement_excluded") is not True
                    or model_prediction.get("support_measurement_times") != list(SUPPORT_TIMES)):
                raise ValueError("P61 sealed causal/query contract changed")
            for adaptation in ADAPTATION_ARMS:
                arms[f"{model_arm}/{adaptation}"] = _metrics(
                    model_prediction[adaptation], target, q, p50)
        row = {"seed": identity[0], "distribution": identity[1], "arms": arms,
               "private": binding(private_path)}
        _finite_row(row)
        rows.append(row)
    if len(rows) != 12 or any(len(row["arms"]) != 4 for row in rows):
        raise RuntimeError("P61 private scoring did not retain the full 12x4 denominator")
    if p50_before != model_state_sha256(p50) or sha256(p50_checkpoint) != p50.checkpoint_sha256:
        raise RuntimeError("Frozen P50 changed during P61 evaluation")
    report = {
        "schema": REPORT_SCHEMA, "prediction_seal": binding(seal_path),
        "evaluator_source": binding(Path(__file__)), "frozen_p50": binding(p50_checkpoint),
        "truth_opened_only_after_complete_prediction_seal": True,
        "development_data_opened_previously": True, "independent_test_claim": False,
        "physical_success_claim": False, "physics_integration_steps": 0,
        "invalid_handling": "fail entire bounded study; no rows silently dropped",
        "held_length_use": "evaluator-only label-conditioned value shared by prediction and teacher",
        "query": "fresh public t29 encoding; t29 measurement excluded",
        "rows": rows, "aggregate": _aggregate(rows),
    }
    report_path = output / "REPORT.json"
    write_new(report_path, report)
    with (output / "REPORT.md").open("x") as stream:
        stream.write(_brief(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prediction = sub.add_parser("predict")
    prediction.add_argument("--dataset", type=Path, required=True)
    prediction.add_argument("--frozen-checkpoint", type=Path, required=True)
    prediction.add_argument("--task-checkpoint", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="cuda")
    prediction.add_argument("--training-freeze", type=Path, default=DEFAULT_TRAINING_FREEZE)
    scoring = sub.add_parser("score")
    scoring.add_argument("--dataset", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--p50-checkpoint", type=Path, default=Path(
        "evaluations/fast_adapter_p50_precision_policy_v01/offline_v01/precision_action.pt"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.operation == "predict":
        result = predict(args.dataset, args.frozen_checkpoint, args.task_checkpoint,
                         args.output, args.device, args.training_freeze)
        compact = {"schema": result["schema"], "episodes": len(result["episodes"]),
                   "arms": result["arms"]}
    else:
        result = score(args.output, args.dataset, args.p50_checkpoint)
        compact = {"schema": result["schema"], "aggregate": result["aggregate"]}
    print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
