"""Read-only sealed-artifact summary for P59 B1 offline evaluations.

The statistical unit is an episode/condition, never a frame.  This tool does
not run models, select checkpoints, infer physical success, or modify inputs.
Opened developer material is labelled as development evidence throughout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METRICS = (
    "geometry_position_mm",
    "geometry_rotation_deg",
    "geometry_half_size_mae_mm",
    "geometry_held_length_mm",
    "relation_position_mm",
    "relation_rotation_deg",
    "world_position_mm",
    "world_rotation_deg",
)
WINDOWS = {
    "immediate_t1_t24": lambda row: row.get("query_scope") == "continuous"
    and 1 <= int(row.get("step_index", -1)) <= 24,
    "future_t27_t29": lambda row: row.get("query_scope") == "future_no_adaptation"
    and int(row.get("step_index", -1)) in (27, 29),
}
GROUPS = {
    ("train", "id"): "train_id",
    ("developer", "id"): "opened_dev_id",
    ("developer", "geometry"): "opened_dev_geometry",
}
EXPECTED_CONDITIONS = {"train_id": 96, "opened_dev_id": 6, "opened_dev_geometry": 6}
COMPARISONS = {
    "online_minus_fixed": ("online", "fixed_init"),
    "online_minus_wrong_online": ("online", "wrong_online"),
    "full_history_minus_online": ("full_history", "online"),
}


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _finite(value, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite offline metric: {label}")
    return value


def _condition_values(records: list[dict]) -> dict:
    """Average frames within condition before any cross-condition operation."""
    collected = {}
    seen_steps = {}
    for row in records:
        group = GROUPS.get((row.get("split"), row.get("distribution")))
        if group is None:
            raise ValueError("unknown split/distribution in offline evaluation")
        windows = [name for name, predicate in WINDOWS.items() if predicate(row)]
        if not windows:
            continue
        if len(windows) != 1:
            raise ValueError("offline query belongs to multiple time windows")
        method, episode = row.get("method"), str(row.get("episode_id", ""))
        if not method or not episode:
            raise ValueError("offline record lacks method or episode identity")
        key = (group, windows[0], method, episode)
        step = int(row["step_index"])
        if step in seen_steps.setdefault(key, set()):
            raise ValueError("duplicate condition/method/query step")
        seen_steps[key].add(step)
        values = collected.setdefault(key, {metric: [] for metric in METRICS})
        for metric in METRICS:
            if metric not in row:
                raise ValueError(f"offline record lacks {metric}")
            values[metric].append(_finite(row[metric], metric))
    result = {}
    for key, values in collected.items():
        result[key] = {metric: sum(samples) / len(samples)
                       for metric, samples in values.items()}
    return result


def summarize_records(main_records: list[dict], sidecar_records: list[dict],
                      *, enforce_frozen_counts: bool = True) -> dict:
    if enforce_frozen_counts:
        for records in (main_records, sidecar_records):
            steps = {}
            for row in records:
                group = GROUPS.get((row.get("split"), row.get("distribution")))
                windows = [name for name, predicate in WINDOWS.items() if predicate(row)]
                if group is not None and windows:
                    key = (group, windows[0], row.get("method"), str(row.get("episode_id")))
                    steps.setdefault(key, set()).add(int(row["step_index"]))
            for (_, window, _, _), observed in steps.items():
                expected = set(range(1, 25)) if window == "immediate_t1_t24" else {27, 29}
                if observed != expected:
                    raise ValueError(f"incomplete query-frame set for {window}")
    main = _condition_values(main_records)
    sidecar = _condition_values(sidecar_records)
    condition_values = {**main}
    overlap = set(main) & set(sidecar)
    if overlap:
        raise ValueError("main and sidecar method identities overlap")
    condition_values.update(sidecar)

    identities = {}
    for group in EXPECTED_CONDITIONS:
        main_ids = {key[3] for key in main if key[0] == group}
        sidecar_ids = {key[3] for key in sidecar if key[0] == group}
        if main_ids != sidecar_ids:
            raise ValueError(f"main/FullHistory condition identities differ in {group}")
        if enforce_frozen_counts and len(main_ids) != EXPECTED_CONDITIONS[group]:
            raise ValueError(f"sealed condition count differs in {group}")
        identities[group] = sorted(main_ids)

    summaries = {}
    for group in EXPECTED_CONDITIONS:
        summaries[group] = {}
        for window in WINDOWS:
            methods = sorted({key[2] for key in condition_values
                              if key[0] == group and key[1] == window})
            method_summary = {}
            for method in methods:
                rows = [value for key, value in condition_values.items()
                        if key[:3] == (group, window, method)]
                if enforce_frozen_counts and len(rows) != EXPECTED_CONDITIONS[group]:
                    raise ValueError(f"incomplete {group}/{window}/{method}")
                method_summary[method] = {
                    "conditions": len(rows),
                    "condition_equal_mean": {
                        metric: sum(row[metric] for row in rows) / len(rows)
                        for metric in METRICS
                    },
                }
            summaries[group][window] = {"methods": method_summary}

    comparisons = {}
    for group in EXPECTED_CONDITIONS:
        comparisons[group] = {}
        for window in WINDOWS:
            window_result = {}
            for name, (left, right) in COMPARISONS.items():
                per_metric = {}
                for metric in METRICS:
                    deltas = []
                    for episode in identities[group]:
                        left_key, right_key = ((group, window, left, episode),
                                               (group, window, right, episode))
                        if left_key not in condition_values or right_key not in condition_values:
                            raise ValueError(f"missing paired method for {name}")
                        deltas.append(condition_values[left_key][metric]
                                      - condition_values[right_key][metric])
                    tolerance = 1e-12
                    per_metric[metric] = {
                        "conditions": len(deltas),
                        "mean_paired_delta": sum(deltas) / len(deltas),
                        "improved": sum(delta < -tolerance for delta in deltas),
                        "worsened": sum(delta > tolerance for delta in deltas),
                        "tied": sum(abs(delta) <= tolerance for delta in deltas),
                        "direction": "negative_is_lower_error",
                    }
                window_result[name] = per_metric
            comparisons[group][window] = window_result
    return {"summaries": summaries, "paired_comparisons": comparisons}


def validate_artifacts(main_evaluation: Path, sidecar_evaluation: Path,
                       main_training_freeze: Path, main_training_seal: Path,
                       sidecar_training_freeze: Path, sidecar_training_seal: Path,
                       sidecar_evaluation_freeze: Path, pipeline_seal: Path) -> tuple[dict, dict, dict]:
    main_eval, side_eval = load_json(main_evaluation), load_json(sidecar_evaluation)
    if main_eval.get("schema") != "prove.p59-offline-evaluation.v1":
        raise ValueError("unknown main offline evaluation schema")
    if side_eval.get("schema") != "prove.p59-full-history-offline-evaluation.v1":
        raise ValueError("unknown FullHistory offline evaluation schema")
    main_freeze, main_seal = load_json(main_training_freeze), load_json(main_training_seal)
    side_freeze, side_seal = load_json(sidecar_training_freeze), load_json(sidecar_training_seal)
    eval_freeze, pipeline = load_json(sidecar_evaluation_freeze), load_json(pipeline_seal)
    if main_freeze.get("schema") != "prove.p59-training-freeze.v1" or \
            main_seal.get("schema") != "prove.p59-training-seal.v1":
        raise ValueError("main training freeze/seal schema mismatch")
    if side_freeze.get("schema") != "prove.p59-full-history-freeze.v1" or \
            side_seal.get("schema") != "prove.p59-full-history-seal.v1":
        raise ValueError("FullHistory training freeze/seal schema mismatch")
    if eval_freeze.get("schema") != "prove.p59-full-history-evaluation-freeze.v1":
        raise ValueError("FullHistory evaluation freeze schema mismatch")
    if pipeline.get("schema") not in (
        "prove.p59-post-training-pipeline-seal.v1",
        "prove.p59-post-training-batched-pipeline-seal.v1",
    ):
        raise ValueError("unknown post-training pipeline seal")
    if pipeline.get("main_evaluation_sha256") != sha(main_evaluation) or \
            pipeline.get("sidecar_evaluation_sha256") != sha(sidecar_evaluation):
        raise ValueError("evaluation bytes differ from pipeline seal")
    main_provenance = main_eval.get("provenance", {})
    if main_provenance.get("training_freeze_sha256") != sha(main_training_freeze) or \
            main_provenance.get("checkpoint_sha256") != main_seal.get("checkpoint_sha256"):
        raise ValueError("main evaluation provenance differs from training seal/freeze")
    side_provenance = side_eval.get("provenance", {})
    if side_provenance.get("evaluation_freeze_sha256") != sha(sidecar_evaluation_freeze) or \
            eval_freeze.get("sidecar_checkpoint_sha256") != side_seal.get("checkpoint_sha256") or \
            eval_freeze.get("sidecar_training_freeze_sha256") != sha(sidecar_training_freeze):
        raise ValueError("FullHistory evaluation provenance differs from freezes/seal")
    return main_eval, side_eval, {
        "main_evaluation_sha256": sha(main_evaluation),
        "sidecar_evaluation_sha256": sha(sidecar_evaluation),
        "pipeline_seal_sha256": sha(pipeline_seal),
        "main_training_freeze_sha256": sha(main_training_freeze),
        "sidecar_training_freeze_sha256": sha(sidecar_training_freeze),
        "sidecar_evaluation_freeze_sha256": sha(sidecar_evaluation_freeze),
    }


def chinese_table(summary: dict) -> str:
    group_names = {"train_id": "训练ID", "opened_dev_id": "已打开开发ID",
                   "opened_dev_geometry": "已打开开发几何OOD"}
    window_names = {"immediate_t1_t24": "即时 t1–24",
                    "future_t27_t29": "真正后续 t27/29"}
    lines = [
        "# P59 B1 离线误差汇总",
        "",
        "统计单位为条件：先在每个条件内平均时间帧，再对条件等权平均。已打开开发集仅是开发证据，不是新的独立测试。",
        "",
        "| 分组 | 时间窗 | 方法 | 条件数 | 几何位置 mm | 几何旋转 ° | 世界平移误差 mm | 世界旋转误差 ° |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, windows in summary["summaries"].items():
        for window, value in windows.items():
            for method, method_value in sorted(value["methods"].items()):
                mean = method_value["condition_equal_mean"]
                lines.append(
                    f"| {group_names[group]} | {window_names[window]} | {method} | "
                    f"{method_value['conditions']} | {mean['geometry_position_mm']:.6f} | "
                    f"{mean['geometry_rotation_deg']:.6f} | {mean['world_position_mm']:.6f} | "
                    f"{mean['world_rotation_deg']:.6f} |")
    lines.extend([
        "", "## 条件级配对升降计数", "",
        "差值为前者减后者，负值表示误差下降。", "",
        "| 分组 | 时间窗 | 配对 | 误差项 | 平均配对差值 | 下降 | 上升 | 持平 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    metric_names = {
        "geometry_position_mm": "几何位置 mm",
        "geometry_rotation_deg": "几何旋转 °",
        "world_position_mm": "世界平移 mm",
        "world_rotation_deg": "世界旋转 °",
    }
    for group, windows in summary["paired_comparisons"].items():
        for window, comparisons in windows.items():
            for comparison, metrics in comparisons.items():
                for metric, metric_name in metric_names.items():
                    value = metrics[metric]
                    lines.append(
                        f"| {group_names[group]} | {window_names[window]} | {comparison} | "
                        f"{metric_name} | {value['mean_paired_delta']:.6f} | "
                        f"{value['improved']} | {value['worsened']} | {value['tied']} |")
    lines.extend([
        "",
        "未从离线动作或几何误差推导任务完成比例。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-evaluation", type=Path, required=True)
    parser.add_argument("--sidecar-evaluation", type=Path, required=True)
    parser.add_argument("--main-training-freeze", type=Path, required=True)
    parser.add_argument("--main-training-seal", type=Path, required=True)
    parser.add_argument("--sidecar-training-freeze", type=Path, required=True)
    parser.add_argument("--sidecar-training-seal", type=Path, required=True)
    parser.add_argument("--sidecar-evaluation-freeze", type=Path, required=True)
    parser.add_argument("--pipeline-seal", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    args = parser.parse_args()
    main_eval, side_eval, provenance = validate_artifacts(
        args.main_evaluation, args.sidecar_evaluation,
        args.main_training_freeze, args.main_training_seal,
        args.sidecar_training_freeze, args.sidecar_training_seal,
        args.sidecar_evaluation_freeze, args.pipeline_seal)
    values = summarize_records(main_eval["records"], side_eval["records"])
    result = {
        "schema": "prove.p59-b1-offline-summary.v1",
        "statistical_unit": "condition_equal_after_within_condition_frame_mean",
        "windows": {"immediate": "t1..t24", "future": "t27,t29 without sparse adaptation"},
        "developer_identity": "opened development evidence; not a new independent test",
        "physical_success_inferred": False,
        "bootstrap": "not performed",
        "provenance": provenance,
        **values,
    }
    with args.output_json.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    args.output_table.open("x").write(chinese_table(result))


if __name__ == "__main__":
    main()
