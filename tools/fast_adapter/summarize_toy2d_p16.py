#!/usr/bin/env python3
"""Summarize Fast Adapter P16 Toy2D outputs.

P16-specific assumptions:
- Core eval raw is one row per domain/cell/arm with arrays over 8 query starts.
- Support is common frozen-base for all methods.
- Dev gate is fixed: displacement_fast S=3,U=3 correct_z vs own frozen_z.
- Calibration/heldout prediction and least-squares diagnostics never select models.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TRAIN_SEEDS_STAGE1 = [11, 22, 33]
TRAIN_SEEDS_FINAL = [11, 22, 33, 44, 55]
FAST_METHODS = {"feature_fast", "displacement_fast"}
METHODS = {"feature_fast", "displacement_fast", "history"}
FAST_CELLS = [(3, 0), (3, 1), (3, 3), (3, 5), (1, 3), (5, 3)]
HISTORY_S = [0, 1, 3, 5]
FAST_ARMS = {"base", "frozen_z", "correct_z", "swapped_z"}
HISTORY_ARMS = {"history"}
PRIMARY = {"method": "displacement_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_arm": "frozen_z"}
SPLITS = {
    "dev": {"domains": 128, "query_count": 8, "domain_seed": 410001, "support_seed": 420001, "calibration_seed": 430001, "query_seed": 440001},
    "final": {"domains": 256, "query_count": 8, "domain_seed": 510001, "support_seed": 520001, "calibration_seed": 530001, "query_seed": 540001},
}
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260922
_UNSET = object()


@dataclass(frozen=True)
class DomainMetric:
    split: str
    config_id: str
    method: str
    train_seed: int
    S: int
    U: int | None
    arm: str
    support_mode: str
    domain_seed: int
    support_seed: int
    calibration_seed: int
    query_seed: int
    domain_index: int
    query_count: int
    mean_goal_error: float
    success_rate: float
    first_query_action_mse: float | None
    action_change_vs_frozen: float | None
    query_action_mse: float | None
    delta_norm_last: float | None
    z_delta_norm: float | None
    support_loss: float | None
    alpha_value: float | None


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expand_inputs(items: list[str], filename: str) -> list[Path]:
    out: list[Path] = []
    for item in items:
        matches = [Path(p) for p in glob.glob(item)] or [Path(item)]
        for path in matches:
            if path.is_dir():
                out.extend(sorted(path.rglob(filename)))
            else:
                out.append(path)
    uniq = sorted(dict.fromkeys(p.resolve() for p in out))
    if not uniq:
        raise ValueError(f"no {filename} inputs found")
    missing = [str(p) for p in uniq if not p.exists()]
    if missing:
        raise FileNotFoundError(missing[:5])
    return uniq


def _as_list(value: Any, field: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
    raise ValueError(f"{field} must be an array")


def _bool01(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return 1.0
    if text in {"false", "f", "no", "n", "0"}:
        return 0.0
    number = float(text)
    if number not in {0.0, 1.0}:
        raise ValueError(f"success must be bool/0/1, got {value!r}")
    return number


def _finite_nonnegative(value: float, field: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative, got {value!r}")
    return value


def _finite(value: float, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return value


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot mean empty list")
    return sum(values) / len(values)


def _mean_optional(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _maybe_mean_array(row: dict[str, Any], field: str) -> float | None:
    if field not in row or row[field] in (None, ""):
        return None
    value = row[field]
    if isinstance(value, list) or (isinstance(value, str) and value.strip().startswith("[")):
        values = _as_list(value, field)
        if not values:
            return None
        return _mean([_finite_nonnegative(float(v), field) for v in values])
    return _finite_nonnegative(float(value), field)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "" or str(value).lower() in {"none", "null"}:
        return None
    return int(value)


def _row_to_metric(row: dict[str, Any], path: Path, line_no: int) -> DomainMetric:
    if "S" not in row and "cell_S" in row:
        row["S"] = row["cell_S"]
    if "U" not in row and "cell_U" in row:
        row["U"] = row["cell_U"]
    required = ["split", "config_id", "method", "train_seed", "S", "arm", "support_mode", "domain_seed", "support_seed", "calibration_seed", "query_seed", "domain_index", "query_indices", "final_goal_errors", "successes"]
    missing = [k for k in required if k not in row]
    if missing:
        raise ValueError(f"{path}:{line_no}: missing fields {missing}")
    method = str(row["method"])
    if method not in METHODS:
        raise ValueError(f"{path}:{line_no}: unknown method {method}")
    arm = str(row["arm"])
    if method in FAST_METHODS and arm not in FAST_ARMS:
        raise ValueError(f"{path}:{line_no}: unexpected fast arm {arm}")
    if method == "history" and arm not in HISTORY_ARMS:
        raise ValueError(f"{path}:{line_no}: unexpected history arm {arm}")
    errors = [_finite_nonnegative(float(v), "final_goal_errors") for v in _as_list(row["final_goal_errors"], "final_goal_errors")]
    successes = [_bool01(v) for v in _as_list(row["successes"], "successes")]
    qidx = [int(v) for v in _as_list(row["query_indices"], "query_indices")]
    if len(errors) != len(successes):
        raise ValueError(f"{path}:{line_no}: final_goal_errors/successes length mismatch")
    if qidx != list(range(len(errors))):
        raise ValueError(f"{path}:{line_no}: query_indices must be 0..{len(errors)-1}, got {qidx}")
    return DomainMetric(
        split=str(row["split"]),
        config_id=str(row.get("config_id", method)),
        method=method,
        train_seed=int(row["train_seed"]),
        S=int(row["S"]),
        U=_optional_int(row.get("U", row.get("cell_U"))),
        arm=arm,
        support_mode=str(row["support_mode"]),
        domain_seed=int(row["domain_seed"]),
        support_seed=int(row["support_seed"]),
        calibration_seed=int(row["calibration_seed"]),
        query_seed=int(row["query_seed"]),
        domain_index=int(row["domain_index"]),
        query_count=len(errors),
        mean_goal_error=_mean(errors),
        success_rate=_mean(successes),
        first_query_action_mse=_maybe_mean_array(row, "first_query_action_mse"),
        action_change_vs_frozen=_maybe_mean_array(row, "first_action_change_vs_frozen") if "first_action_change_vs_frozen" in row else _maybe_mean_array(row, "action_change_vs_frozen"),
        query_action_mse=_maybe_mean_array(row, "query_action_mse"),
        delta_norm_last=_maybe_mean_array(row, "delta_norm_last"),
        z_delta_norm=_maybe_mean_array(row, "z_delta_norm") if "z_delta_norm" in row else _maybe_mean_array(row, "latent_delta_norm"),
        support_loss=_maybe_mean_array(row, "support_loss"),
        alpha_value=_maybe_mean_array(row, "alpha_value"),
    )


def load_metrics(inputs: list[Path]) -> list[DomainMetric]:
    metrics: list[DomainMetric] = []
    seen: set[tuple[Any, ...]] = set()
    for path in inputs:
        for line_no, row in _jsonl(path):
            m = _row_to_metric(row, path, line_no)
            key = (m.split, m.config_id, m.method, m.train_seed, m.S, m.U, m.arm, m.support_mode, m.domain_index)
            if key in seen:
                raise ValueError(f"duplicate domain row {key}")
            seen.add(key)
            metrics.append(m)
    if not metrics:
        raise ValueError("no metrics")
    return metrics


def validate_metrics(metrics: list[DomainMetric], *, split: str, require_complete: bool) -> dict[str, Any]:
    exp = SPLITS[split]
    problems: list[str] = []
    if sorted({m.split for m in metrics}) != [split]:
        problems.append(f"expected split {split}, got {sorted({m.split for m in metrics})}")
    for field in ["domain_seed", "support_seed", "calibration_seed", "query_seed"]:
        vals = sorted({getattr(m, field) for m in metrics})
        if vals != [exp[field]]:
            problems.append(f"{field} expected {exp[field]}, got {vals}")
    qcounts = sorted({m.query_count for m in metrics})
    if qcounts != [exp["query_count"]]:
        problems.append(f"query_count expected {exp['query_count']}, got {qcounts}")
    modes = sorted({m.support_mode for m in metrics})
    if modes != ["common_base"]:
        problems.append(f"support_mode expected common_base, got {modes}")
    groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for m in metrics:
        if m.domain_index < 0 or m.domain_index >= exp["domains"]:
            problems.append(f"domain out of range {m.domain_index}")
        groups[(m.method, m.train_seed, m.S, m.U, m.arm)].add(m.domain_index)
    expected_domains = set(range(exp["domains"]))
    for key, domains in sorted(groups.items(), key=str):
        if domains != expected_domains:
            problems.append(f"domain IDs for {key} must be exactly 0..{exp['domains']-1}; missing={sorted(expected_domains-domains)[:5]} extra={sorted(domains-expected_domains)[:5]}")
    if require_complete:
        seeds = TRAIN_SEEDS_STAGE1 if split == "dev" else TRAIN_SEEDS_FINAL
        expected_groups: set[tuple[Any, ...]] = set()
        for seed in seeds:
            for method in ["feature_fast", "displacement_fast"]:
                for S, U in FAST_CELLS:
                    for arm in FAST_ARMS:
                        expected_groups.add((method, seed, S, U, arm))
            for S in HISTORY_S:
                expected_groups.add(("history", seed, S, None, "history"))
        actual_groups = set(groups)
        missing_groups = sorted(expected_groups - actual_groups, key=str)
        extra_groups = sorted(actual_groups - expected_groups, key=str)
        if missing_groups:
            problems.append(f"missing groups {missing_groups[:8]}")
        if extra_groups:
            problems.append(f"extra groups {extra_groups[:8]}")
    if problems:
        raise ValueError("; ".join(problems[:20]))
    return {
        "split": split,
        "domain_count": exp["domains"],
        "query_count": exp["query_count"],
        "groups": len(groups),
        "methods": sorted({m.method for m in metrics}),
        "support_mode": "common_base",
    }


def aggregate(metrics: list[DomainMetric]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[DomainMetric]] = defaultdict(list)
    for m in metrics:
        buckets[(m.split, m.config_id, m.method, m.train_seed, m.S, m.U, m.arm, m.support_mode)].append(m)
    rows: list[dict[str, Any]] = []
    for (split, config_id, method, seed, S, U, arm, support_mode), items in sorted(buckets.items(), key=str):
        rows.append({
            "split": split, "config_id": config_id, "method": method, "train_seed": seed, "S": S, "U": "" if U is None else U, "arm": arm, "support_mode": support_mode,
            "domain_count": len(items),
            "mean_goal_error": _mean([i.mean_goal_error for i in items]),
            "success_rate": _mean([i.success_rate for i in items]),
            "first_query_action_mse": _mean_optional([i.first_query_action_mse for i in items]),
            "action_change_vs_frozen": _mean_optional([i.action_change_vs_frozen for i in items]),
            "query_action_mse": _mean_optional([i.query_action_mse for i in items]),
            "delta_norm_last": _mean_optional([i.delta_norm_last for i in items]),
            "z_delta_norm": _mean_optional([i.z_delta_norm for i in items]),
            "support_loss": _mean_optional([i.support_loss for i in items]),
            "alpha_value": _mean_optional([i.alpha_value for i in items]),
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metric_map(metrics: list[DomainMetric]) -> dict[tuple[Any, ...], DomainMetric]:
    return {(m.method, m.train_seed, m.S, m.U, m.arm, m.domain_index): m for m in metrics}


def paired_by_seed(metrics: list[DomainMetric], *, method: str, S: int, U: int | None, left_arm: str, right_method: str | None = None, right_S: int | None = None, right_U: Any = _UNSET, right_arm: str = "frozen_z") -> dict[int, dict[int, tuple[float, float]]]:
    right_method = right_method or method
    right_S = S if right_S is None else right_S
    if right_U is _UNSET:
        right_U = U
    metric_map = _metric_map(metrics)
    seeds = sorted({m.train_seed for m in metrics if m.method == method} & {m.train_seed for m in metrics if m.method == right_method})
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for seed in seeds:
        left_domains = {m.domain_index for m in metrics if m.method == method and m.train_seed == seed and m.S == S and m.U == U and m.arm == left_arm}
        right_domains = {m.domain_index for m in metrics if m.method == right_method and m.train_seed == seed and m.S == right_S and m.U == right_U and m.arm == right_arm}
        if left_domains != right_domains:
            raise ValueError(f"paired domains differ seed{seed}: {method} S{S} U{U} {left_arm} vs {right_method} S{right_S} U{right_U} {right_arm}")
        if not left_domains:
            raise ValueError(f"empty paired domains seed{seed}: {method} vs {right_method}")
        out[seed] = {}
        for domain in sorted(left_domains):
            left = metric_map[(method, seed, S, U, left_arm, domain)]
            right = metric_map[(right_method, seed, right_S, right_U, right_arm, domain)]
            out[seed][domain] = (left.mean_goal_error - right.mean_goal_error, left.success_rate - right.success_rate)
    return out


def dev_gate(metrics: list[DomainMetric]) -> dict[str, Any]:
    paired = paired_by_seed(metrics, method="displacement_fast", S=3, U=3, left_arm="correct_z", right_arm="frozen_z")
    rows: list[dict[str, Any]] = []
    for seed in TRAIN_SEEDS_STAGE1:
        if seed not in paired:
            raise ValueError(f"missing dev gate seed {seed}")
        values = list(paired[seed].values())
        err = _mean([v[0] for v in values])
        sr = _mean([v[1] for v in values])
        rows.append({"train_seed": seed, "error_delta": err, "success_delta_pp": 100 * sr, "pass": bool(sr > 0 and err < 0)})
    mean_sr = _mean([r["success_delta_pp"] for r in rows])
    passed = all(r["pass"] for r in rows) and mean_sr >= 2.0
    return {
        "format": "prove.fast-adapter-p16-dev-gate.v1",
        "candidate": "displacement_fast_S3_U3_correct_z_minus_frozen_z",
        "mean_success_delta_pp": mean_sr,
        "per_seed": rows,
        "gate_pass": passed,
        "if_false": "stop dev negative/mixed; no final",
        "if_true": "train seeds 44/55 and run final without endpoint/config changes",
    }


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def bootstrap_pair(paired: dict[int, dict[int, tuple[float, float]]]) -> dict[str, Any]:
    rng = random.Random(BOOTSTRAP_SEED)
    seeds = sorted(paired)
    domains = sorted(set.intersection(*(set(paired[s]) for s in seeds)))
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        err = _mean([paired[seed][d][0] for d in domains])
        sr = _mean([paired[seed][d][1] for d in domains])
        per_seed.append({"train_seed": seed, "error_delta": err, "success_delta_pp": 100 * sr, "both": bool(sr > 0 and err < 0)})
    berr: list[float] = []
    bsr: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        sampled_domains = [domains[rng.randrange(len(domains))] for _ in domains]
        values = [paired[s][d] for s in sampled_seeds for d in sampled_domains]
        berr.append(_mean([v[0] for v in values]))
        bsr.append(100 * _mean([v[1] for v in values]))
    both = sum(1 for row in per_seed if row["both"])
    return {
        "seed_count": len(seeds), "domain_count": len(domains),
        "mean_error_delta": _mean([row["error_delta"] for row in per_seed]),
        "mean_success_delta_pp": _mean([row["success_delta_pp"] for row in per_seed]),
        "error_ci95": [percentile(berr, .025), percentile(berr, .975)],
        "success_ci95_pp": [percentile(bsr, .025), percentile(bsr, .975)],
        "per_seed": per_seed, "both_direction_count": both,
        "gate_pass": bool(percentile(bsr, .025) > 0 and percentile(berr, .975) < 0 and both >= 4),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS, "bootstrap_seed": BOOTSTRAP_SEED,
    }


def final_pairs(metrics: list[DomainMetric]) -> list[dict[str, Any]]:
    specs = [
        ("own_frozen", {"method": "displacement_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_arm": "frozen_z"}),
        ("swap", {"method": "displacement_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_arm": "swapped_z"}),
        ("history_S3", {"method": "displacement_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_method": "history", "right_S": 3, "right_U": None, "right_arm": "history"}),
    ]
    rows: list[dict[str, Any]] = []
    own_gate = False
    for name, kwargs in specs:
        stat = bootstrap_pair(paired_by_seed(metrics, **kwargs))
        row = {"comparison": name, **stat}
        rows.append(row)
        if name == "own_frozen":
            own_gate = stat["gate_pass"]
    for row in rows:
        if row["comparison"].startswith("history"):
            row["stronger_vs_history_claim_allowed"] = bool(row["gate_pass"] and own_gate)
    return rows


def load_calibration(inputs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in inputs:
        for line_no, row in _jsonl(path):
            for field in ["split", "config_id", "method", "train_seed", "S", "U", "domain_index", "arm", "prediction_mse", "prediction_units", "domain_seed", "support_seed", "calibration_seed", "query_seed"]:
                if field not in row:
                    raise ValueError(f"{path}:{line_no}: calibration row missing {field}")
            out = {
                "split": str(row["split"]), "config_id": str(row["config_id"]), "method": str(row["method"]), "train_seed": int(row["train_seed"]),
                "S": int(row["S"]), "U": _optional_int(row.get("U")), "domain_index": int(row["domain_index"]), "arm": str(row["arm"]),
                "prediction_mse": _finite_nonnegative(float(row["prediction_mse"]), "prediction_mse"), "prediction_units": str(row["prediction_units"]),
                "domain_seed": int(row["domain_seed"]), "support_seed": int(row["support_seed"]), "calibration_seed": int(row["calibration_seed"]), "query_seed": int(row["query_seed"]),
            }
            key = (out["split"], out["config_id"], out["method"], out["train_seed"], out["S"], out["U"], out["arm"], out["prediction_units"], out["domain_index"])
            if key in seen:
                raise ValueError(f"duplicate calibration row {key}")
            seen.add(key)
            rows.append(out)
    return rows


def summarize_calibration(rows: list[dict[str, Any]], *, split: str, require_complete: bool) -> list[dict[str, Any]]:
    if not rows:
        return []
    exp = SPLITS[split]
    problems: list[str] = []
    expected_domains = set(range(exp["domains"]))
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != split:
            problems.append(f"calibration mixed split {row['split']}")
        for field in ["domain_seed", "support_seed", "calibration_seed", "query_seed"]:
            if row[field] != exp[field]:
                problems.append(f"calibration {field} expected {exp[field]}, got {row[field]}")
        buckets[(row["config_id"], row["method"], row["train_seed"], row["S"], row["U"], row["arm"], row["prediction_units"])].append(row)
    for key, items in sorted(buckets.items(), key=str):
        domains = {item["domain_index"] for item in items}
        if domains != expected_domains:
            problems.append(f"calibration domains for {key} must be exactly 0..{exp['domains']-1}; missing={sorted(expected_domains-domains)[:5]} extra={sorted(domains-expected_domains)[:5]}")
    if require_complete:
        seeds = TRAIN_SEEDS_STAGE1 if split == "dev" else TRAIN_SEEDS_FINAL
        expected_groups: set[tuple[Any, ...]] = set()
        for seed in seeds:
            for method in ["feature_fast", "displacement_fast"]:
                for S, U in FAST_CELLS:
                    for arm in {"frozen_z", "correct_z", "swapped_z"}:
                        expected_groups.add((method, seed, S, U, arm))
        actual_groups = {(k[1], k[2], k[3], k[4], k[5]) for k in buckets}
        missing = sorted(expected_groups - actual_groups, key=str)
        extra = sorted(actual_groups - expected_groups, key=str)
        if missing:
            problems.append(f"calibration missing groups {missing[:8]}")
        if extra:
            problems.append(f"calibration extra groups {extra[:8]}")
    if problems:
        raise ValueError("; ".join(problems[:20]))
    summary: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items(), key=str):
        summary.append({
            "config_id": key[0], "method": key[1], "train_seed": key[2], "S": key[3], "U": "" if key[4] is None else key[4], "arm": key[5], "prediction_units": key[6],
            "domain_count": len(items), "prediction_mse": _mean([item["prediction_mse"] for item in items]),
        })
    return summary


def load_ls(inputs: list[Path], *, split: str, require_complete: bool) -> list[dict[str, Any]]:
    exp = SPLITS[split]
    expected_domains = set(range(exp["domains"]))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    problems: list[str] = []
    for path in inputs:
        for line_no, row in _jsonl(path):
            for field in ["split", "S", "U", "arm", "domain_index", "domain_seed", "support_seed", "calibration_seed", "query_seed", "query_indices", "final_goal_errors", "successes"]:
                if field not in row:
                    raise ValueError(f"{path}:{line_no}: LS row missing {field}")
            errors = [_finite_nonnegative(float(v), "ls final_goal_errors") for v in _as_list(row["final_goal_errors"], "final_goal_errors")]
            successes = [_bool01(v) for v in _as_list(row["successes"], "successes")]
            qidx = [int(v) for v in _as_list(row["query_indices"], "query_indices")]
            if qidx != list(range(len(errors))) or len(errors) != len(successes):
                raise ValueError(f"{path}:{line_no}: LS query arrays invalid")
            out = {
                "split": str(row["split"]), "config_id": str(row.get("config_id", "least_squares")), "method": "least_squares", "train_seed": int(row.get("train_seed") or 0),
                "S": int(row["S"]), "U": _optional_int(row.get("U")), "arm": str(row["arm"]), "domain_index": int(row["domain_index"]),
                "domain_seed": int(row["domain_seed"]), "support_seed": int(row["support_seed"]), "calibration_seed": int(row["calibration_seed"]), "query_seed": int(row["query_seed"]),
                "mean_goal_error": _mean(errors), "success_rate": _mean(successes),
                "heldout_displacement_mse": _maybe_mean_array(row, "heldout_displacement_mse"),
                "saturation_fraction": _maybe_mean_array(row, "saturation_fraction"),
                "first_action_change_vs_base": _maybe_mean_array(row, "first_action_change_vs_base"),
                "identifiable_rate": _mean([1.0 if bool(v) else 0.0 for v in _as_list(row.get("identifiable", []), "identifiable")]) if isinstance(row.get("identifiable"), list) else (1.0 if row.get("identifiable") is True else 0.0 if row.get("identifiable") is False else None),
                "rank_mean": _maybe_mean_array(row, "rank"),
                "condition_mean": _maybe_mean_array(row, "condition"),
            }
            if out["split"] != split:
                problems.append(f"LS mixed split {out['split']}")
            for field in ["domain_seed", "support_seed", "calibration_seed", "query_seed"]:
                if out[field] != exp[field]:
                    problems.append(f"LS {field} expected {exp[field]}, got {out[field]}")
            key = (out["split"], out["train_seed"], out["S"], out["U"], out["arm"], out["domain_index"])
            if key in seen:
                raise ValueError(f"duplicate LS row {key}")
            seen.add(key)
            rows.append(out)
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(row["train_seed"], row["S"], row["U"], row["arm"])].append(row)
    for key, items in sorted(buckets.items(), key=str):
        domains = {item["domain_index"] for item in items}
        if domains != expected_domains:
            problems.append(f"LS domains for {key} must be exactly 0..{exp['domains']-1}; missing={sorted(expected_domains-domains)[:5]} extra={sorted(domains-expected_domains)[:5]}")
    if require_complete:
        expected_s = {1, 3, 5}
        actual_s = {key[1] for key in buckets}
        if not expected_s.issubset(actual_s):
            problems.append(f"LS missing S values {sorted(expected_s - actual_s)}")
    if problems:
        raise ValueError("; ".join(problems[:20]))
    summary: list[dict[str, Any]] = []
    for key, items in sorted(buckets.items(), key=str):
        summary.append({
            "train_seed": key[0], "S": key[1], "U": "" if key[2] is None else key[2], "arm": key[3], "domain_count": len(items),
            "mean_goal_error": _mean([i["mean_goal_error"] for i in items]), "success_rate": _mean([i["success_rate"] for i in items]),
            "heldout_displacement_mse": _mean_optional([i["heldout_displacement_mse"] for i in items]),
            "saturation_fraction": _mean_optional([i["saturation_fraction"] for i in items]),
            "first_action_change_vs_base": _mean_optional([i["first_action_change_vs_base"] for i in items]),
            "identifiable_rate": _mean_optional([i["identifiable_rate"] for i in items]),
            "rank_mean": _mean_optional([i["rank_mean"] for i in items]),
            "condition_mean": _mean_optional([i["condition_mean"] for i in items]),
        })
    return summary


def _group_mean(rows: list[dict[str, Any]], *, method: str, S: int | None = None, U: int | None | object = _UNSET, arm: str | None = None) -> float | None:
    vals = []
    for row in rows:
        if row.get("method") != method:
            continue
        if S is not None and int(row.get("S")) != S:
            continue
        row_u = None if row.get("U") == "" else row.get("U")
        if U is not _UNSET and row_u != U:
            continue
        if arm is not None and row.get("arm") != arm:
            continue
        vals.append(float(row["success_rate"]))
    return _mean(vals) if vals else None



def primary_outcome_rows(agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("feature_fast", 3, 3, "frozen_z", "feature frozen S3/U3"),
        ("feature_fast", 3, 3, "correct_z", "feature correct S3/U3"),
        ("feature_fast", 3, 3, "swapped_z", "feature swapped S3/U3"),
        ("displacement_fast", 3, 3, "frozen_z", "physical frozen S3/U3"),
        ("displacement_fast", 3, 3, "correct_z", "physical correct S3/U3"),
        ("displacement_fast", 3, 3, "swapped_z", "physical swapped S3/U3"),
        ("history", 3, None, "history", "history S3"),
    ]
    rows: list[dict[str, Any]] = []
    for method, S, U, arm, label in specs:
        items = [r for r in agg if r["method"] == method and int(r["S"]) == S and (None if r["U"] == "" else r["U"]) == U and r["arm"] == arm]
        if not items:
            continue
        rows.append({
            "label": label,
            "method": method,
            "S": S,
            "U": "" if U is None else U,
            "arm": arm,
            "seed_count": len(items),
            "success_rate": _mean([float(r["success_rate"]) for r in items]),
            "success_percent": 100 * _mean([float(r["success_rate"]) for r in items]),
            "mean_goal_error": _mean([float(r["mean_goal_error"]) for r in items]),
            "first_query_action_mse": _mean_optional([None if r.get("first_query_action_mse") in (None, "") else float(r["first_query_action_mse"]) for r in items]),
            "action_change_vs_frozen": _mean_optional([None if r.get("action_change_vs_frozen") in (None, "") else float(r["action_change_vs_frozen"]) for r in items]),
            "query_action_mse": _mean_optional([None if r.get("query_action_mse") in (None, "") else float(r["query_action_mse"]) for r in items]),
            "z_delta_norm": _mean_optional([None if r.get("z_delta_norm") in (None, "") else float(r["z_delta_norm"]) for r in items]),
        })
    return rows


def seed_direction_rows(metrics: list[DomainMetric]) -> list[dict[str, Any]]:
    specs = [
        ("physical_correct_minus_frozen", {"method": "displacement_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_arm": "frozen_z"}),
        ("physical_correct_minus_swapped", {"method": "displacement_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_arm": "swapped_z"}),
        ("physical_correct_minus_historyS3", {"method": "displacement_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_method": "history", "right_S": 3, "right_U": None, "right_arm": "history"}),
        ("feature_correct_minus_frozen", {"method": "feature_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_arm": "frozen_z"}),
        ("feature_correct_minus_swapped", {"method": "feature_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_arm": "swapped_z"}),
        ("feature_correct_minus_historyS3", {"method": "feature_fast", "S": 3, "U": 3, "left_arm": "correct_z", "right_method": "history", "right_S": 3, "right_U": None, "right_arm": "history"}),
    ]
    rows: list[dict[str, Any]] = []
    for name, kwargs in specs:
        try:
            paired = paired_by_seed(metrics, **kwargs)
        except ValueError:
            continue
        for seed in sorted(paired):
            vals = list(paired[seed].values())
            err = _mean([v[0] for v in vals])
            sr = _mean([v[1] for v in vals])
            rows.append({"comparison": name, "train_seed": seed, "error_delta": err, "success_delta_pp": 100 * sr, "both_direction": bool(sr > 0 and err < 0)})
    return rows


def prediction_ratio_rows(calibration: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ["feature_fast", "displacement_fast"]:
        for seed in sorted({int(r["train_seed"]) for r in calibration if r["method"] == method}):
            for S, U in [(3, 3)]:
                frozen = [float(r["prediction_mse"]) for r in calibration if r["method"] == method and int(r["train_seed"]) == seed and int(r["S"]) == S and r["U"] == U and r["arm"] == "frozen_z"]
                correct = [float(r["prediction_mse"]) for r in calibration if r["method"] == method and int(r["train_seed"]) == seed and int(r["S"]) == S and r["U"] == U and r["arm"] == "correct_z"]
                swapped = [float(r["prediction_mse"]) for r in calibration if r["method"] == method and int(r["train_seed"]) == seed and int(r["S"]) == S and r["U"] == U and r["arm"] == "swapped_z"]
                if frozen and correct:
                    f = _mean(frozen)
                    c = _mean(correct)
                    rows.append({"method": method, "train_seed": seed, "S": S, "U": U, "arm": "correct_z", "prediction_mse": c, "frozen_prediction_mse": f, "ratio_vs_frozen": c / f if f > 0 else ""})
                if frozen and swapped:
                    f = _mean(frozen)
                    sw = _mean(swapped)
                    rows.append({"method": method, "train_seed": seed, "S": S, "U": U, "arm": "swapped_z", "prediction_mse": sw, "frozen_prediction_mse": f, "ratio_vs_frozen": sw / f if f > 0 else ""})
    return rows

def make_plots(outdir: Path, agg: list[dict[str, Any]], calibration: list[dict[str, Any]], ls_summary: list[dict[str, Any]]) -> list[dict[str, str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (outdir / "PLOT_ERROR.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return []
    artifacts: list[dict[str, str]] = []

    def save(fig: Any, filename: str, title: str) -> None:
        path = outdir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        artifacts.append({"label": title, "path": str(path)})

    primary = [r for r in agg if r["method"] == "displacement_fast" and int(r["S"]) == 3 and r["arm"] == "correct_z"]
    us = sorted({int(r["U"]) for r in primary if r["U"] != ""})
    if us:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for metric, ax, ylabel in [("success_rate", axes[0], "success"), ("first_query_action_mse", axes[1], "first-action MSE")]:
            ys=[]
            for u in us:
                vals=[float(r[metric]) for r in primary if int(r["U"]) == u and r[metric] not in (None, "")]
                ys.append((100 * _mean(vals) if metric == "success_rate" else _mean(vals)) if vals else math.nan)
            ax.plot(us, ys, marker="o", label="displacement correct_z S3, 5-seed mean")
            ax.set_xlabel("U passes at fixed S=3 records")
            ax.set_ylabel("success (%)" if metric == "success_rate" else ylabel)
            ax.grid(True, alpha=.25)
        axes[0].legend(fontsize=8)
        save(fig, "p16_success_first_action_vs_U_S3.png", "success percent + first-action error vs U at fixed S3, 5-seed mean")

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    s_values = sorted({int(r["S"]) for r in agg if r["method"] == "displacement_fast" and r["arm"] == "correct_z" and r["U"] == 3})
    if s_values:
        y=[100 * _mean([float(r["success_rate"]) for r in agg if r["method"] == "displacement_fast" and r["arm"] == "correct_z" and r["U"] == 3 and int(r["S"]) == s]) for s in s_values]
        ax.plot(s_values, y, marker="o", label="displacement correct_z U3, 5-seed mean")
    h_s = sorted({int(r["S"]) for r in agg if r["method"] == "history"})
    if h_s:
        y=[100 * _mean([float(r["success_rate"]) for r in agg if r["method"] == "history" and int(r["S"]) == s]) for s in h_s]
        ax.plot(h_s, y, marker="s", label="history, 5-seed mean")
    ax.set_xlabel("S fixed frozen-base support records")
    ax.set_ylabel("success (%)")
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8)
    save(fig, "p16_success_vs_S_U3_history.png", "success percent vs S at U3 with history, 5-seed mean")

    if calibration:
        fig, ax = plt.subplots(figsize=(6.2, 3.5))
        rows=[r for r in calibration if r["method"] == "displacement_fast" and r["arm"] == "correct_z" and int(r["S"]) == 3]
        for seed in sorted({int(r["train_seed"]) for r in rows}):
            sr=[r for r in rows if int(r["train_seed"]) == seed]
            uvals=sorted({int(r["U"]) for r in sr if r["U"] != ""})
            if not uvals:
                continue
            raw=[_mean([float(r["prediction_mse"]) for r in sr if int(r["U"]) == u]) for u in uvals]
            base=raw[0] if raw and raw[0] > 0 else 1.0
            ax.plot(uvals, [v/base for v in raw], marker="o", label=f"seed{seed}")
        ax.set_xlabel("U passes at S=3")
        ax.set_ylabel("heldout prediction MSE / first U value")
        ax.set_title("same-checkpoint heldout prediction, per-seed normalized")
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=8, ncol=2)
        save(fig, "p16_heldout_prediction_normalized_by_seed.png", "per-seed normalized heldout prediction")

    if ls_summary:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
        svals=sorted({int(r["S"]) for r in ls_summary})
        for metric, ax, ylabel in [("success_rate", axes[0], "LS success"), ("heldout_displacement_mse", axes[1], "heldout displacement MSE")]:
            y=[]
            for s in svals:
                vals=[float(r[metric]) for r in ls_summary if int(r["S"]) == s and r[metric] not in (None, "")]
                y.append(_mean(vals) if vals else math.nan)
            ax.plot(svals, y, marker="o")
            ax.set_xlabel("S support records")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=.25)
        save(fig, "p16_least_squares_diagnostic.png", "least-squares diagnostic")
    selected = primary_outcome_rows(agg)
    if selected:
        labels = [r["label"].replace(" S3/U3", "").replace("history S3", "history") for r in selected]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        axes[0].bar(range(len(selected)), [r["success_percent"] for r in selected])
        axes[0].set_ylabel("success (%)")
        axes[0].set_xticks(range(len(selected)), labels, rotation=35, ha="right")
        axes[0].grid(True, axis="y", alpha=.25)
        axes[1].bar(range(len(selected)), [r["mean_goal_error"] for r in selected])
        axes[1].set_ylabel("mean goal error")
        axes[1].set_xticks(range(len(selected)), labels, rotation=35, ha="right")
        axes[1].grid(True, axis="y", alpha=.25)
        save(fig, "p16_primary_s3u3_absolute.png", "primary S3/U3 absolute success and goal error")
    return artifacts


def write_report(path: Path, *, split: str, coverage: dict[str, Any], gate: dict[str, Any] | None, final_stats: list[dict[str, Any]] | None, plots: list[dict[str, str]], ls_summary: list[dict[str, Any]], primary_rows: list[dict[str, Any]], direction_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> None:
    lines = ["# Fast Adapter P16 Toy2D 汇总", "", f"split: `{split}`。P16 仍是 2D toy 同分布机制实验，不是 robot/RGB/Insert/OOD/完整 RSI 证据。", ""]
    lines.append("## 覆盖")
    lines.append(f"- methods: {coverage.get('methods')}；domain_count: {coverage.get('domain_count')}；groups: {coverage.get('groups')}；support_mode: `{coverage.get('support_mode')}`。")
    lines.append("")

    if primary_rows:
        lines.append("## 主 S3/U3 绝对结果")
        lines.append("| 条目 | Success | Goal error | First-action MSE | Action change vs frozen | z delta |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in primary_rows:
            lines.append(f"| {row['label']} | {row['success_percent']:.3f}% | {row['mean_goal_error']:.6f} | {row.get('first_query_action_mse') if row.get('first_query_action_mse') is not None else ''} | {row.get('action_change_vs_frozen') if row.get('action_change_vs_frozen') is not None else ''} | {row.get('z_delta_norm') if row.get('z_delta_norm') is not None else ''} |")
        lines.append("")
    if direction_rows:
        lines.append("## 每 seed 配对方向")
        lines.append("| comparison | seed | error delta | success delta | both direction |")
        lines.append("|---|---:|---:|---:|---|")
        for row in direction_rows:
            lines.append(f"| {row['comparison']} | {row['train_seed']} | {row['error_delta']:.6f} | {row['success_delta_pp']:.3f} pp | {row['both_direction']} |")
        lines.append("")
    if prediction_rows:
        lines.append("## 同 checkpoint heldout prediction 诊断")
        lines.append("该表是 frozen-base heldout transition 的 prediction MSE 相对本方法同 seed frozen_z 的 ratio。它不参与 gate，也不能单独解释控制提升。")
        lines.append("| method | seed | arm | MSE | ratio vs frozen |")
        lines.append("|---|---:|---|---:|---:|")
        for row in prediction_rows:
            ratio = row.get('ratio_vs_frozen')
            ratio_text = f"{ratio:.6f}" if isinstance(ratio, float) else ""
            lines.append(f"| {row['method']} | {row['train_seed']} | {row['arm']} | {row['prediction_mse']:.6f} | {ratio_text} |")
        lines.append("")
    if gate:
        lines.append("## Dev gate")
        lines.append(f"- candidate: `{gate['candidate']}`")
        lines.append(f"- mean SR delta: {gate['mean_success_delta_pp']:.4f} pp；gate_pass: `{gate['gate_pass']}`")
        lines.append("| seed | error delta | success delta | pass |")
        lines.append("|---:|---:|---:|---|")
        for row in gate["per_seed"]:
            lines.append(f"| {row['train_seed']} | {row['error_delta']:.6f} | {row['success_delta_pp']:.4f} pp | {row['pass']} |")
        lines.append("")
    if final_stats:
        lines.append("## Final paired gates")
        lines.append("| comparison | error delta | error CI | success delta | success CI | seed both | gate |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in final_stats:
            lines.append(f"| {row['comparison']} | {row['mean_error_delta']:.6f} | [{row['error_ci95'][0]:.6f}, {row['error_ci95'][1]:.6f}] | {row['mean_success_delta_pp']:.4f} pp | [{row['success_ci95_pp'][0]:.4f}, {row['success_ci95_pp'][1]:.4f}] pp | {row['both_direction_count']}/5 | {row['gate_pass']} |")
        lines.append("")
    if plots:
        lines.append("## 图表")
        for art in plots:
            lines.append(f"- {art['label']}: `{Path(art['path']).name}`")
        lines.append("")
    if ls_summary:
        lines.append("## LS diagnostic")
        lines.append("LS 只检查证据可辨识性与有界控制容量，不参与训练、选择或 gate。")
        lines.append("| S | success | goal error | heldout displacement MSE | saturation |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in ls_summary:
            lines.append(f"| {row['S']} | {row['success_rate']:.6f} | {row['mean_goal_error']:.6f} | {row.get('heldout_displacement_mse') if row.get('heldout_displacement_mse') is not None else ''} | {row.get('saturation_fraction') if row.get('saturation_fraction') is not None else ''} |")
        lines.append("")
    lines.append("Feature prediction、physical prediction、first-action metrics、z delta、alpha 和 LS 诊断均为解释性指标；dev gate 只使用固定的 displacement_fast S3/U3 correct_z 相对 own frozen_z 的 SR/error。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["dev", "final"], required=True)
    parser.add_argument("--require-complete-grid", action="store_true")
    parser.add_argument("--dev-gate-receipt", type=Path)
    parser.add_argument("--calibration-inputs", nargs="*", default=[])
    parser.add_argument("--ls-inputs", nargs="*", default=[])
    args = parser.parse_args()

    inputs = _expand_inputs(args.inputs, "eval_raw.jsonl")
    metrics = load_metrics(inputs)
    coverage = validate_metrics(metrics, split=args.split, require_complete=args.require_complete_grid)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate(metrics)
    _write_csv(args.output_dir / "aggregate_by_method_seed_cell_arm.csv", agg)
    primary_rows = primary_outcome_rows(agg)
    _write_csv(args.output_dir / "primary_s3u3_absolute.csv", primary_rows)
    direction_rows = seed_direction_rows(metrics)
    _write_csv(args.output_dir / "paired_seed_directions.csv", direction_rows)

    cal_summary: list[dict[str, Any]] = []
    if args.calibration_inputs:
        cal_inputs = _expand_inputs(args.calibration_inputs, "calibration_prediction_loss.jsonl")
        cal_summary = summarize_calibration(load_calibration(cal_inputs), split=args.split, require_complete=args.require_complete_grid)
        _write_csv(args.output_dir / "calibration_prediction_loss.csv", cal_summary)
    else:
        cal_inputs = []
    prediction_rows = prediction_ratio_rows(cal_summary)
    _write_csv(args.output_dir / "prediction_ratio_vs_frozen_s3u3.csv", prediction_rows)

    ls_summary: list[dict[str, Any]] = []
    if args.ls_inputs:
        ls_inputs = _expand_inputs(args.ls_inputs, "ls_raw.jsonl")
        ls_summary = load_ls(ls_inputs, split=args.split, require_complete=args.require_complete_grid)
        _write_csv(args.output_dir / "least_squares_diagnostic.csv", ls_summary)
    else:
        ls_inputs = []

    gate = None
    final_stats = None
    source_hashes = {"eval_raw": {str(p): _sha256(p) for p in inputs}}
    if cal_inputs:
        source_hashes["calibration_prediction_loss"] = {str(p): _sha256(p) for p in cal_inputs}
    if ls_inputs:
        source_hashes["least_squares"] = {str(p): _sha256(p) for p in ls_inputs}

    if args.split == "dev":
        gate = dev_gate(metrics)
        gate["source_hashes"] = source_hashes
        (args.output_dir / "DEV_GATE_RECEIPT.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        if not args.dev_gate_receipt:
            raise ValueError("final summary requires --dev-gate-receipt")
        gate = json.loads(args.dev_gate_receipt.read_text(encoding="utf-8"))
        if not gate.get("gate_pass"):
            raise ValueError("dev gate did not pass; final summary is not authorized")
        final_stats = final_pairs(metrics)
        (args.output_dir / "FINAL_PAIRED_BOOTSTRAP.json").write_text(json.dumps(final_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plots = make_plots(args.output_dir, agg, cal_summary, ls_summary)
    summary = {
        "format": "prove.fast-adapter-p16-summary.v1",
        "split": args.split,
        "input_files": [str(p) for p in inputs],
        "coverage": coverage,
        "calibration_rows": len(cal_summary),
        "least_squares_rows": len(ls_summary),
        "plots": plots,
        "primary_rows": primary_rows,
        "prediction_ratio_rows": prediction_rows,
        "dev_gate": gate,
        "final_pairs": final_stats,
        "source_hashes": source_hashes,
        "notes": [
            "P16 feature_fast is an aligned-feedback reference, not a clean P15 feedback-source-only ablation.",
            "displacement_fast is a target+auxiliary-supervision package, not pure target/parameter-matched proof.",
            "Calibration and heldout prediction never select stage1.",
        ],
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "REPORT_ZH.md", split=args.split, coverage=coverage, gate=gate, final_stats=final_stats, plots=plots, ls_summary=ls_summary, primary_rows=primary_rows, direction_rows=direction_rows, prediction_rows=prediction_rows)


if __name__ == "__main__":
    main()
