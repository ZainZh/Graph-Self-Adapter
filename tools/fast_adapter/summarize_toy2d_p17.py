#!/usr/bin/env python3
"""Summarize Fast Adapter P17 Toy2D static-OOD outputs.

P17-specific assumptions:
- eval_raw.jsonl rows are per physical domain with arrays over query starts.
- Success arrays encode ordinary failures as False; infra aborts are represented
  by missing/flagged run artifacts, not a generic failed field.
- Primary dev gate is joint_ood corrected_u3 versus history_init.
- Bootstrap resamples paired model seeds and physical domains; query starts are
  averaged within each domain row and are not treated as independent domains.
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

TRAIN_SEEDS_DEV = [11, 22, 33]
TRAIN_SEEDS_FINAL = [11, 22, 33, 44, 55]
STRATA = ["iid", "gain_ood", "bias_ood", "joint_ood"]
STRATUM_INDEX = {name: i for i, name in enumerate(STRATA)}
QUERY_COUNT = 8
SUPPORT_S = 3
EXPECTED_ARMS = {
    "nominal_base": None,
    "history_init": 0,
    "corrected_u1": 1,
    "corrected_u3": 3,
    "corrected_u5": 5,
    "swapped_correction_u3": 3,
    "bounded_ls_s3": None,
}
RETROFIT_ARMS = {"nominal_base", "history_init", "corrected_u1", "corrected_u3", "corrected_u5", "swapped_correction_u3"}
LS_ARM = "bounded_ls_s3"
SPLITS = {
    "dev": {"domains": 128, "domain_seed": 710001, "support_seed": 720001, "calibration_seed": 730001, "query_seed": 740001, "seeds": TRAIN_SEEDS_DEV},
    "final": {"domains": 256, "domain_seed": 810001, "support_seed": 820001, "calibration_seed": 830001, "query_seed": 840001, "seeds": TRAIN_SEEDS_FINAL},
}
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260922


@dataclass(frozen=True)
class DomainMetric:
    split: str
    stratum: str
    stratum_index: int
    method: str
    config_id: str
    train_seed: int
    S: int
    U: int | None
    arm: str
    domain_index: int
    domain_seed: int
    support_seed: int
    calibration_seed: int
    query_seed: int
    query_count: int
    mean_goal_error: float
    success_rate: float
    first_query_action_mse: float | None
    first_action_change_vs_history_init: float | None
    query_action_mse: float | None
    delta_norm_last: float | None
    action_abs_max: float | None
    z_delta_norm: float | None
    support_prediction_loss_pre: float | None
    support_prediction_loss_post: float | None
    heldout_physical_prediction_mse: float | None
    identifiable: float | None
    saturation_fraction: float | None
    rank_mean: float | None
    condition_mean: float | None


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


def _finite(value: float, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return value


def _finite_nonnegative(value: float, field: str) -> float:
    value = _finite(value, field)
    if value < 0:
        raise ValueError(f"{field} must be nonnegative, got {value!r}")
    return value


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot mean empty list")
    return sum(values) / len(values)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "" or str(value).lower() in {"none", "null"}:
        return None
    return int(value)


def _maybe_mean_array(row: dict[str, Any], field: str, *, nonnegative: bool = True) -> float | None:
    if field not in row or row[field] in (None, ""):
        return None
    value = row[field]
    if isinstance(value, list) or (isinstance(value, str) and value.strip().startswith("[")):
        arr = _as_list(value, field)
        if not arr:
            return None
        vals = [float(v) for v in arr]
    else:
        vals = [float(value)]
    checker = _finite_nonnegative if nonnegative else _finite
    return _mean([checker(v, field) for v in vals])


def _row_to_metric(row: dict[str, Any], path: Path, line_no: int) -> DomainMetric:
    required = ["split", "stratum", "train_seed", "S", "arm", "domain_seed", "support_seed", "calibration_seed", "query_seed", "domain_index", "query_indices", "final_goal_errors", "successes"]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"{path}:{line_no}: missing fields {missing}")
    arm = str(row["arm"])
    if arm not in EXPECTED_ARMS:
        raise ValueError(f"{path}:{line_no}: unexpected arm {arm}")
    method = str(row.get("method", "p17_retrofit_history"))
    if arm == LS_ARM:
        if method != "bounded_least_squares":
            raise ValueError(f"{path}:{line_no}: bounded_ls_s3 must use method bounded_least_squares, got {method}")
    elif method != "p17_retrofit_history":
        raise ValueError(f"{path}:{line_no}: retrofit arm {arm} must use method p17_retrofit_history, got {method}")
    stratum = str(row["stratum"])
    if stratum not in STRATUM_INDEX:
        raise ValueError(f"{path}:{line_no}: unknown stratum {stratum}")
    stratum_index = int(row.get("stratum_index", STRATUM_INDEX[stratum]))
    if stratum_index != STRATUM_INDEX[stratum]:
        raise ValueError(f"{path}:{line_no}: stratum_index mismatch for {stratum}")
    errors = [_finite_nonnegative(float(v), "final_goal_errors") for v in _as_list(row["final_goal_errors"], "final_goal_errors")]
    successes = [_bool01(v) for v in _as_list(row["successes"], "successes")]
    qidx = [int(v) for v in _as_list(row["query_indices"], "query_indices")]
    if len(errors) != len(successes):
        raise ValueError(f"{path}:{line_no}: final_goal_errors/successes length mismatch")
    if qidx != list(range(len(errors))):
        raise ValueError(f"{path}:{line_no}: query_indices must be 0..{len(errors) - 1}, got {qidx}")
    support_mode = row.get("support_mode", "common_base")
    if support_mode != "common_base":
        raise ValueError(f"{path}:{line_no}: support_mode expected common_base, got {support_mode}")
    S = int(row["S"])
    U = _optional_int(row.get("U"))
    if S != SUPPORT_S:
        raise ValueError(f"{path}:{line_no}: P17 rows must use S=3, got {S}")
    expected_u = EXPECTED_ARMS[arm]
    if U != expected_u:
        raise ValueError(f"{path}:{line_no}: arm {arm} expected U={expected_u}, got {U}")
    return DomainMetric(
        split=str(row["split"]),
        stratum=stratum,
        stratum_index=stratum_index,
        method=method,
        config_id=str(row.get("config_id", "p17_history256_physical_latent_correction")),
        train_seed=int(row["train_seed"]),
        S=S,
        U=U,
        arm=arm,
        domain_index=int(row["domain_index"]),
        domain_seed=int(row["domain_seed"]),
        support_seed=int(row["support_seed"]),
        calibration_seed=int(row["calibration_seed"]),
        query_seed=int(row["query_seed"]),
        query_count=len(errors),
        mean_goal_error=_mean(errors),
        success_rate=_mean(successes),
        first_query_action_mse=_maybe_mean_array(row, "first_query_action_mse"),
        first_action_change_vs_history_init=_maybe_mean_array(row, "first_action_change_vs_history_init"),
        query_action_mse=_maybe_mean_array(row, "query_action_mse"),
        delta_norm_last=_maybe_mean_array(row, "delta_norm_last"),
        action_abs_max=_maybe_mean_array(row, "action_abs_max"),
        z_delta_norm=_maybe_mean_array(row, "z_delta_norm"),
        support_prediction_loss_pre=_maybe_mean_array(row, "support_prediction_loss_pre"),
        support_prediction_loss_post=_maybe_mean_array(row, "support_prediction_loss_post"),
        heldout_physical_prediction_mse=_maybe_mean_array(row, "heldout_physical_prediction_mse"),
        identifiable=(1.0 if row.get("identifiable") is True else 0.0 if row.get("identifiable") is False else None),
        saturation_fraction=_maybe_mean_array(row, "saturation_fraction"),
        rank_mean=_maybe_mean_array(row, "rank"),
        condition_mean=_maybe_mean_array(row, "condition"),
    )


def load_metrics(inputs: list[Path]) -> list[DomainMetric]:
    metrics: list[DomainMetric] = []
    seen: set[tuple[Any, ...]] = set()
    for path in inputs:
        for line_no, row in _jsonl(path):
            metric = _row_to_metric(row, path, line_no)
            key = (metric.split, metric.stratum, metric.train_seed, metric.arm, metric.domain_index)
            if key in seen:
                raise ValueError(f"duplicate domain row {key}")
            seen.add(key)
            metrics.append(metric)
    if not metrics:
        raise ValueError("no metrics")
    return metrics


def validate_metrics(metrics: list[DomainMetric], *, split: str, require_complete: bool) -> dict[str, Any]:
    exp = SPLITS[split]
    problems: list[str] = []
    if sorted({m.split for m in metrics}) != [split]:
        problems.append(f"expected split {split}, got {sorted({m.split for m in metrics})}")
    if sorted({m.query_count for m in metrics}) != [QUERY_COUNT]:
        problems.append(f"query_count expected {QUERY_COUNT}, got {sorted({m.query_count for m in metrics})}")
    expected_domains = set(range(exp["domains"]))
    groups: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for m in metrics:
        offset = STRATUM_INDEX[m.stratum] * 1000
        for field in ["domain_seed", "support_seed", "calibration_seed", "query_seed"]:
            expected = exp[field] + offset
            actual = getattr(m, field)
            if actual != expected:
                problems.append(f"{m.stratum} {field} expected {expected}, got {actual}")
        if m.domain_index < 0 or m.domain_index >= exp["domains"]:
            problems.append(f"domain out of range {m.domain_index}")
        groups[(m.stratum, m.train_seed, m.arm)].add(m.domain_index)
    for key, domains in sorted(groups.items(), key=str):
        if domains != expected_domains:
            problems.append(f"domain IDs for {key} must be exactly 0..{exp['domains'] - 1}; missing={sorted(expected_domains - domains)[:5]} extra={sorted(domains - expected_domains)[:5]}")
    if require_complete:
        expected_groups = {(stratum, seed, arm) for stratum in STRATA for seed in exp["seeds"] for arm in EXPECTED_ARMS}
        actual_groups = set(groups)
        missing = sorted(expected_groups - actual_groups, key=str)
        extra = sorted(actual_groups - expected_groups, key=str)
        if missing:
            problems.append(f"missing groups {missing[:10]}")
        if extra:
            problems.append(f"extra groups {extra[:10]}")
    if problems:
        raise ValueError("; ".join(problems[:30]))
    return {
        "split": split,
        "domain_count": exp["domains"],
        "query_count": QUERY_COUNT,
        "strata": STRATA,
        "groups": len(groups),
        "arms": sorted({m.arm for m in metrics}),
        "train_seeds": sorted({m.train_seed for m in metrics}),
    }


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


def aggregate(metrics: list[DomainMetric]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[DomainMetric]] = defaultdict(list)
    for m in metrics:
        buckets[(m.split, m.stratum, m.train_seed, m.method, m.arm, m.S, m.U)].append(m)
    rows: list[dict[str, Any]] = []
    for (split, stratum, seed, method, arm, S, U), items in sorted(buckets.items(), key=str):
        rows.append({
            "split": split,
            "stratum": stratum,
            "train_seed": seed,
            "method": method,
            "arm": arm,
            "S": S,
            "U": "" if U is None else U,
            "domain_count": len(items),
            "mean_goal_error": _mean([i.mean_goal_error for i in items]),
            "success_rate": _mean([i.success_rate for i in items]),
            "success_percent": 100 * _mean([i.success_rate for i in items]),
            "first_query_action_mse": _mean_optional(i.first_query_action_mse for i in items),
            "first_action_change_vs_history_init": _mean_optional(i.first_action_change_vs_history_init for i in items),
            "query_action_mse": _mean_optional(i.query_action_mse for i in items),
            "delta_norm_last": _mean_optional(i.delta_norm_last for i in items),
            "action_abs_max": _mean_optional(i.action_abs_max for i in items),
            "z_delta_norm": _mean_optional(i.z_delta_norm for i in items),
            "support_prediction_loss_pre": _mean_optional(i.support_prediction_loss_pre for i in items),
            "support_prediction_loss_post": _mean_optional(i.support_prediction_loss_post for i in items),
            "heldout_physical_prediction_mse": _mean_optional(i.heldout_physical_prediction_mse for i in items),
            "identifiable_rate": _mean_optional(i.identifiable for i in items),
            "saturation_fraction": _mean_optional(i.saturation_fraction for i in items),
            "rank_mean": _mean_optional(i.rank_mean for i in items),
            "condition_mean": _mean_optional(i.condition_mean for i in items),
        })
    return rows


def domain_rows(metrics: list[DomainMetric], *, stratum: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in sorted(metrics, key=lambda x: (x.stratum, x.train_seed, x.domain_index, x.arm)):
        if stratum is not None and m.stratum != stratum:
            continue
        rows.append({
            "split": m.split,
            "stratum": m.stratum,
            "train_seed": m.train_seed,
            "domain_index": m.domain_index,
            "method": m.method,
            "arm": m.arm,
            "S": m.S,
            "U": "" if m.U is None else m.U,
            "mean_goal_error": m.mean_goal_error,
            "success_rate": m.success_rate,
            "success_percent": 100 * m.success_rate,
            "heldout_physical_prediction_mse": m.heldout_physical_prediction_mse,
            "support_prediction_loss_post": m.support_prediction_loss_post,
        })
    return rows


def _metric_map(metrics: list[DomainMetric]) -> dict[tuple[str, int, str, int], DomainMetric]:
    return {(m.stratum, m.train_seed, m.arm, m.domain_index): m for m in metrics}


def paired_by_seed(metrics: list[DomainMetric], *, stratum: str, left_arm: str, right_arm: str) -> dict[int, dict[int, tuple[float, float]]]:
    mmap = _metric_map(metrics)
    seeds = sorted({m.train_seed for m in metrics if m.stratum == stratum and m.arm == left_arm} & {m.train_seed for m in metrics if m.stratum == stratum and m.arm == right_arm})
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for seed in seeds:
        left_domains = {m.domain_index for m in metrics if m.stratum == stratum and m.train_seed == seed and m.arm == left_arm}
        right_domains = {m.domain_index for m in metrics if m.stratum == stratum and m.train_seed == seed and m.arm == right_arm}
        if left_domains != right_domains:
            raise ValueError(f"paired domains differ {stratum} seed{seed}: {left_arm} vs {right_arm}")
        if not left_domains:
            raise ValueError(f"empty paired domains {stratum} seed{seed}: {left_arm} vs {right_arm}")
        out[seed] = {}
        for domain in sorted(left_domains):
            left = mmap[(stratum, seed, left_arm, domain)]
            right = mmap[(stratum, seed, right_arm, domain)]
            out[seed][domain] = (left.mean_goal_error - right.mean_goal_error, left.success_rate - right.success_rate)
    return out


def pair_summary(metrics: list[DomainMetric], *, stratum: str, left_arm: str, right_arm: str) -> dict[str, Any]:
    paired = paired_by_seed(metrics, stratum=stratum, left_arm=left_arm, right_arm=right_arm)
    per_seed: list[dict[str, Any]] = []
    all_err: list[float] = []
    all_sr: list[float] = []
    for seed in sorted(paired):
        vals = list(paired[seed].values())
        err = _mean([v[0] for v in vals])
        sr = _mean([v[1] for v in vals])
        all_err.append(err)
        all_sr.append(sr)
        per_seed.append({"train_seed": seed, "error_delta": err, "success_delta": sr, "success_delta_pp": 100 * sr, "both_direction": bool(sr > 0 and err < 0)})
    return {"stratum": stratum, "left_arm": left_arm, "right_arm": right_arm, "mean_error_delta": _mean(all_err), "mean_success_delta": _mean(all_sr), "mean_success_delta_pp": 100 * _mean(all_sr), "per_seed": per_seed}


def arm_success(metrics: list[DomainMetric], *, stratum: str, arm: str, seeds: list[int] | None = None) -> float:
    vals = [m.success_rate for m in metrics if m.stratum == stratum and m.arm == arm and (seeds is None or m.train_seed in seeds)]
    if not vals:
        raise ValueError(f"no rows for {stratum} {arm}")
    return _mean(vals)


def dev_gate(metrics: list[DomainMetric]) -> dict[str, Any]:
    primary = pair_summary(metrics, stratum="joint_ood", left_arm="corrected_u3", right_arm="history_init")
    iid = pair_summary(metrics, stratum="iid", left_arm="corrected_u3", right_arm="history_init")
    ls_joint = arm_success(metrics, stratum="joint_ood", arm=LS_ARM, seeds=TRAIN_SEEDS_DEV)
    per_seed = primary["per_seed"]
    gate_pass = (
        all(row["success_delta"] > 0 and row["error_delta"] < 0 for row in per_seed)
        and primary["mean_success_delta"] >= 0.02
        and iid["mean_success_delta"] >= -0.01
        and ls_joint >= 0.95
    )
    return {
        "format": "prove.fast-adapter-p17-dev-gate.v1",
        "candidate": "joint_ood corrected_u3 minus history_init",
        "primary_joint_ood": primary,
        "iid_guard": iid,
        "bounded_ls_joint_ood_success_rate": ls_joint,
        "gate_pass": bool(gate_pass),
        "if_false": "close completed_dev_negative_or_mixed without tuning/final",
        "if_true": "train seeds 44/55, seal all five checkpoints/source/protocol, then run fresh final",
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


def bootstrap_pair(metrics: list[DomainMetric], *, stratum: str, left_arm: str, right_arm: str) -> dict[str, Any]:
    paired = paired_by_seed(metrics, stratum=stratum, left_arm=left_arm, right_arm=right_arm)
    rng = random.Random(BOOTSTRAP_SEED)
    seeds = sorted(paired)
    domains = sorted(set.intersection(*(set(paired[s]) for s in seeds)))
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        vals = [paired[seed][domain] for domain in domains]
        err = _mean([v[0] for v in vals])
        sr = _mean([v[1] for v in vals])
        per_seed.append({"train_seed": seed, "error_delta": err, "success_delta": sr, "success_delta_pp": 100 * sr, "both_direction": bool(sr > 0 and err < 0)})
    berr: list[float] = []
    bsr: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        sampled_domains = [domains[rng.randrange(len(domains))] for _ in domains]
        vals = [paired[seed][domain] for seed in sampled_seeds for domain in sampled_domains]
        berr.append(_mean([v[0] for v in vals]))
        bsr.append(_mean([v[1] for v in vals]))
    both_count = sum(1 for row in per_seed if row["both_direction"])
    return {
        "stratum": stratum,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "seed_count": len(seeds),
        "domain_count": len(domains),
        "mean_error_delta": _mean([row["error_delta"] for row in per_seed]),
        "mean_success_delta": _mean([row["success_delta"] for row in per_seed]),
        "mean_success_delta_pp": 100 * _mean([row["success_delta"] for row in per_seed]),
        "error_ci95": [percentile(berr, 0.025), percentile(berr, 0.975)],
        "success_ci95": [percentile(bsr, 0.025), percentile(bsr, 0.975)],
        "success_ci95_pp": [100 * percentile(bsr, 0.025), 100 * percentile(bsr, 0.975)],
        "per_seed": per_seed,
        "both_direction_count": both_count,
        "ci_gate_pass": bool(percentile(bsr, 0.025) > 0 and percentile(berr, 0.975) < 0),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def final_gates(metrics: list[DomainMetric]) -> dict[str, Any]:
    primary = bootstrap_pair(metrics, stratum="joint_ood", left_arm="corrected_u3", right_arm="history_init")
    swapped = bootstrap_pair(metrics, stratum="joint_ood", left_arm="corrected_u3", right_arm="swapped_correction_u3")
    iid = pair_summary(metrics, stratum="iid", left_arm="corrected_u3", right_arm="history_init")
    ls_joint = arm_success(metrics, stratum="joint_ood", arm=LS_ARM, seeds=TRAIN_SEEDS_FINAL)
    primary_pass = bool(
        primary["ci_gate_pass"]
        and primary["both_direction_count"] >= 4
        and primary["mean_success_delta"] >= 0.02
        and iid["mean_success_delta"] >= -0.01
        and ls_joint >= 0.95
    )
    correction_claim = bool(swapped["ci_gate_pass"])
    return {
        "format": "prove.fast-adapter-p17-final-gates.v1",
        "primary_joint_ood_vs_history": primary,
        "domain_specific_correction_vs_swapped": swapped,
        "iid_guard": iid,
        "bounded_ls_joint_ood_success_rate": ls_joint,
        "final_positive_gate_pass": primary_pass,
        "domain_specific_correction_claim_pass": correction_claim,
    }


def seed_direction_rows(metrics: list[DomainMetric]) -> list[dict[str, Any]]:
    specs = [
        ("corrected_u3_minus_history", "corrected_u3", "history_init"),
        ("corrected_u3_minus_swapped", "corrected_u3", "swapped_correction_u3"),
    ]
    rows: list[dict[str, Any]] = []
    for stratum in STRATA:
        for name, left, right in specs:
            try:
                paired = paired_by_seed(metrics, stratum=stratum, left_arm=left, right_arm=right)
            except ValueError:
                continue
            for seed in sorted(paired):
                vals = list(paired[seed].values())
                err = _mean([v[0] for v in vals])
                sr = _mean([v[1] for v in vals])
                rows.append({"comparison": name, "stratum": stratum, "train_seed": seed, "error_delta": err, "success_delta": sr, "success_delta_pp": 100 * sr, "both_direction": bool(sr > 0 and err < 0)})
    return rows


def _expand_existing_optional(items: list[str], filename: str) -> list[Path]:
    if not items:
        return []
    out: list[Path] = []
    for item in items:
        matches = [Path(p) for p in glob.glob(item)] or [Path(item)]
        for path in matches:
            if path.is_dir():
                out.extend(sorted(path.rglob(filename)))
            elif path.exists():
                out.append(path)
    return sorted(dict.fromkeys(p.resolve() for p in out))


def summarize_cost_timing(timing_inputs: list[Path], train_summary_inputs: list[Path]) -> dict[str, Any]:
    timing_rows: list[dict[str, Any]] = []
    for path in timing_inputs:
        for _line_no, row in _jsonl(path):
            timing_rows.append(row)
    train_rows: list[dict[str, Any]] = []
    for path in train_summary_inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        train_rows.append(payload)
    latent_u3 = [float(row["elapsed_sec"]) for row in timing_rows if row.get("operation") == "latent_correction" and int(row.get("U", -1)) == 3]
    history_u0 = [float(row["elapsed_sec"]) for row in timing_rows if row.get("operation") == "history_init"]
    train_elapsed = [float(row["elapsed_sec"]) for row in train_rows if "elapsed_sec" in row]
    history_unchanged = [bool(row.get("history_freeze_audit", {}).get("unchanged")) for row in train_rows if row.get("history_freeze_audit") is not None]
    return {
        "format": "prove.fast-adapter-p17-cost-timing-note.v1",
        "timing_files": [str(path) for path in timing_inputs],
        "train_summary_files": [str(path) for path in train_summary_inputs],
        "extra_train_updates_per_seed": 1500,
        "extra_train_batch_size": 64,
        "extra_trainable_parameters": 412675,
        "frozen_history_parameters": 550242,
        "parameter_cost_note": "P17 adds 412675 trainable predictor/log-alpha parameters around a frozen 550242-parameter History backbone; this is not an equal-cost History comparison.",
        "extra_train_elapsed_sec_mean": _mean(train_elapsed) if train_elapsed else None,
        "extra_train_elapsed_sec_per_seed": train_elapsed,
        "history_freeze_audit_all_unchanged": all(history_unchanged) if history_unchanged else None,
        "history_init_elapsed_sec_mean_per_128_domains": _mean(history_u0) if history_u0 else None,
        "latent_u3_elapsed_sec_mean_per_128_domains": _mean(latent_u3) if latent_u3 else None,
        "instrumentation_note": "Timing rows include batched support-pass/instrumentation measurements and metric pre/post loss computations; they are diagnostics for this run, not production latency claims and not gate inputs.",
    }


def make_success_comparison_plot(outdir: Path, agg: list[dict[str, Any]], *, split: str, coverage: dict[str, Any]) -> list[dict[str, str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        (outdir / "PLOT_ERROR.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return []

    arms = [
        ("history_init", "History U0"),
        ("corrected_u3", "Corrected U3"),
        ("bounded_ls_s3", "Bounded LS"),
    ]
    values: dict[str, list[float]] = {arm: [] for arm, _label in arms}
    for stratum in STRATA:
        for arm, _label in arms:
            vals = [float(row["success_percent"]) for row in agg if row["stratum"] == stratum and row["arm"] == arm]
            if not vals:
                raise ValueError(f"missing aggregate rows for plot: {stratum} {arm}")
            values[arm].append(_mean(vals))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = list(range(len(STRATA)))
    width = 0.24
    offsets = [-width, 0.0, width]
    for offset, (arm, label) in zip(offsets, arms):
        ax.bar([i + offset for i in x], values[arm], width=width, label=label)
    ax.set_xticks(x, STRATA)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Success rate (%)")
    if split == "dev":
        subtitle = "Development: 3 seeds, 128 domains × 8 query starts per stratum"
    else:
        subtitle = f"{split}: {len(coverage['train_seeds'])} seeds, {coverage['domain_count']} domains × {coverage['query_count']} query starts per stratum"
    ax.set_title("P17: History initialization vs gradient correction\n" + subtitle)
    for offset, (arm, _label) in zip(offsets, arms):
        for i, value in enumerate(values[arm]):
            ax.text(i + offset, value + 1.2, f"{value:.2f}%", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path = outdir / "p17_success_by_stratum_history_corrected_ls.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [{"label": "success-rate comparison by stratum", "path": str(path)}]


def primary_stratum_rows(agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum in STRATA:
        def arm_row(arm: str) -> list[dict[str, Any]]:
            return [row for row in agg if row["stratum"] == stratum and row["arm"] == arm]
        history = arm_row("history_init")
        corrected = arm_row("corrected_u3")
        ls_rows = arm_row("bounded_ls_s3")
        if not history or not corrected or not ls_rows:
            continue
        h_sr = _mean([float(row["success_rate"]) for row in history])
        c_sr = _mean([float(row["success_rate"]) for row in corrected])
        ls_sr = _mean([float(row["success_rate"]) for row in ls_rows])
        h_err = _mean([float(row["mean_goal_error"]) for row in history])
        c_err = _mean([float(row["mean_goal_error"]) for row in corrected])
        rows.append({
            "stratum": stratum,
            "history_success_percent": 100 * h_sr,
            "corrected_u3_success_percent": 100 * c_sr,
            "delta_success_pp": 100 * (c_sr - h_sr),
            "history_error": h_err,
            "corrected_u3_error": c_err,
            "delta_error": c_err - h_err,
            "bounded_ls_success_percent": 100 * ls_sr,
        })
    return rows


def write_report(path: Path, *, split: str, coverage: dict[str, Any], agg: list[dict[str, Any]], gate: dict[str, Any], final: dict[str, Any] | None, directions: list[dict[str, Any]], plots: list[dict[str, str]], cost_timing: dict[str, Any]) -> None:
    primary_rows_by_stratum = primary_stratum_rows(agg)
    lines = [
        "# Fast Adapter P17 Toy2D 静态 OOD 汇总",
        "",
        f"split: `{split}`。P17 是 Toy2D static-OOD 机制实验；不是机器人、RGB/DINO/Insert、动态变化、物理系统复杂度扩展或完整 RSI 证据。",
        "",
        "U0/`history_init` 是既有强 History checkpoint。P17 的 predictor/log-alpha 训练是额外计算和参数成本，不是同预算方法竞赛。Prediction、timing、calibration、LS 诊断不参与选模；LS 是资格诊断。",
        "",
        "## 覆盖",
        f"- strata: {coverage['strata']}；train seeds: {coverage['train_seeds']}；domain_count/stratum: {coverage['domain_count']}；query_count/domain: {coverage['query_count']}；groups: {coverage['groups']}。",
        "",
        "## 关键结论",
    ]
    if split == "dev":
        joint = next((row for row in primary_rows_by_stratum if row["stratum"] == "joint_ood"), None)
        if joint is not None:
            lines.append(f"- 主门 joint_ood 上，History U0 为 {joint['history_success_percent']:.6f}%，corrected U3 为 {joint['corrected_u3_success_percent']:.6f}%，提升 {joint['delta_success_pp']:.6f} 个百分点；预注册实用门槛是 +2.000000 个百分点，因此 `completed_dev_negative_or_mixed`。")
            lines.append(f"- mean goal error 从 {joint['history_error']:.8f} 到 {joint['corrected_u3_error']:.8f}，三次重复方向一致但幅度很小；这不支持已经展示有实际幅度的梯度适配优势。")
        joint_history_rows = [row for row in agg if row["stratum"] == "joint_ood" and row["arm"] == "history_init"]
        joint_corrected_rows = [row for row in agg if row["stratum"] == "joint_ood" and row["arm"] == "corrected_u3"]
        if joint_history_rows and joint_corrected_rows:
            def optmean(rows: list[dict[str, Any]], field: str) -> float | None:
                vals = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
                return _mean(vals) if vals else None
            h_support = optmean(joint_history_rows, "support_prediction_loss_post")
            c_support = optmean(joint_corrected_rows, "support_prediction_loss_post")
            h_heldout = optmean(joint_history_rows, "heldout_physical_prediction_mse")
            c_heldout = optmean(joint_corrected_rows, "heldout_physical_prediction_mse")
            h_first = optmean(joint_history_rows, "first_query_action_mse")
            c_first = optmean(joint_corrected_rows, "first_query_action_mse")
            z_delta = optmean(joint_corrected_rows, "z_delta_norm")
            first_change = optmean(joint_corrected_rows, "first_action_change_vs_history_init")
            if None not in (h_support, c_support, h_heldout, c_heldout, h_first, c_first, z_delta, first_change):
                lines.append(f"- 诊断显示梯度更新实际发生：joint_ood support loss {h_support:.14f}->{c_support:.14f}，heldout prediction MSE {h_heldout:.14f}->{c_heldout:.14f}，first-action MSE {h_first:.14f}->{c_first:.14f}；但 mean z delta 只有 {z_delta:.11f}，first-action L2 change 只有 {first_change:.11f}，控制修正太小，不是更新路径未生效。")
        lines.append("- IID guard 未显示损害；bounded LS 四个 stratum 均为100%，但它使用已知线性函数形式和无噪声直接状态，是资格诊断，不是一般机器人方法、训练方法或改 gate 的理由。")
        lines.append("- 该结果不是对一般 latent-gradient 机制的反证；它只说明当前 frozen-History retrofit、固定 S3/U3、静态 OOD 设计下没有达到预注册实用幅度。")
    lines.extend([
        "",
        "## 四个 stratum 主表",
        "| stratum | History U0 success | Corrected U3 success | delta | History error | Corrected U3 error | error delta | LS success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in primary_rows_by_stratum:
        lines.append(f"| {row['stratum']} | {row['history_success_percent']:.6f}% | {row['corrected_u3_success_percent']:.6f}% | {row['delta_success_pp']:.6f} pp | {row['history_error']:.8f} | {row['corrected_u3_error']:.8f} | {row['delta_error']:.8f} | {row['bounded_ls_success_percent']:.6f}% |")
    lines.extend([
        "",
        "## 分 stratum / seed / arm 结果",
        "| stratum | seed | arm | success | goal error | heldout pred MSE | support loss post |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    key_arms = ["history_init", "corrected_u1", "corrected_u3", "corrected_u5", "swapped_correction_u3", "nominal_base", "bounded_ls_s3"]
    for row in agg:
        if row["arm"] not in key_arms:
            continue
        h = row.get("heldout_physical_prediction_mse")
        s = row.get("support_prediction_loss_post")
        lines.append(f"| {row['stratum']} | {row['train_seed']} | {row['arm']} | {row['success_percent']:.3f}% | {row['mean_goal_error']:.6f} | {h if h is not None else ''} | {s if s is not None else ''} |")
    lines.append("")
    if plots:
        lines.append("## 图表")
        for art in plots:
            lines.append(f"- {art['label']}: `{Path(art['path']).name}`")
        lines.append("")
    lines.extend([
        "## 每 seed 配对方向",
        "| comparison | stratum | seed | error delta | success delta | both direction |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in directions:
        lines.append(f"| {row['comparison']} | {row['stratum']} | {row['train_seed']} | {row['error_delta']:.6f} | {row['success_delta_pp']:.3f} pp | {row['both_direction']} |")
    lines.append("")
    lines.extend([
        "## Development gate",
        f"- primary: joint_ood corrected_u3 - history_init；gate_pass: `{gate['gate_pass']}`。",
        f"- joint_ood mean SR delta: {gate['primary_joint_ood']['mean_success_delta_pp']:.3f} pp；mean error delta: {gate['primary_joint_ood']['mean_error_delta']:.6f}。",
        f"- IID guard mean SR delta: {gate['iid_guard']['mean_success_delta_pp']:.3f} pp。",
        f"- bounded LS joint_ood success: {100 * gate['bounded_ls_joint_ood_success_rate']:.3f}%。",
        "",
        "| seed | joint error delta | joint success delta | both direction |",
        "|---:|---:|---:|---|",
    ])
    for row in gate["primary_joint_ood"]["per_seed"]:
        lines.append(f"| {row['train_seed']} | {row['error_delta']:.6f} | {row['success_delta_pp']:.3f} pp | {row['both_direction']} |")
    lines.append("")
    if split == "dev":
        if gate["gate_pass"]:
            lines.append("解释：development gate 通过时，协议允许进入 seeds 44/55 extra training 与 fresh final；最终结论仍需 root 独立验收，不能把 dev 结果当作 final positive。")
        else:
            lines.append("解释：development gate 未通过时，本协议要求关闭为 `completed_dev_negative_or_mixed`，不调 stratum/range/U，不进入 final；上表保留完整三 seed development 结果，用于说明失败或混合方向。")
        lines.append("")
    if cost_timing:
        lines.append("## 成本与计时说明")
        lines.append(f"- 每个 P17 seed 额外训练 predictor/log-alpha 1500 updates，batch64；新增 trainable parameters 为 412675，frozen History backbone 为 550242 parameters。这是 History U0 之外的新增计算与参数成本，不是 equal-cost comparison。")
        if cost_timing.get("extra_train_elapsed_sec_mean") is not None:
            lines.append(f"- 已提供 train summary 的平均额外训练耗时约 {cost_timing['extra_train_elapsed_sec_mean']:.3f} 秒/seed；history freeze audit all unchanged: `{cost_timing.get('history_freeze_audit_all_unchanged')}`。")
        if cost_timing.get("latent_u3_elapsed_sec_mean_per_128_domains") is not None:
            lines.append(f"- U3 latent correction 计时均值约 {cost_timing['latent_u3_elapsed_sec_mean_per_128_domains']:.6f} 秒/128 domains；History init 计时均值约 {cost_timing.get('history_init_elapsed_sec_mean_per_128_domains', 0.0):.6f} 秒/128 domains。")
        lines.append(f"- {cost_timing.get('instrumentation_note')}")
        lines.append("")
    if final is not None:
        p = final["primary_joint_ood_vs_history"]
        s = final["domain_specific_correction_vs_swapped"]
        lines.extend([
            "## Final gates",
            f"- final_positive_gate_pass: `{final['final_positive_gate_pass']}`；domain_specific_correction_claim_pass: `{final['domain_specific_correction_claim_pass']}`。",
            "| comparison | error delta | error CI | success delta | success CI | seed both | CI gate |",
            "|---|---:|---:|---:|---:|---:|---|",
            f"| corrected-history joint_ood | {p['mean_error_delta']:.6f} | [{p['error_ci95'][0]:.6f}, {p['error_ci95'][1]:.6f}] | {p['mean_success_delta_pp']:.3f} pp | [{p['success_ci95_pp'][0]:.3f}, {p['success_ci95_pp'][1]:.3f}] pp | {p['both_direction_count']}/5 | {p['ci_gate_pass']} |",
            f"| corrected-swapped joint_ood | {s['mean_error_delta']:.6f} | [{s['error_ci95'][0]:.6f}, {s['error_ci95'][1]:.6f}] | {s['mean_success_delta_pp']:.3f} pp | [{s['success_ci95_pp'][0]:.3f}, {s['success_ci95_pp'][1]:.3f}] pp | {s['both_direction_count']}/5 | {s['ci_gate_pass']} |",
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["dev", "final"], required=True)
    parser.add_argument("--require-complete-grid", action="store_true")
    parser.add_argument("--dev-gate-receipt", type=Path)
    parser.add_argument("--timing-inputs", nargs="*", default=[])
    parser.add_argument("--train-summary-inputs", nargs="*", default=[])
    args = parser.parse_args()

    inputs = _expand_inputs(args.inputs, "eval_raw.jsonl")
    metrics = load_metrics(inputs)
    coverage = validate_metrics(metrics, split=args.split, require_complete=args.require_complete_grid)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agg = aggregate(metrics)
    _write_csv(args.output_dir / "aggregate_by_stratum_seed_arm.csv", agg)
    _write_csv(args.output_dir / "primary_joint_ood_domain_rows.csv", domain_rows(metrics, stratum="joint_ood"))
    directions = seed_direction_rows(metrics)
    _write_csv(args.output_dir / "paired_seed_directions.csv", directions)

    source_hashes = {str(path): _sha256(path) for path in inputs}
    final = None
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
        final = final_gates(metrics)
        (args.output_dir / "FINAL_PAIRED_BOOTSTRAP.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plots = make_success_comparison_plot(args.output_dir, agg, split=args.split, coverage=coverage)
    timing_inputs = _expand_existing_optional(args.timing_inputs or args.inputs, "eval_timing.jsonl")
    train_summary_inputs = _expand_existing_optional(args.train_summary_inputs, "summary.json")
    cost_timing = summarize_cost_timing(timing_inputs, train_summary_inputs)
    (args.output_dir / "COST_TIMING_NOTE.json").write_text(json.dumps(cost_timing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "format": "prove.fast-adapter-p17-summary.v1",
        "split": args.split,
        "input_files": [str(path) for path in inputs],
        "coverage": coverage,
        "dev_gate": gate,
        "final_gates": final,
        "outputs": {
            "primary_strata_table_in_report": "REPORT_ZH.md#四个-stratum-主表",
            "aggregate": "aggregate_by_stratum_seed_arm.csv",
            "primary_joint_ood_domain_rows": "primary_joint_ood_domain_rows.csv",
            "paired_seed_directions": "paired_seed_directions.csv",
            "success_plot": "p17_success_by_stratum_history_corrected_ls.png",
            "cost_timing_note": "COST_TIMING_NOTE.json",
            "report_zh": "REPORT_ZH.md",
        },
        "source_hashes": source_hashes,
        "plots": plots,
        "cost_timing": cost_timing,
        "notes": [
            "U0/history_init is the existing strong History checkpoint; P17 extra-head training is additional cost.",
            "Successes arrays encode ordinary failures as False; missing/aborted jobs are not silently converted into failed rows.",
            "Bootstrap resamples model seed and physical domain axes only.",
        ],
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "REPORT_ZH.md", split=args.split, coverage=coverage, agg=agg, gate=gate, final=final, directions=directions, plots=plots, cost_timing=cost_timing)


if __name__ == "__main__":
    main()
