#!/usr/bin/env python3
"""Summarize Fast Adapter P1.5 Toy2D ablation outputs.

Reads core P1.5 eval_raw.jsonl files where each row is one domain and contains
length-N query arrays (final_goal_errors/successes).  The script aggregates
arrays immediately to domain metrics and never expands query rows into a large
in-memory table.  It supports dev selection, final summaries from a sealed dev
selection receipt, cross-config paired comparisons, two-dimensional bootstrap,
and static figures.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260921
PRIMARY_MODE = "common_base"
PRIMARY_K = 3
TRAIN_SEEDS = [11, 22, 33, 44, 55]
K_VALUES = [0, 1, 2, 3, 5, 10]
SPLIT_EXPECTATIONS = {
    "dev": {"domains": 128, "domain_seed": 131000, "support_seed": 132000, "query_seed": 141000, "calibration_seed": 133000},
    "final": {"domains": 256, "domain_seed": 191000, "support_seed": 192000, "query_seed": 201000, "calibration_seed": 193000},
}
FAST_CONFIGS = [
    "z4_alpha5",
    "z8_alpha5",
    "z16_alpha5",
    "z32_alpha5",
    "z64_alpha5",
    "z32_alpha0p05",
    "z32_alpha0p5",
    "z32_alpha1",
    "z32_alpha_learned",
]
HISTORY_CONFIGS = ["history_width256", "history_width640"]
FAST_ARMS = ["base", "frozen_z", "correct_z", "swapped_z"]
HISTORY_ARMS = ["history"]


@dataclass(frozen=True)
class DomainMetric:
    split: str
    train_seed: int
    config_id: str
    method_family: str
    support_mode: str
    domain_seed: int
    support_seed: int
    query_seed: int
    calibration_seed: int
    domain_index: int
    arm: str
    k: int
    query_count: int
    mean_goal_error: float
    success_rate: float
    z_source_domain_index: int | None = None
    latent_delta_norm: float | None = None
    delta_norm_last: float | None = None
    first_query_action_mse: float | None = None
    query_action_mse: float | None = None


def _method_family(config_id: str) -> str:
    if config_id.startswith("history_"):
        return "history"
    if config_id.startswith("z"):
        return "fast"
    raise ValueError(f"cannot infer method_family from config_id={config_id!r}")


def _fast_sort_key(config_id: str) -> tuple[int, int, float, str]:
    if config_id == "z32_alpha_learned":
        return (32, 1, 0.5, config_id)
    m = re.fullmatch(r"z(\d+)_alpha(.+)", config_id)
    if not m:
        return (9999, 9, 9999.0, config_id)
    dim = int(m.group(1))
    alpha_text = m.group(2).replace("p", ".")
    return (dim, 0, float(alpha_text), config_id)


def _history_sort_key(config_id: str) -> tuple[int, str]:
    m = re.fullmatch(r"history_width(\d+)", config_id)
    return (int(m.group(1)) if m else 999999, config_id)


def _as_list(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
    raise ValueError(f"{field} must be an array in P1.5 core raw")


def _bool01(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return 1.0
    if text in {"false", "f", "no", "n", "0"}:
        return 0.0
    number = float(text)
    if number not in (0.0, 1.0):
        raise ValueError(f"success value must be bool/0/1, got {value!r}")
    return number


def _maybe_mean_array(row: dict[str, Any], field: str) -> float | None:
    if field not in row or row[field] is None:
        return None
    try:
        values = _as_list(row[field], field=field)
    except ValueError:
        try:
            return float(row[field])
        except Exception:
            return None
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return int(value)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc


def _metric_from_row(row: dict[str, Any], *, path: Path, line_number: int) -> DomainMetric:
    required = [
        "split",
        "train_seed",
        "config_id",
        "support_mode",
        "domain_seed",
        "support_seed",
        "query_seed",
        "calibration_seed",
        "domain_index",
        "arm",
        "K",
        "final_goal_errors",
        "successes",
    ]
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"{path}:{line_number}: missing required fields {missing}")
    errors = [float(v) for v in _as_list(row["final_goal_errors"], field="final_goal_errors")]
    successes = [_bool01(v) for v in _as_list(row["successes"], field="successes")]
    if len(errors) != len(successes):
        raise ValueError(f"{path}:{line_number}: final_goal_errors/successes length mismatch")
    if not errors:
        raise ValueError(f"{path}:{line_number}: empty query arrays")
    query_indices = row.get("query_indices")
    if query_indices is not None:
        indices = [int(v) for v in _as_list(query_indices, field="query_indices")]
        expected_indices = list(range(len(errors)))
        if indices != expected_indices:
            raise ValueError(f"{path}:{line_number}: query_indices must be {expected_indices}, got {indices}")
    if len(errors) != 8:
        raise ValueError(f"{path}:{line_number}: P1.5 expects 8 query values per domain row, got {len(errors)}")
    config_id = str(row["config_id"])
    family = _method_family(config_id)
    arm = str(row["arm"])
    if family == "fast" and arm not in FAST_ARMS:
        raise ValueError(f"{path}:{line_number}: unexpected fast arm {arm!r}")
    if family == "history" and arm not in HISTORY_ARMS:
        raise ValueError(f"{path}:{line_number}: unexpected history arm {arm!r}")
    return DomainMetric(
        split=str(row["split"]),
        train_seed=int(row["train_seed"]),
        config_id=config_id,
        method_family=family,
        support_mode=str(row["support_mode"]),
        domain_seed=int(row["domain_seed"]),
        support_seed=int(row["support_seed"]),
        query_seed=int(row["query_seed"]),
        calibration_seed=int(row["calibration_seed"]),
        domain_index=int(row["domain_index"]),
        arm=arm,
        k=int(row["K"]),
        query_count=len(errors),
        mean_goal_error=sum(errors) / len(errors),
        success_rate=sum(successes) / len(successes),
        z_source_domain_index=_optional_int(row.get("z_source_domain_index")),
        latent_delta_norm=_maybe_mean_array(row, "latent_delta_norm"),
        delta_norm_last=_maybe_mean_array(row, "delta_norm_last"),
        first_query_action_mse=_maybe_mean_array(row, "first_query_action_mse"),
        query_action_mse=_maybe_mean_array(row, "query_action_mse"),
    )


def _expand_inputs(paths_or_globs: list[str], *, filename: str = "eval_raw.jsonl") -> list[Path]:
    found: list[Path] = []
    for item in paths_or_globs:
        matches = [Path(p) for p in glob.glob(item)]
        if not matches:
            matches = [Path(item)]
        for path in matches:
            if path.is_dir():
                found.extend(sorted(path.rglob(filename)))
            else:
                found.append(path)
    unique = sorted(dict.fromkeys(p.resolve() for p in found))
    if not unique:
        raise ValueError(f"no {filename} inputs found")
    for path in unique:
        if not path.exists():
            raise FileNotFoundError(path)
    return unique


def load_metrics(inputs: list[Path]) -> list[DomainMetric]:
    metrics: list[DomainMetric] = []
    seen: set[tuple[Any, ...]] = set()
    splits: set[str] = set()
    domain_seeds: set[int] = set()
    for path in inputs:
        for line_number, row in enumerate(_read_jsonl(path), start=1):
            metric = _metric_from_row(row, path=path, line_number=line_number)
            key = (
                metric.split,
                metric.train_seed,
                metric.config_id,
                metric.support_mode,
                metric.domain_index,
                metric.arm,
                metric.k,
            )
            if key in seen:
                raise ValueError(f"duplicate domain row for {key}")
            seen.add(key)
            splits.add(metric.split)
            domain_seeds.add(metric.domain_seed)
            metrics.append(metric)
    if len(splits) != 1:
        raise ValueError(f"refuse mixed splits: {sorted(splits)}")
    if len(domain_seeds) != 1:
        raise ValueError(f"refuse mixed domain_seed values: {sorted(domain_seeds)}")
    return metrics


def validate_grid(metrics: list[DomainMetric], *, require_complete: bool) -> dict[str, Any]:
    configs = sorted({m.config_id for m in metrics})
    split = sorted({m.split for m in metrics})[0]
    modes = sorted({m.support_mode for m in metrics})
    ks = sorted({m.k for m in metrics})
    seeds = sorted({m.train_seed for m in metrics})
    query_counts = sorted({m.query_count for m in metrics})
    expected_split = SPLIT_EXPECTATIONS.get(split)
    if expected_split:
        for field, expected_value in [("domain_seed", expected_split["domain_seed"]), ("support_seed", expected_split["support_seed"]), ("query_seed", expected_split["query_seed"]), ("calibration_seed", expected_split["calibration_seed"])]:
            observed = sorted({getattr(m, field) for m in metrics})
            if observed != [expected_value]:
                problems.append(f"{split} {field} must be {expected_value}, got {observed}")
        if query_counts != [8]:
            problems.append(f"{split} query_count must be [8], got {query_counts}")
    coverage: dict[tuple[int, str, str, str, int], set[int]] = defaultdict(set)
    for m in metrics:
        coverage[(m.train_seed, m.config_id, m.support_mode, m.arm, m.k)].add(m.domain_index)
    problems: list[str] = []
    reference_domains = set(next(iter(coverage.values())))
    for (seed, config, mode, arm, k), domains in sorted(coverage.items()):
        if domains != reference_domains:
            missing = sorted(reference_domains - domains)[:8]
            extra = sorted(domains - reference_domains)[:8]
            problems.append(f"domain set mismatch {(seed, config, mode, arm, k)}; missing={missing}, extra={extra}")
    if require_complete:
        expected_configs = FAST_CONFIGS + HISTORY_CONFIGS
        missing_configs = sorted(set(expected_configs) - set(configs))
        if missing_configs:
            problems.append(f"missing configs {missing_configs}")
        missing_seeds = sorted(set(TRAIN_SEEDS) - set(seeds))
        if missing_seeds:
            problems.append(f"missing train seeds {missing_seeds}")
        missing_modes = sorted(set(["common_base", "on_policy"]) - set(modes))
        if missing_modes:
            problems.append(f"missing support modes {missing_modes}")
        missing_k = sorted(set(K_VALUES) - set(ks))
        if missing_k:
            problems.append(f"missing K values {missing_k}")
        expected_domain_count = expected_split["domains"] if expected_split else len(reference_domains)
        if len(reference_domains) != expected_domain_count:
            problems.append(f"{split} domain set must contain {expected_domain_count} domains, got {len(reference_domains)}")
        for config in FAST_CONFIGS:
            for seed in TRAIN_SEEDS:
                for mode in ["common_base", "on_policy"]:
                    for k in K_VALUES:
                        for arm in FAST_ARMS:
                            n = len(coverage.get((seed, config, mode, arm, k), set()))
                            if n != expected_domain_count:
                                problems.append(f"incomplete fast group seed={seed} config={config} mode={mode} arm={arm} K={k} domains={n}/{expected_domain_count}")
        for config in HISTORY_CONFIGS:
            for seed in TRAIN_SEEDS:
                for mode in ["common_base", "on_policy"]:
                    for k in K_VALUES:
                        n = len(coverage.get((seed, config, mode, "history", k), set()))
                        if n != expected_domain_count:
                            problems.append(f"incomplete history group seed={seed} config={config} mode={mode} K={k} domains={n}/{expected_domain_count}")
    if problems:
        raise ValueError("; ".join(problems[:20]))
    return {
        "split": split,
        "configs": configs,
        "train_seeds": seeds,
        "support_modes": modes,
        "K_values": ks,
        "query_counts": query_counts,
        "domain_count_min": min(len(v) for v in coverage.values()),
        "domain_count_max": max(len(v) for v in coverage.values()),
    }


def aggregate_rows(metrics: list[DomainMetric]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[DomainMetric]] = defaultdict(list)
    for m in metrics:
        buckets[(m.split, m.train_seed, m.config_id, m.method_family, m.support_mode, m.arm, m.k)].append(m)
    rows = []
    for (split, seed, config, family, mode, arm, k), items in sorted(buckets.items()):
        rows.append({
            "split": split,
            "train_seed": seed,
            "config_id": config,
            "method_family": family,
            "support_mode": mode,
            "arm": arm,
            "K": k,
            "domain_count": len(items),
            "query_count_min": min(i.query_count for i in items),
            "query_count_max": max(i.query_count for i in items),
            "mean_goal_error": sum(i.mean_goal_error for i in items) / len(items),
            "success_rate": sum(i.success_rate for i in items) / len(items),
        })
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty mean")
    return sum(values) / len(values)


def score_config(metrics: list[DomainMetric], *, config_id: str, arm: str, mode: str, k: int) -> dict[str, Any]:
    selected = [m for m in metrics if m.config_id == config_id and m.arm == arm and m.support_mode == mode and m.k == k]
    by_seed: dict[int, list[DomainMetric]] = defaultdict(list)
    for m in selected:
        by_seed[m.train_seed].append(m)
    seed_rows = []
    for seed in sorted(by_seed):
        items = by_seed[seed]
        seed_rows.append({
            "train_seed": seed,
            "domain_count": len(items),
            "mean_goal_error": _mean([m.mean_goal_error for m in items]),
            "success_rate": _mean([m.success_rate for m in items]),
        })
    if not seed_rows:
        raise ValueError(f"no rows for {config_id}/{arm}/{mode}/K{k}")
    return {
        "config_id": config_id,
        "arm": arm,
        "support_mode": mode,
        "K": k,
        "seed_count": len(seed_rows),
        "mean_goal_error": _mean([r["mean_goal_error"] for r in seed_rows]),
        "success_rate": _mean([r["success_rate"] for r in seed_rows]),
        "per_seed": seed_rows,
    }


def select_dev(metrics: list[DomainMetric]) -> dict[str, Any]:
    split = sorted({m.split for m in metrics})[0]
    if split != "dev":
        raise ValueError("dev selection may only run on split=dev")
    fast_scores = [score_config(metrics, config_id=c, arm="correct_z", mode=PRIMARY_MODE, k=PRIMARY_K) for c in FAST_CONFIGS if any(m.config_id == c for m in metrics)]
    history_scores = [score_config(metrics, config_id=c, arm="history", mode=PRIMARY_MODE, k=PRIMARY_K) for c in HISTORY_CONFIGS if any(m.config_id == c for m in metrics)]
    for row in fast_scores + history_scores:
        if row["seed_count"] != len(TRAIN_SEEDS):
            raise ValueError(f"selection requires five train seeds for {row['config_id']}, got {row['seed_count']}")
    tol = 1e-5
    best_fast_error = min(r["mean_goal_error"] for r in fast_scores)
    fast_ties = [r for r in fast_scores if r["mean_goal_error"] <= best_fast_error + tol]
    fast_winner = sorted(fast_ties, key=lambda r: _fast_sort_key(r["config_id"]))[0]
    best_history_error = min(r["mean_goal_error"] for r in history_scores)
    hist_ties = [r for r in history_scores if r["mean_goal_error"] <= best_history_error + tol]
    history_winner = sorted(hist_ties, key=lambda r: _history_sort_key(r["config_id"]))[0]
    fixed_scores = [score_config(metrics, config_id=c, arm="frozen_z", mode=PRIMARY_MODE, k=0) for c in FAST_CONFIGS if any(m.config_id == c for m in metrics)]
    best_fixed_error = min(r["mean_goal_error"] for r in fixed_scores)
    fixed_ties = [r for r in fixed_scores if r["mean_goal_error"] <= best_fixed_error + tol]
    fixed_winner = sorted(fixed_ties, key=lambda r: _fast_sort_key(r["config_id"]))[0]
    return {
        "format": "prove.fast-adapter-p15-dev-selection.v1",
        "selection_split": "dev",
        "primary_support_mode": PRIMARY_MODE,
        "primary_K": PRIMARY_K,
        "tie_tolerance": tol,
        "fast_winner": fast_winner,
        "history_winner": history_winner,
        "strongest_fixed_adapter_background": fixed_winner,
        "fast_scores": sorted(fast_scores, key=lambda r: (r["mean_goal_error"], _fast_sort_key(r["config_id"]))),
        "history_scores": sorted(history_scores, key=lambda r: (r["mean_goal_error"], _history_sort_key(r["config_id"]))),
        "fixed_k0_scores": sorted(fixed_scores, key=lambda r: (r["mean_goal_error"], _fast_sort_key(r["config_id"]))),
        "rules": [
            "Fast and History winners are selected only by five-seed mean dev common_base K3 goal error.",
            "Ties within 1e-5 prefer smaller z_dim, fixed alpha over learned, then smaller alpha; history ties prefer width256.",
            "Strongest fixed adapter is background only, selected by dev common_base K0 frozen_z goal error.",
        ],
    }


def _metric_map(metrics: list[DomainMetric]) -> dict[tuple[int, str, str, str, int, int], DomainMetric]:
    return {(m.train_seed, m.config_id, m.support_mode, m.arm, m.k, m.domain_index): m for m in metrics}


def paired_domain_values(metrics: list[DomainMetric], *, left_config: str, left_arm: str, right_config: str, right_arm: str, mode: str, k: int) -> dict[int, dict[int, tuple[float, float]]]:
    by_key = _metric_map(metrics)
    seeds = sorted(set(TRAIN_SEEDS) & {m.train_seed for m in metrics if m.config_id in {left_config, right_config}})
    result: dict[int, dict[int, tuple[float, float]]] = {}
    for seed in seeds:
        left_domains = {m.domain_index for m in metrics if m.train_seed == seed and m.config_id == left_config and m.support_mode == mode and m.arm == left_arm and m.k == k}
        right_domains = {m.domain_index for m in metrics if m.train_seed == seed and m.config_id == right_config and m.support_mode == mode and m.arm == right_arm and m.k == k}
        if left_domains != right_domains:
            missing = sorted(left_domains - right_domains)[:8]
            extra = sorted(right_domains - left_domains)[:8]
            raise ValueError(f"paired comparison domain sets differ for seed={seed} {left_config}:{left_arm} vs {right_config}:{right_arm}; missing_right={missing}, extra_right={extra}")
        domains = sorted(left_domains)
        result[seed] = {}
        for domain in domains:
            left = by_key[(seed, left_config, mode, left_arm, k, domain)]
            right = by_key[(seed, right_config, mode, right_arm, k, domain)]
            # Deltas are Fast/left minus comparator/right. Error <0 and success >0 are improvements.
            result[seed][domain] = (left.mean_goal_error - right.mean_goal_error, left.success_rate - right.success_rate)
    return result


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("empty percentile")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def bootstrap_pair(paired: dict[int, dict[int, tuple[float, float]]], *, iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    import random

    seeds = sorted(paired)
    common_domains = sorted(set.intersection(*(set(paired[s]) for s in seeds))) if seeds else []
    if not seeds or not common_domains:
        raise ValueError("paired bootstrap requires common seeds and domains")
    per_seed = []
    for s in seeds:
        e = _mean([paired[s][d][0] for d in common_domains])
        sr = _mean([paired[s][d][1] for d in common_domains])
        per_seed.append({"train_seed": s, "error_delta": e, "success_delta_pp": 100.0 * sr, "both_improve": bool(sr > 0 and e < 0)})
    mean_error = _mean([r["error_delta"] for r in per_seed])
    mean_success_pp = _mean([r["success_delta_pp"] for r in per_seed])
    rng = random.Random(seed)
    cross_error: list[float] = []
    cross_success: list[float] = []
    conditional_error: list[float] = []
    conditional_success: list[float] = []
    for _ in range(iterations):
        sampled_seed_ids = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        sampled_domains = [common_domains[rng.randrange(len(common_domains))] for _ in common_domains]
        vals = [paired[s][d] for s in sampled_seed_ids for d in sampled_domains]
        cross_error.append(_mean([v[0] for v in vals]))
        cross_success.append(100.0 * _mean([v[1] for v in vals]))
        vals_cond = [paired[s][d] for s in seeds for d in sampled_domains]
        conditional_error.append(_mean([v[0] for v in vals_cond]))
        conditional_success.append(100.0 * _mean([v[1] for v in vals_cond]))
    return {
        "seed_count": len(seeds),
        "domain_count": len(common_domains),
        "mean_error_delta": mean_error,
        "mean_success_delta_pp": mean_success_pp,
        "cross_bootstrap_95ci_error_delta": [_percentile(cross_error, 0.025), _percentile(cross_error, 0.975)],
        "cross_bootstrap_95ci_success_delta_pp": [_percentile(cross_success, 0.025), _percentile(cross_success, 0.975)],
        "conditional_domain_95ci_error_delta": [_percentile(conditional_error, 0.025), _percentile(conditional_error, 0.975)],
        "conditional_domain_95ci_success_delta_pp": [_percentile(conditional_success, 0.025), _percentile(conditional_success, 0.975)],
        "per_seed": per_seed,
        "per_seed_both_improve_count": sum(1 for r in per_seed if r["both_improve"]),
        "superiority_gate_pass": bool(_percentile(cross_success, 0.025) > 0 and _percentile(cross_error, 0.975) < 0 and sum(1 for r in per_seed if r["both_improve"]) >= 4),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }


def build_primary_pairs(metrics: list[DomainMetric], *, fast_winner: str, history_winner: str) -> list[dict[str, Any]]:
    specs = [
        ("fast_correct_minus_same_checkpoint_frozen_z", fast_winner, "correct_z", fast_winner, "frozen_z"),
        ("fast_correct_minus_same_checkpoint_swapped_z", fast_winner, "correct_z", fast_winner, "swapped_z"),
        ("fast_correct_minus_history_winner", fast_winner, "correct_z", history_winner, "history"),
    ]
    # Also report both history widths against the Fast winner when present.
    for history in HISTORY_CONFIGS:
        if history != history_winner and any(m.config_id == history for m in metrics):
            specs.append((f"fast_correct_minus_{history}", fast_winner, "correct_z", history, "history"))
    rows = []
    for comparison, lc, la, rc, ra in specs:
        paired = paired_domain_values(metrics, left_config=lc, left_arm=la, right_config=rc, right_arm=ra, mode=PRIMARY_MODE, k=PRIMARY_K)
        stat = bootstrap_pair(paired)
        rows.append({"comparison": comparison, "left_config": lc, "left_arm": la, "right_config": rc, "right_arm": ra, **stat})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    return obj


def _curve_rows(aggregate: list[dict[str, Any]], *, configs: list[str] | None = None) -> list[dict[str, Any]]:
    rows = aggregate
    if configs is not None:
        rows = [r for r in rows if r["config_id"] in configs]
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["config_id"], r["method_family"], r["support_mode"], r["arm"], r["K"])].append(r)
    out = []
    for (config, family, mode, arm, k), items in sorted(grouped.items()):
        out.append({
            "config_id": config,
            "method_family": family,
            "support_mode": mode,
            "arm": arm,
            "K": k,
            "seed_count": len(items),
            "mean_goal_error": _mean([float(i["mean_goal_error"]) for i in items]),
            "goal_error_min_seed": min(float(i["mean_goal_error"]) for i in items),
            "goal_error_max_seed": max(float(i["mean_goal_error"]) for i in items),
            "success_rate": _mean([float(i["success_rate"]) for i in items]),
            "success_min_seed": min(float(i["success_rate"]) for i in items),
            "success_max_seed": max(float(i["success_rate"]) for i in items),
        })
    return out


def read_calibration(inputs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in inputs:
        # infer config/seed from sibling summary or directory names when not present
        for row in _read_jsonl(path):
            item = dict(row)
            item.setdefault("source_path", str(path))
            for parent in [path.parent, path.parent.parent]:
                name = parent.name
                m = re.search(r"seed(\d+)", name)
                if m:
                    item.setdefault("train_seed", int(m.group(1)))
                for config in FAST_CONFIGS + HISTORY_CONFIGS:
                    if config in name:
                        item.setdefault("config_id", config)
            if "config_id" not in item:
                # Try eval summary beside calibration path.
                summary_path = path.parent / "eval_summary.json"
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text())
                        if "config_id" in summary:
                            item["config_id"] = summary["config_id"]
                    except Exception:
                        pass
            rows.append(item)
    return rows


def summarize_calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        if "config_id" not in row:
            continue
        mode = str(row.get("support_mode", "unknown"))
        arm = str(row.get("arm", "unknown"))
        k = int(row.get("K", row.get("k", -1)))
        value = None
        for field in ["adapt", "loss", "adapt_loss", "mean_loss", "L_adapt", "l_adapt"]:
            if field in row:
                value = float(row[field])
                break
        if value is None:
            continue
        buckets[(str(row["config_id"]), mode, k, arm)].append(value)
    return [
        {"config_id": c, "support_mode": m, "K": k, "arm": a, "mean_adapt_loss": _mean(v), "count": len(v)}
        for (c, m, k, a), v in sorted(buckets.items())
    ]


def write_figures(output_dir: Path, curve_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (output_dir / "FIGURE_WARNING.txt").write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return

    def plot_metric(metric: str, low: str, high: str, ylabel: str, stem: str, configs: list[str] | None = None) -> None:
        rows = [r for r in curve_rows if r["support_mode"] == PRIMARY_MODE]
        if configs is not None:
            rows = [r for r in rows if r["config_id"] in configs]
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[(r["config_id"], r["arm"])].append(r)
        plt.figure(figsize=(11, 6))
        for (config, arm), items in sorted(groups.items()):
            if arm not in {"correct_z", "frozen_z", "swapped_z", "history"}:
                continue
            items = sorted(items, key=lambda r: r["K"])
            x = [r["K"] for r in items]
            y = [r[metric] for r in items]
            ylo = [r[low] for r in items]
            yhi = [r[high] for r in items]
            label = f"{config}:{arm}"
            plt.plot(x, y, marker="o", linewidth=1.3, label=label)
            plt.fill_between(x, ylo, yhi, alpha=0.12)
        plt.xlabel("support K (transitions)")
        plt.ylabel(ylabel)
        plt.title(f"P1.5 {ylabel} vs K, common_base; line=5-seed mean, shade=seed min-max")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        for suffix in ["png", "svg"]:
            plt.savefig(output_dir / f"{stem}.{suffix}")
        plt.close()

    plot_metric("success_rate", "success_min_seed", "success_max_seed", "success rate", "success_all_configs_common_base")
    plot_metric("mean_goal_error", "goal_error_min_seed", "goal_error_max_seed", "Toy normalized final goal error", "goal_error_all_configs_common_base")

    # Dim and alpha focused figures use correct_z only.
    for family_name, configs in {
        "dim_sweep": ["z4_alpha5", "z8_alpha5", "z16_alpha5", "z32_alpha5", "z64_alpha5"],
        "alpha_sweep": ["z32_alpha0p05", "z32_alpha0p5", "z32_alpha1", "z32_alpha5", "z32_alpha_learned"],
    }.items():
        subset = [r for r in curve_rows if r["config_id"] in configs and r["arm"] == "correct_z" and r["support_mode"] == PRIMARY_MODE]
        if not subset:
            continue
        for metric, low, high, ylabel, stem in [
            ("success_rate", "success_min_seed", "success_max_seed", "success rate", f"success_{family_name}_common_base"),
            ("mean_goal_error", "goal_error_min_seed", "goal_error_max_seed", "Toy normalized final goal error", f"goal_error_{family_name}_common_base"),
        ]:
            plt.figure(figsize=(8, 5))
            for config in configs:
                items = sorted([r for r in subset if r["config_id"] == config], key=lambda r: r["K"])
                if not items:
                    continue
                x = [r["K"] for r in items]
                plt.plot(x, [r[metric] for r in items], marker="o", label=config)
                plt.fill_between(x, [r[low] for r in items], [r[high] for r in items], alpha=0.12)
            plt.xlabel("support K (transitions)")
            plt.ylabel(ylabel)
            plt.title(f"P1.5 {family_name}: line=5-seed mean, shade=seed min-max")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            for suffix in ["png", "svg"]:
                plt.savefig(output_dir / f"{stem}.{suffix}")
            plt.close()

    def plot_selected(configs: list[str]) -> None:
        if not configs:
            return
        rows = [r for r in curve_rows if r["config_id"] in configs and r["arm"] in {"correct_z", "frozen_z", "swapped_z", "history"}]
        if not rows:
            return
        for metric, low, high, ylabel, stem in [
            ("success_rate", "success_min_seed", "success_max_seed", "success rate", "success_selected_fast_history"),
            ("mean_goal_error", "goal_error_min_seed", "goal_error_max_seed", "Toy normalized final goal error", "goal_error_selected_fast_history"),
        ]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
            for ax, mode in zip(axes, ["common_base", "on_policy"]):
                panel = [r for r in rows if r["support_mode"] == mode]
                for (config, arm), _ in sorted({(r["config_id"], r["arm"]): None for r in panel}.items()):
                    items = sorted([r for r in panel if r["config_id"] == config and r["arm"] == arm], key=lambda r: r["K"])
                    if not items:
                        continue
                    x = [r["K"] for r in items]
                    ax.plot(x, [r[metric] for r in items], marker="o", linewidth=1.6, label=f"{config}:{arm}")
                    ax.fill_between(x, [r[low] for r in items], [r[high] for r in items], alpha=0.14)
                ax.set_title(mode)
                ax.set_xlabel("support K (transitions)")
                ax.grid(True, alpha=0.25)
            axes[0].set_ylabel(ylabel)
            handles, labels = axes[0].get_legend_handles_labels()
            handles2, labels2 = axes[1].get_legend_handles_labels()
            by_label = dict(zip(labels + labels2, handles + handles2))
            fig.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.03))
            fig.suptitle(f"P1.5 selected comparison: line=5-seed mean, shade=seed min-max", y=1.12)
            fig.tight_layout()
            for suffix in ["png", "svg"]:
                fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")
            plt.close(fig)

    selected_path = output_dir / "selected_configs_for_figures.json"
    selected_configs: list[str] = []
    if selected_path.exists():
        try:
            selected_payload = json.loads(selected_path.read_text())
            selected_configs = [c for c in selected_payload.get("configs", []) if isinstance(c, str)]
        except Exception:
            selected_configs = []
    plot_selected(selected_configs)

    if calibration_rows:
        plt.figure(figsize=(9, 5))
        for config in sorted({r["config_id"] for r in calibration_rows}):
            items = sorted([r for r in calibration_rows if r["config_id"] == config and r["support_mode"] == PRIMARY_MODE], key=lambda r: (r["arm"], r["K"]))
            for arm in sorted({r["arm"] for r in items}):
                arm_items = [r for r in items if r["arm"] == arm]
                if not arm_items:
                    continue
                plt.plot([r["K"] for r in arm_items], [r["mean_adapt_loss"] for r in arm_items], marker="o", label=f"{config}:{arm}")
        plt.xlabel("support K (transitions)")
        plt.ylabel("held-out L_adapt on same base transitions")
        plt.title("P1.5 calibration loss; compare mainly within checkpoint across K")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        for suffix in ["png", "svg"]:
            plt.savefig(output_dir / f"calibration_l_adapt_common_base.{suffix}")
        plt.close()



def read_support_metrics(inputs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        for row in _read_jsonl(path):
            item = dict(row)
            item.setdefault("source_path", str(path))
            for parent in [path.parent, path.parent.parent]:
                name = parent.name
                m = re.search(r"seed(\d+)", name)
                if m:
                    item.setdefault("train_seed", int(m.group(1)))
                for config in FAST_CONFIGS:
                    if config in name:
                        item.setdefault("config_id", config)
            rows.append(item)
    return rows


def summarize_support_prefix(rows: list[dict[str, Any]], *, full_prefix_k: int = 10) -> list[dict[str, Any]]:
    """Use K=10 support metrics only so repeated prefixes from smaller K are not independent."""
    filtered = [r for r in rows if int(r.get("K", -1)) == full_prefix_k]
    buckets: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        config = str(row.get("config_id", "unknown"))
        seed = int(row.get("train_seed", -1))
        mode = str(row.get("support_mode", "unknown"))
        support_index = int(row.get("support_index", -1))
        buckets[(config, seed, mode, support_index)].append(row)
    out = []
    numeric_fields = [
        "pre_adapt", "post_adapt", "pre_mse", "post_mse", "pre_cos", "post_cos",
        "pre_z_reg", "post_z_reg", "grad_z_norm", "latent_delta_norm", "inner_lr",
    ]
    for (config, seed, mode, support_index), items in sorted(buckets.items()):
        record = {"config_id": config, "train_seed": seed, "support_mode": mode, "support_index": support_index, "source_K": full_prefix_k, "row_count": len(items)}
        for field in numeric_fields:
            vals = []
            for item in items:
                if field in item:
                    try:
                        vals.append(float(item[field]))
                    except Exception:
                        pass
            if vals:
                record[f"mean_{field}"] = _mean(vals)
        out.append(record)
    return out


def learned_alpha_values(search_roots: list[Path]) -> list[dict[str, Any]]:
    """Read learned-alpha final values from train summaries, not eval init config.

    Core stores the learned-alpha initial value in eval_summary.train_config as
    inner_lr=0.5.  The learned final value is train summary last.inner_lr beside
    the checkpoint path, e.g. train_fast/z32_alpha_learned/seed*/summary.json.
    """
    rows: list[dict[str, Any]] = []
    eval_summaries: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            eval_summaries.extend(root.rglob("eval_summary.json"))
        elif root.name == "eval_summary.json":
            eval_summaries.append(root)
    seen: set[Path] = set()
    for eval_summary in sorted(eval_summaries):
        try:
            data = json.loads(eval_summary.read_text())
        except Exception:
            continue
        train_config = data.get("train_config", {})
        fast_config = train_config.get("fast_config", {}) if isinstance(train_config, dict) else {}
        if fast_config.get("config_id") != "z32_alpha_learned":
            continue
        checkpoint = Path(str(data.get("checkpoint", "")))
        train_summary = checkpoint.parent / "summary.json"
        if not train_summary.exists():
            # Fallback for relative oddities.
            candidate = eval_summary.parents[3] / "train_fast" / "z32_alpha_learned" / eval_summary.parent.name / "summary.json"
            train_summary = candidate
        if train_summary in seen or not train_summary.exists():
            continue
        seen.add(train_summary)
        train_data = json.loads(train_summary.read_text())
        seed = train_config.get("train_seed") or fast_config.get("train_seed")
        m = re.search(r"seed(\d+)", str(train_summary))
        if seed is None and m:
            seed = int(m.group(1))
        rows.append({
            "train_seed": int(seed) if seed is not None else None,
            "config_id": "z32_alpha_learned",
            "initial_inner_lr": float(fast_config.get("inner_lr", 0.5)),
            "final_inner_lr": float(train_data.get("last", {}).get("inner_lr")),
            "train_summary_path": str(train_summary),
            "eval_summary_path": str(eval_summary),
            "finite": bool(train_data.get("last", {}).get("finite", True)) and bool(train_data.get("final_parameter_allfinite", True)),
        })
    return sorted(rows, key=lambda r: (-1 if r["train_seed"] is None else r["train_seed"]))


def write_report_zh(path: Path, *, summary: dict[str, Any], calibration_rows: list[dict[str, Any]], support_prefix_rows: list[dict[str, Any]]) -> None:
    lines = []
    lines.append("# Fast Adapter P1.5 Toy2D 汇总草稿")
    lines.append("")
    lines.append(f"状态：统计产物由 `{summary['split']}` split 生成。主端点固定为 `common_base` K=3；曲线阴影为训练 seed 最小-最大范围，不是 CI。")
    lines.append("")
    lines.append("## 覆盖")
    cov = summary["coverage"]
    lines.append(f"- configs: {len(cov['configs'])}；train seeds: {cov['train_seeds']}；support modes: {cov['support_modes']}；K: {cov['K_values']}。")
    lines.append(f"- 每组 domain count 范围：{cov['domain_count_min']}–{cov['domain_count_max']}；query count: {cov['query_counts']}。")
    lines.append("")
    if summary.get("selection_available"):
        sel = summary["selection"]
        lines.append("## Development 选择")
        lines.append(f"- Fast winner: `{summary['selected_fast_config']}`，规则为五个训练 seed 的 dev common_base K3 mean final goal error 最低。")
        lines.append(f"- History winner: `{summary['selected_history_config']}`，规则同上；history tie 优先 width256。")
        bg = sel.get("strongest_fixed_adapter_background", {})
        if bg:
            lines.append(f"- strongest fixed adapter 背景项：`{bg.get('config_id')}`，只按 dev common_base K0 frozen_z 选择，不作为主结论。")
    else:
        lines.append("## Development 选择")
        lines.append(f"当前输入不足以选择 winner：`{summary.get('selection_error')}`。partial dev smoke 不得用于选择或结论。")
    lines.append("")
    if summary.get("primary_pairs"):
        lines.append("## 主配对比较")
        lines.append("")
        lines.append("| comparison | error delta | error cross CI | success delta | success cross CI | seed同向 | gate |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in summary["primary_pairs"]:
            lines.append(
                f"| {row['comparison']} | {row['mean_error_delta']:.6f} | "
                f"[{row['cross_bootstrap_95ci_error_delta'][0]:.6f}, {row['cross_bootstrap_95ci_error_delta'][1]:.6f}] | "
                f"{row['mean_success_delta_pp']:.4f} pp | "
                f"[{row['cross_bootstrap_95ci_success_delta_pp'][0]:.4f}, {row['cross_bootstrap_95ci_success_delta_pp'][1]:.4f}] pp | "
                f"{row['per_seed_both_improve_count']}/5 | {row['superiority_gate_pass']} |"
            )
        lines.append("")
        lines.append("error delta 与 success delta 均为左侧 Fast correct_z 减右侧 comparator；error < 0 且 success > 0 表示改善。")
    lines.append("")
    if Path(path).with_name("learned_alpha_values.json").exists():
        lines.append("## Learned alpha")
        lines.append("`z32_alpha_learned` 的初始 alpha 为 0.5；报告中的 final inner_lr 来自训练 summary 的 `last.inner_lr`，不是 eval config 的初始化值。")
        lines.append("")
    lines.append("## Calibration / support 说明")
    lines.append(f"- calibration rows: {len(calibration_rows)}。L_adapt 使用同一 held-out base transitions，主要看同 checkpoint 随 K 的变化或归一化趋势，不能按不同 encoder 的绝对 loss 选模型。")
    lines.append(f"- support prefix rows: {len(support_prefix_rows)}。support_transition_metrics 只取 K=10 完整 prefix，避免把各 K 重复 prefix 计为独立样本。")
    lines.append("")
    lines.append("## 边界")
    lines.append("P1.5 仍是 2D toy 同分布机制消融，不是 RGB/DINO、Insert、机器人 rollout、OOD 或完整 RSI 证据。final split 必须读取 dev selection receipt，不允许从 final 选择 winner。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="eval_raw.jsonl files, directories, or globs")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["dev", "final"], default=None, help="optional expected split")
    parser.add_argument("--require-complete-grid", action="store_true")
    parser.add_argument("--write-dev-selection-receipt", type=Path)
    parser.add_argument("--dev-selection-receipt", type=Path, help="required for final summaries; gives selected winners")
    parser.add_argument("--calibration-inputs", nargs="*", default=[])
    parser.add_argument("--support-metrics-inputs", nargs="*", default=[])
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    if args.write_dev_selection_receipt and not args.require_complete_grid:
        raise ValueError("--write-dev-selection-receipt requires --require-complete-grid so all 11 configs x 5 seeds are present")
    inputs = _expand_inputs(args.inputs, filename="eval_raw.jsonl")
    metrics = load_metrics(inputs)
    split = sorted({m.split for m in metrics})[0]
    if args.split and split != args.split:
        raise ValueError(f"expected split {args.split}, got {split}")
    if split == "final" and not args.dev_selection_receipt:
        raise ValueError("final summaries must provide --dev-selection-receipt; never select winners from final data")
    coverage = validate_grid(metrics, require_complete=args.require_complete_grid)
    aggregate = aggregate_rows(metrics)
    curves = _curve_rows(aggregate)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "aggregate_by_config_seed_mode_arm_k.csv", aggregate)
    _write_csv(output_dir / "curves_by_config_mode_arm_k.csv", curves)

    selection = None
    selection_error = None
    if split == "dev":
        try:
            selection = select_dev(metrics)
        except Exception as exc:
            selection_error = str(exc)
            if args.require_complete_grid or args.write_dev_selection_receipt:
                raise
        if selection and args.write_dev_selection_receipt:
            args.write_dev_selection_receipt.parent.mkdir(parents=True, exist_ok=True)
            args.write_dev_selection_receipt.write_text(json.dumps(_jsonable(selection), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        selection = json.loads(args.dev_selection_receipt.read_text(encoding="utf-8"))

    primary_pairs = []
    if selection:
        fast_winner = selection["fast_winner"]["config_id"]
        history_winner = selection["history_winner"]["config_id"]
        primary_pairs = build_primary_pairs(metrics, fast_winner=fast_winner, history_winner=history_winner)
    else:
        fast_winner = None
        history_winner = None
    pair_flat = []
    per_seed_flat = []
    for row in primary_pairs:
        flat = {k: v for k, v in row.items() if k != "per_seed"}
        pair_flat.append(flat)
        for seed_row in row["per_seed"]:
            per_seed_flat.append({"comparison": row["comparison"], **seed_row})
    _write_csv(output_dir / "primary_paired_bootstrap.csv", pair_flat)
    _write_csv(output_dir / "primary_per_seed_directions.csv", per_seed_flat)

    calibration_rows = []
    if args.calibration_inputs:
        cal_inputs = _expand_inputs(args.calibration_inputs, filename="calibration_adapt_loss.jsonl")
        calibration_rows = summarize_calibration(read_calibration(cal_inputs))
        _write_csv(output_dir / "calibration_l_adapt.csv", calibration_rows)

    support_prefix_rows = []
    if args.support_metrics_inputs:
        support_inputs = _expand_inputs(args.support_metrics_inputs, filename="support_transition_metrics.jsonl")
        support_prefix_rows = summarize_support_prefix(read_support_metrics(support_inputs), full_prefix_k=10)
        _write_csv(output_dir / "support_prefix_k10_metrics.csv", support_prefix_rows)

    learned_rows = learned_alpha_values([Path(p) for p in args.inputs if Path(p).exists()])
    if learned_rows:
        (output_dir / "learned_alpha_values.json").write_text(json.dumps(_jsonable(learned_rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if selection:
        selected_configs_for_figures = [selection["fast_winner"]["config_id"], selection["history_winner"]["config_id"]]
        for cfg in HISTORY_CONFIGS:
            if cfg not in selected_configs_for_figures and any(m.config_id == cfg for m in metrics):
                selected_configs_for_figures.append(cfg)
        (output_dir / "selected_configs_for_figures.json").write_text(json.dumps({"configs": selected_configs_for_figures}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_figures:
        write_figures(output_dir, curves, calibration_rows)

    summary = {
        "format": "prove.fast-adapter-p15-summary.v1",
        "split": split,
        "input_files": [str(p) for p in inputs],
        "coverage": coverage,
        "selection_source": "dev_computed" if split == "dev" else str(args.dev_selection_receipt),
        "selected_fast_config": fast_winner,
        "selected_history_config": history_winner,
        "selection_available": selection is not None,
        "selection_error": selection_error,
        "selection": selection,
        "primary_pairs": primary_pairs,
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED},
        "notes": [
            "Rows are aggregated at domain level from per-domain query arrays; query rows are not treated as independent domains.",
            "Primary P1.5 endpoint is common_base K3. on_policy is secondary.",
            "Final summaries refuse to select winners from final data and require a dev selection receipt.",
            "Curve shading, when generated, is train-seed min-max range, not a confidence interval.",
            "Calibration L_adapt uses same held-out transitions and is mainly interpretable within checkpoint or after K0 normalization; do not select models by absolute cross-encoder calibration loss.",
            "Support transition metrics use only K=10 full prefixes when summarized, because smaller K reruns duplicate the same prefix steps.",
        ],
    }
    (output_dir / "SUMMARY.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report_zh(output_dir / "REPORT_ZH.md", summary=summary, calibration_rows=calibration_rows, support_prefix_rows=support_prefix_rows)


if __name__ == "__main__":
    main()
