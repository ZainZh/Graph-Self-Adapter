#!/usr/bin/env python3
"""Summarize Fast Adapter toy2d final/eval rows without training or evaluation.

Expected core raw schema, one row per query rollout:
split, train_seed, domain_seed, support_seed, query_seed, domain_index,
query_index, arm, K, z_source_domain_index, final_goal_error, success,
first_query_action_mse, query_action_mse, delta_norm_last, latent_delta_norm,
support_loss

The script first averages the 8 query rows within each (train_seed, domain_id,
arm, k), then computes per-seed domain means and same-domain paired differences.
It never treats query_id rows as independent domains.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RAW_SCHEMA_COLUMNS = [
    "split",
    "train_seed",
    "domain_seed",
    "support_seed",
    "query_seed",
    "domain_index",
    "query_index",
    "arm",
    "K",
    "z_source_domain_index",
    "final_goal_error",
    "success",
    "first_query_action_mse",
    "query_action_mse",
    "delta_norm_last",
    "latent_delta_norm",
    "support_loss",
]
REQUIRED_COLUMNS = set(RAW_SCHEMA_COLUMNS)
LEGACY_COLUMNS = {
    "train_seed",
    "domain_id",
    "query_id",
    "arm",
    "k",
    "goal_error",
    "success",
}


@dataclass(frozen=True)
class DomainMetric:
    train_seed: int
    domain_id: str
    arm: str
    k: int
    query_count: int
    mean_goal_error: float
    success_rate: float


def _parse_bool(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return 1.0
    if text in {"0", "false", "f", "no", "n"}:
        return 0.0
    number = float(text)
    if number not in {0.0, 1.0}:
        raise ValueError(f"success must be boolean/0/1, got {value!r}")
    return number


def _read_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    if not isinstance(rows, list):
        raise ValueError("raw input must be a row list")
    normalized = []
    for row in rows:
        keys = set(row)
        if REQUIRED_COLUMNS <= keys:
            item = dict(row)
            item["domain_id"] = f"{row['domain_seed']}:{row['domain_index']}"
            item["query_id"] = row["query_index"]
            item["k"] = row["K"]
            item["goal_error"] = row["final_goal_error"]
            normalized.append(item)
        elif LEGACY_COLUMNS <= keys:
            item = dict(row)
            item.setdefault("split", "unknown")
            item.setdefault("domain_seed", "legacy")
            item.setdefault("domain_index", row["domain_id"])
            item.setdefault("query_index", row["query_id"])
            item.setdefault("K", row["k"])
            item.setdefault("final_goal_error", row["goal_error"])
            item["domain_id"] = f"{item['domain_seed']}:{item['domain_index']}"
            normalized.append(item)
        else:
            missing = sorted(REQUIRED_COLUMNS - keys)
            raise ValueError(f"raw rows missing required columns: {missing}")
    _validate_raw_rows(normalized)
    return normalized


def _validate_raw_rows(rows: list[dict[str, object]]) -> None:
    splits = {str(row["split"]) for row in rows}
    if len(splits) != 1:
        raise ValueError(f"refuse mixed split raw rows: {sorted(splits)}")
    domain_seeds = {str(row["domain_seed"]) for row in rows}
    if len(domain_seeds) != 1:
        raise ValueError(f"refuse mixed domain_seed raw rows: {sorted(domain_seeds)}")
    seen = set()
    coverage: dict[tuple[int, str, int], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        key = (
            int(row["train_seed"]),
            str(row["domain_id"]),
            str(row["query_id"]),
            str(row["arm"]),
            int(row["k"]),
        )
        if key in seen:
            raise ValueError(f"duplicate raw row for train_seed/domain/query/arm/K: {key}")
        seen.add(key)
        coverage[(int(row["train_seed"]), str(row["arm"]), int(row["k"]))].add((str(row["domain_id"]), str(row["query_id"])))
    by_seed: dict[int, dict[tuple[str, int], set[tuple[str, str]]]] = defaultdict(dict)
    for (seed, arm, k), samples in coverage.items():
        by_seed[seed][(arm, k)] = samples
    for seed, groups in by_seed.items():
        reference_key, reference = next(iter(groups.items()))
        for group_key, samples in groups.items():
            if samples != reference:
                missing = sorted(reference - samples)[:5]
                extra = sorted(samples - reference)[:5]
                raise ValueError(
                    f"inconsistent domain/query coverage for train_seed {seed}: "
                    f"{group_key} differs from {reference_key}; missing={missing}, extra={extra}"
                )


def _domain_metrics(rows: Iterable[dict[str, object]]) -> list[DomainMetric]:
    grouped: dict[tuple[int, str, str, int], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["train_seed"]),
            str(row["domain_id"]),
            str(row["arm"]),
            int(row["k"]),
        )
        grouped[key].append((float(row["goal_error"]), _parse_bool(row["success"])))
    metrics = []
    for (train_seed, domain_id, arm, k), values in sorted(grouped.items()):
        if not values:
            continue
        metrics.append(
            DomainMetric(
                train_seed=train_seed,
                domain_id=domain_id,
                arm=arm,
                k=k,
                query_count=len(values),
                mean_goal_error=sum(v[0] for v in values) / len(values),
                success_rate=sum(v[1] for v in values) / len(values),
            )
        )
    return metrics


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _bootstrap_ci(values: list[float], *, iterations: int, seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def _summarize_by_seed(metrics: list[DomainMetric], *, bootstrap_iterations: int, bootstrap_seed: int) -> list[dict[str, object]]:
    buckets: dict[tuple[int, str, int], list[DomainMetric]] = defaultdict(list)
    for metric in metrics:
        buckets[(metric.train_seed, metric.arm, metric.k)].append(metric)
    rows = []
    for (train_seed, arm, k), items in sorted(buckets.items()):
        errors = [item.mean_goal_error for item in items]
        successes = [item.success_rate for item in items]
        rows.append(
            {
                "train_seed": train_seed,
                "arm": arm,
                "k": k,
                "domain_count": len(items),
                "query_count_min": min(item.query_count for item in items),
                "query_count_max": max(item.query_count for item in items),
                "mean_goal_error": _mean(errors),
                "goal_error_domain_bootstrap_ci95": _bootstrap_ci(errors, iterations=bootstrap_iterations, seed=bootstrap_seed + train_seed * 1009 + k),
                "success_rate": _mean(successes),
                "success_domain_bootstrap_ci95": _bootstrap_ci(successes, iterations=bootstrap_iterations, seed=bootstrap_seed + train_seed * 2003 + k),
            }
        )
    return rows


def _lookup(metrics: list[DomainMetric]) -> dict[tuple[int, str, str, int], DomainMetric]:
    return {(m.train_seed, m.domain_id, m.arm, m.k): m for m in metrics}


def _available_k(index: dict[tuple[int, str, str, int], DomainMetric], seed: int, domain: str, arm: str, preferred_k: int) -> int | None:
    if (seed, domain, arm, preferred_k) in index:
        return preferred_k
    if (seed, domain, arm, 0) in index:
        return 0
    ks = sorted(k for s, d, a, k in index if s == seed and d == domain and a == arm)
    return ks[0] if ks else None


def _pair_specs(fixed_arm: str, history_arm: str, shuffle_arm: str) -> list[tuple[str, str]]:
    return [
        ("z3_minus_fixed_z0", fixed_arm),
        ("ours_minus_history", history_arm),
        ("ours_minus_shuffle", shuffle_arm),
    ]


def _paired_delta_maps(
    metrics: list[DomainMetric],
    *,
    ours_arm: str,
    other_arm: str,
    pair_k: int,
) -> tuple[dict[int, dict[str, tuple[float, float]]], list[int]]:
    index = _lookup(metrics)
    output: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
    rhs_ks = set()
    for seed in sorted({m.train_seed for m in metrics}):
        domains = sorted({m.domain_id for m in metrics if m.train_seed == seed})
        for domain in domains:
            ours = index.get((seed, domain, ours_arm, pair_k))
            rhs_k = _available_k(index, seed, domain, other_arm, pair_k)
            other = index.get((seed, domain, other_arm, rhs_k)) if rhs_k is not None else None
            if ours is None or other is None:
                continue
            rhs_ks.add(rhs_k)
            output[seed][domain] = (ours.mean_goal_error - other.mean_goal_error, ours.success_rate - other.success_rate)
    return output, sorted(rhs_ks)


def _paired_rows(
    metrics: list[DomainMetric],
    *,
    ours_arm: str,
    fixed_arm: str,
    history_arm: str,
    shuffle_arm: str,
    pair_k: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    pairs = _pair_specs(fixed_arm, history_arm, shuffle_arm)
    seeds = sorted({m.train_seed for m in metrics})
    output = []
    for seed in seeds:
        for pair_name, other_arm in pairs:
            maps, rhs_ks = _paired_delta_maps(metrics, ours_arm=ours_arm, other_arm=other_arm, pair_k=pair_k)
            values = maps.get(seed, {})
            error_deltas = [delta[0] for _, delta in sorted(values.items())]
            success_deltas = [delta[1] for _, delta in sorted(values.items())]
            output.append(
                {
                    "train_seed": seed,
                    "pair": pair_name,
                    "ours_arm": ours_arm,
                    "ours_k": pair_k,
                    "other_arm": other_arm,
                    "other_k_values": rhs_ks,
                    "paired_domain_count": len(error_deltas),
                    "mean_goal_error_delta_ours_minus_other": _mean(error_deltas),
                    "mean_goal_error_improvement_other_minus_ours": None if not error_deltas else -_mean(error_deltas),
                    "goal_error_delta_domain_bootstrap_ci95": _bootstrap_ci(error_deltas, iterations=bootstrap_iterations, seed=bootstrap_seed + seed * 3001 + len(pair_name)),
                    "mean_success_delta_ours_minus_other": _mean(success_deltas),
                    "success_delta_domain_bootstrap_ci95": _bootstrap_ci(success_deltas, iterations=bootstrap_iterations, seed=bootstrap_seed + seed * 4001 + len(pair_name)),
                }
            )
    return output


def _paired_across_seed_conditional(
    metrics: list[DomainMetric],
    *,
    ours_arm: str,
    fixed_arm: str,
    history_arm: str,
    shuffle_arm: str,
    pair_k: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, object]]:
    rows = []
    for pair_name, other_arm in _pair_specs(fixed_arm, history_arm, shuffle_arm):
        maps, rhs_ks = _paired_delta_maps(metrics, ours_arm=ours_arm, other_arm=other_arm, pair_k=pair_k)
        seeds = sorted(seed for seed, values in maps.items() if values)
        if not seeds:
            rows.append({
                "pair": pair_name, "train_seed_count": 0, "common_domain_count": 0,
                "mean_goal_error_delta_ours_minus_other": None,
                "mean_goal_error_improvement_other_minus_ours": None,
                "goal_error_delta_conditional_domain_bootstrap_ci95": [None, None],
                "mean_success_delta_ours_minus_other": None,
                "success_delta_conditional_domain_bootstrap_ci95": [None, None],
                "per_seed_goal_error_delta_range": [None, None],
                "per_seed_success_delta_range": [None, None],
                "other_k_values": rhs_ks,
            })
            continue
        common_domains = sorted(set.intersection(*(set(maps[seed]) for seed in seeds)))
        per_seed_error_means = []
        per_seed_success_means = []
        for seed in seeds:
            errors = [maps[seed][domain][0] for domain in common_domains]
            successes = [maps[seed][domain][1] for domain in common_domains]
            if errors:
                per_seed_error_means.append(sum(errors) / len(errors))
                per_seed_success_means.append(sum(successes) / len(successes))
        mean_error = _mean(per_seed_error_means)
        mean_success = _mean(per_seed_success_means)
        error_draws = []
        success_draws = []
        if common_domains:
            rng = random.Random(bootstrap_seed + len(pair_name) * 5003)
            n = len(common_domains)
            for _ in range(bootstrap_iterations):
                sampled = [common_domains[rng.randrange(n)] for _ in range(n)]
                seed_error_means = []
                seed_success_means = []
                for seed in seeds:
                    seed_error_means.append(sum(maps[seed][domain][0] for domain in sampled) / n)
                    seed_success_means.append(sum(maps[seed][domain][1] for domain in sampled) / n)
                error_draws.append(sum(seed_error_means) / len(seed_error_means))
                success_draws.append(sum(seed_success_means) / len(seed_success_means))
        rows.append({
            "pair": pair_name,
            "train_seed_count": len(seeds),
            "train_seeds": seeds,
            "common_domain_count": len(common_domains),
            "ours_arm": ours_arm,
            "ours_k": pair_k,
            "other_arm": other_arm,
            "other_k_values": rhs_ks,
            "mean_goal_error_delta_ours_minus_other": mean_error,
            "mean_goal_error_improvement_other_minus_ours": None if mean_error is None else -mean_error,
            "goal_error_delta_conditional_domain_bootstrap_ci95": [_percentile(error_draws, 0.025), _percentile(error_draws, 0.975)],
            "mean_success_delta_ours_minus_other": mean_success,
            "success_delta_conditional_domain_bootstrap_ci95": [_percentile(success_draws, 0.025), _percentile(success_draws, 0.975)],
            "per_seed_goal_error_delta_range": [min(per_seed_error_means), max(per_seed_error_means)] if per_seed_error_means else [None, None],
            "per_seed_success_delta_range": [min(per_seed_success_means), max(per_seed_success_means)] if per_seed_success_means else [None, None],
        })
    return rows


def _train_seed_ranges(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["arm"]), int(row["k"]))].append(row)
    ranges = []
    for (arm, k), rows in sorted(grouped.items()):
        errors = [row["mean_goal_error"] for row in rows if row["mean_goal_error"] is not None]
        successes = [row["success_rate"] for row in rows if row["success_rate"] is not None]
        ranges.append(
            {
                "arm": arm,
                "k": k,
                "train_seed_count": len(rows),
                "mean_goal_error_minmax_across_train_seeds": [min(errors), max(errors)] if errors else [None, None],
                "success_rate_minmax_across_train_seeds": [min(successes), max(successes)] if successes else [None, None],
            }
        )
    return ranges


def _plot_curves(seed_rows: list[dict[str, object]], output_dir: Path, *, title_suffix: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional environment package
        return [f"plot_skipped: {exc}"]

    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["arm"]), int(row["k"]))].append(row)
    arms = sorted({arm for arm, _ in grouped})
    paths = []
    for field, ylabel, stem in (
        ("mean_goal_error", "query mean final L2 goal error", "goal_error_curve"),
        ("success_rate", "success rate", "success_curve"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        for arm in arms:
            ks = sorted(k for a, k in grouped if a == arm)
            means = []
            lows = []
            highs = []
            for k in ks:
                values = [float(row[field]) for row in grouped[(arm, k)] if row[field] is not None]
                means.append(sum(values) / len(values) if values else math.nan)
                lows.append(min(values) if values else math.nan)
                highs.append(max(values) if values else math.nan)
            ax.plot(ks, means, marker="o", label=arm)
            if len([x for x in lows if not math.isnan(x)]) >= 2:
                ax.fill_between(ks, lows, highs, alpha=0.12)
        ax.set_xlabel("support transitions K")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Fast Adapter toy2d {title_suffix}".strip())
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            out = output_dir / f"{stem}.{suffix}"
            fig.savefig(out)
            paths.append(str(out))
        plt.close(fig)
    return paths


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(summary: dict[str, object], path: Path) -> None:
    lines = [
        "# Fast Adapter toy2d 统计草稿",
        "",
        "本文由 `tools/fast_adapter/summarize_toy2d.py` 从 raw query rows 自动生成。它只汇总数据，不替 root 作科学验收结论。",
        "",
        "## 口径",
        "",
        "- raw schema: `split, train_seed, domain_seed, support_seed, query_seed, domain_index, query_index, arm, K, z_source_domain_index, final_goal_error, success, first_query_action_mse, query_action_mse, delta_norm_last, latent_delta_norm, support_loss`。",
        "- 先对每个 train seed / domain / arm / K 的 8 个 query 求均值，再按 domain 做统计和配对。",
        "- 单 seed 指标的 CI 在该训练模型内对 domain bootstrap；跨 seed 配对差使用相同 domain 抽样后先算每个训练 seed 的均值，再对已训练 seed 求平均。",
        "- 训练 seed 间范围单列；不把 seed * domain 展平成独立 bootstrap 样本。",
        "- `shuffle-domain-zK` 仅是机制特异性诊断，不用于选择最好 K 或 checkpoint。",
        "- 该 toy 是同分布新 domain 组合，不是 OOD；不是 RGB、不是 Insert、不是完整 RSI 证据。",
        "",
        "## 每 seed 指标",
        "",
        "| seed | arm | K | domains | mean final L2 | success |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary["per_seed_metrics"]:
        lines.append(
            f"| {row['train_seed']} | {row['arm']} | {row['k']} | {row['domain_count']} | {_fmt(row['mean_goal_error'])} | {_fmt(row['success_rate'])} |"
        )
    lines.extend([
        "",
        "## 同 domain 配对差",
        "",
        "正的 `error improvement` 表示 other arm 的 goal error 高于 ours。",
        "",
        "| seed | pair | paired domains | error improvement | success delta ours-other |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in summary["paired_differences"]:
        lines.append(
            f"| {row['train_seed']} | {row['pair']} | {row['paired_domain_count']} | {_fmt(row['mean_goal_error_improvement_other_minus_ours'])} | {_fmt(row['mean_success_delta_ours_minus_other'])} |"
        )
    lines.extend([
        "",
        "## 跨 seed 条件 paired bootstrap",
        "",
        "这里每次 bootstrap 使用同一组 sampled domains，并在每个训练 seed 内求均值后再跨 seed 平均。",
        "",
        "| pair | seeds | common domains | error improvement | success delta ours-other | seed error range |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in summary["paired_differences_across_seed_conditional"]:
        lines.append(
            f"| {row['pair']} | {row['train_seed_count']} | {row['common_domain_count']} | {_fmt(row['mean_goal_error_improvement_other_minus_ours'])} | {_fmt(row['mean_success_delta_ours_minus_other'])} | {_fmt(row['per_seed_goal_error_delta_range'])} |"
        )
    lines.extend([
        "",
        "## 图",
        "",
    ])
    for plot in summary.get("plots", []):
        lines.append(f"- `{plot}`")
    lines.extend([
        "",
        "## 待 root 验收后填写",
        "",
        "- 是否支持 adaptation 优于同 checkpoint fixed z0。",
        "- 是否支持 gradient adaptation 优于 history-conditioned 无梯度方法。",
        "- 是否存在 prediction loss 降低但 query control 未提升的分离情况。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(args: argparse.Namespace) -> dict[str, object]:
    rows = _read_rows(args.input)
    metrics = _domain_metrics(rows)
    seed_rows = _summarize_by_seed(metrics, bootstrap_iterations=args.bootstrap_iterations, bootstrap_seed=args.bootstrap_seed)
    pairs = _paired_rows(
        metrics,
        ours_arm=args.ours_arm,
        fixed_arm=args.fixed_arm,
        history_arm=args.history_arm,
        shuffle_arm=args.shuffle_arm,
        pair_k=args.pair_k,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    paired_across_seed = _paired_across_seed_conditional(
        metrics,
        ours_arm=args.ours_arm,
        fixed_arm=args.fixed_arm,
        history_arm=args.history_arm,
        shuffle_arm=args.shuffle_arm,
        pair_k=args.pair_k,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = _plot_curves(seed_rows, args.output_dir, title_suffix=args.title_suffix)
    summary = {
        "format": "prove.fast-adapter-toy2d-summary.v1",
        "raw_schema": RAW_SCHEMA_COLUMNS,
        "input_path": str(args.input),
        "domain_aggregation": "mean over query rows within each train_seed/domain_id/arm/k before domain statistics",
        "query_rows": len(rows),
        "domain_cells": len(metrics),
        "bootstrap": {"unit": "domain within train_seed for per-seed metrics; common-domain resampling then averaging across train seeds for paired cross-seed deltas", "iterations": args.bootstrap_iterations, "seed": args.bootstrap_seed},
        "arms": {"ours": args.ours_arm, "fixed": args.fixed_arm, "history": args.history_arm, "shuffle": args.shuffle_arm},
        "pair_k": args.pair_k,
        "per_seed_metrics": seed_rows,
        "paired_differences": pairs,
        "paired_differences_across_seed_conditional": paired_across_seed,
        "train_seed_ranges": _train_seed_ranges(seed_rows),
        "plots": plots,
        "claim_boundary": [
            "same-distribution new domain combinations, not OOD",
            "toy2d state control, not RGB perception",
            "not Insert validation",
            "not complete RSI evidence",
            "summary does not choose a best K or make root acceptance claims",
        ],
    }
    (args.output_dir / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(summary, args.output_dir / "REPORT_ZH.md")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV, JSON, or JSONL raw query rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ours-arm", default="ours")
    parser.add_argument("--fixed-arm", default="frozen_z")
    parser.add_argument("--history-arm", default="history")
    parser.add_argument("--shuffle-arm", default="shuffle_z")
    parser.add_argument("--pair-k", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260921)
    parser.add_argument("--title-suffix", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
