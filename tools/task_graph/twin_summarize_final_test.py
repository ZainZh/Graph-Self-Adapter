#!/usr/bin/env python3
"""Validate and summarize baseline and selected Twin RSI final-test cohorts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.task_graph import twin_export_packages as export  # noqa: E402

RUN_SCHEMA = "graph-self-adapter.twin-run.v1"
COHORT_SCHEMA = "graph-self-adapter.twin-cohort-summary.v1"
PAIRED_SCHEMA = "graph-self-adapter.twin-paired-adaptation.v1"


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _dump_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _strict_success(report: Mapping[str, Any]) -> bool:
    outcome = report.get("original_task_outcome", {})
    return bool(report.get("graph_completed") and not report.get("graph_failure") and
                outcome.get("numerical_safety", {}).get("passed") and
                outcome.get("task_outcome", {}).get("passed"))


def _failure_stage(report: Mapping[str, Any]) -> str:
    if str(report.get("graph_failure", "")).startswith("entry"):
        records = report.get("entry_records", [])
        return str(records[-1].get("stage_id", "entry")) if records else "entry"
    trace = report.get("trace", [])
    return str(trace[-1].get("stage_id", "entry")) if trace else "entry"


def _selection(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = path.resolve()
    document = _load(path)
    if (document.get("schema") != "graph-self-adapter.twin-final-selection.v1" or
            document.get("status") != "frozen"):
        raise ValueError(f"invalid final selection: {path}")
    rows = {str(row["task"]): dict(row) for row in document.get("tasks", [])}
    if not rows or len(rows) != len(document.get("tasks", [])):
        raise ValueError(f"selection task rows are missing or duplicated: {path}")
    return document, rows


def _validate_execution(directory: Path, selection_path: Path,
                        expected_controllers: set[str], trials: int,
                        execution_review: Path | None = None) -> dict[str, Any] | None:
    freeze_path, summary_path = directory / "COHORT_FREEZE.json", directory / "EXECUTION_SUMMARY.json"
    freeze, summary = _load(freeze_path), _load(summary_path)
    if not freeze.get("frozen_before_rollouts"):
        raise ValueError(f"cohort was not frozen before execution: {directory}")
    if freeze.get("selection", {}).get("sha256") != export.sha256(selection_path):
        raise ValueError(f"cohort selection hash mismatch: {directory}")
    if set(freeze.get("controllers", [])) != expected_controllers:
        raise ValueError(f"cohort controller set mismatch: {directory}")
    if int(freeze.get("trials_per_task_per_arm", -1)) != trials:
        raise ValueError(f"cohort trial count mismatch: {directory}")
    if summary.get("cohort_freeze_sha256") != export.sha256(freeze_path):
        raise ValueError(f"execution summary does not bind cohort freeze: {directory}")
    if (summary.get("binding_changes_during_execution") or
            summary.get("invalid_if_bindings_changed") or
            not summary.get("pre_post_bindings_match", False)):
        raise ValueError(f"cohort execution is invalid: {directory}")
    errors = summary.get("errors", [])
    run_errors = [row for row in summary.get("runs", [])
                  if row.get("returncode") or row.get("error")]
    error_rows = {f"{row['task']}:{int(row['trial'])}:{row['controller']}": row
                  for row in errors}
    run_error_rows = {f"{row['task']}:{int(row['trial'])}:{row['controller']}": row
                      for row in run_errors}
    if error_rows != run_error_rows or len(error_rows) != len(errors):
        raise ValueError(f"execution error lists disagree: {directory}")
    if not errors:
        if execution_review is not None:
            raise ValueError("execution review supplied for a cohort with no errors")
        return None
    if execution_review is None:
        raise ValueError(f"cohort errors require an execution review: {directory}")
    review_path = execution_review.resolve()
    review = _load(review_path)
    if (review.get("schema") != "graph-self-adapter.twin-execution-review.v1" or
            review.get("status") != "complete" or review.get("unresolved_errors") or
            review.get("source_and_assets_unchanged") is not True):
        raise ValueError(f"invalid execution review: {review_path}")
    if (review.get("execution_summary", {}).get("sha256") != export.sha256(summary_path) or
            review.get("cohort_freeze", {}).get("sha256") != export.sha256(freeze_path) or
            review.get("selection", {}).get("sha256") != export.sha256(selection_path)):
        raise ValueError(f"execution review binding mismatch: {review_path}")
    expected_keys = {f"{row['task']}:{int(row['trial'])}:{row['controller']}" for row in errors}
    adjudications = review.get("adjudications", [])
    if {row.get("key") for row in adjudications} != expected_keys or len(adjudications) != len(errors):
        raise ValueError(f"execution review does not uniquely resolve all errors: {review_path}")
    for row in adjudications:
        if (row.get("disposition") !=
                "completed_strict_failure_with_post_payload_mujoco_qacc_warning" or
                row.get("counts_as_strict_success") is not False or
                row.get("warning", {}).get("diagnostic") != "numerical_instability"):
            raise ValueError(f"unsupported execution adjudication: {row.get('key')}")
        for name in ("stdout", "stderr", "report", "visual_trace"):
            binding = row.get(name, {})
            path = Path(binding.get("path", ""))
            if not path.is_file() or export.sha256(path) != binding.get("sha256"):
                raise ValueError(f"execution review {name} hash mismatch: {row.get('key')}")
    return review


def load_reports(directory: Path, selection_path: Path, controllers: set[str],
                 trials: int, execution_review: Path | None = None
                 ) -> tuple[dict[str, Any], dict[tuple[str, str, int], tuple[Path, dict[str, Any]]]]:
    directory, selection_path = directory.resolve(), selection_path.resolve()
    selection, selected_rows = _selection(selection_path)
    _validate_execution(directory, selection_path, controllers, trials, execution_review)
    selection_sha = export.sha256(selection_path)
    reports: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    for path in directory.glob("*.json"):
        report = _load(path)
        if report.get("schema") != RUN_SCHEMA:
            continue
        task, controller = str(report.get("task")), str(report.get("controller"))
        condition = report.get("condition", {})
        trial = condition.get("trial")
        if task not in selected_rows or controller not in controllers or not isinstance(trial, int):
            raise ValueError(f"unexpected report identity: {path}")
        key = (task, controller, trial)
        if key in reports:
            raise ValueError(f"duplicate report key {key}: {path}")
        row = selected_rows[task]
        if report.get("candidate_id") != row["candidate_id"]:
            raise ValueError(f"candidate mismatch: {path}")
        if report.get("condition_set") != "test" or condition.get("seed") != 71005:
            raise ValueError(f"report is not from frozen test conditions: {path}")
        if condition.get("task") != task or trial not in range(trials):
            raise ValueError(f"condition task/trial mismatch: {path}")
        if report.get("selection_row") != row:
            raise ValueError(f"runtime selection input row mismatch: {path}")
        bindings = report.get("bindings", {})
        required = {"graph_sha256": row["graph_sha256"],
                    "model_sha256": row["model_sha256"],
                    "candidate_freeze_sha256": row["candidate_freeze_sha256"],
                    "selection_freeze_sha256": selection_sha,
                    "condition_sha256": _canonical_sha(condition)}
        for name, expected in required.items():
            if bindings.get(name) != expected:
                raise ValueError(f"{name} mismatch: {path}")
        solution = export.solution_path(row["packages"], task, selection_path.parent)
        if export.sha256(solution / "refs/visible_centroids.json") != row["reference_sha256"]:
            raise ValueError(f"reference input hash mismatch: {path}")
        freeze = _load(solution / "CANDIDATE_FREEZE.json")
        if (freeze.get("references_sha256") != row["reference_sha256"] or
                freeze.get("graph_sha256") != row["graph_sha256"] or
                freeze.get("model_sha256") != row["model_sha256"]):
            raise ValueError(f"candidate input freeze mismatch: {path}")
        trace = bindings.get("visual_trace", {})
        trace_path = Path(trace.get("path", ""))
        if not trace_path.is_file() or export.sha256(trace_path) != trace.get("sha256"):
            raise ValueError(f"visual trace hash mismatch: {path}")
        if _strict_success(report) != report.get("strict_success"):
            raise ValueError(f"strict success is inconsistent: {path}")
        reports[key] = (path, report)
    expected = {(task, controller, trial) for task in selected_rows
                for controller in controllers for trial in range(trials)}
    if set(reports) != expected:
        missing, extra = sorted(expected - set(reports)), sorted(set(reports) - expected)
        raise ValueError(f"incomplete cohort {directory}; missing={missing}, extra={extra}")
    return selection, reports


def cohort_summary(selection: Mapping[str, Any], reports: Mapping,
                   controller: str, trials: int,
                   execution_review: Mapping[str, Any] | None = None) -> dict[str, Any]:
    tasks = [str(row["task"]) for row in selection["tasks"]]
    result = {"schema": COHORT_SCHEMA, "candidate_id": selection["candidate_id"],
              "condition_set": "test", "controller": controller,
              "trials": len(tasks) * trials, "successes": 0, "tasks": [],
              "condition_seed": 71005, "includes_nominal_case": False,
              "independent_test": True, "source_unchanged_during_execution": True,
              "strict_success": "original goal AND numerical safety AND valid complete graph",
              "task_candidates": {row["task"]: row["candidate_id"]
                                  for row in selection["tasks"]},
              "interpretation": "Complete-task independent test results; baseline and selected denominators remain separate."}
    diagnostics = [] if execution_review is None else [
        row for row in execution_review.get("adjudications", [])
        if row.get("controller") == controller]
    result["numerical_instability_diagnostics"] = {
        "episodes": len(diagnostics),
        "run_keys": [row["key"] for row in diagnostics],
        "interpretation": ("Post-payload MuJoCo QACC warnings are retained as numerical-instability diagnostics; "
                           "these episodes remain strict failures and evaluator safety fields are not restated as "
                           "100% numerical stability.")}
    for task in tasks:
        rows = [reports[(task, controller, trial)] for trial in range(trials)]
        successes = sum(bool(report["strict_success"]) for _, report in rows)
        failures = Counter((report.get("graph_failure") or
                            "original_goal_or_safety_failed")
                           for _, report in rows if not report["strict_success"])
        stages = Counter(_failure_stage(report) for _, report in rows
                         if not report["strict_success"])
        result["tasks"].append({
            "task": task, "candidate_id": selection["tasks"][tasks.index(task)]["candidate_id"],
            "trials": trials, "successes": successes, "rate": successes / trials,
            "physical_goal_successes": sum(bool(report["original_task_outcome"]["task_outcome"]["passed"])
                                           for _, report in rows),
            "safety_passes": sum(bool(report["original_task_outcome"]["numerical_safety"]["passed"])
                                 for _, report in rows),
            "failures": dict(failures), "failure_stages": dict(stages),
            "reports": [{"path": str(path), "sha256": export.sha256(path)}
                        for path, _ in rows]})
        result["successes"] += successes
    return result


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _binomial_two_sided(discordant_a: int, discordant_b: int) -> float:
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(discordant_a, discordant_b) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def selected_paired(selection: Mapping[str, Any], reports: Mapping,
                    trials: int,
                    execution_review: Mapping[str, Any] | None = None) -> dict[str, Any]:
    task_rows, totals = [], Counter()
    for selected in selection["tasks"]:
        task, outcomes = selected["task"], Counter()
        updates, loss_nonincrease, action_changes = 0, [], []
        for trial in range(trials):
            fixed = reports[(task, "fixed_z", trial)][1]
            online = reports[(task, "online_z", trial)][1]
            for key in ("condition_sha256", "graph_sha256", "model_sha256",
                        "candidate_freeze_sha256", "selection_freeze_sha256"):
                if fixed["bindings"][key] != online["bindings"][key]:
                    raise ValueError(f"selected pair {task}/{trial} mismatches {key}")
            if fixed["condition"] != online["condition"] or fixed["selection_row"] != online["selection_row"]:
                raise ValueError(f"selected pair {task}/{trial} mismatches condition/input")
            f, o = bool(fixed["strict_success"]), bool(online["strict_success"])
            label = "both_success" if f and o else "fixed_only" if f else "online_only" if o else "both_fail"
            outcomes[label] += 1
            totals[label] += 1
            for sample in online.get("trace", []):
                controller = sample.get("controller", {})
                latent = controller.get("latent", {})
                before, after = latent.get("loss_before"), latent.get("loss_after")
                if latent.get("gradient_local") is not None:
                    if latent.get("admitted") is not True:
                        raise ValueError(
                            f"gradient update was not admitted: {task}/{trial}")
                    if before is None or after is None:
                        raise ValueError(
                            f"gradient update has no matched loss: {task}/{trial}")
                    updates += 1
                    loss_nonincrease.append(float(after) <= float(before) + 1e-12)
                change = controller.get("action_online_minus_fixed_l2_rad")
                if change is not None:
                    action_changes.append(float(change))
        task_rows.append({"task": task, "candidate_id": selected["candidate_id"],
                          "pairs": trials, "outcomes": dict(outcomes),
                          "online_gradient_updates": updates,
                          "updated_loss_nonincrease_fraction": _mean([float(x) for x in loss_nonincrease]),
                          "mean_action_change_rad": _mean(action_changes),
                          "max_action_change_rad": max(action_changes) if action_changes else None})
    diagnostics = [] if execution_review is None else execution_review.get("adjudications", [])
    return {"schema": PAIRED_SCHEMA, "candidate_id": selection["candidate_id"],
            "condition_set": "test", "pairs": len(selection["tasks"]) * trials,
            "task_candidates": {row["task"]: row["candidate_id"] for row in selection["tasks"]},
            "outcomes": dict(totals),
            "mcnemar_exact_two_sided_p": _binomial_two_sided(totals["fixed_only"], totals["online_only"]),
            "numerical_instability_diagnostics": {
                "episodes": len(diagnostics),
                "run_keys": [row["key"] for row in diagnostics],
                "interpretation": "Reviewed QACC-warning episodes remain strict failures."},
            "tasks": task_rows}


def summarize(baseline_selection: Path, baseline_directory: Path,
              selected_selection: Path, selected_directory: Path,
              output_directory: Path, *, trials: int = 10,
              baseline_execution_review: Path | None = None,
              selected_execution_review: Path | None = None) -> dict[str, Path]:
    baseline_review = (None if baseline_execution_review is None else
                       _load(baseline_execution_review.resolve()))
    selected_review = (None if selected_execution_review is None else
                       _load(selected_execution_review.resolve()))
    baseline_selection_doc, baseline = load_reports(
        baseline_directory, baseline_selection, {"fixed_z"}, trials,
        baseline_execution_review)
    selected_selection_doc, selected = load_reports(
        selected_directory, selected_selection, {"fixed_z", "online_z"}, trials,
        selected_execution_review)
    if {row["task"] for row in baseline_selection_doc["tasks"]} != {
            row["task"] for row in selected_selection_doc["tasks"]}:
        raise ValueError("baseline and selected task sets differ")
    # Test conditions are candidate-independent and must match across outer comparison.
    for task in (row["task"] for row in selected_selection_doc["tasks"]):
        for trial in range(trials):
            left, right = baseline[(task, "fixed_z", trial)][1], selected[(task, "fixed_z", trial)][1]
            if (left["bindings"]["condition_sha256"] != right["bindings"]["condition_sha256"] or
                    left["condition"] != right["condition"]):
                raise ValueError(f"baseline/selected condition mismatch: {task}/{trial}")
    outputs = {
        "baseline": output_directory / "INITIAL_GRAPH_FIXED_REPORT.json",
        "selected_fixed": output_directory / "SELECTED_FIXED_REPORT.json",
        "selected_online": output_directory / "SELECTED_ONLINE_REPORT.json",
        "selected_paired": output_directory / "SELECTED_PAIRED_REPORT.json",
    }
    if output_directory.exists() and any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"report output already exists: {output_directory}")
    _dump_new(outputs["baseline"], cohort_summary(
        baseline_selection_doc, baseline, "fixed_z", trials, baseline_review))
    _dump_new(outputs["selected_fixed"], cohort_summary(
        selected_selection_doc, selected, "fixed_z", trials, selected_review))
    _dump_new(outputs["selected_online"], cohort_summary(
        selected_selection_doc, selected, "online_z", trials, selected_review))
    _dump_new(outputs["selected_paired"], selected_paired(
        selected_selection_doc, selected, trials, selected_review))
    return outputs


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-selection", type=Path, required=True)
    p.add_argument("--baseline-directory", type=Path, required=True)
    p.add_argument("--selected-selection", type=Path, required=True)
    p.add_argument("--selected-directory", type=Path, required=True)
    p.add_argument("--baseline-execution-review", type=Path)
    p.add_argument("--selected-execution-review", type=Path)
    p.add_argument("--output-directory", type=Path, required=True)
    p.add_argument("--trials", type=int, default=10)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    outputs = summarize(args.baseline_selection, args.baseline_directory,
                        args.selected_selection, args.selected_directory,
                        args.output_directory, trials=args.trials,
                        baseline_execution_review=args.baseline_execution_review,
                        selected_execution_review=args.selected_execution_review)
    json.dump({name: str(path) for name, path in outputs.items()}, sys.stdout,
              indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
