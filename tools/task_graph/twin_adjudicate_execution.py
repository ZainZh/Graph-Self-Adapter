#!/usr/bin/env python3
"""Adjudicate a narrow post-JSON MuJoCo warning without rewriting cohort evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.task_graph import twin_export_packages as export  # noqa: E402
from tools.task_graph import twin_summarize_final_test as summary  # noqa: E402

SCHEMA = "graph-self-adapter.twin-execution-review.v1"
WARNING_RE = re.compile(
    r"\s*WARNING: Nan, Inf or huge value in QACC at DOF (?P<dof>\d+)\. "
    r"The simulation is unstable\. Time = (?P<time>\d+(?:\.\d+)?)\.\s*\Z")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def parse_stdout(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    stripped = text.lstrip()
    payload, end = json.JSONDecoder().raw_decode(stripped)
    if not isinstance(payload, dict):
        raise ValueError("runner stdout prefix must be a JSON object")
    suffix = stripped[end:]
    match = WARNING_RE.fullmatch(suffix)
    if match is None:
        raise ValueError("stdout suffix is not the single approved MuJoCo QACC warning")
    return payload, {"text": suffix.strip(), "dof": int(match.group("dof")),
                     "physical_time_sec": float(match.group("time")),
                     "diagnostic": "numerical_instability"}


def _key(row: Mapping[str, Any]) -> str:
    return f"{row['task']}:{int(row['trial'])}:{row['controller']}"


def _validate_frozen_execution(directory: Path, selection_path: Path) -> tuple[dict, dict]:
    freeze_path = directory / "COHORT_FREEZE.json"
    execution_path = directory / "EXECUTION_SUMMARY.json"
    freeze, execution = _load(freeze_path), _load(execution_path)
    if not freeze.get("frozen_before_rollouts"):
        raise ValueError("cohort was not frozen before execution")
    if freeze.get("selection", {}).get("sha256") != export.sha256(selection_path):
        raise ValueError("cohort selection binding mismatch")
    if execution.get("cohort_freeze_sha256") != export.sha256(freeze_path):
        raise ValueError("execution summary does not bind COHORT_FREEZE")
    if (execution.get("binding_changes_during_execution") or
            execution.get("invalid_if_bindings_changed") or
            not execution.get("pre_post_bindings_match", False)):
        raise ValueError("source/assets changed during cohort")
    pre = {(row["path"], row["kind"]): row["sha256"] for row in freeze.get("bindings", [])}
    post = {(row["path"], row["kind"]): row["sha256"]
            for row in execution.get("post_execution_bindings", [])}
    if pre != post:
        raise ValueError("pre/post source and asset bindings differ")
    for (raw, _), digest in pre.items():
        path = Path(raw)
        if not path.is_file() or export.sha256(path) != digest:
            raise ValueError(f"frozen source/asset changed after execution: {path}")
    return freeze, execution


def _validate_report(report_path: Path, report: Mapping[str, Any], payload: Mapping[str, Any],
                     row: Mapping[str, Any], selection_path: Path) -> None:
    task, controller, trial = row["task"], row["controller"], int(row["trial"])
    if report.get("schema") != summary.RUN_SCHEMA:
        raise ValueError(f"wrong run-report schema: {report_path}")
    if (report.get("task") != task or report.get("candidate_id") != row["candidate_id"] or
            report.get("condition_set") != "test" or report.get("controller") != controller or
            report.get("condition", {}).get("trial") != trial or
            report.get("condition", {}).get("task") != task or
            report.get("condition", {}).get("seed") != 71005):
        raise ValueError(f"run report identity mismatch: {report_path}")
    if payload.get("output") != str(report_path.resolve()):
        raise ValueError("stdout payload output does not identify sealed run report")
    if (payload.get("strict_success") is not False or report.get("strict_success") is not False or
            summary._strict_success(report) is not False):
        raise ValueError("QACC-warning adjudication is restricted to strict failures")
    if payload.get("failure") != report.get("graph_failure"):
        raise ValueError("stdout failure differs from sealed report")

    selection = _load(selection_path)
    matches = [selected for selected in selection.get("tasks", [])
               if selected.get("task") == task and
               selected.get("candidate_id") == row["candidate_id"]]
    if len(matches) != 1 or report.get("selection_row") != matches[0]:
        raise ValueError("sealed report does not match its unique selection row")
    selected = matches[0]
    bindings = report.get("bindings", {})
    expected = {"selection_freeze_sha256": export.sha256(selection_path),
                "graph_sha256": selected["graph_sha256"],
                "model_sha256": selected["model_sha256"],
                "candidate_freeze_sha256": selected["candidate_freeze_sha256"],
                "condition_sha256": _canonical_sha(report["condition"])}
    if any(bindings.get(key) != value for key, value in expected.items()):
        raise ValueError("sealed report artifact/condition bindings mismatch")
    solution = export.solution_path(selected["packages"], task, selection_path.parent)
    if export.sha256(solution / "refs/visible_centroids.json") != selected["reference_sha256"]:
        raise ValueError("selected reference hash mismatch")
    candidate = _load(solution / "CANDIDATE_FREEZE.json")
    if any(candidate.get(key) != selected[name] for key, name in
           (("graph_sha256", "graph_sha256"), ("model_sha256", "model_sha256"),
            ("references_sha256", "reference_sha256"))):
        raise ValueError("candidate freeze artifact hashes mismatch")
    trace = bindings.get("visual_trace", {})
    trace_path = Path(trace.get("path", ""))
    if not trace_path.is_file() or export.sha256(trace_path) != trace.get("sha256"):
        raise ValueError("visual trace binding mismatch")


def adjudicate(directory: Path, selection_path: Path, output: Path) -> dict[str, Any]:
    directory, selection_path, output = directory.resolve(), selection_path.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite execution review: {output}")
    freeze, execution = _validate_frozen_execution(directory, selection_path)
    errors = execution.get("errors", [])
    if not isinstance(errors, list):
        raise ValueError("execution errors must be a list")
    adjudications = []
    for row in errors:
        if row.get("returncode") != 0 or not str(row.get("error", "")).startswith(
                "runner stdout was not JSON: Extra data:"):
            raise ValueError(f"unreviewable execution error: {_key(row)}")
        stem = f"{row['task']}_{int(row['trial']):02d}_{row['controller']}"
        stdout, stderr = directory / f"{stem}.stdout", directory / f"{stem}.stderr"
        if not stdout.is_file() or not stderr.is_file() or stderr.read_bytes() != b"":
            raise ValueError(f"review requires present stdout and empty stderr: {_key(row)}")
        payload, warning = parse_stdout(stdout.read_text())
        report_path = Path(payload.get("output", "")).resolve()
        if not report_path.is_file() or report_path.parent != directory:
            raise ValueError(f"sealed report is missing/outside cohort: {_key(row)}")
        report = _load(report_path)
        _validate_report(report_path, report, payload, row, selection_path)
        trace_path = Path(report["bindings"]["visual_trace"]["path"])
        adjudications.append({
            "key": _key(row), "task": row["task"], "trial": int(row["trial"]),
            "controller": row["controller"], "candidate_id": row["candidate_id"],
            "disposition": "completed_strict_failure_with_post_payload_mujoco_qacc_warning",
            "counts_as_strict_success": False, "warning": warning,
            "stdout": {"path": str(stdout), "sha256": export.sha256(stdout)},
            "stderr": {"path": str(stderr), "sha256": export.sha256(stderr)},
            "report": {"path": str(report_path), "sha256": export.sha256(report_path)},
            "visual_trace": {"path": str(trace_path), "sha256": export.sha256(trace_path)},
            "payload": payload})
    document = {
        "schema": SCHEMA, "status": "complete", "cohort": freeze.get("cohort"),
        "adjudicator_source": {"path": str(Path(__file__).resolve()),
                               "sha256": export.sha256(Path(__file__).resolve())},
        "selection": {"path": str(selection_path), "sha256": export.sha256(selection_path)},
        "cohort_freeze": {"path": str(directory / "COHORT_FREEZE.json"),
                          "sha256": export.sha256(directory / "COHORT_FREEZE.json")},
        "execution_summary": {"path": str(directory / "EXECUTION_SUMMARY.json"),
                              "sha256": export.sha256(directory / "EXECUTION_SUMMARY.json")},
        "source_and_assets_unchanged": True,
        "adjudication_policy": "only strict failures with one exact post-JSON MuJoCo QACC warning",
        "numerical_instability_diagnostic_count": len(adjudications),
        "adjudications": adjudications, "unresolved_errors": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = adjudicate(args.directory, args.selection, args.output)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
