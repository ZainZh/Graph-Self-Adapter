#!/usr/bin/env python3
"""Plan or execute hash-bound mixed-candidate Twin RSI final cohorts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.task_graph import twin_export_packages as export  # noqa: E402

SCHEMA = "graph-self-adapter.twin-final-cohort-freeze.v1"
SUMMARY_SCHEMA = "graph-self-adapter.twin-final-execution.v1"
SELECTION_SCHEMA = "graph-self-adapter.twin-final-selection.v1"
DEFAULT_PYTHON = "/home/zheng/github/Mujoco-RSI/.venv/bin/python"


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _canonical_sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def _add_binding(bindings: dict[str, dict[str, str]], path: Path,
                 expected: str | None, kind: str) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = export.sha256(path)
    if expected is not None and actual != export.require_hash(expected, str(path)):
        raise ValueError(f"hash mismatch for {kind}: {path}")
    key = str(path)
    previous = bindings.get(key)
    if previous is not None and previous["sha256"] != actual:
        raise ValueError(f"conflicting hashes for shared input: {path}")
    bindings[key] = {"path": key, "sha256": actual, "kind": kind}


def _validate_selection(path: Path, arms: tuple[str, ...], trials: int,
                        expected_tasks: set[str] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    document = _json(path)
    if document.get("schema") != SELECTION_SCHEMA or document.get("status") != "frozen":
        raise ValueError(f"selection must be frozen final-selection.v1: {path}")
    conditions = document.get("conditions", {})
    if conditions.get("test_seed") != 71005:
        raise ValueError(f"selection does not bind test_seed=71005: {path}")
    if int(conditions.get("trials_per_task_per_arm", -1)) != trials:
        raise ValueError(f"selection trial count differs from requested {trials}: {path}")
    declared_arms = set(document.get("arms", []))
    if not set(arms).issubset(declared_arms):
        raise ValueError(f"selection does not permit arms {arms}: {path}")
    rows = document.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"selection has no task rows: {path}")
    task_names = [row.get("task") for row in rows if isinstance(row, Mapping)]
    if len(task_names) != len(rows) or len(task_names) != len(set(task_names)):
        raise ValueError(f"selection tasks must be unique objects: {path}")
    if expected_tasks is not None and set(task_names) != expected_tasks:
        raise ValueError("baseline and selected selections must bind the same task set")

    bindings: dict[str, dict[str, str]] = {}
    _add_binding(bindings, path, None, "selection")
    scope_binding = document.get("scope_binding")
    if not isinstance(scope_binding, Mapping):
        raise ValueError(f"selection scope_binding is required: {path}")
    _add_binding(bindings, Path(scope_binding["path"]), scope_binding["sha256"],
                 "scientific_scope")
    evaluator_sources = document.get("external_evaluator_sources")
    if not isinstance(evaluator_sources, list) or len(evaluator_sources) != 2:
        raise ValueError(f"selection must bind both external evaluator sources: {path}")
    for binding in evaluator_sources:
        _add_binding(bindings, Path(binding["path"]), binding["sha256"],
                     "external_evaluator_source")
    verified = []
    base = path.parent
    for row in rows:
        task = str(row["task"])
        candidate = str(row.get("candidate_id", ""))
        if not candidate:
            raise ValueError(f"{task}: candidate_id is required")
        snapshot, embedded_sha = export.source_snapshot_path(row.get("source_snapshot"), base)
        expected_source_freeze = row.get("source_freeze_sha256", embedded_sha)
        source_hashes, source_freeze_sha = export.verify_snapshot(
            snapshot, None if expected_source_freeze is None else
            export.require_hash(expected_source_freeze, f"{task}.source_freeze_sha256"))
        solution = export.solution_path(row.get("packages"), task, base)
        hashes = export.verify_solution(row, solution, source_hashes)

        _add_binding(bindings, snapshot / "SOURCE_FREEZE.json", source_freeze_sha,
                     "source_freeze")
        for relative, digest in source_hashes.items():
            if relative in export.REQUIRED_SNAPSHOT_CODE:
                _add_binding(bindings, snapshot / relative, digest, "source_code")
        package_files = {
            "graph": (solution / "graph/task_graph.json", row["graph_sha256"]),
            "model": (solution / "models/local_skill.npz", row["model_sha256"]),
            "reference": (solution / "refs/visible_centroids.json", row["reference_sha256"]),
            "candidate_freeze": (solution / "CANDIDATE_FREEZE.json",
                                  row["candidate_freeze_sha256"]),
            "manifest": (solution / "MANIFEST.json", hashes["manifest"]),
            "external_bindings": (solution / "EXTERNAL_BINDINGS.json",
                                  hashes["external_bindings"]),
        }
        for kind, (bound_path, digest) in package_files.items():
            _add_binding(bindings, bound_path, digest, kind)

        externals = _json(solution / "EXTERNAL_BINDINGS.json")
        for label, binding in externals.get("sources", {}).items():
            _add_binding(bindings, Path(binding["path"]), binding["sha256"],
                         f"external_source:{label}")
        for label in ("external_assets", "revision_override", "root_plan"):
            binding = externals.get(label)
            if isinstance(binding, Mapping):
                _add_binding(bindings, Path(binding["path"]), binding["sha256"], label)
        runtime_contracts = {
            "snapshot_root_plan": (
                snapshot / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json",
                externals.get("root_plan", {}).get("sha256")),
            "snapshot_external_assets": (
                snapshot / "evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json",
                externals.get("external_assets", {}).get("sha256")),
            "snapshot_scope": (
                snapshot / "evaluations/twin_rsi_v01/SCOPE.json",
                scope_binding.get("sha256")),
        }
        for kind, (runtime_path, expected_digest) in runtime_contracts.items():
            if expected_digest is None:
                raise ValueError(f"{task}: missing expected hash for {kind}")
            _add_binding(bindings, runtime_path, expected_digest, kind)
        catalog_path = snapshot / "evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json"
        catalog = _json(catalog_path)
        task_assets = catalog.get("task_assets", {}).get(task)
        if not isinstance(task_assets, Mapping):
            raise ValueError(f"{task}: external asset catalog entry missing")
        for raw, digest in task_assets.get("external_asset_hashes", {}).items():
            _add_binding(bindings, Path(raw), digest, "external_asset")

        package_root = solution.parent.parent
        expected_solution = package_root / f"twin_{task}" / "solution"
        if expected_solution.resolve() != solution:
            raise ValueError(
                f"{task}: package layout must be <root>/twin_<task>/solution for runner")
        verified.append({
            "task": task, "candidate_id": candidate,
            "solution": str(solution), "packages_root": str(package_root),
            "source_snapshot": str(snapshot), "source_freeze_sha256": source_freeze_sha,
            "graph_sha256": row["graph_sha256"], "model_sha256": row["model_sha256"],
            "reference_sha256": row["reference_sha256"],
            "candidate_freeze_sha256": row["candidate_freeze_sha256"],
        })
    return {"path": str(path), "sha256": export.sha256(path),
            "candidate_id": document.get("candidate_id"), "tasks": verified,
            "bindings": sorted(bindings.values(), key=lambda row: row["path"])}


def build_plan(selected_selection: Path, baseline_selection: Path, final_scope: Path,
               selected_output: Path, baseline_output: Path, *, trials: int = 10,
               python: str = DEFAULT_PYTHON) -> dict[str, Any]:
    selected = _validate_selection(selected_selection, ("fixed_z", "online_z"), trials)
    task_set = {row["task"] for row in selected["tasks"]}
    baseline = _validate_selection(baseline_selection, ("fixed_z",), trials, task_set)
    final_scope = final_scope.expanduser().resolve()
    scope = _json(final_scope)
    if (scope.get("schema") != "graph-self-adapter.twin-final-test-scope.v1" or
            scope.get("status") != "result_free" or scope.get("test_opened_before_freeze") is not False):
        raise ValueError("final-test scope must be frozen result-free and unopened")
    if (scope.get("test_seed") != 71005 or
            int(scope.get("trials_per_task_per_arm", -1)) != trials or
            set(scope.get("tasks", [])) != task_set or scope.get("planned_episodes") != 3 * trials * len(task_set)):
        raise ValueError("final-test scope does not match selections/trials")
    arm_bindings = {(row["name"], Path(row["selection"]["path"]).resolve(),
                     row["selection"]["sha256"]) for row in scope.get("arms", [])}
    required = {("initial_graph_fixed_z", Path(baseline["path"]), baseline["sha256"]),
                ("selected_fixed_z", Path(selected["path"]), selected["sha256"]),
                ("selected_online_z", Path(selected["path"]), selected["sha256"])}
    if arm_bindings != required:
        raise ValueError("final-test scope arm/selection bindings do not match")
    for selection in (selected, baseline):
        selection["bindings"].append({"path": str(final_scope),
                                      "sha256": export.sha256(final_scope),
                                      "kind": "final_test_scope"})
        driver = Path(__file__).resolve()
        selection["bindings"].append({"path": str(driver),
                                      "sha256": export.sha256(driver),
                                      "kind": "cohort_driver"})
    cohorts = (("selected", selected, Path(selected_output).resolve(),
                ("fixed_z", "online_z")),
               ("baseline", baseline, Path(baseline_output).resolve(), ("fixed_z",)))
    jobs = []
    for cohort, selection, output, controllers in cohorts:
        for row in selection["tasks"]:
            for trial in range(trials):
                for controller in controllers:
                    command = [python,
                               str(Path(row["source_snapshot"]) /
                                   "tools/task_graph/twin_rsi.py"),
                               "--plan", str(Path(row["source_snapshot"]) /
                                             "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json"),
                               "--assets", str(Path(row["source_snapshot"]) /
                                               "evaluations/twin_rsi_v01/EXTERNAL_ASSET_BINDINGS.json"),
                               "--packages", row["packages_root"], "run",
                               "--task", row["task"], "--candidate-id", row["candidate_id"],
                               "--condition-set", "test", "--trial", str(trial),
                               "--output", str(output), "--selection-freeze", selection["path"]]
                    if controller == "online_z":
                        command.append("--online-z")
                    jobs.append({"cohort": cohort, "task": row["task"],
                                 "candidate_id": row["candidate_id"], "trial": trial,
                                 "controller": controller, "cwd": row["source_snapshot"],
                                 "output": str(output), "command": command})
    return {"schema": "graph-self-adapter.twin-final-test-plan.v1",
            "condition_set": "test", "trials_per_task_per_arm": trials,
            "final_test_scope": {"path": str(final_scope),
                                 "sha256": export.sha256(final_scope)},
            "selected": selected, "baseline": baseline, "jobs": jobs,
            "job_counts": {"selected": len(selected["tasks"]) * trials * 2,
                           "baseline": len(baseline["tasks"]) * trials,
                           "total": len(jobs)}}


def _freeze_for(plan: Mapping[str, Any], cohort: str, output: Path,
                workers: int) -> dict[str, Any]:
    selection = plan[cohort]
    controllers = ["fixed_z", "online_z"] if cohort == "selected" else ["fixed_z"]
    bindings = {row["path"]: row for row in selection["bindings"]}
    return {"schema": SCHEMA, "cohort": cohort, "frozen_before_rollouts": True,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "condition_set": "test", "test_seed": 71005,
            "trials_per_task_per_arm": plan["trials_per_task_per_arm"],
            "controllers": controllers, "workers": workers,
            "selection": {"path": selection["path"], "sha256": selection["sha256"],
                          "candidate_id": selection["candidate_id"]},
            "tasks": selection["tasks"], "bindings": sorted(bindings.values(),
                                                               key=lambda row: row["path"]),
            "bindings_sha256": _canonical_sha(sorted(bindings.values(),
                                                       key=lambda row: row["path"])),
            "output": str(output)}


def _check_bindings(rows: list[Mapping[str, str]]) -> list[dict[str, str]]:
    changes = []
    for row in rows:
        path = Path(row["path"])
        actual = export.sha256(path) if path.is_file() else None
        if actual != row["sha256"]:
            changes.append({"path": str(path), "expected_sha256": row["sha256"],
                            "actual_sha256": actual, "kind": row["kind"]})
    return changes


def _post_bindings(rows: list[Mapping[str, str]]) -> list[dict[str, str | None]]:
    return [{"path": row["path"], "kind": row["kind"],
             "sha256": export.sha256(Path(row["path"]))
             if Path(row["path"]).is_file() else None} for row in rows]


def _run_job(job: Mapping[str, Any], output: Path) -> dict[str, Any]:
    start = time.monotonic()
    stem = f"{job['task']}_{job['trial']:02d}_{job['controller']}"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    completed = subprocess.run(job["command"], cwd=job["cwd"], env=env,
                               capture_output=True, text=True, check=False)
    (output / f"{stem}.stdout").write_text(completed.stdout)
    (output / f"{stem}.stderr").write_text(completed.stderr)
    result = {key: job[key] for key in
              ("cohort", "task", "candidate_id", "trial", "controller")}
    result["returncode"] = completed.returncode
    result["wall_sec"] = round(time.monotonic() - start, 3)
    if completed.returncode:
        result["error"] = completed.stderr[-4000:]
        return result
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        result["error"] = f"runner stdout was not JSON: {exc}"
        return result
    result.update(strict_success=payload.get("strict_success"),
                  failure=payload.get("failure"), output=payload.get("output"))
    return result


def execute(plan: Mapping[str, Any], *, workers: int) -> dict[str, Any]:
    outputs = {name: Path(plan[name + "_output"]).resolve()
               for name in ("selected", "baseline")} if "selected_output" in plan else {
                   "selected": Path(next(job["output"] for job in plan["jobs"]
                                         if job["cohort"] == "selected")),
                   "baseline": Path(next(job["output"] for job in plan["jobs"]
                                         if job["cohort"] == "baseline"))}
    for output in outputs.values():
        if output.exists():
            raise FileExistsError(f"refusing to overwrite cohort output: {output}")
    freezes = {}
    for name, output in outputs.items():
        output.mkdir(parents=True)
        freezes[name] = _freeze_for(plan, name, output, workers)
        _dump(output / "COHORT_FREEZE.json", freezes[name])

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_job, job, outputs[job["cohort"]]): job
                   for job in plan["jobs"]}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    invalid = False
    summaries = {}
    for name, output in outputs.items():
        cohort_results = [row for row in results if row["cohort"] == name]
        changes = _check_bindings(freezes[name]["bindings"])
        post_bindings = _post_bindings(freezes[name]["bindings"])
        errors = [row for row in cohort_results if row.get("error") or row["returncode"]]
        summary = {"schema": SUMMARY_SCHEMA, "cohort": name,
                   "cohort_freeze_sha256": export.sha256(output / "COHORT_FREEZE.json"),
                   "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "runs": sorted(cohort_results,
                                  key=lambda row: (row["task"], row["trial"],
                                                   row["controller"])),
                   "binding_changes_during_execution": changes,
                   "post_execution_bindings": post_bindings,
                   "post_execution_bindings_sha256": _canonical_sha(post_bindings),
                   "pre_post_bindings_match": not changes,
                   "invalid_if_bindings_changed": bool(changes),
                   "errors": errors}
        _dump(output / "EXECUTION_SUMMARY.json", summary)
        summaries[name] = summary
        invalid |= bool(changes or errors)
    if invalid:
        raise RuntimeError("final cohort failed or frozen source/assets changed")
    return summaries


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("plan", "run"))
    p.add_argument("--selected-selection", type=Path, required=True)
    p.add_argument("--baseline-selection", type=Path, required=True)
    p.add_argument("--final-test-scope", type=Path, required=True)
    p.add_argument("--selected-output", type=Path, required=True)
    p.add_argument("--baseline-output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--python", default=DEFAULT_PYTHON)
    p.add_argument("--plan-output", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.trials <= 0 or args.workers <= 0:
        raise ValueError("trials and workers must be positive")
    plan = build_plan(args.selected_selection, args.baseline_selection,
                      args.final_test_scope,
                      args.selected_output, args.baseline_output,
                      trials=args.trials, python=args.python)
    if args.mode == "plan":
        if args.plan_output:
            if args.plan_output.exists():
                raise FileExistsError(args.plan_output)
            _dump(args.plan_output, plan)
        json.dump(plan, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    execute(plan, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
