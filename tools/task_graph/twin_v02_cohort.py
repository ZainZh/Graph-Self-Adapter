#!/usr/bin/env python3
"""Freeze, plan, and execute a hash-bound twin v02 cohort.

The campaign is root-authored and result-free.  Each arm supplies per-task
solution and source-snapshot paths so parent and revised graphs run their own
frozen runner.  ``--dry-run`` validates bindings and emits commands without
starting MuJoCo. Conditions are materialized only after selection gates pass.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


SCHEMA = "graph-self-adapter.twin-v02-campaign.v1"
FREEZE_SCHEMA = "graph-self-adapter.twin-v02-cohort-freeze.v1"
SUMMARY_SCHEMA = "graph-self-adapter.twin-v02-execution-summary.v1"
TWIN_PYTHON = Path("/home/zheng/github/Mujoco-RSI/.venv/bin/python3.12")
WARNING_RE = re.compile(
    r"WARNING: Nan, Inf or huge value in Q(?:POS|VEL|ACC) at DOF \d+\. "
    r"The simulation is unstable\. Time = [0-9.eE+-]+\."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def resolve_bound(base: Path, binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding.get("path", "")))
    if not path.is_absolute():
        path = base / path
    if not path.is_file() or sha256(path) != binding.get("sha256"):
        raise ValueError(f"{label} binding mismatch: {path}")
    return path.resolve()


def parse_stdout(raw: str) -> tuple[dict[str, Any], list[str]]:
    decoder = json.JSONDecoder()
    stripped = raw.lstrip()
    value, end = decoder.raw_decode(stripped)
    if not isinstance(value, dict):
        raise ValueError("runner stdout first JSON payload must be an object")
    tail = stripped[end:].strip()
    warnings = [] if not tail else [line.strip() for line in tail.splitlines() if line.strip()]
    if any(WARNING_RE.fullmatch(line) is None for line in warnings):
        raise ValueError("runner stdout has non-MuJoCo trailing output")
    return value, warnings


def _solution_bindings(solution: Path, row: Mapping[str, Any]) -> list[dict[str, str]]:
    expected = {
        "graph": (solution / "graph/task_graph.json", row.get("graph_sha256")),
        "model": (solution / "models/local_skill.npz", row.get("model_sha256")),
        "reference": (solution / "refs/visible_centroids.json", row.get("reference_sha256")),
        "candidate_freeze": (solution / "CANDIDATE_FREEZE.json", row.get("candidate_freeze_sha256")),
    }
    out = []
    for label, (path, selected) in expected.items():
        actual = sha256(path)
        if selected is not None and selected != actual:
            raise ValueError(f"{row['task']} {label} selection binding mismatch")
        out.append({"label": label, "path": str(path.resolve()), "sha256": actual})
    return out


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def add_binding(bindings: list, path: Path, label: str, expected: str | None = None):
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"{label} binding mismatch: {path}")
    bindings.append({"label": label, "path": str(path.resolve()), "sha256": actual})



def build_plan(campaign_path: Path) -> dict[str, Any]:
    campaign = load(campaign_path)
    if campaign.get("schema") != SCHEMA or campaign.get("status") != "frozen":
        raise ValueError("campaign must be a frozen twin-v02-campaign.v1")
    base = campaign_path.resolve().parent
    scope = resolve_bound(base, campaign["scope_binding"], "scope")
    runtime = resolve_bound(base, campaign["runtime_contract"], "runtime contract")
    plan = resolve_bound(base, campaign["root_plan"], "root plan")
    assets = resolve_bound(base, campaign["external_assets"], "external assets")
    selection_rule = resolve_bound(base, campaign["selection_rule"], "selection rule")
    runtime_value = load(runtime)
    if (runtime_value.get("schema") != "graph-self-adapter.twin-runtime-contract.v2" or
            runtime_value.get("cup_release_control") != "portable_tcp_displacement_v1" or
            runtime_value.get("numerical_integrity_fail_closed") is not True):
        raise ValueError("campaign runtime contract is not the portable fail-closed v02 contract")
    tasks = list(campaign.get("tasks", ()))
    if sorted(tasks) != sorted(("put_knife_board_new", "put_mug_on_box", "sorting_cube_bimanual")):
        raise ValueError("v02 campaign must contain the exact three scoped tasks")
    trials = list(campaign.get("trials", ()))
    if trials != list(range(10)):
        raise ValueError("v02 campaign trials must be exactly 0..9")
    seed = int(campaign["condition_seed"])
    condition_set = str(campaign["condition_set"])
    if condition_set == "test_v02" and seed != 92005:
        raise ValueError("test_v02 seed must be exactly 92005")
    if condition_set not in ("development_v02", "test_v02"):
        raise ValueError("unsupported v02 condition set")
    if condition_set == "development_v02" and seed != 51005:
        raise ValueError("development_v02 must use the opened seed 51005")
    output_root = Path(campaign["output_root"]).resolve()
    if "twin_rsi_v02" not in output_root.parts:
        raise ValueError("output root must be inside a twin_rsi_v02 directory")
    arms = campaign.get("arms", [])
    arm_ids = [arm.get("arm_id") for arm in arms]
    if not arms or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in arm_ids):
        raise ValueError("arms require nonempty safe arm ids")
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("arm ids must be unique")
    jobs = []
    bindings = [
        {"label": "campaign", "path": str(campaign_path.resolve()), "sha256": sha256(campaign_path)},
        {"label": "selection_rule", "path": str(selection_rule), "sha256": sha256(selection_rule)},
        {"label": "scope", "path": str(scope), "sha256": sha256(scope)},
        {"label": "runtime_contract", "path": str(runtime), "sha256": sha256(runtime)},
        {"label": "root_plan", "path": str(plan), "sha256": sha256(plan)},
        {"label": "external_assets", "path": str(assets), "sha256": sha256(assets)},
    ]
    selection = None
    if condition_set == "test_v02":
        selection = resolve_bound(base, campaign["selection_freeze"], "selection freeze")
        bindings.append({"label": "selection_freeze", "path": str(selection),
                         "sha256": sha256(selection)})
    export = load_module(Path(__file__).with_name("twin_export_packages.py"), "_v02_export")
    add_binding(bindings, Path(__file__), "cohort_driver")
    add_binding(bindings, Path(__file__).with_name("twin_export_packages.py"), "binding_validator")
    root_tasks = {row["task"]: row for row in load(plan)["tasks"]}
    catalog = load(assets)
    frame_hashes = {}
    if selection is not None and load(selection).get("status") != "frozen":
        raise ValueError("test selection must be frozen before condition generation")
    for arm in arms:
        arm_id = str(arm["arm_id"])
        controller = str(arm["controller"])
        if controller not in ("fixed_z", "online_z"):
            raise ValueError(f"unsupported controller {controller}")
        rows = {row["task"]: row for row in arm.get("tasks", ())}
        if len(arm.get("tasks", ())) != len(tasks) or sorted(rows) != sorted(tasks):
            raise ValueError(f"arm {arm_id} must bind each scoped task exactly once")
        for task in tasks:
            row = rows[task]
            solution = Path(row["solution"]).resolve()
            snapshot = Path(row["source_snapshot"]).resolve()
            runner = snapshot / "tools/task_graph/twin_rsi.py"
            if not runner.is_file():
                raise ValueError(f"missing frozen runner: {runner}")
            bindings.extend(_solution_bindings(solution, row))
            source_hashes, source_freeze_sha = export.verify_snapshot(snapshot, row.get("source_freeze_sha256"))
            add_binding(bindings, snapshot / "SOURCE_FREEZE.json", "source_freeze", source_freeze_sha)
            for relative in export.REQUIRED_SNAPSHOT_CODE:
                add_binding(bindings, snapshot / relative, "source_code", source_hashes[relative])
            bound_row = dict(row)
            for key, relative in (("graph_sha256", "graph/task_graph.json"),
                                  ("model_sha256", "models/local_skill.npz"),
                                  ("reference_sha256", "refs/visible_centroids.json"),
                                  ("candidate_freeze_sha256", "CANDIDATE_FREEZE.json")):
                bound_row.setdefault(key, sha256(solution / relative))
            export.verify_solution(bound_row, solution, source_hashes)
            if load(solution / "CANDIDATE_FREEZE.json").get("runtime_contract_sha256") != sha256(runtime):
                raise ValueError("candidate freeze must bind the campaign runtime contract")
            for relative in ("MANIFEST.json", "EXTERNAL_BINDINGS.json"):
                add_binding(bindings, solution / relative, "package_metadata")
            externals = load(solution / "EXTERNAL_BINDINGS.json")
            for label, binding in externals.get("sources", {}).items():
                add_binding(bindings, Path(binding["path"]), "external_source:" + label, binding["sha256"])
            for label in ("root_plan", "external_assets", "revision_override", "runtime_contract"):
                binding = externals.get(label)
                if isinstance(binding, Mapping):
                    add_binding(bindings, Path(binding["path"]), label, binding["sha256"])
            for label, binding in root_tasks[task]["sources"].items():
                add_binding(bindings, Path(binding["path"]), "root_source:" + label, binding["sha256"])
            for raw, expected in catalog["task_assets"][task]["external_asset_hashes"].items():
                add_binding(bindings, Path(raw), "external_mesh_or_asset", expected)
            refs = load(solution / "refs/visible_centroids.json")
            materialization = refs.get("runtime_materialization", {})
            runtime_canonical = hashlib.sha256(canonical(runtime_value).rstrip(b"\n")).hexdigest()
            if (materialization.get("cup_release_control") != runtime_value["cup_release_control"] or
                    materialization.get("runtime_contract_canonical_sha256") != runtime_canonical):
                raise ValueError("all arms require portable references with runtime provenance")
            frames_hash = hashlib.sha256(canonical(refs["frames"])).hexdigest()
            if task in frame_hashes and frame_hashes[task] != frames_hash:
                raise ValueError("arms must share identical visible reference frames")
            frame_hashes[task] = frames_hash
            frozen_runner = load_module(runner, "_v02_runner_" + str(len(jobs)))
            selection_row = None
            if selection is not None:
                selection_row = frozen_runner.validate_test_selection(
                    load(selection), task=task, candidate_id=row["candidate_id"],
                    solution=solution, selection_path=selection)
            packages = solution.parents[1]
            for trial in trials:
                condition = frozen_runner.condition_manifest(
                    root_tasks[task], seed=seed, trial=trial,
                    nominal=condition_set == "development_v02" and trial == 0,
                    condition_design=runtime_value["condition_design"])
                if selection is not None:
                    frozen_runner.validate_v02_test_condition(
                        load(selection), row=selection_row, selection_path=selection,
                        runtime_contract_path=runtime, seed=seed, trial=trial, condition=condition)
                output = Path(campaign["output_root"]).resolve() / arm_id
                command = [str(TWIN_PYTHON), str(runner), "--plan", str(plan), "--assets", str(assets),
                           "--packages", str(packages), "run", "--task", task,
                           "--condition-set", condition_set, "--condition-seed", str(seed),
                           "--runtime-contract", str(runtime), "--trial", str(trial),
                           "--candidate-id", str(row["candidate_id"]), "--output", str(output)]
                if controller == "online_z": command.append("--online-z")
                if condition_set == "test_v02":
                    assert selection is not None
                    command.extend(("--selection-freeze", str(selection)))
                jobs.append({"arm_id": arm_id, "controller": controller, "task": task,
                             "trial": trial, "candidate_id": row["candidate_id"],
                             "output": str(output), "command": command,
                             "condition_set": condition_set, "condition_seed": seed,
                             "condition_sha256": frozen_runner.canonical_json_sha256(condition),
                             "runtime_contract_sha256": sha256(runtime)})
    by_path = {}
    for binding in bindings:
        if binding["path"] in by_path and by_path[binding["path"]]["sha256"] != binding["sha256"]:
            raise ValueError("conflicting hashes for one bound path")
        by_path[binding["path"]] = binding
    unique = {(x["path"], x["sha256"]): x for x in bindings}
    return {"schema": SCHEMA, "campaign": campaign, "jobs": jobs,
            "bindings": sorted(unique.values(), key=lambda x: (x["label"], x["path"]))}


def _run_job(job: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(job["output"]); output.mkdir(parents=True, exist_ok=True)
    stem = f"{job['task']}_{job['condition_set']}_{job['trial']:02d}_{job['controller']}"
    stdout_path, stderr_path = output / f"{stem}.stdout", output / f"{stem}.stderr"
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(f"refusing to overwrite process logs: {stem}")
    completed = subprocess.run(job["command"], text=True, capture_output=True)
    stdout_path.write_text(completed.stdout); stderr_path.write_text(completed.stderr)
    result = {k: job[k] for k in ("arm_id", "controller", "task", "trial", "candidate_id")}
    result.update(returncode=completed.returncode, stdout_sha256=sha256(stdout_path),
                  stderr_sha256=sha256(stderr_path), error=None)
    try:
        payload, warnings = parse_stdout(completed.stdout)
        result["payload"] = payload; result["native_warning_tail"] = warnings
        report_path = Path(payload["output"])
        report = load(report_path)
        result["report"] = {"path": str(report_path), "sha256": sha256(report_path)}
        if (report.get("task") != job["task"] or report.get("candidate_id") != job["candidate_id"] or
                report.get("controller") != job["controller"]):
            raise ValueError("run report identity mismatch")
        condition = report.get("condition", {})
        condition_sha = hashlib.sha256(canonical(condition).rstrip(b"\n")).hexdigest()
        if (report.get("condition_set") != job["condition_set"] or
                condition.get("trial") != job["trial"] or condition.get("seed") != job["condition_seed"] or
                condition_sha != job["condition_sha256"] or
                report.get("bindings", {}).get("condition_sha256") != job["condition_sha256"] or
                report.get("bindings", {}).get("runtime_contract_sha256") != job["runtime_contract_sha256"]):
            raise ValueError("run report trial/condition/runtime binding mismatch")
        if report_path.resolve().parent != output.resolve():
            raise ValueError("run report outside declared output")
        if warnings:
            safety = report.get("original_task_outcome", {}).get("numerical_safety", {})
            if report.get("strict_success") is not False or safety.get("passed") is not False:
                raise ValueError("native numerical warning was not fail-closed")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    if completed.returncode and result["error"] is None:
        result["error"] = f"runner exited {completed.returncode}"
    return result


def execute(plan: Mapping[str, Any], output: Path, workers: int) -> dict[str, Any]:
    campaign = plan["campaign"]
    if campaign.get("execution_authorized") is not True:
        raise ValueError("campaign execution_authorized must be true")
    if not 1 <= workers <= 8:
        raise ValueError("workers must be within 1..8")
    if output.resolve() != Path(campaign["output_root"]).resolve():
        raise ValueError("--output must equal campaign.output_root")
    for binding in plan["bindings"]:
        if sha256(Path(binding["path"])) != binding["sha256"]:
            raise ValueError("pre-execution binding changed")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite cohort output: {output}")
    output.mkdir(parents=True)
    freeze = {"schema": FREEZE_SCHEMA, "frozen_before_rollouts": True,
              "campaign_sha256": next(x["sha256"] for x in plan["bindings"]
                                       if x["label"] == "campaign"),
              "jobs": plan["jobs"], "bindings": plan["bindings"]}
    freeze_path = output / "COHORT_FREEZE.json"; freeze_path.write_bytes(canonical(freeze))
    before = {x["path"]: sha256(Path(x["path"])) for x in plan["bindings"]}
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_job, job): job for job in plan["jobs"]}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                job = futures[future]
                results.append({**{k: job[k] for k in ("arm_id", "controller", "task", "trial", "candidate_id")},
                                "returncode": None, "error": f"{type(exc).__name__}: {exc}",
                                "strict_success": False})
    after = {}
    for binding in plan["bindings"]:
        try:
            after[binding["path"]] = sha256(Path(binding["path"]))
        except OSError:
            after[binding["path"]] = None
    changes = [path for path in before if before[path] != after[path]]
    summary = {"schema": SUMMARY_SCHEMA, "cohort_freeze_sha256": sha256(freeze_path),
               "declared_jobs": len(plan["jobs"]), "completed_jobs": len(results),
               "binding_changes": changes, "post_execution_bindings": after,
               "task_denominators": {task: sum(j["task"] == task for j in plan["jobs"]) for task in campaign["tasks"]}, "runs": sorted(results, key=lambda x: (
                   x["arm_id"], x["task"], x["trial"])),
               "errors": sum(bool(x.get("error") or x.get("returncode")) for x in results)}
    summary_path = output / "EXECUTION_SUMMARY.json"; summary_path.write_bytes(canonical(summary))
    if changes or summary["errors"] or len(results) != len(plan["jobs"]):
        raise RuntimeError(f"v02 cohort incomplete or invalid; inspect {summary_path}")
    return {"freeze": str(freeze_path), "summary": str(summary_path), "jobs": len(results)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    plan = build_plan(args.campaign)
    if args.dry_run:
        result = {"schema": SCHEMA, "dry_run": True, "jobs": plan["jobs"],
                  "bindings": plan["bindings"]}
    else:
        if args.output is None:
            raise ValueError("--output is required for execution")
        result = execute(plan, args.output, args.workers)
    json.dump(result, sys.stdout, indent=2, sort_keys=True); print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
