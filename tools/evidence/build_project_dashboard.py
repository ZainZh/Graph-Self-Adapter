#!/usr/bin/env python3
"""Validate dashboard bindings and render a small static project page."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS = REPO_ROOT / "dashboard" / "STATUS.json"
DEFAULT_OUTPUT = REPO_ROOT / "dashboard" / "index.html"
REQUIRED_BINDINGS = (
    "task_graph", "readme", "engineering_acceptance", "science_report", "video"
)
TWIN_BINDINGS = ("twin_scope", "twin_plan", "twin_status")
TWIN_V02_BINDINGS = (
    "twin_v02_scope", "twin_v02_plan", "twin_v02_input_audit",
    "twin_v02_status", "twin_v01_input_erratum",
)
TWIN_V02_DEVELOPMENT_REPORTS = (
    ("twin_v02_development_b0_a1_report", "B0 / A1 fixed-z"),
    ("twin_v02_development_a2_report", "A2 fixed-z"),
    ("twin_v02_development_a3_report", "A3 fixed-z"),
    ("twin_v02_selected_online_development_report", "选版 online-z"),
)
TWIN_V02_FINAL_REPORT = "twin_v02_final_report"
TWIN_V02_ARM_ORDER = ("B0", "A1", "A2", "A3")
TWIN_V02_ARM_LABELS = {
    "B0": "B0 统一后端父图",
    "A1": "A1 支撑平面先验",
    "A2": "A2 抓持确认与时间参数化",
    "A3": "A3",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_status(status_path: Path, root: Path) -> dict:
    status = json.loads(status_path.read_text())
    bindings = status.get("bindings", {})
    missing = [key for key in REQUIRED_BINDINGS if key not in bindings]
    if missing:
        raise ValueError(f"missing dashboard bindings: {missing}")
    twin_present = [key for key in TWIN_BINDINGS if key in bindings]
    if twin_present and len(twin_present) != len(TWIN_BINDINGS):
        raise ValueError(f"Twin dashboard bindings must be all-or-none: {TWIN_BINDINGS}")
    twin_v02_present = [key for key in TWIN_V02_BINDINGS if key in bindings]
    if twin_v02_present and len(twin_v02_present) != len(TWIN_V02_BINDINGS):
        raise ValueError(
            f"Twin v02 dashboard bindings must be all-or-none: {TWIN_V02_BINDINGS}")
    for key, binding in bindings.items():
        path = (root / binding["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"binding escapes repository: {key}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != binding["sha256"]:
            raise ValueError(f"binding hash mismatch for {key}: {actual}")
    return status


def load_cohort_reports(twin: dict, task_order: list[str]) -> list[dict]:
    reports = []
    for binding in twin.get("report_bindings", []):
        if not isinstance(binding, dict):
            raise ValueError("twin_status.report_bindings entries must be objects")
        path = (REPO_ROOT / binding["path"]).resolve()
        if not path.is_relative_to(REPO_ROOT.resolve()) or not path.is_file():
            raise ValueError(f"invalid cohort report path: {binding.get('path')}")
        if sha256(path) != binding.get("sha256"):
            raise ValueError(f"cohort report hash mismatch: {path}")
        report = json.loads(path.read_text())
        if report.get("schema") != "graph-self-adapter.twin-cohort-summary.v1":
            raise ValueError(f"unsupported cohort report schema: {path}")
        for key in ("candidate_id", "condition_set"):
            if report.get(key) is not None and report[key] != binding.get(key):
                raise ValueError(f"cohort report {key} differs from binding: {path}")
        rows = report.get("tasks", [])
        if [row.get("task") for row in rows] != task_order:
            raise ValueError(f"cohort report must contain the ordered task set: {path}")
        by_task = {}
        total_trials = total_successes = 0
        for row in rows:
            trials, successes = row.get("trials"), row.get("successes")
            if (not isinstance(trials, int) or isinstance(trials, bool) or trials < 0 or
                    not isinstance(successes, int) or isinstance(successes, bool) or
                    not 0 <= successes <= trials):
                raise ValueError(f"invalid episode counts in cohort report: {path}")
            rate = row.get("rate")
            if trials:
                if not isinstance(rate, (int, float)) or isinstance(rate, bool) or abs(rate - successes / trials) > 1e-9:
                    raise ValueError(f"rate does not match successes/trials: {path}")
            elif rate is not None:
                raise ValueError(f"zero-trial task rate must be null: {path}")
            sources = row.get("reports", [])
            if len(sources) != trials:
                raise ValueError(f"task report count does not match trials: {path}")
            source_successes = 0
            for source_binding in sources:
                source = (REPO_ROOT / source_binding["path"]).resolve()
                if not source.is_relative_to(REPO_ROOT.resolve()) or not source.is_file():
                    raise ValueError(f"invalid bound task report: {source}")
                if sha256(source) != source_binding.get("sha256"):
                    raise ValueError(f"bound task report hash mismatch: {source}")
                source_report = json.loads(source.read_text())
                if source_report.get("schema") != "graph-self-adapter.twin-run.v1":
                    raise ValueError(f"unsupported bound task report schema: {source}")
                source_successes += bool(source_report.get("strict_success"))
            if source_successes != successes:
                raise ValueError(f"task success count does not match bound reports: {path}")
            total_trials += trials
            total_successes += successes
            by_task[row["task"]] = row
        if report.get("trials") != total_trials or report.get("successes") != total_successes:
            raise ValueError(f"cohort aggregate counts do not match task rows: {path}")
        controller = report.get("controller") or report.get("adapter_mode") or ""
        custom_label = binding.get("label")
        if custom_label is not None and (not isinstance(custom_label, str) or not custom_label.strip()):
            raise ValueError("cohort report label must be a non-empty string")
        label = custom_label.strip() if custom_label is not None else str(binding["candidate_id"])
        reports.append({
            "label": label, "path": binding["path"], "tasks": by_task,
            "trials": total_trials, "successes": total_successes,
            "candidate_id": binding["candidate_id"],
            "condition_set": binding["condition_set"], "controller": controller,
        })
    return reports


def _repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()) or not path.is_file():
        raise ValueError(f"invalid repository evidence path: {path_value}")
    return path


def _checked_binding(binding: dict) -> Path:
    path = _repo_path(binding["path"])
    if sha256(path) != binding.get("sha256"):
        raise ValueError(f"evidence binding hash mismatch: {path}")
    if "bytes" in binding and path.stat().st_size != binding["bytes"]:
        raise ValueError(f"evidence binding size mismatch: {path}")
    return path


SENSOR_COLUMNS = (("initial_gt", "v03 初始 GT 诊断对照（重用）"),
                  ("initial_sensor", "v03 初始传感判定"),
                  ("repair1_sensor", "v03 repair1"), ("v04_sensor", "v04 Align 交接许可"))


def _sensor_json(binding, schema):
    value = json.loads(_checked_binding(binding).read_text())
    if value.get("schema") != schema:
        raise ValueError("unsupported sensor report schema")
    return value


def _sensor_counts(report, *, final=False, initial=False):
    runs = report.get("source_reports" if final else "runs")
    if not isinstance(runs, list):
        raise ValueError("sensor source runs must be a list")
    counts, seen = {}, set()
    for run in runs:
        task, trial = run.get("task"), run.get("trial")
        mode = run.get("judge_mode") if initial else "sensor_only"
        key = (task, mode)
        if not isinstance(task, str) or type(trial) is not int or trial < 0 or (key, trial) in seen:
            raise ValueError("invalid or duplicate sensor source identity")
        seen.add((key, trial))
        source = _sensor_json(run["report"], "graph-self-adapter.twin-run.v1")
        if (source.get("task") != task or source.get("condition", {}).get("trial") != trial
                or source.get("condition", {}).get("seed") != 51005 or source.get("controller") != "fixed_z"):
            raise ValueError("sensor source task/trial/controller mismatch")
        for field in ("strict_success", "graph_completed"):
            if type(run.get(field)) is not bool or source.get(field) is not run[field]:
                raise ValueError("sensor source result mismatch")
        if initial and source.get("sensor_backend", {}).get("judge_mode") != mode:
            raise ValueError("sensor source judge mode mismatch")
        seal_binding = run.get("sensor_prediction_seal" if final else "prediction_seal")
        seal = _sensor_json(seal_binding, "graph-self-adapter.sensor-prediction-seal.v1")
        if not seal.get("sealed_before_finalize_evaluator"):
            raise ValueError("sensor decisions were not sealed before evaluation")
        backend_binding = source.get("sensor_backend", {}).get("prediction_seal")
        if backend_binding is None or _checked_binding(backend_binding) != _checked_binding(seal_binding):
            raise ValueError("sensor source seal mismatch")
        archive = seal.get("public_sensor_packet_archive")
        archive_path = _checked_binding(archive)
        if final:
            if seal.get("execution_variant") != "sensor_transition_v04":
                raise ValueError("wrong sensor transition execution variant")
            if _checked_binding(run["public_sensor_packet_archive"]) != archive_path:
                raise ValueError("sensor packet archive mismatch")
        aggregate = counts.setdefault(key, {"successes": 0, "denominator": 0, "graph": 0, "physical": 0, "safety": 0})
        aggregate["denominator"] += 1
        aggregate["successes"] += int(run["strict_success"])
        aggregate["graph"] += int(run["graph_completed"])
        if final:
            outcome = source.get("original_task_outcome", {})
            for field, evidence_key, count_key in (("original_physical_goal_passed", "task_outcome", "physical"),
                                                  ("original_safety_passed", "numerical_safety", "safety")):
                if type(run.get(field)) is not bool or outcome.get(evidence_key, {}).get("passed") is not run[field]:
                    raise ValueError("post-seal sensor physical/safety result mismatch")
                aggregate[count_key] += int(run[field])
    task_keys = set()
    for row in report.get("tasks", []):
        key = (row.get("task"), row.get("judge_mode") if initial else "sensor_only")
        if key in task_keys or key not in counts:
            raise ValueError("sensor task row mismatch")
        task_keys.add(key)
        count = counts[key]
        if (type(row.get("denominator")) is not int or row["denominator"] != count["denominator"]
                or row.get("strict_successes") != count["successes"] or row.get("graph_successes") != count["graph"]):
            raise ValueError("sensor task counts differ from raw reports")
        if final and (row.get("post_seal_original_physical_goal_successes") != count["physical"]
                      or row.get("post_seal_original_safety_successes") != count["safety"]):
            raise ValueError("post-seal sensor task counts mismatch")
    if task_keys != set(counts):
        raise ValueError("sensor task rows omit source runs")
    if final and (report.get("denominator") != len(runs) or report.get("errors") != 0
                  or report.get("strict_successes") != sum(c["successes"] for c in counts.values())
                  or report.get("graph_successes") != sum(c["graph"] for c in counts.values())):
        raise ValueError("sensor final totals mismatch")
    return counts


def load_sensor_transition_report(bindings):
    report = _sensor_json(bindings["sensor_transition_final_report"],
                          "graph-self-adapter.sensor-transition-final-report.v1")
    source_bindings = report.get("source_bindings", {})
    for binding in source_bindings.values():
        _checked_binding(binding)
    for key, source_key in (("sensor_stage_initial_report", "initial_report"),
                            ("sensor_stage_repair_report", "repair1_report")):
        if key in bindings and _checked_binding(bindings[key]) != _checked_binding(source_bindings[source_key]):
            raise ValueError("sensor dashboard source differs from final report")
    initial = _sensor_json(source_bindings["initial_report"], "graph-self-adapter.sensor-initial-report.v1")
    repair = _sensor_json(source_bindings["repair1_report"], "graph-self-adapter.sensor-repair1-report.v1")
    initial_counts = _sensor_counts(initial, initial=True)
    repair_counts = _sensor_counts(repair)
    final_counts = _sensor_counts(report, final=True)
    versions = {"initial_gt": (initial_counts, "privileged_gt_control"),
                "initial_sensor": (initial_counts, "sensor_only"),
                "repair1_sensor": (repair_counts, "sensor_only"), "v04_sensor": (final_counts, "sensor_only")}
    table = report.get("comparison_table")
    if not isinstance(table, list) or len({r.get("task") for r in table}) != len(table):
        raise ValueError("invalid sensor comparison task rows")
    if {r["task"] for r in table} != {key[0] for key in initial_counts}:
        raise ValueError("sensor comparison task coverage mismatch")
    for row in table:
        for column, (counts, mode) in versions.items():
            expected = counts.get((row["task"], mode))
            actual = row.get(column)
            if expected is None:
                if actual is not None: raise ValueError("unverified sensor task must be null")
            elif (not isinstance(actual, dict) or type(actual.get("successes")) is not int
                  or type(actual.get("denominator")) is not int
                  or actual != {k: expected[k] for k in ("successes", "denominator")}):
                raise ValueError("sensor comparison differs from source counts")
    return report


def render_sensor_transition_table(report):
    headers = "".join(f"<th>{escape(label)}</th>" for _, label in SENSOR_COLUMNS)
    rows = []
    for row in report["comparison_table"]:
        cells = []
        for key, _ in SENSOR_COLUMNS:
            count = row[key]
            text = "未验证" if count is None else f"{count['successes']}/{count['denominator']}"
            cells.append(f"<td>{escape(text)}</td>")
        rows.append(f"<tr><td><code>{escape(row['task'])}</code></td>{''.join(cells)}</tr>")
    post = "".join(f"<tr><td><code>{escape(row['task'])}</code></td>"
                   f"<td>{row['post_seal_original_physical_goal_successes']}/{row['denominator']}</td>"
                   f"<td>{row['post_seal_original_safety_successes']}/{row['denominator']}</td></tr>"
                   for row in report['tasks'])
    return (f'<div class="table-scroll"><table><thead><tr><th>任务严格完成</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            '<p>v04 封存后独立物理检查另列；物理目标通过但整图未完成仍不计严格成功，不能据此断言全部动作失败。</p>'
            f'<table><thead><tr><th>v04 任务</th><th>封存后物理目标通过</th><th>封存后安全检查通过</th></tr></thead><tbody>{post}</tbody></table>')


def load_v02_comparison_videos(binding: dict, task_order: list[str]) -> list[dict]:
    """Read only the sealed v02 manifest and verify its local media provenance."""
    manifest = json.loads(_checked_binding(binding).read_text())
    if manifest.get("schema") != "graph-self-adapter.twin-v02-comparison-videos.v1":
        raise ValueError("unsupported v02 video manifest schema")
    if (manifest.get("offline_visual_replay_only") is not True or
            manifest.get("controller_adapter_evaluator_or_mj_step_executed") is not False):
        raise ValueError("v02 videos must be offline sealed-trace replay")
    for key in ("render_source", "final_report", "final_selection", "plan"):
        _checked_binding(manifest[key])
    clips = {}
    for clip in manifest.get("clips", []):
        video = _checked_binding(clip["video"])
        if not video.is_relative_to((REPO_ROOT / "media/twin_rsi_v02").resolve()):
            raise ValueError("v02 video must be under media/twin_rsi_v02")
        sidecar_path = _checked_binding(clip["render_sidecar"])
        if sidecar_path != video.with_suffix(".render.json"):
            raise ValueError("v02 render sidecar path differs from video")
        sidecar = json.loads(sidecar_path.read_text())
        if (sidecar.get("schema") != "graph-self-adapter.twin-video.v1" or
                sidecar.get("offline_visual_replay_only") is not True or
                sidecar.get("policy_or_adapter_executed") is not False or
                clip.get("offline_only") is not True):
            raise ValueError("unsupported or non-offline v02 render sidecar")
        if sidecar.get("video") != clip["video"]:
            raise ValueError("v02 sidecar/video binding mismatch")
        inputs = clip["sealed_inputs"]
        for key in ("report", "visual_trace", "scene", "condition", "initialization"):
            if sidecar.get(key) != inputs.get(key):
                raise ValueError(f"v02 sidecar sealed input mismatch: {key}")
        for key in ("report", "visual_trace"):
            source = _checked_binding(inputs[key])
            if not source.is_relative_to((REPO_ROOT / "runs/twin_rsi_v02").resolve()):
                raise ValueError("v02 video source must be v02 evidence")
        initialization = inputs["initialization"]
        if initialization.get("mj_step_executed") is not False:
            raise ValueError("v02 render initialization executed physical steps")
        for key in ("candidate_freeze", "physics_source"):
            _checked_binding(initialization[key])
        report = json.loads(_repo_path(inputs["report"]["path"]).read_text())
        if report.get("schema") != "graph-self-adapter.twin-run.v1":
            raise ValueError("unsupported v02 video source report")
        canonical = hashlib.sha256(json.dumps(
            report.get("condition"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if canonical != inputs["condition"].get("embedded_canonical_sha256"):
            raise ValueError("v02 video source condition hash mismatch")
        if report.get("bindings", {}).get("condition_sha256") != inputs["condition"].get("sha256"):
            raise ValueError("v02 source report condition binding mismatch")
        if video in clips:
            raise ValueError("duplicate v02 source video")
        clips[video] = (clip, report)
    comparisons = manifest.get("comparisons", [])
    if [row.get("task") for row in comparisons] != task_order:
        raise ValueError("v02 comparisons must match ordered focused tasks")
    result = []
    for row in comparisons:
        video = _checked_binding(row["video"])
        if not video.is_relative_to((REPO_ROOT / "media/twin_rsi_v02").resolve()):
            raise ValueError("v02 comparison video path escapes media")
        if not isinstance(row.get("label"), str) or not row["label"].strip():
            raise ValueError("v02 comparison label is missing")
        sides = []
        for key in ("left_parent", "right_selected"):
            path = _checked_binding(row[key])
            if path not in clips or clips[path][0]["video"] != row[key]:
                raise ValueError("v02 comparison references unbound source clip")
            clip, report = clips[path]
            if report.get("task") != row["task"] or report.get("condition", {}).get("trial") != row["trial"]:
                raise ValueError("v02 comparison task/trial mismatch")
            if report.get("controller") != "fixed_z":
                raise ValueError("v02 comparison must use fixed-z source runs")
            sides.append((clip, report))
        if row.get("source_render_sidecars") != [item[0]["render_sidecar"] for item in sides]:
            raise ValueError("v02 comparison sidecars mismatch")
        if sides[0][0]["sealed_inputs"]["condition"] != sides[1][0]["sealed_inputs"]["condition"]:
            raise ValueError("v02 comparison conditions differ")
        result.append({**row, "path": str(video.relative_to(REPO_ROOT.resolve()))})
    return result


def load_v02_result_report(binding: dict, task_order: list[str], *, final: bool = False) -> dict:
    """Validate a v02 aggregate and every available source run it counts."""
    path = _repo_path(binding["path"])
    if sha256(path) != binding.get("sha256"):
        raise ValueError(f"v02 report hash mismatch: {path}")
    report = json.loads(path.read_text())
    expected = ("graph-self-adapter.twin-v02-final-report.v1" if final else
                "graph-self-adapter.twin-v02-development-batch-report.v1")
    if report.get("schema") != expected:
        raise ValueError(f"unsupported v02 report schema: {path}")
    rows = report.get("tasks")
    runs = report.get("runs")
    if not isinstance(rows, list) or not isinstance(runs, list):
        raise ValueError(f"v02 report requires task and run lists: {path}")

    grouped: dict[tuple[str, str], list[dict]] = {}
    arm_meta: dict[str, dict[str, set[str]]] = {}
    execution_errors = 0
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError(f"v02 run entries must be objects: {path}")
        arm, task = run.get("arm_id"), run.get("task")
        if not isinstance(arm, str) or not arm or task not in task_order:
            raise ValueError(f"invalid v02 run identity: {path}")
        if not isinstance(run.get("trial"), int) or isinstance(run.get("trial"), bool):
            raise ValueError(f"invalid v02 run trial: {path}")
        error = run.get("error") or run.get("execution_error")
        source = run.get("report")
        if source is None:
            if not isinstance(error, str) or not error.strip():
                raise ValueError(f"v02 run without source report requires an explicit error: {path}")
            if run.get("strict_success") not in (None, False):
                raise ValueError(f"v02 execution error cannot be successful: {path}")
        else:
            if not isinstance(source, dict) or not isinstance(source.get("path"), str):
                raise ValueError(f"invalid v02 source report binding: {path}")
            source_path = _repo_path(source["path"])
            if sha256(source_path) != source.get("sha256"):
                raise ValueError(f"v02 source report hash mismatch: {source_path}")
            source_report = json.loads(source_path.read_text())
            if source_report.get("schema") != "graph-self-adapter.twin-run.v1":
                raise ValueError(f"unsupported v02 source run schema: {source_path}")
            if source_report.get("task") != task:
                raise ValueError(f"v02 source run task mismatch: {source_path}")
            if source_report.get("candidate_id") != run.get("candidate_id"):
                raise ValueError(f"v02 source run candidate mismatch: {source_path}")
            if source_report.get("controller") != run.get("controller"):
                raise ValueError(f"v02 source run controller mismatch: {source_path}")
            if source_report.get("strict_success") is not run.get("strict_success"):
                raise ValueError(f"v02 source run success mismatch: {source_path}")
        if error:
            execution_errors += 1
        grouped.setdefault((arm, task), []).append(run)
        meta = arm_meta.setdefault(arm, {"controllers": set(), "candidates": set()})
        if run.get("controller"):
            meta["controllers"].add(str(run["controller"]))
        if run.get("candidate_id"):
            meta["candidates"].add(str(run["candidate_id"]))

    by_key = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"v02 task rows must be objects: {path}")
        key = (row.get("arm_id"), row.get("task"))
        if key in by_key or key[0] not in arm_meta or key[1] not in task_order:
            raise ValueError(f"invalid or duplicate v02 task row: {path}")
        denominator, successes = row.get("denominator"), row.get("strict_successes")
        if (not isinstance(denominator, int) or isinstance(denominator, bool) or denominator <= 0 or
                not isinstance(successes, int) or isinstance(successes, bool) or
                not 0 <= successes <= denominator):
            raise ValueError(f"invalid v02 task denominator: {path}")
        source_runs = grouped.get(key, [])
        source_successes = sum(run.get("strict_success") is True for run in source_runs)
        if len(source_runs) != denominator or source_successes != successes:
            raise ValueError(f"v02 task row differs from source runs: {path} {key}")
        by_key[key] = dict(row, execution_errors=sum(
            bool(run.get("error") or run.get("execution_error")) for run in source_runs))

    expected_keys = {(arm, task) for arm in arm_meta for task in task_order}
    if set(by_key) != expected_keys or set(grouped) != expected_keys:
        raise ValueError(f"v02 report must contain every task for every arm: {path}")
    if report.get("execution_errors", 0) != execution_errors:
        raise ValueError(f"v02 execution error count mismatch: {path}")

    arms = []
    totals = report.get("arm_totals", {})
    order = {arm: index for index, arm in enumerate(TWIN_V02_ARM_ORDER)}
    for arm in sorted(arm_meta, key=lambda value: (order.get(value, len(order)), value)):
        controllers = arm_meta[arm]["controllers"]
        candidates = arm_meta[arm]["candidates"]
        if len(controllers) != 1:
            raise ValueError(f"v02 arm must have one controller: {path} {arm}")
        candidate_by_task = {}
        for task in task_order:
            task_candidates = {run.get("candidate_id") for run in grouped[(arm, task)]}
            if len(task_candidates) != 1 or not all(isinstance(x, str) and x for x in task_candidates):
                raise ValueError(f"v02 arm/task must have one candidate: {path} {arm}/{task}")
            candidate_by_task[task] = next(iter(task_candidates))
        denominator = sum(by_key[(arm, task)]["denominator"] for task in task_order)
        successes = sum(by_key[(arm, task)]["strict_successes"] for task in task_order)
        if (not isinstance(totals.get(arm), dict) or
                totals[arm].get("denominator") != denominator or
                totals[arm].get("strict_successes") != successes):
            raise ValueError(f"v02 arm total differs from task rows: {path} {arm}")
        arms.append({
            "arm_id": arm, "controller": next(iter(controllers)),
            "candidate_id": (next(iter(candidates)) if len(candidates) == 1 else "mixed_per_task_selection"),
            "candidate_label": (next(iter(candidates)) if len(candidates) == 1 else "逐任务混合选版"),
            "candidate_by_task": candidate_by_task, "denominator": denominator,
            "strict_successes": successes,
            "execution_errors": sum(by_key[(arm, task)]["execution_errors"] for task in task_order),
            "tasks": {task: by_key[(arm, task)] for task in task_order},
        })
    return {"path": binding["path"], "label": binding.get("label", path.stem),
            "schema": expected, "arms": arms}


def render_v02_result_table(columns: list[dict], task_order: list[str], *, pending: list[str]) -> str:
    headers = []
    for column in columns:
        arm = column["arm"]
        headers.append(
            f'<th><a href="../{escape(column["report_path"], quote=True)}">{escape(column["label"])}</a>'
            f'<small>{escape(arm["arm_id"])} · {escape(arm.get("candidate_label", arm["candidate_id"]))} · '
            f'{escape(arm["controller"])}</small></th>')
    headers.extend(f'<th>{escape(label)}<small>待绑定</small></th>' for label in pending)
    body = []
    for task in task_order:
        cells = []
        for column in columns:
            row = column["arm"]["tasks"][task]
            failed = row["denominator"] - row["strict_successes"]
            suffix = f'；失败 {failed}'
            if row["execution_errors"]:
                suffix += f'（执行错误 {row["execution_errors"]}）'
            cells.append(
                f'<td>{row["strict_successes"]}/{row["denominator"]}'
                f'（{row["strict_successes"] / row["denominator"]:.0%}）{escape(suffix)}</td>')
        cells.extend('<td class="pending">待测</td>' for _ in pending)
        body.append(f'<tr><td><code>{escape(task)}</code></td>{"".join(cells)}</tr>')
    totals = []
    for column in columns:
        arm = column["arm"]
        failed = arm["denominator"] - arm["strict_successes"]
        text = f'{arm["strict_successes"]}/{arm["denominator"]}；失败 {failed}'
        if arm["execution_errors"]:
            text += f'（执行错误 {arm["execution_errors"]}）'
        totals.append(f'<td><strong>{escape(text)}</strong></td>')
    totals.extend('<td class="pending"><strong>待测</strong></td>' for _ in pending)
    return (
        '<div class="table-scroll"><table><thead><tr><th>任务</th>'
        f'{"".join(headers)}</tr></thead><tbody>{"".join(body)}</tbody>'
        f'<tfoot><tr><th>按完整回合汇总</th>{"".join(totals)}</tr></tfoot></table></div>')


def load_task_videos(twin: dict, task_order: list[str]) -> dict[str, list[dict]]:
    """Validate annotated videos and the exact run reports they visualize."""
    grouped = {task: [] for task in task_order}
    for binding in twin.get("task_videos", []):
        if not isinstance(binding, dict):
            raise ValueError("twin_status.task_videos entries must be objects")
        required = {"task", "label", "path", "sha256", "report_path", "report_sha256"}
        if required - set(binding):
            raise ValueError(f"task video binding missing fields: {sorted(required - set(binding))}")
        task, label = binding["task"], binding["label"]
        if task not in grouped:
            raise ValueError(f"task video has unknown task: {task}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("task video label must be a non-empty string")
        checked = {}
        for field, hash_field in (("path", "sha256"), ("report_path", "report_sha256")):
            path = (REPO_ROOT / binding[field]).resolve()
            if not path.is_relative_to(REPO_ROOT.resolve()) or not path.is_file():
                raise ValueError(f"invalid task video binding path: {binding[field]}")
            if sha256(path) != binding[hash_field]:
                raise ValueError(f"task video binding hash mismatch: {path}")
            checked[field] = path
        report = json.loads(checked["report_path"].read_text())
        if report.get("schema") != "graph-self-adapter.twin-run.v1":
            raise ValueError(f"unsupported task video run report: {checked['report_path']}")
        if report.get("task") != task:
            raise ValueError(f"task video/report task mismatch: {task}")
        grouped[task].append(dict(binding, label=label.strip()))
    return {task: rows for task, rows in grouped.items() if rows}


def render_cohort_table(reports: list[dict], *, task_rows: list[dict],
                        planned: dict[str, dict], empty_text: str) -> str:
    if not reports:
        return f'<p class="empty">{escape(empty_text)}</p>'
    headers = []
    for report in reports:
        metadata = " · ".join(x for x in (report["condition_set"], report["controller"]) if x)
        headers.append(
            f'<th><a href="../{escape(report["path"], quote=True)}">{escape(report["label"])}</a>'
            f'<small>{escape(metadata)}</small></th>')
    body = []
    for row in task_rows:
        task = row["task"]
        cells = []
        for report in reports:
            result = report["tasks"][task]
            trials, successes = result["trials"], result["successes"]
            text = "待测" if trials == 0 else f"{successes}/{trials}（{successes / trials:.0%}）"
            cells.append(f"<td>{escape(text)}</td>")
        body.append(
            "<tr>"
            f"<td>{escape(planned[task].get('label', task))}<br><code>{escape(task)}</code></td>"
            f"<td>{len(planned[task].get('cycles', []))}</td>"
            f"<td>{'已通过' if row.get('physics_qualified') else '未通过'}</td>"
            f"{''.join(cells)}</tr>")
    totals = []
    for report in reports:
        trials, successes = report["trials"], report["successes"]
        text = "待测" if trials == 0 else f"{successes}/{trials}（{successes / trials:.0%}）"
        totals.append(f"<td><strong>{escape(text)}</strong></td>")
    return (
        '<div class="table-scroll"><table><thead><tr><th>任务</th><th>交互循环</th><th>物理场景</th>'
        f"{''.join(headers)}</tr></thead><tbody>{''.join(body)}</tbody>"
        f"<tfoot><tr><th>按回合汇总</th><td></td><td></td>{''.join(totals)}</tr></tfoot></table></div>")


def render_agent_live(bindings: dict) -> str:
    if "agent_live_final_report" not in bindings:
        return ""
    report = _sensor_json(bindings["agent_live_final_report"], "graph-self-adapter.agent-live-final-report.v1")
    episodes = report.get("episodes", [])
    if not episodes or len({e.get("task") for e in episodes}) != len(episodes):
        raise ValueError("invalid live Agent episode identities")
    cells = []
    sources = {}
    for episode in episodes:
        source_path = _checked_binding(episode["report"])
        _checked_binding(episode["agent_seal"])
        source = json.loads(source_path.read_text())
        if source.get("task") != episode["task"] or any(source.get(k) is not episode[k] for k in ("strict_success", "graph_completed")):
            raise ValueError("Agent live source result mismatch")
        events = episode.get("events", [])
        if episode["request_count"] != len(events):
            raise ValueError("Agent live request count mismatch")
        unknown = sum(event.get("decision") == "unknown" for event in events)
        if unknown != episode.get("decision_counts", {}).get("unknown", 0):
            raise ValueError("Agent live UNKNOWN count mismatch")
        stage_count = len(set(row["stage_id"] for row in source.get("trace", [])))
        if stage_count != episode["completed_stage_entries"]:
            raise ValueError("Agent live stage count mismatch")
        sources[episode["task"]] = source_path
        cells.append("<tr>" + f"<td><code>{escape(episode['task'])}</code></td>"
            + f"<td>{stage_count}</td><td>{episode['request_count']}</td><td>{unknown}</td>"
            + f"<td>{'通过' if episode['strict_success'] else '未通过'}</td></tr>")
    videos = []
    if "agent_live_video_manifest" in bindings:
        manifest = _sensor_json(bindings["agent_live_video_manifest"], "graph-self-adapter.agent-live-video-manifest.v1")
        seen = set()
        for item in manifest["videos"]:
            task = item["task"]
            if task in seen or task not in sources or _checked_binding(item["source_report"]) != sources[task]:
                raise ValueError("Agent video source task mismatch")
            seen.add(task)
            path = _checked_binding(item["video"]); _checked_binding(item["render"])
            if not path.is_relative_to((REPO_ROOT / "media/agent_stage_v06").resolve()) or item.get("offline_reconstruction_no_new_rollouts") is not True:
                raise ValueError("Agent video reconstruction/path mismatch")
            relative = str(path.relative_to(REPO_ROOT.resolve()))
            videos.append(f'<figure><figcaption>{escape(task)}</figcaption><video controls preload="metadata" src="../{escape(relative, quote=True)}"></video></figure>')
        if seen != set(sources):
            raise ValueError("Agent live videos do not cover declared episodes")
    resources = []
    for key, label in (("agent_live_final_report", "连续闭环封存报告"),
                       ("agent_live_video_manifest", "封存轨迹视频清单"),
                       ("xtrainer_root_report", "Xtrainer 接入报告"),
                       ("xtrainer_readme", "接入说明与命令"), ("xtrainer_status", "迁移状态")):
        if key in bindings:
            path = _checked_binding(bindings[key]).relative_to(REPO_ROOT.resolve())
            resources.append(f'<a href="../{escape(str(path), quote=True)}">{escape(label)}</a>')
    return ('<section class="active-study"><h2>连续 Agent 闭环与 Xtrainer 接入</h2>'
        + '<p>' + ' · '.join(resources) + '</p>'
        + f'<p>{len(episodes)} 个已开放名义开发工程案例；逐任务列出，不作为成功率或独立准确率结论。</p>'
        + '<table><thead><tr><th>任务</th><th>阶段</th><th>Root 图像请求</th><th>UNKNOWN</th><th>原严格目标与安全</th></tr></thead><tbody>'
        + ''.join(cells) + '</tbody></table><div class="video-grid">' + ''.join(videos) + '</div>'
        + '<p>视频仅重建封存轨迹，Root 墙钟等待已压缩；不是实时性能展示。模拟器等判断时暂停物理时间，真实世界需要持续反馈检查与回复后重新取图。</p></section>')


def render_agent_position(bindings: dict) -> str:
    if "agent_position_status" not in bindings:
        return ""
    status = _sensor_json(bindings["agent_position_status"], "graph-self-adapter.agent-position-status.v1")
    completed, planned = status.get("complete_task_episodes"), status.get("planned_task_episodes")
    if type(completed) is not int or not 0 <= completed <= 10 or planned != 10 or type(planned) is not int:
        raise ValueError("invalid position regression progress")
    resources = []
    for key, label in (("agent_position_status", "公开执行状态"), ("agent_position_scope", "位置回归范围"),
                       ("agent_position_final_report", "封存汇总"), ("agent_position_root_report", "Root 报告"),
                       ("agent_position_video_manifest", "封存视频清单")):
        if key in bindings:
            path = _checked_binding(bindings[key]).relative_to(REPO_ROOT.resolve())
            resources.append(f'<a href="../{escape(str(path), quote=True)}">{escape(label)}</a>')
    results = '<p>最终结果尚未封存；完成数是执行进度，不是任务成绩。</p>'
    if "agent_position_final_report" in bindings:
        final = _sensor_json(bindings["agent_position_final_report"], "graph-self-adapter.agent-position-final-report.v1")
        episodes = final.get("episodes")
        if final.get("task") != "put_mug_on_box" or type(final.get("denominator")) is not int or final["denominator"] != 10:
            raise ValueError("wrong position regression task/denominator")
        if not isinstance(episodes, list) or len(episodes) != 10 or completed != 10:
            raise ValueError("position final report requires all ten completed episodes")
        totals = dict(strict_successes=0, graph_completed_count=0, root_request_count=0, unknown_count=0)
        seen = set()
        for episode in episodes:
            trial = episode.get("trial")
            if type(trial) is not int or trial not in range(10) or trial in seen:
                raise ValueError("duplicate or invalid position episode trial")
            seen.add(trial)
            if any(type(episode.get(k)) is not bool for k in ("strict_success", "graph_completed")):
                raise ValueError("position episode result must be Boolean")
            if episode["strict_success"] and not episode["graph_completed"]:
                raise ValueError("strict success requires graph completion")
            if any(type(episode.get(k)) is not int or episode[k] < 0 for k in ("root_request_count", "unknown_count")):
                raise ValueError("invalid position Root/UNKNOWN counts")
            if episode["unknown_count"] > episode["root_request_count"]:
                raise ValueError("UNKNOWN count exceeds Root requests")
            _checked_binding(episode["termination_file"])
            totals["strict_successes"] += int(episode["strict_success"])
            totals["graph_completed_count"] += int(episode["graph_completed"])
            totals["root_request_count"] += episode["root_request_count"]
            totals["unknown_count"] += episode["unknown_count"]
        if any(type(final.get(k)) is not int or final[k] != v for k, v in totals.items()):
            raise ValueError("position statistics disagree with episode evidence")
        results = ('<table><thead><tr><th>严格成功</th><th>整图完成</th><th>Root 请求</th><th>UNKNOWN</th></tr></thead><tbody><tr>'
            + f'<td>{totals["strict_successes"]}/10</td><td>{totals["graph_completed_count"]}/10</td>'
            + f'<td>{totals["root_request_count"]}</td><td>{totals["unknown_count"]}</td></tr></tbody></table>')
    return ('<section class="active-study"><h2>双杯放箱：已打开位置变化回归</h2>'
        + '<p>' + ' · '.join(resources) + '</p>'
        + f'<p>单任务 <code>put_mug_on_box</code>；完成 {completed}/{planned} 个计划整任务。</p>'
        + results + '<p>同条件历史 8/10 仅作不同阶段判定接口的参考；条件已经打开，不是盲评或因果归因。</p></section>')


def render(status: dict) -> str:
    bindings = status["bindings"]
    graph = json.loads((REPO_ROOT / bindings["task_graph"]["path"]).read_text())
    routing = graph["solver_routing"]
    rows = []
    for stage in graph["stage_ids"]:
        sid = stage["id"]
        rows.append(
            "<tr>"
            f"<td><code>{escape(sid)}</code></td>"
            f"<td>{escape(stage.get('entry', ''))}</td>"
            f"<td>{escape(stage.get('exit', ''))}</td>"
            f"<td>{escape(routing[sid])}</td>"
            "</tr>"
        )
    def link(key: str, label: str) -> str:
        path = bindings[key]["path"]
        return f'<a href="../{escape(path, quote=True)}">{escape(label)}</a>'
    video = "../" + bindings["video"]["path"]
    position_html = render_agent_position(bindings)
    live_html = render_agent_live(bindings)
    agent_html = ""
    if "agent_stage_review_receipt" in bindings:
        receipt = _sensor_json(bindings["agent_stage_review_receipt"],
                               "graph-self-adapter.agent-stage-review-receipt.v1")
        for key in ("request", "root_reply"):
            _checked_binding(receipt[key])
        decision = receipt.get("actual_consumer_result", {}).get("decision")
        if decision not in ("continue", "complete", "local_failure", "structural_failure", "unknown"):
            raise ValueError("invalid Agent review decision")
        flags = ("live_executor_integrated", "unattended_model_service_integrated")
        if any(type(receipt.get(key)) is not bool for key in flags):
            raise ValueError("invalid Agent review integration flags")
        resources = " · ".join(link(key, label) for key, label in (
            ("agent_stage_report", "阶段判读报告"), ("agent_stage_review_receipt", "请求校验与判读收据"),
            ("agent_stage_panel", "六帧图像面板"), ("agent_stage_scope", "离线范围"),
            ("agent_stage_status", "当前状态")) if key in bindings)
        live = "已接入" if receipt["live_executor_integrated"] else "未接入"
        service = "已接入" if receipt["unattended_model_service_integrated"] else "未接入"
        agent_html = f"""<section class="active-study"><h2>当前方向：Agent 语义阶段判定</h2>
<p>{resources}</p><p>实际判断：<strong>{escape(decision.upper())}</strong>；实时执行器：{live}；无人值守模型服务：{service}。</p>
<p>当前 Root Agent 直接查看图像并回写判断，文件桥本身不生成判断。一次已开放条件的离线判读不改变任何成功率，不是自动模型服务或真机结果；没有新增物理回合。</p></section>"""
    sensor_html = ""
    if "sensor_transition_final_report" in bindings:
        sensor_report = load_sensor_transition_report(bindings)
        sensor_links = (("sensor_transition_final_report", "封存结果与原始来源"),
            ("sensor_transition_status", "当前状态"), ("sensor_transition_scope", "实验范围"),
            ("sensor_stage_initial_report", "v03 初始报告"), ("sensor_stage_repair_report", "v03 repair1 报告"),
            ("sensor_integration_guide", "传感接口与接入缺口"),
            ("sensor_transition_root_report", "中文科学报告"), ("sensor_transition_acceptance", "验收"))
        resources = " · ".join(link(key, label) for key, label in sensor_links if key in bindings)
        sensor_html = f"""<section class="active-study"><h2>已完成的传感原型：Sensor v03 / v04</h2>
<p>{resources}</p>{render_sensor_transition_table(sensor_report)}
<p>各版本分别统计，不合并分母；初始 GT 诊断对照为重用结果。条件已打开，只作小样本开发／回归，默认 fixed-z。</p>
<p>Align 暂定交接许可不等于物理支撑或完成；最终 Release 与完成仍采用严格传感判定。</p>
<p>目前使用模拟器语义 mask 与理想同步本体反馈。真实采集反馈、SAM 和硬件驱动尚未接入；本机 SAM 代码和权重已存在，Graph 未集成、未测端到端延迟。结果不构成硬件资格。</p></section>"""
    v02_html = ""
    if all(key in bindings for key in TWIN_V02_BINDINGS):
        v02_scope = json.loads((REPO_ROOT / bindings["twin_v02_scope"]["path"]).read_text())
        v02_status = json.loads((REPO_ROOT / bindings["twin_v02_status"]["path"]).read_text())
        v01_erratum = json.loads((REPO_ROOT / bindings["twin_v01_input_erratum"]["path"]).read_text())
        if v02_scope.get("schema") != "graph-self-adapter.twin-v02-scope.v1":
            raise ValueError("unsupported twin_v02_scope schema")
        if v02_status.get("schema") != "graph-self-adapter.twin-v02-status.v1":
            raise ValueError("unsupported twin_v02_status schema")
        if v01_erratum.get("schema") != "graph-self-adapter.twin-v01-input-permission-erratum.v1":
            raise ValueError("unsupported twin_v01_input_erratum schema")
        focused = v02_scope.get("focused_tasks")
        if focused != [row.get("task") for row in v02_status.get("focused_tasks", [])]:
            raise ValueError("Twin v02 scope and status focused tasks differ")
        if v02_status.get("results_available"):
            result_text = "已有结果（请以绑定状态为准）"
        else:
            result_text = "结果为空"
        test_text = "已打开" if v02_status.get("independent_test_opened") else "未打开"
        authorization_text = "已许可" if v02_status.get("execution_authorized") else "待 Root 执行许可"
        task_items = "".join(f"<li><code>{escape(task)}</code></li>" for task in focused)
        development_columns = []
        pending_development = []
        for key, default_label in TWIN_V02_DEVELOPMENT_REPORTS:
            if key not in bindings:
                pending_development.append(default_label)
                continue
            report = load_v02_result_report(bindings[key], focused)
            for arm in report["arms"]:
                development_columns.append({
                    "report_path": report["path"], "arm": arm,
                    "label": TWIN_V02_ARM_LABELS.get(
                        arm["arm_id"], f'{report["label"]} · {arm["arm_id"]}'),
                })
        development_v02_table = render_v02_result_table(
            development_columns, focused, pending=pending_development)
        final_columns = []
        pending_final = []
        if TWIN_V02_FINAL_REPORT in bindings:
            final_report = load_v02_result_report(
                bindings[TWIN_V02_FINAL_REPORT], focused, final=True)
            for arm in final_report["arms"]:
                final_columns.append({
                    "report_path": final_report["path"], "arm": arm,
                    "label": TWIN_V02_ARM_LABELS.get(
                        arm["arm_id"], f'{final_report["label"]} · {arm["arm_id"]}'),
                })
        else:
            pending_final.append("v02 独立测试")
        final_v02_table = render_v02_result_table(final_columns, focused, pending=pending_final)
        resource_labels = (
            ("twin_v02_root_report", "v02 科学报告"),
            ("twin_v02_task_package", "选定任务复用包"),
            ("twin_v02_package_verification", "复用包检查与 dry-run"),
            ("twin_v02_video_manifest", "封存比较视频清单"),
        )
        resource_links = " · ".join(
            link(key, label) for key, label in resource_labels if key in bindings)
        comparison_html = ""
        if "twin_v02_video_manifest" in bindings:
            comparisons = load_v02_comparison_videos(bindings["twin_v02_video_manifest"], focused)
            figures = "".join(
                '<figure><figcaption>' + escape(item["label"]) + '</figcaption>'
                '<video controls preload="metadata" src="../' + escape(item["path"], quote=True)
                + '"></video></figure>' for item in comparisons)
            comparison_html = ('<h3>封存轨迹比较示例</h3><p>左侧父图、右侧选版；'
                               '刀具和杯子展示改善例，分类展示退化例。示例不代表总体成功率；'
                               '较慢的阶段时间参数化涉及可靠性与运行时间的权衡。</p>'
                               '<div class="video-grid">' + figures + '</div>')
        v02_html = f"""
<section class="active-study"><h2>已封存历史：Twin RSI v02（与 v01 分开统计）</h2>
<div class="warning"><strong>v01 输入边界勘误：</strong>v01 不只使用特权实例 mask；冻结的低层 release 伺服还读取连续物体真值来改变松爪时序，接触、支撑和阶段完成 Boolean 也来自模拟器真值。v01 历史结果保持冻结，不能据此声称真机接口已经具备。</div>
<p>{link('twin_v02_scope','v02 范围')} · {link('twin_v02_plan','根计划')} · {link('twin_v02_input_audit','输入审计与真机缺口')} · {link('twin_v02_status','v02 状态')} · {link('twin_v01_input_erratum','v01 输入权限勘误')}</p>
<div class="cards"><div><strong>{escape(authorization_text)}</strong><span>执行门</span></div><div><strong>{escape(result_text)}</strong><span>v02 结果</span></div><div><strong>{escape(test_text)}</strong><span>独立测试 seed 92005</span></div><div><strong>{v02_status.get('completed_algorithm_episodes', 0)}</strong><span>已执行算法回合</span></div></div>
<p>v02 只研究以下三个弱任务；其余七个任务继续保留 v01 冻结包：</p><ul>{task_items}</ul>
<h3>开发批次</h3><p>数字只从已绑定的 canonical report 读取。UNKNOWN、运行错误及其他非严格成功均保留在失败分母中。</p>{development_v02_table}
<h3>独立测试</h3>{final_v02_table}
<p>{resource_links}</p>{comparison_html}
<p>真机 Camera、RoleTracker、ContactEstimator、StageJudge、RobotCommand、SafetySupervisor 和独立 Evaluator 均未实现；本机 SAM 代码和权重已存在，但 Graph 未接入、未测端到端延迟。</p>
</section>"""
    twin_html = ""
    if all(key in bindings for key in TWIN_BINDINGS):
        scope = json.loads((REPO_ROOT / bindings["twin_scope"]["path"]).read_text())
        plan = json.loads((REPO_ROOT / bindings["twin_plan"]["path"]).read_text())
        twin = json.loads((REPO_ROOT / bindings["twin_status"]["path"]).read_text())
        if scope.get("schema") != "graph-self-adapter.twin-scope.v1":
            raise ValueError("unsupported twin_scope schema")
        if plan.get("schema") != "graph-self-adapter.twin-root-plan.v1":
            raise ValueError("unsupported twin_plan schema")
        if twin.get("schema") != "graph-self-adapter.twin-status.v1":
            raise ValueError("unsupported twin_status schema")
        planned = {row["task"]: row for row in plan.get("tasks", [])}
        scoped = list(scope.get("tasks", []))
        task_rows = twin.get("tasks", [])
        current = [row.get("task") for row in task_rows]
        if len(scoped) != 10 or set(scoped) != set(planned) or current != scoped:
            raise ValueError("Twin scope, plan and status must describe the same ordered 10 tasks")
        cohort_reports = load_cohort_reports(twin, scoped)
        test_reports = [report for report in cohort_reports if report["condition_set"] == "test"]
        development_reports = [report for report in cohort_reports if report["condition_set"] != "test"]
        task_videos = load_task_videos(twin, scoped)
        nominal = twin.get("engineering_nominal_passed")
        if isinstance(nominal, bool):
            nominal_text = f"{len(scoped)}/{len(scoped)}" if nominal else "未通过"
        elif isinstance(nominal, int):
            nominal_text = f"{nominal}/{len(scoped)}"
        elif isinstance(nominal, list):
            nominal_text = f"{sum(bool(value) for value in nominal)}/{len(scoped)}"
        else:
            nominal_text = "待记录"
        resource_labels = (
            ("twin_root_report", "最终科学报告"),
            ("twin_task_package", "可复用任务包"),
            ("twin_paired_report", "固定／在线配对报告"),
            ("twin_video_manifest", "最终注释视频清单"),
            ("twin_final_chart_svg", "逐任务结果图（SVG）"),
        )
        resource_links = " · ".join(
            link(key, label) for key, label in resource_labels if key in bindings)
        chart_html = ""
        if "twin_final_chart_png" in bindings:
            chart_path = bindings["twin_final_chart_png"]["path"]
            chart_html = (
                '<figure class="result-chart">'
                f'<img src="../{escape(chart_path, quote=True)}" '
                'alt="三配置逐任务独立测试严格成功数柱状图">'
                '<figcaption>逐任务严格整任务成功数；统计由封存汇总报告生成。</figcaption>'
                '</figure>')
        test_table = render_cohort_table(
            test_reports, task_rows=task_rows, planned=planned,
            empty_text="尚无已绑定的独立测试报告。")
        development_table = render_cohort_table(
            development_reports, task_rows=task_rows, planned=planned,
            empty_text="尚无已绑定的研发队列报告。")
        task_video_html = ""
        if task_videos:
            groups = []
            for task in scoped:
                if task not in task_videos:
                    continue
                figures = []
                for item in task_videos[task]:
                    figures.append(
                        '<figure>'
                        f'<figcaption>{escape(item["label"])} · '
                        f'<a href="../{escape(item["report_path"], quote=True)}">源运行报告</a></figcaption>'
                        f'<video controls preload="metadata" src="../{escape(item["path"], quote=True)}"></video>'
                        '</figure>')
                groups.append(
                    f'<article><h3>{escape(planned[task].get("label", task))} '
                    f'<code>{escape(task)}</code></h3>{"".join(figures)}</article>')
            task_video_html = (
                '<section class="task-videos"><h2>每任务注释视频</h2>'
                '<div class="video-grid">' + ''.join(groups) + '</div></section>')
        twin_html = f"""
<section><h2>当前实验：{len(scoped)} 个完整物理任务</h2>
<p>{link('twin_scope','实验范围')} · {link('twin_plan','根任务计划')} · {link('twin_status','当前状态')}</p>
<p>{resource_links}</p>
<div class="cards"><div><strong>{len(scoped)}</strong><span>范围内任务</span></div><div><strong>{escape(nominal_text)}</strong><span>工程名义回放</span></div><div><strong>{len(test_reports)}</strong><span>独立测试报告</span></div><div><strong>{len(development_reports)}</strong><span>研发报告</span></div></div>
<h2>独立测试结果</h2>{chart_html}{test_table}
<details class="development"><summary>展开研发版本结果（与独立测试分开统计）</summary>{development_table}</details>
</section>{task_video_html}"""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(status['headline'])}</title>
<style>body{{max-width:1560px;margin:2rem auto;padding:0 1.25rem;font:16px/1.5 system-ui;color:#17202a}}table{{border-collapse:collapse;width:100%;min-width:900px}}th,td{{border:1px solid #ccd1d1;padding:.55rem;text-align:left;vertical-align:top}}th{{background:#eef2f3}}th small{{display:block;font-weight:400;color:#566573;white-space:nowrap}}td.pending{{color:#566573;background:#f7f9f9}}video,img{{max-width:100%}}.active-study{{padding:1.25rem;border:2px solid #2471a3;border-radius:.6rem;background:#f4f9fc}}.warning{{padding:1rem;border-left:5px solid #b03a2e;background:#fdf2e9;color:#641e16}}.result-chart{{margin:1rem 0;padding:1rem;border:1px solid #d5dbdb;border-radius:.5rem}}.result-chart figcaption{{font-weight:400;color:#566573}}code{{white-space:nowrap}}section{{margin:2rem 0}}details{{margin-top:2.5rem;border:1px solid #d5dbdb;border-radius:.5rem;padding:1rem}}details.development{{margin-top:1.25rem}}summary{{font-weight:700;cursor:pointer}}.table-scroll{{overflow-x:auto}}.empty{{padding:1rem;background:#f7f9f9;border-radius:.4rem}}.cards{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}}.cards div{{min-width:180px;padding:1rem;background:#eef6f5;border-radius:.5rem}}.cards strong,.cards span{{display:block}}.cards strong{{font-size:1.5rem}}.video-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}.video-grid article{{border:1px solid #d5dbdb;border-radius:.5rem;padding:1rem}}figure{{margin:1rem 0}}figcaption{{margin-bottom:.45rem;font-weight:600}}@media(max-width:720px){{body{{padding:0 .65rem}}.video-grid{{grid-template-columns:1fr}}}}</style></head>
<body><h1>{escape(status['headline'])}</h1><p>{escape(status['summary'])}</p>
<p>{link('readme','仓库说明')} · {link('engineering_acceptance','迁移验收')}</p>
{position_html}
{live_html}
{agent_html}
{sensor_html}
{v02_html}
{twin_html}
<details><summary>历史证据：P70 R3 图、科学报告与工程视频</summary>
<p>{link('science_report','历史科学报告')} · {link('task_graph','历史冻结任务图')}</p>
<h2>P70 R3 阶段路由</h2><table><thead><tr><th>Stage</th><th>Entry</th><th>Exit</th><th>Solver</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>历史工程回放</h2><video controls preload="metadata" src="{escape(video, quote=True)}"></video></details>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    status = load_status(args.status.resolve(), REPO_ROOT)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(status))
    print(output)


if __name__ == "__main__":
    main()
