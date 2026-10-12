#!/usr/bin/env python3
"""Plan or render annotated videos from sealed Twin RSI final-test traces."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.task_graph import twin_export_packages as export  # noqa: E402
from tools.task_graph import twin_summarize_final_test as summary  # noqa: E402

SCHEMA = "graph-self-adapter.twin-final-video-plan.v1"
MANIFEST_SCHEMA = "graph-self-adapter.twin-final-video-manifest.v1"
DEFAULT_PYTHON = "/home/zheng/github/Mujoco-RSI/.venv/bin/python"
COMPARISON_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
EXTRA_FAILURE_TASKS = (
    ("put_knife_board_new", "刀任务失败"),
    ("put_mug_on_box", "双杯放箱任务失败"),
    ("sorting_cube_bimanual", "双臂任务失败"),
)
COMPARISON_TASKS = (
    "collect_cubes", "open_drawer_and_place_object", "sweep_objects")
REPORT_NAMES = {
    "baseline": "INITIAL_GRAPH_FIXED_REPORT.json",
    "selected_fixed": "SELECTED_FIXED_REPORT.json",
    "selected_online": "SELECTED_ONLINE_REPORT.json",
    "selected_paired": "SELECTED_PAIRED_REPORT.json",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _dump_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n")


def _report_sources(cohort: Mapping[str, Any]) -> list[Path]:
    paths = []
    for task in cohort.get("tasks", []):
        for binding in task.get("reports", []):
            path = Path(binding["path"])
            path = path if path.is_absolute() else REPO_ROOT / path
            path = path.resolve()
            if not path.is_file() or export.sha256(path) != binding.get("sha256"):
                raise ValueError(f"cohort source report binding mismatch: {path}")
            paths.append(path)
    return paths


def _validate_final_reports(report_dir: Path, baseline_selection: Path,
                            baseline_results: Path, selected_selection: Path,
                            selected_results: Path, baseline_execution_review: Path,
                            selected_execution_review: Path,
                            trials: int) -> tuple[dict, dict]:
    report_dir = report_dir.resolve()
    paths = {key: report_dir / name for key, name in REPORT_NAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "all four sealed final-test reports are required before video planning: " +
            ", ".join(missing))
    documents = {key: _load(path) for key, path in paths.items()}
    if any(documents[key].get("schema") != summary.COHORT_SCHEMA
           for key in ("baseline", "selected_fixed", "selected_online")):
        raise ValueError("three cohort reports must use twin-cohort-summary.v1")
    if documents["selected_paired"].get("schema") != summary.PAIRED_SCHEMA:
        raise ValueError("selected paired report has the wrong schema")

    baseline_selection_doc, baseline = summary.load_reports(
        baseline_results, baseline_selection, {"fixed_z"}, trials,
        baseline_execution_review)
    selected_selection_doc, selected = summary.load_reports(
        selected_results, selected_selection, {"fixed_z", "online_z"}, trials,
        selected_execution_review)
    baseline_review = _load(baseline_execution_review.resolve())
    selected_review = _load(selected_execution_review.resolve())
    regenerated = {
        "baseline": summary.cohort_summary(
            baseline_selection_doc, baseline, "fixed_z", trials, baseline_review),
        "selected_fixed": summary.cohort_summary(
            selected_selection_doc, selected, "fixed_z", trials, selected_review),
        "selected_online": summary.cohort_summary(
            selected_selection_doc, selected, "online_z", trials, selected_review),
        "selected_paired": summary.selected_paired(
            selected_selection_doc, selected, trials, selected_review),
    }
    for key in regenerated:
        if documents[key] != regenerated[key]:
            raise ValueError(f"sealed final report differs from validated source runs: {paths[key]}")
    # Force validation of every source binding retained in the published summaries.
    for key in ("baseline", "selected_fixed", "selected_online"):
        _report_sources(documents[key])
    return baseline, selected


def choose_selected(reports: Mapping[tuple[str, str, int], tuple[Path, dict]],
                    tasks: list[str], trials: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    used: set[Path] = set()
    for task in tasks:
        candidates = [reports[(task, "online_z", trial)] for trial in range(trials)]
        selected = next((item for item in candidates if item[1]["strict_success"]),
                        candidates[0])
        path, report = selected
        chosen.append({"task": task, "report": path, "report_data": report,
                       "label": "选版 online 首个严格成功" if report["strict_success"]
                       else "选版 online 无成功，首个失败",
                       "slug": f"{task}__selected_online__primary"})
        used.add(path.resolve())
    for task, label in EXTRA_FAILURE_TASKS:
        failures = [reports[(task, "online_z", trial)] for trial in range(trials)
                    if not reports[(task, "online_z", trial)][1]["strict_success"]]
        if not failures:
            continue
        path, report = failures[0]
        if path.resolve() in used:
            continue
        chosen.append({"task": task, "report": path, "report_data": report,
                       "label": label, "slug": f"{task}__selected_online__failure"})
        used.add(path.resolve())
    if len(chosen) > 13:
        raise AssertionError("selected video cap exceeded")
    return chosen


def choose_comparisons(
    baseline: Mapping[tuple[str, str, int], tuple[Path, dict]],
    selected: Mapping[tuple[str, str, int], tuple[Path, dict]], trials: int,
) -> list[dict[str, Any]]:
    chosen = []
    for task in COMPARISON_TASKS:
        match = None
        for trial in range(trials):
            left = baseline[(task, "fixed_z", trial)]
            right = selected[(task, "fixed_z", trial)]
            if (not left[1]["strict_success"] and right[1]["strict_success"] and
                    left[1]["condition"] == right[1]["condition"] and
                    left[1]["bindings"]["condition_sha256"] ==
                    right[1]["bindings"]["condition_sha256"]):
                match = (trial, left, right)
                break
        if match is None:
            raise ValueError(
                f"no baseline-failure/selected-fixed-success matched pair for {task}")
        trial, left, right = match
        chosen.append({"task": task, "trial": trial, "baseline": left,
                       "selected": right,
                       "label": "单案例：初始任务图失败 / 外层修订任务图 fixed 成功",
                       "slug": f"{task}__baseline_vs_selected_fixed"})
    return chosen


def choose_inner_comparison(
    selected: Mapping[tuple[str, str, int], tuple[Path, dict]], trials: int,
) -> dict[str, Any]:
    task = "sorting_cube_bimanual"
    for trial in range(trials):
        fixed = selected[(task, "fixed_z", trial)]
        online = selected[(task, "online_z", trial)]
        if (not fixed[1]["strict_success"] and online[1]["strict_success"] and
                fixed[1]["condition"] == online[1]["condition"] and
                fixed[1]["bindings"]["condition_sha256"] ==
                online[1]["bindings"]["condition_sha256"] and
                all(fixed[1]["bindings"][key] == online[1]["bindings"][key]
                    for key in ("graph_sha256", "model_sha256",
                                "candidate_freeze_sha256"))):
            return {"task": task, "trial": trial, "baseline": fixed,
                    "selected": online,
                    "label": "单案例：同图同模型 fixed-z 失败 / online-z 成功",
                    "slug": f"{task}__selected_fixed_vs_online",
                    "comparison_kind": "inner_fixed_vs_online",
                    "left_title": "同图同模型 fixed-z · 失败",
                    "right_title": "同图同模型 online-z · 成功",
                    "caption": "仅单例；总体 69% vs 70%，尚无稳定额外收益"}
    raise ValueError("no selected fixed-failure/online-success matched bimanual pair")


def _selection_rows(path: Path) -> dict[str, dict[str, Any]]:
    document = _load(path.resolve())
    return {row["task"]: row for row in document["tasks"]}


def _context(task: str, report: Mapping[str, Any], selection_path: Path,
             selection_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    row = selection_rows[task]
    if report.get("selection_row") != row or report.get("candidate_id") != row["candidate_id"]:
        raise ValueError(f"{task}: run does not match selected task row")
    base = selection_path.resolve().parent
    snapshot, expected_freeze = export.source_snapshot_path(row["source_snapshot"], base)
    source_hashes, actual_freeze = export.verify_snapshot(
        snapshot, row.get("source_freeze_sha256", expected_freeze))
    solution = export.solution_path(row["packages"], task, base)
    export.verify_solution(row, solution, source_hashes)
    candidate_freeze = solution / "CANDIDATE_FREEZE.json"
    physics = snapshot / "prove/task_graph/twin_physics.py"
    external = _load(solution / "EXTERNAL_BINDINGS.json")
    root_binding = external.get("root_plan", {})
    root_plan = snapshot / "evaluations/twin_rsi_v01/ROOT_TASK_PLAN.json"
    if (not root_plan.is_file() or export.sha256(root_plan) != root_binding.get("sha256")):
        raise ValueError(f"{task}: frozen ROOT_TASK_PLAN binding mismatch")
    task_rows = [item for item in _load(root_plan).get("tasks", [])
                 if item.get("task") == task]
    if len(task_rows) != 1:
        raise ValueError(f"{task}: frozen ROOT_TASK_PLAN task is not unique")
    scene_binding = task_rows[0].get("sources", {}).get("scene.xml", {})
    scene = Path(scene_binding.get("path", "")).resolve()
    if not scene.is_file() or export.sha256(scene) != scene_binding.get("sha256"):
        raise ValueError(f"{task}: frozen scene binding mismatch")
    if report.get("bindings", {}).get("scene_sha256") != scene_binding["sha256"]:
        raise ValueError(f"{task}: run scene differs from frozen ROOT_TASK_PLAN")
    return {"scene": str(scene), "physics_source": str(physics),
            "candidate_freeze": str(candidate_freeze),
            "source_snapshot": str(snapshot), "source_freeze_sha256": actual_freeze,
            "root_plan": str(root_plan), "root_plan_sha256": export.sha256(root_plan)}


def build_plan(report_dir: Path, baseline_selection: Path, baseline_results: Path,
               selected_selection: Path, selected_results: Path,
               baseline_execution_review: Path, selected_execution_review: Path,
               output_root: Path,
               *, trials: int = 10, python: str = DEFAULT_PYTHON) -> dict[str, Any]:
    baseline, selected = _validate_final_reports(
        report_dir, baseline_selection, baseline_results,
        selected_selection, selected_results, baseline_execution_review,
        selected_execution_review, trials)
    selected_doc = _load(selected_selection.resolve())
    tasks = [row["task"] for row in selected_doc["tasks"]]
    selected_rows = _selection_rows(selected_selection)
    baseline_rows = _selection_rows(baseline_selection)
    output_root = output_root.resolve()
    if not output_root.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("video evidence output must remain inside this repository")
    renderer = REPO_ROOT / "tools/evidence/render_twin_graph_video.py"
    selected_jobs = []
    for item in choose_selected(selected, tasks, trials):
        context = _context(item["task"], item["report_data"],
                           selected_selection, selected_rows)
        output = output_root / "selected" / f"{item['slug']}.mp4"
        command = [python, str(renderer), "--report", str(item["report"]),
                   "--scene", context["scene"], "--physics-source",
                   context["physics_source"], "--candidate-freeze",
                   context["candidate_freeze"], "--output", str(output),
                   "--language", "zh", "--fps", "10"]
        selected_jobs.append({**{k: v for k, v in item.items() if k != "report_data"},
                              "report": str(item["report"]), "output": str(output),
                              "context": context, "command": command})
    comparison_jobs = []
    for item in choose_comparisons(baseline, selected, trials):
        task = item["task"]
        left_path, left_report = item["baseline"]
        right_path, right_report = item["selected"]
        left_context = _context(task, left_report, baseline_selection, baseline_rows)
        right_context = _context(task, right_report, selected_selection, selected_rows)
        raw_dir = output_root / "comparisons/raw"
        left_output = raw_dir / f"{item['slug']}__baseline.mp4"
        right_output = raw_dir / f"{item['slug']}__selected.mp4"
        def render_command(path: Path, report_path: Path, context: Mapping[str, str]):
            return [python, str(renderer), "--report", str(report_path),
                    "--scene", context["scene"], "--physics-source",
                    context["physics_source"], "--candidate-freeze",
                    context["candidate_freeze"], "--output", str(path),
                    "--language", "zh", "--fps", "10"]
        comparison_jobs.append({
            "task": task, "trial": item["trial"], "label": item["label"],
            "slug": item["slug"], "baseline_report": str(left_path),
            "selected_report": str(right_path), "baseline_context": left_context,
            "selected_context": right_context, "baseline_raw": str(left_output),
            "selected_raw": str(right_output),
            "output": str(output_root / "comparisons" / f"{item['slug']}.mp4"),
            "baseline_command": render_command(left_output, left_path, left_context),
            "selected_command": render_command(right_output, right_path, right_context),
            "comparison_kind": "outer_initial_vs_revised",
            "left_title": "初始任务图 fixed · 失败",
            "right_title": "外层修订任务图 fixed · 成功",
            "caption": "单案例封存轨迹对比（非总体统计）",
        })
    inner = choose_inner_comparison(selected, trials)
    task = inner["task"]
    left_path, left_report = inner["baseline"]
    right_path, right_report = inner["selected"]
    left_context = _context(task, left_report, selected_selection, selected_rows)
    right_context = _context(task, right_report, selected_selection, selected_rows)
    raw_dir = output_root / "comparisons/raw"
    left_output = raw_dir / f"{inner['slug']}__fixed.mp4"
    right_output = raw_dir / f"{inner['slug']}__online.mp4"
    comparison_jobs.append({
        "task": task, "trial": inner["trial"], "label": inner["label"],
        "slug": inner["slug"], "comparison_kind": inner["comparison_kind"],
        "left_title": inner["left_title"], "right_title": inner["right_title"],
        "caption": inner["caption"], "baseline_report": str(left_path),
        "selected_report": str(right_path), "baseline_context": left_context,
        "selected_context": right_context, "baseline_raw": str(left_output),
        "selected_raw": str(right_output),
        "output": str(output_root / "comparisons" / f"{inner['slug']}.mp4"),
        "baseline_command": render_command(left_output, left_path, left_context),
        "selected_command": render_command(right_output, right_path, right_context),
    })
    rendered_by_report = {job["report"]: job["output"] for job in selected_jobs}
    for job in comparison_jobs:
        reused = rendered_by_report.get(job["selected_report"])
        if reused is not None:
            job["selected_raw"] = reused
            job["selected_command"] = None
    targets = [Path(job["output"]) for job in selected_jobs + comparison_jobs]
    targets += [Path(job[key]) for job in comparison_jobs
                for key, command_key in (("baseline_raw", "baseline_command"),
                                         ("selected_raw", "selected_command"))
                if job[command_key] is not None]
    targets += [output_root / "FINAL_TEST_VIDEO_MANIFEST.json",
                output_root / "TASK_VIDEOS.json"]
    existing = [str(path) for path in targets
                if path.exists() or path.with_suffix(".render.json").exists()]
    if existing:
        raise FileExistsError("refusing to overwrite video evidence: " + ", ".join(existing))
    fixed_summary = _load((report_dir / REPORT_NAMES["selected_fixed"]).resolve())
    online_summary = _load((report_dir / REPORT_NAMES["selected_online"]).resolve())
    fixed_rate = 100. * fixed_summary["successes"] / fixed_summary["trials"]
    online_rate = 100. * online_summary["successes"] / online_summary["trials"]
    return {"schema": SCHEMA, "offline_visual_replay_only": True,
            "policy_or_adapter_executed": False, "output_root": str(output_root),
            "selected_selection": {"path": str(selected_selection.resolve()),
                                   "sha256": export.sha256(selected_selection.resolve())},
            "baseline_selection": {"path": str(baseline_selection.resolve()),
                                   "sha256": export.sha256(baseline_selection.resolve())},
            "execution_reviews": {
                "baseline": {"path": str(baseline_execution_review.resolve()),
                             "sha256": export.sha256(baseline_execution_review.resolve())},
                "selected": {"path": str(selected_execution_review.resolve()),
                             "sha256": export.sha256(selected_execution_review.resolve())}},
            "final_reports": {key: {"path": str((report_dir / name).resolve()),
                                     "sha256": export.sha256((report_dir / name).resolve())}
                              for key, name in REPORT_NAMES.items()},
            "aggregate_context": {
                "selected_fixed_successes": fixed_summary["successes"],
                "selected_fixed_trials": fixed_summary["trials"],
                "selected_online_successes": online_summary["successes"],
                "selected_online_trials": online_summary["trials"],
                "difference_percentage_points": online_rate - fixed_rate,
                "interpretation": "Descriptive difference only; no established stable online-z benefit."},
            "selected_jobs": selected_jobs, "comparison_jobs": comparison_jobs,
            "counts": {"selected": len(selected_jobs),
                       "comparisons": len(comparison_jobs)}}


def _run(command: list[str]) -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["MUJOCO_GL"] = "egl"
    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])


def _video_duration(sidecar: Path) -> float:
    render = _load(sidecar)["render"]
    return float(render["frames"]) / float(render["fps"])


def _compose_pair(job: Mapping[str, Any]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required")
    if not Path(COMPARISON_FONT).is_file():
        raise FileNotFoundError(f"comparison CJK font is required: {COMPARISON_FONT}")
    left, right, output = Path(job["baseline_raw"]), Path(job["selected_raw"]), Path(job["output"])
    left_duration = _video_duration(left.with_suffix(".render.json"))
    right_duration = _video_duration(right.with_suffix(".render.json"))
    left_pad, right_pad = max(0., right_duration - left_duration), max(0., left_duration - right_duration)
    caption = str(job["caption"])
    filter_graph = (
        f"[0:v]tpad=stop_mode=clone:stop_duration={left_pad:.3f},"
        "pad=iw:ih+80:0:42:color=black,"
        f"drawtext=fontfile='{COMPARISON_FONT}':text='{job['left_title']}':"
        "x=18:y=10:fontsize=22:fontcolor=white:expansion=none[l];"
        f"[1:v]tpad=stop_mode=clone:stop_duration={right_pad:.3f},"
        "pad=iw:ih+80:0:42:color=black,"
        f"drawtext=fontfile='{COMPARISON_FONT}':text='{job['right_title']}':"
        "x=18:y=10:fontsize=22:fontcolor=white:expansion=none[r];"
        "[l][r]hstack=inputs=2[stack];"
        f"[stack]drawtext=fontfile='{COMPARISON_FONT}':text='{caption}':"
        "x=(w-text_w)/2:y=h-30:fontsize=20:fontcolor=white:expansion=none[out]")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-nostdin", "-v", "error", "-i", str(left), "-i", str(right),
               "-filter_complex", filter_graph, "-map", "[out]", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr[-4000:])
    sidecar = output.with_suffix(".render.json")
    sidecar.write_text(json.dumps({
        "schema": "graph-self-adapter.twin-comparison-video.v1",
        "offline_visual_replay_only": True, "policy_or_adapter_executed": False,
        "caption": caption,
        "baseline": {"video": str(left), "sha256": export.sha256(left),
                     "report": job["baseline_report"],
                     "report_sha256": export.sha256(Path(job["baseline_report"]))},
        "selected": {"video": str(right), "sha256": export.sha256(right),
                     "report": job["selected_report"],
                     "report_sha256": export.sha256(Path(job["selected_report"]))},
        "timing": {"baseline_duration_sec": left_duration,
                   "selected_duration_sec": right_duration,
                   "baseline_last_frame_hold_sec": left_pad,
                   "selected_last_frame_hold_sec": right_pad,
                   "playback_speed_changed": False},
        "video": {"path": str(output), "sha256": export.sha256(output)}},
        indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def execute(plan: Mapping[str, Any], workers: int) -> dict[str, Any]:
    if not 1 <= workers <= 2:
        raise ValueError("render workers must be 1 or 2")
    commands = [job["command"] for job in plan["selected_jobs"]]
    commands += [job[key] for job in plan["comparison_jobs"]
                 for key in ("baseline_command", "selected_command")
                 if job[key] is not None]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, command) for command in commands]
        for future in as_completed(futures):
            future.result()
    for job in plan["comparison_jobs"]:
        _compose_pair(job)

    bindings, records = [], []
    for job in plan["selected_jobs"]:
        video, report = Path(job["output"]), Path(job["report"])
        bindings.append({"task": job["task"], "label": job["label"],
                         "path": str(video.relative_to(REPO_ROOT)),
                         "sha256": export.sha256(video),
                         "report_path": str(report.relative_to(REPO_ROOT)),
                         "report_sha256": export.sha256(report)})
        records.append({**job, "video_sha256": export.sha256(video),
                        "render_sidecar": str(video.with_suffix(".render.json")),
                        "render_sidecar_sha256": export.sha256(video.with_suffix(".render.json"))})
    for job in plan["comparison_jobs"]:
        video, report = Path(job["output"]), Path(job["selected_report"])
        bindings.append({"task": job["task"], "label": job["label"],
                         "path": str(video.relative_to(REPO_ROOT)),
                         "sha256": export.sha256(video),
                         "report_path": str(report.relative_to(REPO_ROOT)),
                         "report_sha256": export.sha256(report)})
        records.append({**job, "video_sha256": export.sha256(video),
                        "baseline_raw_sha256": export.sha256(Path(job["baseline_raw"])),
                        "selected_raw_sha256": export.sha256(Path(job["selected_raw"])),
                        "comparison_sidecar": str(video.with_suffix(".render.json")),
                        "comparison_sidecar_sha256": export.sha256(video.with_suffix(".render.json")),
                        "caption_boundary": "single sealed case; not aggregate evidence"})
    output_root = Path(plan["output_root"])
    manifest = {"schema": MANIFEST_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
                "offline_visual_replay_only": True, "policy_or_adapter_executed": False,
                "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True,
                                                          separators=(",", ":")).encode()).hexdigest(),
                "selected_selection": plan["selected_selection"],
                "baseline_selection": plan["baseline_selection"],
                "execution_reviews": plan["execution_reviews"],
                "final_reports": plan["final_reports"],
                "aggregate_context": plan["aggregate_context"], "records": records}
    _dump_new(output_root / "TASK_VIDEOS.json", {"task_videos": bindings})
    _dump_new(output_root / "FINAL_TEST_VIDEO_MANIFEST.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reports-dir", type=Path, required=True)
    p.add_argument("--baseline-selection", type=Path, required=True)
    p.add_argument("--baseline-results", type=Path, required=True)
    p.add_argument("--selected-selection", type=Path, required=True)
    p.add_argument("--selected-results", type=Path, required=True)
    p.add_argument("--baseline-execution-review", type=Path, required=True)
    p.add_argument("--selected-execution-review", type=Path, required=True)
    p.add_argument("--output-root", type=Path,
                   default=REPO_ROOT / "media/twin_rsi_v01")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--python", default=DEFAULT_PYTHON)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--render", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.workers <= 2:
        raise ValueError("workers must be 1 or 2")
    plan = build_plan(args.reports_dir, args.baseline_selection,
                      args.baseline_results, args.selected_selection,
                      args.selected_results, args.baseline_execution_review,
                      args.selected_execution_review, args.output_root,
                      trials=args.trials, python=args.python)
    if args.plan_only:
        json.dump(plan, sys.stdout, indent=2, sort_keys=True, default=str)
        print()
    else:
        execute(plan, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
