#!/usr/bin/env python3
"""Render a sealed Twin Graph run from its saved full-state visual trace.

This is an evidence-only visualizer.  Assigning saved qpos values is permitted
here solely to reconstruct frames; this module never imports or calls a policy,
controller, adapter, evaluator, or rollout runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPORT_SCHEMA = "graph-self-adapter.twin-run.v1"
VIDEO_SCHEMA = "graph-self-adapter.twin-video.v1"
TRACE_KEYS = (
    "timestamps", "qpos", "stage_id", "stage_index", "arm", "object_role",
    "target_role", "controller_kind", "z", "bilateral_holding", "failure",
)
ZH_TASKS = {
    "collect_apple": "收集苹果",
    "collect_cubes": "收集方块",
    "hanging_mug": "悬挂杯子",
    "open_drawer_and_place_object": "打开抽屉并放入物体",
    "put_knife_board_new": "把刀放到砧板",
    "put_mug_on_box": "把杯子放到盒子",
    "sort_two_spoon": "分拣两把勺子",
    "sorting_cube_bimanual": "双臂分拣方块",
    "stack_cups": "叠放杯子",
    "sweep_objects": "清扫物体",
}

ZH_PHASES = {"approach":"接近", "grasp":"抓持", "lift":"抬升", "move":"运输", "align":"对齐",
             "release":"松爪", "wait":"等待", "insert":"插入", "push":"推动", "pull":"拉动", "retreat":"撤离"}
ZH_ROLES = {"left":"左臂", "right":"右臂", "red_knife":"红刀", "blue_knife":"蓝刀", "board":"砧板",
            "knife_holder":"刀架", "red_mug":"红杯", "blue_mug":"蓝杯", "red_cube":"红方块",
            "blue_cube":"蓝方块", "yellow_cube":"黄方块", "white_mug":"白杯", "tray":"托盘", "box":"盒子", "table_handover_region":"桌面交接区"}


def chinese_stage(stage: str) -> str:
    parts = stage.split(".")
    return ".".join(parts[:-1]) + " · " + ZH_PHASES.get(parts[-1], parts[-1])


def v02_timing_annotation(report: Mapping[str, Any], stage: str) -> str:
    candidate = report.get("candidate_id", "")
    phase = stage.split(".")[-1]
    speed = {"grasp":.5, "lift":.5, "align":.7}.get(phase, 1.) if candidate in ("v02_a2","v02_a3") else 1.
    confirmation = " · 抓持确认≥0.2秒" if phase=="grasp" and candidate in ("v02_a2","v02_a3") else ""
    return f"{speed:g}×参考速度{confirmation}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _binding(report: Mapping[str, Any], name: str) -> tuple[str | None, str]:
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("report.bindings must be an object")
    value = bindings.get(name)
    if isinstance(value, Mapping):
        expected = value.get("sha256")
        path = value.get("path")
    else:
        expected = bindings.get(f"{name}_sha256")
        path = bindings.get(f"{name}_path")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"report binding {name!r} lacks a SHA-256")
    return (str(path) if path is not None else None), expected


def _resolve_bound_path(raw: str, report_path: Path) -> Path:
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [report_path.parent / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"bound file does not exist: {raw}")


def _strings(array: np.ndarray, key: str, count: int) -> np.ndarray:
    if array.shape != (count,):
        raise ValueError(f"trace {key} must have shape ({count},), got {array.shape}")
    if array.dtype.kind not in "US":
        raise ValueError(f"trace {key} must use a non-object Unicode/bytes dtype")
    return array.astype(str)


def load_inputs(report_path: Path, trace_override: Path | None, scene: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = _read_json(report_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"unsupported report schema: {report.get('schema')!r}")
    trace_raw, trace_expected = _binding(report, "visual_trace")
    if trace_override is None:
        if trace_raw is None:
            raise ValueError("visual_trace binding must include path when --trace is omitted")
        trace_path = _resolve_bound_path(trace_raw, report_path)
    else:
        trace_path = trace_override.expanduser().resolve()
    if not trace_path.is_file() or sha256(trace_path) != trace_expected:
        raise ValueError("visual trace is missing or its SHA-256 does not match the sealed report")
    _, scene_expected = _binding(report, "scene")
    scene = scene.expanduser().resolve()
    if not scene.is_file() or sha256(scene) != scene_expected:
        raise ValueError("scene is missing or its SHA-256 does not match the sealed report")

    with np.load(trace_path, allow_pickle=False) as archive:
        missing = sorted(set(TRACE_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"visual trace lacks required arrays: {missing}")
        trace = {key: archive[key].copy() for key in archive.files}
    timestamps = np.asarray(trace["timestamps"], dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError("trace timestamps must be a finite vector with at least two samples")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("trace timestamps must be strictly increasing")
    count = len(timestamps)
    qpos = np.asarray(trace["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] != count or not np.isfinite(qpos).all():
        raise ValueError("trace qpos must be finite [N,nq]")
    stage_index = np.asarray(trace["stage_index"])
    if stage_index.shape != (count,) or stage_index.dtype.kind not in "iu":
        raise ValueError("trace stage_index must be integer [N]")
    z = np.asarray(trace["z"], dtype=np.float64)
    if z.shape != (count, 3) or not np.isfinite(z).all():
        raise ValueError("trace z must be finite [N,3]")
    holding = np.asarray(trace["bilateral_holding"])
    if holding.shape != (count,) or holding.dtype.kind != "b":
        raise ValueError("trace bilateral_holding must be Boolean [N]")
    for key in ("stage_id", "arm", "object_role", "target_role", "controller_kind", "failure"):
        trace[key] = _strings(np.asarray(trace[key]), key, count)
    trace["timestamps"], trace["qpos"], trace["stage_index"] = timestamps, qpos, stage_index
    trace["z"], trace["bilateral_holding"] = z, holding
    trace["_path"] = np.asarray(str(trace_path))
    return report, trace


def condition_binding(report: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    condition = report.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError("report.condition must be an object")
    canonical = json_sha256(condition)
    bindings = report.get("bindings", {})
    value = bindings.get("condition") if isinstance(bindings, Mapping) else None
    expected = None
    raw_path = None
    if isinstance(value, Mapping):
        expected, raw_path = value.get("sha256"), value.get("path")
    elif isinstance(bindings, Mapping):
        expected, raw_path = bindings.get("condition_sha256"), bindings.get("condition_path")
    record: dict[str, Any] = {"embedded_canonical_sha256": canonical}
    if raw_path is not None:
        source = _resolve_bound_path(str(raw_path), report_path)
        actual = sha256(source)
        if expected is not None and actual != expected:
            raise ValueError("condition file SHA-256 does not match the sealed report")
        if _read_json(source) != dict(condition):
            raise ValueError("bound condition file differs from embedded report.condition")
        record.update(path=str(source), sha256=actual)
    elif expected is not None:
        if expected != canonical:
            raise ValueError("embedded condition canonical SHA-256 does not match the sealed report")
        record["sha256"] = expected
    else:
        raise ValueError("sealed report lacks bindings.condition_sha256")
    return record


def apply_fixed_initial_translation(model: mujoco.MjModel, condition: Mapping[str, Any]) -> dict[str, list[float]]:
    translations = condition.get("initial_translation", {})
    if translations is None:
        return {}
    if not isinstance(translations, Mapping):
        raise ValueError("condition.initial_translation must be an object")
    actual = condition.get("actual_initial_translation_m")
    if actual is not None and actual != translations:
        raise ValueError("condition requested and actual initial translations differ")
    applied: dict[str, list[float]] = {}
    for role, raw_delta in translations.items():
        delta = np.asarray(raw_delta, dtype=np.float64)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError(f"initial_translation[{role!r}] must be finite XYZ")
        body_name = str(role) if str(role).startswith("object_") else f"object_{role}"
        try:
            body = model.body(body_name)
        except KeyError as exc:
            raise ValueError(f"condition refers to unknown body {body_name!r}") from exc
        start = int(model.body_jntadr[body.id])
        count = int(model.body_jntnum[body.id])
        joint_ids = range(start, start + count)
        if any(model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE for joint_id in joint_ids):
            # Free-object translations are already represented by every saved
            # qpos sample and must not be applied a second time.
            continue
        model.body_pos[body.id] += delta
        applied[str(role)] = delta.tolist()
    return applied


def _condition_translations(condition: Mapping[str, Any]) -> dict[str, list[float]]:
    translations = condition.get("initial_translation", {})
    if not isinstance(translations, Mapping):
        raise ValueError("condition.initial_translation must be an object")
    actual = condition.get("actual_initial_translation_m")
    if actual is not None and actual != translations:
        raise ValueError("condition requested and actual initial translations differ")
    checked: dict[str, list[float]] = {}
    for role, value in translations.items():
        delta = np.asarray(value, dtype=np.float64)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError(f"initial_translation[{role!r}] must be finite XYZ")
        checked[str(role)] = delta.tolist()
    return checked


def _initial_transform_sha256(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    digest = hashlib.sha256()
    for value in (model.body_pos, model.body_quat, data.qpos, data.xpos, data.xquat):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def frozen_initial_state(
    report: Mapping[str, Any],
    report_path: Path,
    scene_path: Path,
    physics_source: Path,
    candidate_freeze: Path,
) -> tuple[Any, mujoco.MjModel, mujoco.MjData, dict[str, Any]]:
    """Build the visualization state with the run's hash-bound frozen backend."""
    physics_source = physics_source.expanduser().resolve()
    candidate_freeze = candidate_freeze.expanduser().resolve()
    if not physics_source.is_file() or not candidate_freeze.is_file():
        raise FileNotFoundError("frozen physics source and candidate freeze must exist")
    _, sealed_freeze_sha = _binding(report, "candidate_freeze")
    freeze_sha = sha256(candidate_freeze)
    if freeze_sha != sealed_freeze_sha:
        raise ValueError("candidate freeze SHA-256 does not match the sealed report")
    freeze = _read_json(candidate_freeze)
    expected_physics = freeze.get("code", {}).get("prove/task_graph/twin_physics.py")
    if not isinstance(expected_physics, str) or len(expected_physics) != 64:
        raise ValueError("candidate freeze lacks the frozen twin_physics.py hash")
    physics_sha = sha256(physics_source)
    if physics_sha != expected_physics:
        raise ValueError("physics source SHA-256 does not match CANDIDATE_FREEZE.code")
    for key in ("task", "candidate_id"):
        if report.get(key) is not None and freeze.get(key) != report.get(key):
            raise ValueError(f"candidate freeze {key} does not match the sealed report")

    module_name = f"_twin_video_physics_{physics_sha[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, physics_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen physics source: {physics_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    translations = _condition_translations(report["condition"])
    world = module.TwinWorld.from_run(
        scene_path.parent, render=False, initial_translation=translations)
    if Path(scene_path.parent / "scene.xml").resolve() != scene_path.resolve():
        world.close()
        raise ValueError("--scene must be the scene.xml inside the frozen source run")
    if float(world.data.time) != 0.:
        world.close()
        raise ValueError("frozen backend initialization advanced physical time")
    record = {
        "mode": "frozen_twin_physics",
        "candidate_freeze": {"path": str(candidate_freeze), "sha256": freeze_sha},
        "physics_source": {"path": str(physics_source), "sha256": physics_sha},
        "requested_world_translations": translations,
        "initial_body_transforms_sha256": _initial_transform_sha256(
            world.model, world.data),
        "physical_time_sec": float(world.data.time),
        "mj_step_executed": False,
    }
    return world, world.model, world.data, record


def _font(path: Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        path,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _truth(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key in ("passed", "strict_success", "success"):
            if key in value:
                return bool(value[key])
        nested = value.get("task_outcome")
        return _truth(nested) if nested is not None else False
    return bool(value)


def _adapter_mode(report: Mapping[str, Any], report_path: Path) -> str:
    condition = report.get("condition", {})
    if isinstance(condition, Mapping):
        for key in ("adapter_mode", "mode", "adaptation"):
            value = condition.get(key)
            if isinstance(value, str) and value:
                return value
        if isinstance(condition.get("online"), bool):
            return "online_z" if condition["online"] else "fixed_z"
    return "online_z" if report_path.stem.endswith("_online") else "fixed_z"


def _controller_label(kind: str, mode: str, language: str) -> str:
    neural = "mlp" in kind.lower() or "neural" in kind.lower()
    if language == "zh":
        base = "短时冻结 MLP" if neural else "代码控制器"
        return f"{base} + 在线 z" if neural and "online" in mode else base
    base = "short frozen MLP" if neural else "code controller"
    return f"{base} + online z" if neural and "online" in mode else base


def _holding_gate_annotation(
    report: Mapping[str, Any], trace: Mapping[str, np.ndarray], index: int,
    language: str,
) -> tuple[str, str]:
    rows = report.get("trace")
    if isinstance(rows, list) and index < len(rows) and isinstance(rows[index], Mapping):
        controller = rows[index].get("controller")
        diagnostics = (controller.get("predicate_diagnostics")
                       if isinstance(controller, Mapping) else None)
        if isinstance(diagnostics, list):
            holding = [row for row in diagnostics if isinstance(row, Mapping) and
                       row.get("predicate") == "holding"]
            if holding:
                value = holding[-1].get("value")
                if value is True:
                    text = "满足" if language == "zh" else "satisfied"
                elif value is False:
                    text = "未满足" if language == "zh" else "not satisfied"
                else:
                    text = "未知" if language == "zh" else "unknown"
                label = ("保持条件（夹持/目标支撑）" if language == "zh"
                         else "Hold gate (grip/target support)")
                return label, text
    legacy = bool(trace["bilateral_holding"][index])
    return (("双指门记录", "是" if legacy else "否") if language == "zh" else
            ("Recorded bilateral gate", "yes" if legacy else "no"))


def _camera_name(model: mujoco.MjModel, requested: str) -> str:
    names = [model.camera(index).name or "" for index in range(model.ncam)]
    if requested == "auto":
        for preferred in ("overview", "main"):
            if preferred in names:
                return preferred
        if names:
            return names[0]
    elif requested in names:
        return requested
    raise ValueError(f"camera {requested!r} unavailable; scene cameras: {names}")


def _short(value: Any, width: int = 34) -> str:
    text = str(value or "—")
    return text if len(text) <= width else text[: width - 1] + "…"


def _draw_frame(
    rgb: np.ndarray,
    *,
    report: Mapping[str, Any],
    trace: Mapping[str, np.ndarray],
    index: int,
    mode: str,
    language: str,
    fonts: tuple[Any, Any, Any],
) -> Image.Image:
    title_font, body_font, small_font = fonts
    canvas = Image.new("RGB", (960, 540), "#101722")
    canvas.paste(Image.fromarray(rgb).resize((640, 360), Image.Resampling.LANCZOS), (0, 0))
    draw = ImageDraw.Draw(canvas)
    count = len(trace["timestamps"])
    condition = report.get("condition") if isinstance(report.get("condition"), Mapping) else {}
    task = str(report.get("task") or condition.get("task") or "unknown_task")
    strict = bool(report.get("strict_success", False))
    graph = bool(report.get("graph_completed", False))
    original = report.get("original_task_outcome")
    task_ok = _truth(original)
    outcome = report.get("outcome", {})
    safety_value = report.get("numerical_safety", False)
    if isinstance(original, Mapping) and "numerical_safety" in original:
        safety_value = original["numerical_safety"]
    if isinstance(outcome, Mapping) and "numerical_safety" in outcome:
        safety_value = outcome["numerical_safety"]
    safety = _truth(safety_value)
    stage = trace["stage_id"][index]
    failure = trace["failure"][index] or report.get("graph_failure") or ""
    z = trace["z"][index]
    dz = 0. if index == 0 else float(np.linalg.norm(z - trace["z"][index - 1]))
    holding_label, holding_value = _holding_gate_annotation(
        report, trace, index, language)
    controller = _controller_label(trace["controller_kind"][index], mode, language)
    current_time = float(trace["timestamps"][index])
    duration = float(trace["timestamps"][-1])

    if language == "zh":
        title = f"{ZH_TASKS.get(task, task)}｜{'严格成功' if strict else '未通过严格成功门'}"
        subtitle = "封存回合最终结果标记 · 当前进度见物理时间 · 未重新执行策略"
        if str(report.get("candidate_id", "")).startswith("v02_"):
            subtitle = f"{report['candidate_id']} · 真实物理时间标注 · 减速是耗时权衡，无新增恢复"
        display_stage = chinese_stage(str(stage))
        rows = [
            ("阶段", f"{int(trace['stage_index'][index])} · {_short(display_stage, 27)}"),
            ("控制", controller),
            ("角色", f"{ZH_ROLES.get(trace['arm'][index], trace['arm'][index])} · {ZH_ROLES.get(trace['object_role'][index], trace['object_role'][index])} → {ZH_ROLES.get(trace['target_role'][index], trace['target_role'][index])}"),
            ("在线状态", f"z {z[0]:+.3f}/{z[1]:+.3f}/{z[2]:+.3f} | Δ {dz:.4f}"),
            (holding_label, holding_value),
            ("物理时间", f"{current_time:.2f} / {duration:.2f} 秒"),
        ]
        if str(report.get("candidate_id", "")).startswith("v02_") and mode == "fixed_z":
            rows[3] = ("阶段速度 / 确认", v02_timing_annotation(report, str(stage)))
        gates = [("最终图完成", graph), ("最终任务结果", task_ok),
                 ("最终数值安全", safety), ("最终严格成功", strict)]
        failure_label = "失败状态"
    else:
        title = f"{task} | {'STRICT SUCCESS' if strict else 'STRICT GATE NOT PASSED'}"
        subtitle = "Sealed final-result markers · current progress is stage/time · policy not re-executed"
        rows = [
            ("Stage", f"{int(trace['stage_index'][index])} · {_short(stage, 27)}"),
            ("Control", controller),
            ("Roles", f"{_short(trace['arm'][index], 8)} · {_short(trace['object_role'][index], 12)} → {_short(trace['target_role'][index], 12)}"),
            ("Online state", f"z {z[0]:+.3f}/{z[1]:+.3f}/{z[2]:+.3f} | Δ {dz:.4f}"),
            (holding_label, holding_value),
            ("Physical time", f"{current_time:.2f} / {duration:.2f} s"),
        ]
        gates = [("Final graph", graph), ("Final task", task_ok),
                 ("Final safety", safety), ("Final strict", strict)]
        failure_label = "Failure state"

    draw.rectangle((0, 360, 960, 540), fill="#101722")
    draw.rectangle((640, 0, 960, 540), fill="#162131")
    draw.text((18, 374), title, font=title_font, fill="#f3f7fc")
    draw.text((18, 414), subtitle, font=small_font, fill="#94a8bd")
    x0, x1 = 18, 622
    progress = 0. if count == 1 else index / (count - 1)
    draw.rounded_rectangle((x0, 464, x1, 478), radius=7, fill="#29394b")
    draw.rounded_rectangle((x0, 464, x0 + int((x1 - x0) * progress), 478), radius=7, fill="#42c7a5")
    draw.text((18, 492), f"{index + 1}/{count}", font=small_font, fill="#c8d4df")

    draw.text((660, 18), _short(chinese_stage(str(stage)) if language == "zh" else stage, 25), font=title_font, fill="#ffffff")
    y = 67
    for label, value in rows:
        draw.text((660, y), label, font=small_font, fill="#7f9bb5")
        draw.text((660, y + 20), _short(value, 31), font=body_font, fill="#e7eef5")
        y += 52
    gate_y = 382
    for gate_index, (label, passed) in enumerate(gates):
        gx = 660 + (gate_index % 2) * 145
        gy = gate_y + (gate_index // 2) * 40
        color = "#38c995" if passed else "#e26d75"
        draw.ellipse((gx, gy + 5, gx + 12, gy + 17), fill=color)
        draw.text((gx + 18, gy), label, font=small_font, fill="#dbe5ee")
    draw.text((660, 468), failure_label, font=small_font, fill="#7f9bb5")
    draw.text((660, 493), _short(failure or ("无" if language == "zh" else "none"), 31),
              font=body_font, fill="#e26d75" if failure else "#a7bacb")
    return canvas


def render(
    report_path: Path,
    scene_path: Path,
    output: Path,
    *,
    trace_path: Path | None = None,
    camera: str = "auto",
    fps: int = 10,
    language: str = "zh",
    font: Path | None = None,
    physics_source: Path | None = None,
    candidate_freeze: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if fps <= 0 or fps > 60:
        raise ValueError("fps must be in 1..60")
    report_path, scene_path, output = report_path.resolve(), scene_path.resolve(), output.resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must use the .mp4 suffix")
    sidecar = output.with_suffix(".render.json")
    if not force and (output.exists() or sidecar.exists()):
        raise FileExistsError(f"output already exists: {output} or {sidecar}")
    report, trace = load_inputs(report_path, trace_path, scene_path)
    condition_record = condition_binding(report, report_path)
    if (physics_source is None) != (candidate_freeze is None):
        raise ValueError("--physics-source and --candidate-freeze must be provided together")
    frozen_world = None
    if physics_source is not None and candidate_freeze is not None:
        frozen_world, model, data, initialization = frozen_initial_state(
            report, report_path, scene_path, physics_source, candidate_freeze)
        fixed_translations = initialization["requested_world_translations"]
    else:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        fixed_translations = apply_fixed_initial_translation(model, report["condition"])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        initialization = {
            "mode": "legacy_fixed_body_translation",
            "requested_world_translations": _condition_translations(report["condition"]),
            "fixed_initial_translation_applied": fixed_translations,
            "initial_body_transforms_sha256": _initial_transform_sha256(model, data),
            "physical_time_sec": float(data.time),
            "mj_step_executed": False,
        }
    try:
        if trace["qpos"].shape[1] != model.nq:
            raise ValueError(
                f"trace qpos width {trace['qpos'].shape[1]} != scene nq {model.nq}")
        selected_camera = _camera_name(model, camera)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required")
    except BaseException:
        if frozen_world is not None:
            frozen_world.close()
        raise
    command = [
        ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", "960x540", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    fonts = (_font(font, 24), _font(font, 16), _font(font, 15))
    mode = _adapter_mode(report, report_path)
    renderer = mujoco.Renderer(model, width=640, height=360)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for index, qpos in enumerate(trace["qpos"]):
            # Visual reconstruction only. This state is never returned to a
            # controller and does not continue through mj_step.
            data.qpos[:] = qpos
            if "ctrl" in trace:
                ctrl = np.asarray(trace["ctrl"][index], dtype=np.float64)
                if ctrl.shape == (model.nu,) and np.isfinite(ctrl).all():
                    data.ctrl[:] = ctrl
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=selected_camera)
            rgb = renderer.render().copy()
            frame = _draw_frame(rgb, report=report, trace=trace, index=index,
                                mode=mode, language=language, fonts=fonts)
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        process.stdin.close()
        code = process.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited with status {code}")
        os.replace(temporary, output)
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()
        if frozen_world is not None:
            frozen_world.close()

    trace_actual = sha256(Path(str(trace["_path"])))
    record = {
        "schema": VIDEO_SCHEMA,
        "offline_visual_replay_only": True,
        "policy_or_adapter_executed": False,
        "report": {"path": str(report_path), "sha256": sha256(report_path)},
        "visual_trace": {"path": str(trace["_path"]), "sha256": trace_actual},
        "scene": {"path": str(scene_path), "sha256": sha256(scene_path)},
        "condition": condition_record,
        "initialization": initialization,
        "render": {
            "camera": selected_camera, "fps": fps, "frames": len(trace["timestamps"]),
            "physical_start_sec": float(trace["timestamps"][0]),
            "physical_end_sec": float(trace["timestamps"][-1]),
            "language": language, "resolution": [960, 540],
            "codec": "H.264", "pixel_format": "yuv420p", "faststart": True,
            "fixed_initial_translation_applied": fixed_translations,
        },
        "video": {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size},
    }
    sidecar.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="sealed graph-self-adapter.twin-run.v1 report")
    parser.add_argument("--scene", required=True, type=Path, help="read-only scene.xml bound by the report")
    parser.add_argument("--trace", type=Path, help="override bound trace location; its sealed hash must still match")
    parser.add_argument("--output", required=True, type=Path, help="destination .mp4")
    parser.add_argument("--camera", default="auto", help="auto, main, overview, or another named scene camera")
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--physics-source", type=Path,
                        help="frozen twin_physics.py used by the sealed candidate")
    parser.add_argument("--candidate-freeze", type=Path,
                        help="CANDIDATE_FREEZE.json whose hash is sealed by the report")
    parser.add_argument("--force", action="store_true", help="replace an existing video and render sidecar")
    args = parser.parse_args()
    record = render(args.report, args.scene, args.output, trace_path=args.trace,
                    camera=args.camera, fps=args.fps, language=args.language,
                    font=args.font, physics_source=args.physics_source,
                    candidate_freeze=args.candidate_freeze, force=args.force)
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
