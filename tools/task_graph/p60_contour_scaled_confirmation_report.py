#!/usr/bin/env python3
"""Post-seal P60-B2 confirmation audit, statistics, and declared videos.

This tool never imports or constructs a physics environment.  It refuses a
partial 64-cell cohort, verifies every seal, recomputes P50 commands from the
sealed estimate/proprio/controller state, and renders only the first declared
condition of each split from the already recorded image frames.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prove.fast_adapter.geometry3d_adapter import pose
from prove.task_graph.precision_action_policy import PrecisionActionController, load_frozen_head
from tools.task_graph.p50_precision_pilot import controller_state, shadow_command
from tools.task_graph.p60_contour_scaled_confirmation import SCHEMA, cell_name, cells

ROOT = Path(__file__).resolve().parents[2]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
PRESENTATION_SCHEMA = "prove.p60-contour-scaled-independent-confirmation-presentation.v1"
AUDIT_SCHEMA = "prove.p60-contour-scaled-independent-confirmation-sealed-action-audit.v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n")


def _max_difference(actual, expected):
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    if isinstance(actual, (float, int, np.floating, np.integer)) and isinstance(
            expected, (float, int, np.floating, np.integer)):
        return abs(float(actual) - float(expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Controller-state keys changed")
        return max((_max_difference(actual[key], expected[key]) for key in actual), default=0.0)
    if actual != expected:
        raise ValueError(f"Non-numeric sealed value changed: {actual!r} != {expected!r}")
    return 0.0


def load_complete(freeze_path):
    frozen = json.loads(freeze_path.read_text())
    if frozen.get("schema") != SCHEMA or frozen.get("cells") != cells():
        raise ValueError("Expected the exact frozen 64-cell P60-B2 confirmation cohort")
    records = []
    freeze_sha = sha(freeze_path)
    for index, cell in enumerate(frozen["cells"]):
        directory = freeze_path.parent / "cells" / cell_name(index, cell)
        if not (directory / "RESULT.json").is_file() or not (directory / "SEAL.json").is_file():
            raise ValueError(f"All 64 sealed P60-B2 confirmation cells required: {directory.name}")
        result = json.loads((directory / "RESULT.json").read_text())
        seal = json.loads((directory / "SEAL.json").read_text())
        if (result.get("cell") != cell or result.get("cell_index") != index
                or result.get("seal_sha256") != sha(directory / "SEAL.json")
                or seal.get("freeze", {}).get("sha256") != freeze_sha
                or seal.get("neural_parameters_before") != seal.get("neural_parameters_after")):
            raise ValueError(f"Result/seal mismatch: {directory.name}")
        for name, digest in seal["files"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"Sealed file changed: {directory / name}")
        records.append({"cell": cell, "result": result, "directory": str(directory),
                        "result_sha256": sha(directory / "RESULT.json"),
                        "seal_sha256": sha(directory / "SEAL.json")})
    return frozen, records


def audit_actions(frozen, records, freeze_path):
    head_path = Path(frozen["files"]["action_head"]["path"])
    if not head_path.is_absolute():
        head_path = ROOT / head_path
    if sha(head_path) != frozen["files"]["action_head"]["sha256"]:
        raise ValueError("Frozen P50 action head changed")
    head = load_frozen_head(head_path)
    rows = []
    maxima = dict(command_position=0.0, command_rotation_matrix=0.0,
                  controller_state=0.0, shadow_position=0.0,
                  shadow_rotation_matrix=0.0)
    for record in records:
        directory = Path(record["directory"])
        trace = np.load(directory / "TRACE.npy", allow_pickle=True).tolist()
        if len(trace) != 240:
            raise ValueError("P60-B2 action audit requires every one of 240 sealed actions")
        first = trace[0]["controller_before"]
        controller = PrecisionActionController(
            copy.deepcopy(head), np.asarray(first["start_pos"]),
            np.asarray(first["start_R"]), length=float(first["length"]))
        neural_steps = shadow_steps = 0
        for t, item in enumerate(trace):
            maxima["controller_state"] = max(
                maxima["controller_state"],
                _max_difference(controller_state(controller), item["controller_before"]))
            update = item.get("adaptation")
            if update is not None:
                shadow = shadow_command(controller, t,
                                        pose(np.asarray(update["pre_update_points"])),
                                        item["proprio"])
                if "shadow_command_position" in update:
                    maxima["shadow_position"] = max(
                        maxima["shadow_position"],
                        float(np.max(np.abs(shadow[0] - update["shadow_command_position"]))))
                    maxima["shadow_rotation_matrix"] = max(
                        maxima["shadow_rotation_matrix"],
                        float(np.max(np.abs(shadow[1] - update["shadow_command_rotation"]))))
                    if (bool(shadow[2]) != bool(item["closed"])) != bool(
                            update["same_state_closed_changed"]):
                        raise ValueError("Stored shadow gripper comparison changed")
                    if (shadow[3] != item["stage"]) != bool(update["same_state_stage_changed"]):
                        raise ValueError("Stored shadow stage comparison changed")
                    shadow_steps += 1
            command = controller.command(t, item["estimate"], item["proprio"])
            maxima["command_position"] = max(
                maxima["command_position"],
                float(np.max(np.abs(command[0] - item["command_p"]))))
            maxima["command_rotation_matrix"] = max(
                maxima["command_rotation_matrix"],
                float(np.max(np.abs(command[1] - item["command_R"]))))
            if bool(command[2]) != bool(item["closed"]) or command[3] != item["stage"]:
                raise ValueError(f"P50 discrete action mismatch: {directory.name} step {t}")
            maxima["controller_state"] = max(
                maxima["controller_state"],
                _max_difference(controller_state(controller), item["controller_after"]))
            neural_steps += int(item.get("neural_action") is not None)
        rows.append({"cell": record["cell"], "trace_sha256": sha(directory / "TRACE.npy"),
                     "actions_recomputed": len(trace), "neural_actions": neural_steps,
                     "shadow_actions_recomputed": shadow_steps})
    if max(maxima.values()) > 1e-6:
        raise ValueError(f"Sealed P50 action replay mismatch: {maxima}")
    return {"schema": AUDIT_SCHEMA, "freeze_sha256": sha(freeze_path),
            "cells": len(rows), "actions_recomputed": sum(r["actions_recomputed"] for r in rows),
            "all_seals_verified": True, "no_physics_replay": True,
            "maximum_errors": maxima, "rows": rows}


def paired_counts(records):
    def wilson(successes, total, z=1.959963984540054):
        p = successes / total
        denominator = 1.0 + z * z / total
        center = (p + z * z / (2.0 * total)) / denominator
        radius = z * ((p * (1.0 - p) / total + z * z / (4.0 * total * total)) ** .5) / denominator
        return [center - radius, center + radius]

    groups, pairs = [], []
    for split in ("id", "geometry"):
        selected = [row for row in records if row["cell"]["split"] == split]
        by_arm = {}
        for arm in ("contour_scaled_fixed", "contour_scaled_continuous"):
            arm_rows = [row for row in selected if row["cell"]["arm"] == arm]
            if len(arm_rows) != 16:
                raise ValueError("Sixteen sealed conditions required per split and arm")
            by_arm[arm] = {row["cell"]["condition_seed"]:
                           bool(row["result"]["validated_success"]) for row in arm_rows}
            success = sum(by_arm[arm].values())
            groups.append({"split": split, "arm": arm, "n": 16,
                           "validated_success": success,
                           "success_rate": success / 16.0,
                           "wilson_95": wilson(success, 16),
                           "mechanical_anomalies": sum(not row["result"]["mechanical_valid"]
                                                       for row in arm_rows)})
        fixed, online = by_arm["contour_scaled_fixed"], by_arm["contour_scaled_continuous"]
        if set(fixed) != set(online):
            raise ValueError("P60-B2 paired seeds differ")
        pairs.append({"split": split, "n": 16,
                      "rescued": sum(not fixed[key] and online[key] for key in fixed),
                      "harmed": sum(fixed[key] and not online[key] for key in fixed),
                      "both_success": sum(fixed[key] and online[key] for key in fixed),
                      "both_failure": sum(not fixed[key] and not online[key] for key in fixed)})
    return {"groups": groups, "paired": pairs}


def _fit(canvas, rgb, box):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    image.thumbnail((box[2], box[3]), Image.Resampling.LANCZOS)
    canvas.paste(image, (box[0] + (box[2] - image.width) // 2,
                         box[1] + (box[3] - image.height) // 2))


def _phase(stage):
    return {"shared observation motion": "共同公开观察",
            "neural align": "冻结小策略：对齐",
            "neural approach": "冻结小策略：接近槽口",
            "neural insert": "冻结小策略：插入",
            "neural hold": "冻结小策略：保持"}.get(stage, str(stage))


def render_pair(fixed, online, output):
    loaded = []
    for record in (fixed, online):
        directory = Path(record["directory"])
        with np.load(directory / "FRAMES.npz") as package:
            frames = package["frames"].copy()
        with np.load(directory / "DETAIL_FRAMES.npz") as package:
            detail = package["frames"].copy()
        trace = np.load(directory / "TRACE.npy", allow_pickle=True).tolist()
        if len(frames) != 241 or len(detail) != 241 or len(trace) != 240:
            raise ValueError("Prespecified P60-B2 media must contain all 240 actions plus terminal")
        loaded.append((frames, detail, trace))
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 18)
    process = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", "1440x950", "-r", "20", "-i", "-",
         "-an", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for t in range(241):
            canvas = Image.new("RGB", (1440, 950), "#111924")
            draw = ImageDraw.Draw(canvas)
            split = "ID" if fixed["cell"]["split"] == "id" else "几何 OOD"
            seed = fixed["cell"]["condition_seed"]
            draw.text((18, 12), f"P60-B2 独立确认 · {split} · 条件 {seed} · 已持物 Panda 局部插入",
                      font=font, fill="white")
            for column, (record, arrays) in enumerate(zip((fixed, online), loaded)):
                x = column * 720
                frames, detail, trace = arrays
                label = "曲率归一化 fixed：z 不更新" if column == 0 else "曲率归一化 continuous：每次 3 步更新"
                draw.text((x + 16, 52), label, font=font, fill="#93cff0")
                _fit(canvas, frames[t], (x + 10, 90, 700, 300))
                _fit(canvas, detail[t], (x + 10, 430, 700, 300))
                draw.text((x + 16, 395), "上：封存全局＋工具视角；下：封存细节视角",
                          font=small, fill="white")
                row = trace[min(t, 239)]
                if t == 240:
                    phase = "终态（最后动作之后）"
                    status = ("通过严格物理成功判据" if record["result"]["validated_success"]
                              else "未通过；失败仍计入分母")
                else:
                    phase = f"阶段：{_phase(row['stage'])}"
                    update = row.get("adaptation")
                    status = ("本帧轮廓未获准" if update is None else
                              f"支持观测 {update['S']} · latent 更新步 {update['U']} · "
                              f"|z-z0|={update['z_from_initial']:.4f}")
                draw.text((x + 16, 748), phase, font=font, fill="white")
                draw.text((x + 16, 791), status, font=font, fill="#ffce96")
                draw.text((x + 16, 835),
                          (f"动作前帧 {t}/240；P45网络与P50动作头始终冻结" if t < 240
                           else "结果来自封存轨迹评价；视频不参与选择或判定"),
                          font=small, fill="white")
            draw.text((18, 900), "预先指定每个分布首个条件；仅使用封存真实帧，不按结果选视频。",
                      font=font, fill="#b7c5d5")
            process.stdin.write(canvas.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode()
        if process.wait():
            raise RuntimeError(error)
    except BaseException:
        process.kill()
        process.wait()
        raise
    manifest = {"schema": "prove.p60-contour-scaled-independent-confirmation-paired-video.v1",
                "video_sha256": sha(output), "frame_count": 241, "fps": 20,
                "display_fps_not_physics_timing": True,
                "selection": "prespecified first condition of split, independent of outcome",
                "post_seal_derivative": True, "no_synthetic_frames": True,
                "sources": [{"cell": row["cell"], "result_sha256": row["result_sha256"],
                             "seal_sha256": row["seal_sha256"],
                             "files": {name: sha(Path(row["directory"]) / name)
                                       for name in ("FRAMES.npz", "DETAIL_FRAMES.npz", "TRACE.npy")}}
                            for row in (fixed, online)]}
    dump(output.with_suffix(".manifest.json"), manifest)
    return manifest


def build(freeze_path, output):
    frozen, records = load_complete(freeze_path)
    output.mkdir(parents=True, exist_ok=False)
    audit = audit_actions(frozen, records, freeze_path)
    dump(output / "ACTION_AUDIT.json", audit)
    summary = paired_counts(records)
    summary.update({"schema": PRESENTATION_SCHEMA, "freeze_sha256": sha(freeze_path),
                    "episodes": 64, "development_only": False,
                    "fresh_test_claim": True,
                    "independent_success_rate_claim": True,
                    "action_audit_sha256": sha(output / "ACTION_AUDIT.json"),
                    "records": records})
    dump(output / "SUMMARY.json", summary)
    for split, seed in (("id", 609100), ("geometry", 609200)):
        pair = {row["cell"]["arm"]: row for row in records
                if row["cell"]["split"] == split and row["cell"]["condition_seed"] == seed}
        render_pair(pair["contour_scaled_fixed"], pair["contour_scaled_continuous"],
                    output / f"{split}_{seed}_contour_scaled_pair_zh.mp4")
    dump(output / "MANIFEST.json", {
        "schema": "prove.p60-contour-scaled-independent-confirmation-presentation-manifest.v1",
        "freeze_sha256": sha(freeze_path), "presenter_sha256": sha(__file__),
        "files": {path.name: sha(path) for path in output.iterdir() if path.is_file()}})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(args.freeze.resolve(), args.output.resolve())
    print(json.dumps({"groups": value["groups"], "paired": value["paired"]},
                     ensure_ascii=False, indent=2))
