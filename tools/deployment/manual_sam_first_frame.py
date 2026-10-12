#!/usr/bin/env python3
"""Operator GUI for SAM prompts on one frozen INITIAL_RGB.png.

This module only records human box/point prompts.  It does not open a camera,
run SAM, or send robot commands.  A successful confirmation atomically writes
an INITIAL_BOXES.json bound to the exact PNG bytes by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import uuid


BOX_STEPS = (
    ("white_mug", "白杯（完整物体）"),
    ("red_mug", "红杯（完整物体）"),
    ("box", "盒子（放置目标）"),
    ("white_handle", "白杯抓持小区域（把手/可抓部位）"),
    ("table_patch", "白杯旁的桌面小区域"),
)
POINT_STEP = ("white_handle", "在白杯可抓部位内部点一个前景点")
BOX_ROLES = tuple(name for name, _ in BOX_STEPS)


class AnnotationCancelled(RuntimeError):
    """The operator explicitly cancelled or closed the annotation window."""


class FrameChangedError(RuntimeError):
    """The frame bytes changed between annotation and confirmation."""


def configure_local_desktop_environment(
    environ: dict[str, str] | None = None,
    *,
    x_socket_dir: Path = Path("/tmp/.X11-unix"),
) -> dict[str, str]:
    """Fill DISPLAY/XAUTHORITY for a launcher started outside the GUI shell."""
    target = os.environ if environ is None else environ
    if not target.get("DISPLAY") and x_socket_dir.is_dir():
        sockets = sorted(
            (path for path in x_socket_dir.iterdir() if path.name.startswith("X")),
            key=lambda path: path.name,
        )
        if sockets:
            target["DISPLAY"] = f":{sockets[-1].name[1:]}"
    if not target.get("XAUTHORITY"):
        candidate = Path(f"/run/user/{os.getuid()}/gdm/Xauthority")
        if candidate.is_file():
            target["XAUTHORITY"] = str(candidate)
    return target


def frame_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_annotations(
    boxes: Mapping[str, Sequence[object]],
    points: Mapping[str, Sequence[object]],
    *,
    width: int,
    height: int,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    if set(boxes) != set(BOX_ROLES):
        missing = sorted(set(BOX_ROLES) - set(boxes))
        extra = sorted(set(boxes) - set(BOX_ROLES))
        raise ValueError(f"标注框不完整；缺少={missing}，多余={extra}")
    if set(points) != {POINT_STEP[0]}:
        raise ValueError("必须且只能提供一个 white_handle 前景点")

    clean_boxes: dict[str, list[int]] = {}
    for role in BOX_ROLES:
        box = boxes[role]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"{role} 必须是 [x0,y0,x1,y1]")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box):
            raise ValueError(f"{role} 坐标必须是数字")
        values = [int(round(float(v))) for v in box]
        x0, y0, x1, y1 = values
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"{role} 框为空或超出图像")
        clean_boxes[role] = values

    point = points[POINT_STEP[0]]
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError("white_handle 前景点必须是 [x,y]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in point):
        raise ValueError("white_handle 前景点坐标必须是数字")
    x, y = (int(round(float(v))) for v in point)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("white_handle 前景点超出图像")
    hx0, hy0, hx1, hy1 = clean_boxes["white_handle"]
    if not (hx0 <= x < hx1 and hy0 <= y < hy1):
        raise ValueError("前景点必须位于 white_handle 抓持小框内")
    return clean_boxes, {POINT_STEP[0]: [x, y]}


def build_payload(
    frame_path: Path,
    boxes: Mapping[str, Sequence[object]],
    points: Mapping[str, Sequence[object]],
    *,
    expected_sha256: str,
) -> dict[str, object]:
    import cv2

    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码首帧：{frame_path}")
    height, width = image.shape[:2]
    clean_boxes, clean_points = validate_annotations(
        boxes, points, width=width, height=height
    )
    actual_sha = frame_sha256(frame_path)
    if actual_sha != expected_sha256:
        raise FrameChangedError("标注期间 INITIAL_RGB.png 已变化，请对新画面重新标注")
    return {
        "frame_sha256": actual_sha,
        "role_boxes": clean_boxes,
        "role_points": clean_points,
    }


def atomic_write_json(destination: Path, payload: Mapping[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class _View:
    scale: float
    image_x: int
    image_y: int
    image_width: int
    image_height: int

    def to_image(self, x: int, y: int) -> tuple[int, int] | None:
        ix = int((x - self.image_x) / self.scale)
        iy = int((y - self.image_y) / self.scale)
        if 0 <= ix < int(self.image_width / self.scale) and 0 <= iy < int(
            self.image_height / self.scale
        ):
            return ix, iy
        return None


class ManualPromptGUI:
    """Small OpenCV desktop UI; imported lazily so validation stays headless."""

    COLORS = {
        "white_mug": (255, 220, 80),
        "red_mug": (60, 80, 255),
        "box": (80, 220, 80),
        "white_handle": (255, 80, 220),
        "table_patch": (80, 220, 255),
    }

    def __init__(self, image, *, window_name: str = "Manual-SAM-Grounding"):
        import cv2

        self.cv2 = cv2
        self.image = image
        self.window_name = window_name
        self.boxes: dict[str, list[int]] = {}
        self.points: dict[str, list[int]] = {}
        self.step = 0
        self.drag_start: tuple[int, int] | None = None
        self.drag_now: tuple[int, int] | None = None
        self.cancelled = False
        self.confirmed = False
        self.status = "请按顺序完成 5 个框和 1 个前景点"
        self.view: _View | None = None
        self.buttons: dict[str, tuple[int, int, int, int]] = {}

    @property
    def complete(self) -> bool:
        return set(self.boxes) == set(BOX_ROLES) and set(self.points) == {
            POINT_STEP[0]
        }

    def _current(self) -> tuple[str, str, str] | None:
        if self.step < len(BOX_STEPS):
            role, label = BOX_STEPS[self.step]
            return "box", role, label
        if self.step == len(BOX_STEPS):
            role, label = POINT_STEP
            return "point", role, label
        return None

    def undo(self) -> None:
        if self.step > len(BOX_STEPS):
            self.points.pop(POINT_STEP[0], None)
            self.step = len(BOX_STEPS)
        elif self.step == len(BOX_STEPS) and POINT_STEP[0] in self.points:
            self.points.pop(POINT_STEP[0], None)
        elif self.step > 0:
            self.step -= 1
            self.boxes.pop(BOX_STEPS[self.step][0], None)
        else:
            self.boxes.pop(BOX_STEPS[0][0], None)
        self.status = "已退回，请重新选择当前步骤"

    def _mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        del flags
        if event == self.cv2.EVENT_LBUTTONDOWN:
            for action, (x0, y0, x1, y1) in self.buttons.items():
                if x0 <= x <= x1 and y0 <= y <= y1:
                    if action == "undo":
                        self.undo()
                    elif action == "cancel":
                        self.cancelled = True
                    elif action == "confirm" and self.complete:
                        self.confirmed = True
                    return
            current = self._current()
            point = self.view.to_image(x, y) if self.view else None
            if current is None or point is None:
                return
            if current[0] == "point":
                hx0, hy0, hx1, hy1 = self.boxes["white_handle"]
                if not (hx0 <= point[0] < hx1 and hy0 <= point[1] < hy1):
                    self.status = "前景点必须点在紫色的白杯抓持小框内"
                    return
                self.points[current[1]] = [point[0], point[1]]
                self.step += 1
                self.status = "标注完整；请检查后点“确定保存”"
            else:
                self.drag_start = point
                self.drag_now = point
        elif event == self.cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            point = self.view.to_image(x, y) if self.view else None
            if point is not None:
                self.drag_now = point
        elif event == self.cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            point = self.view.to_image(x, y) if self.view else None
            start = self.drag_start
            self.drag_start = self.drag_now = None
            current = self._current()
            if point is None or current is None or current[0] != "box":
                return
            x0, x1 = sorted((start[0], point[0]))
            y0, y1 = sorted((start[1], point[1]))
            if x1 - x0 < 2 or y1 - y0 < 2:
                self.status = "框太小，请重新拖动"
                return
            self.boxes[current[1]] = [x0, y0, x1 + 1, y1 + 1]
            self.step += 1
            self.status = "已记录；继续下一步"

    def _draw_text(self, canvas, text: str, origin, *, scale=0.62, color=(245, 245, 245)):
        try:
            from PIL import Image, ImageDraw, ImageFont

            font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            font = ImageFont.truetype(font_path, max(13, int(28 * scale)))
            rgb = self.cv2.cvtColor(canvas, self.cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(pil_image)
            draw.text(origin, text, font=font, fill=tuple(reversed(color)))
            canvas[:] = self.cv2.cvtColor(
                __import__("numpy").asarray(pil_image), self.cv2.COLOR_RGB2BGR
            )
        except (ImportError, OSError):
            fallback = text if text.isascii() else "SAM prompt"
            self.cv2.putText(
                canvas,
                fallback,
                origin,
                self.cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                1,
                self.cv2.LINE_AA,
            )

    def render(self):
        import numpy as np

        height, width = self.image.shape[:2]
        max_width, max_height = 1180, 760
        scale = min(max_width / width, (max_height - 150) / height, 1.5)
        shown_w, shown_h = max(1, int(width * scale)), max(1, int(height * scale))
        canvas_w = max(720, shown_w)
        image_x = (canvas_w - shown_w) // 2
        canvas = np.zeros((shown_h + 150, canvas_w, 3), dtype=np.uint8)
        shown = self.cv2.resize(self.image, (shown_w, shown_h))
        canvas[72 : 72 + shown_h, image_x : image_x + shown_w] = shown
        self.view = _View(scale, image_x, 72, shown_w, shown_h)

        current = self._current()
        step_text = (
            f"步骤 {min(self.step + 1, 6)}/6：{current[2]}"
            if current
            else "全部标注完成：请检查彩色框和红色前景点"
        )
        self._draw_text(canvas, step_text, (12, 28), scale=0.7)
        self._draw_text(canvas, self.status, (12, 56), scale=0.5)

        for role, box in self.boxes.items():
            color = self.COLORS[role]
            x0, y0, x1, y1 = box
            p0 = (image_x + int(x0 * scale), 72 + int(y0 * scale))
            p1 = (image_x + int(x1 * scale), 72 + int(y1 * scale))
            self.cv2.rectangle(canvas, p0, p1, color, 2)
            self._draw_text(canvas, role, (p0[0] + 3, max(86, p0[1] + 18)), scale=0.5, color=color)
        if POINT_STEP[0] in self.points:
            x, y = self.points[POINT_STEP[0]]
            center = (image_x + int(x * scale), 72 + int(y * scale))
            self.cv2.drawMarker(canvas, center, (20, 20, 255), self.cv2.MARKER_CROSS, 18, 3)
        if self.drag_start and self.drag_now:
            a, b = self.drag_start, self.drag_now
            self.cv2.rectangle(
                canvas,
                (image_x + int(a[0] * scale), 72 + int(a[1] * scale)),
                (image_x + int(b[0] * scale), 72 + int(b[1] * scale)),
                (255, 255, 255),
                2,
            )

        top = 82 + shown_h
        button_w = max(120, canvas_w // 4)
        specs = (
            ("undo", 10, "重选上一步 [R]"),
            ("cancel", (canvas_w - button_w) // 2, "取消 [Esc]"),
            ("confirm", canvas_w - button_w - 10, "确定保存 [Enter]"),
        )
        self.buttons = {}
        for action, left, label in specs:
            rect = (left, top, left + button_w, top + 46)
            self.buttons[action] = rect
            enabled = action != "confirm" or self.complete
            color = (50, 130, 50) if action == "confirm" and enabled else (75, 75, 75)
            if action == "cancel":
                color = (70, 70, 150)
            self.cv2.rectangle(canvas, rect[:2], rect[2:], color, -1)
            self._draw_text(canvas, label, (left + 8, top + 29), scale=0.55)
        return canvas

    def run(self) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        window_created = False
        try:
            self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_AUTOSIZE)
            window_created = True
            self.cv2.imshow(self.window_name, self.render())
            self.cv2.waitKey(1)
            self.cv2.setMouseCallback(self.window_name, self._mouse)
            while not self.cancelled and not self.confirmed:
                self.cv2.imshow(self.window_name, self.render())
                key = self.cv2.waitKey(20) & 0xFF
                if key in (27, ord("c"), ord("C")):
                    self.cancelled = True
                elif key in (8, 127, ord("r"), ord("R")):
                    self.undo()
                elif key in (10, 13) and self.complete:
                    self.confirmed = True
                try:
                    visible = self.cv2.getWindowProperty(
                        self.window_name, self.cv2.WND_PROP_VISIBLE
                    )
                    if visible < 1:
                        self.cancelled = True
                except self.cv2.error:
                    self.cancelled = True
            if self.cancelled:
                raise AnnotationCancelled("用户取消了标注；未生成 INITIAL_BOXES.json")
            return dict(self.boxes), dict(self.points)
        finally:
            if window_created:
                self.cv2.destroyWindow(self.window_name)


def annotate(
    frame_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    import cv2

    configure_local_desktop_environment()
    frame_path = frame_path.resolve()
    output_path = (output_path or frame_path.with_name("INITIAL_BOXES.json")).resolve()
    initial_sha = frame_sha256(frame_path)
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码首帧：{frame_path}")
    boxes, points = ManualPromptGUI(image).run()
    payload = build_payload(
        frame_path, boxes, points, expected_sha256=initial_sha
    )
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path, help="实际首帧 INITIAL_RGB.png")
    parser.add_argument("--output", type=Path, help="默认同目录 INITIAL_BOXES.json")
    args = parser.parse_args(argv)
    try:
        payload = annotate(args.frame, args.output)
    except AnnotationCancelled as exc:
        print(str(exc))
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
