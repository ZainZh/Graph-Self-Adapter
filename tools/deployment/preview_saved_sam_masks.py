#!/usr/bin/env python3
"""Preview already-saved SAM masks over their RGB frame; runs no inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Mapping, Sequence
import uuid

import cv2
import numpy as np

try:
    from tools.deployment.manual_sam_first_frame import configure_local_desktop_environment
except ModuleNotFoundError:
    from manual_sam_first_frame import configure_local_desktop_environment


COLORS_BGR = {
    "white_mug": (255, 210, 50),
    "red_mug": (40, 50, 255),
    "box": (40, 220, 70),
    "white_handle": (255, 60, 220),
    "table_patch": (60, 220, 255),
}


def load_saved_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def overlay_saved_masks(
    image: np.ndarray,
    masks: Mapping[str, np.ndarray],
    *,
    alpha: float = 0.42,
) -> np.ndarray:
    """Return a BGR overlay that depicts mask pixels, not prompt boxes."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB preview image must have shape HxWx3")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")
    height, width = image.shape[:2]
    result = image.copy()
    for index, (role, raw_mask) in enumerate(masks.items()):
        mask = np.squeeze(raw_mask)
        if mask.shape != (height, width):
            raise ValueError(
                f"mask {role} has shape {mask.shape}, expected {(height, width)}"
            )
        selected = mask.astype(bool)
        color = np.asarray(
            COLORS_BGR.get(
                role,
                ((53 * index + 80) % 256, (97 * index + 120) % 256, (173 * index + 60) % 256),
            ),
            dtype=np.float32,
        )
        if selected.any():
            pixels = result[selected].astype(np.float32)
            result[selected] = np.clip(
                pixels * (1.0 - alpha) + color * alpha, 0, 255
            ).astype(np.uint8)
            contours, _ = cv2.findContours(
                selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, tuple(int(v) for v in color), 2)
    return result


def render_saved_mask_preview(rgb_path: Path, masks_path: Path) -> np.ndarray:
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode RGB frame: {rgb_path}")
    masks = load_saved_masks(masks_path)
    if not masks:
        raise ValueError(f"saved mask archive is empty: {masks_path}")
    return overlay_saved_masks(image, masks)


def atomic_write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary), image):
            raise OSError(f"failed to encode preview: {temporary}")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def show_preview(image: np.ndarray) -> None:
    configure_local_desktop_environment()
    title = "Saved-SAM-Mask-Preview"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        cv2.imshow(title, image)
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (10, 13, 27, ord("q"), ord("Q")):
                return
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return
    finally:
        cv2.destroyWindow(title)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgb", type=Path, help="与 masks 对应的已保存 RGB")
    parser.add_argument("masks", type=Path, help="运行时导出的 MASKS_*.npz")
    parser.add_argument("--output", type=Path, help="可选的叠加预览 PNG")
    parser.add_argument("--show", action="store_true", help="在本机桌面显示，Enter/Esc 关闭")
    args = parser.parse_args(argv)
    preview = render_saved_mask_preview(args.rgb, args.masks)
    if args.output:
        atomic_write_png(args.output, preview)
        print(args.output)
    if args.show:
        show_preview(preview)
    if not args.output and not args.show:
        parser.error("至少提供 --output 或 --show")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
