import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import pytest

from tools.deployment.manual_sam_first_frame import (
    AnnotationCancelled,
    FrameChangedError,
    ManualPromptGUI,
    annotate,
    atomic_write_json,
    build_payload,
    configure_local_desktop_environment,
    validate_annotations,
)
from tools.deployment.wait_for_manual_sam import wait_for_frame
from tools.deployment.preview_saved_sam_masks import (
    atomic_write_png,
    overlay_saved_masks,
    render_saved_mask_preview,
)


BOXES = {
    "white_mug": [10, 10, 50, 60],
    "red_mug": [60, 10, 100, 60],
    "box": [110, 10, 160, 70],
    "white_handle": [35, 25, 48, 45],
    "table_patch": [15, 65, 55, 85],
}
POINTS = {"white_handle": [40, 30]}


def _png(path: Path, value: int = 20) -> bytes:
    image = np.full((100, 180, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path.read_bytes()


def test_complete_geometry_and_foreground_seed_are_required():
    boxes, points = validate_annotations(BOXES, POINTS, width=180, height=100)
    assert boxes["table_patch"] == [15, 65, 55, 85]
    assert points == POINTS
    with pytest.raises(ValueError, match="缺少"):
        validate_annotations(
            {k: v for k, v in BOXES.items() if k != "table_patch"},
            POINTS,
            width=180,
            height=100,
        )
    with pytest.raises(ValueError, match="抓持小框内"):
        validate_annotations(BOXES, {"white_handle": [5, 5]}, width=180, height=100)


def test_payload_is_bound_to_exact_frame_and_rejects_changed_frame(tmp_path):
    frame = tmp_path / "INITIAL_RGB.png"
    raw = _png(frame)
    digest = hashlib.sha256(raw).hexdigest()
    payload = build_payload(frame, BOXES, POINTS, expected_sha256=digest)
    assert payload["frame_sha256"] == digest
    _png(frame, value=80)
    with pytest.raises(FrameChangedError):
        build_payload(frame, BOXES, POINTS, expected_sha256=digest)


def test_atomic_write_replaces_complete_json_and_leaves_no_temp(tmp_path):
    output = tmp_path / "INITIAL_BOXES.json"
    output.write_text('{"old": true}\n', encoding="utf-8")
    payload = {"frame_sha256": "a" * 64, "role_boxes": BOXES, "role_points": POINTS}
    atomic_write_json(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob(".INITIAL_BOXES.json.*.tmp")) == []


def test_cancel_never_creates_output(tmp_path, monkeypatch):
    frame = tmp_path / "INITIAL_RGB.png"
    _png(frame)
    output = tmp_path / "INITIAL_BOXES.json"

    def cancel(_self):
        raise AnnotationCancelled("cancelled")

    monkeypatch.setattr(ManualPromptGUI, "run", cancel)
    with pytest.raises(AnnotationCancelled):
        annotate(frame, output)
    assert not output.exists()


def test_launcher_discovers_local_x_display(tmp_path):
    sockets = tmp_path / ".X11-unix"
    sockets.mkdir()
    (sockets / "X1").touch()
    environment = {}
    configure_local_desktop_environment(environment, x_socket_dir=sockets)
    assert environment["DISPLAY"] == ":1"


def test_gui_fixture_renders_all_boxes_point_and_disabled_confirm():
    gui = ManualPromptGUI(np.zeros((100, 180, 3), dtype=np.uint8))
    assert gui.window_name.isascii()
    first = gui.render()
    assert first.shape[0] > 100 and not gui.complete
    gui.boxes = {key: list(value) for key, value in BOXES.items()}
    gui.points = {key: list(value) for key, value in POINTS.items()}
    gui.step = 6
    complete = gui.render()
    assert complete.shape == first.shape and gui.complete


def test_waiter_requires_new_stable_frame_when_baseline_is_given(tmp_path):
    frame = tmp_path / "INITIAL_RGB.png"
    baseline = hashlib.sha256(_png(frame)).hexdigest()

    def replace():
        time.sleep(0.04)
        _png(frame, value=90)

    thread = threading.Thread(target=replace)
    thread.start()
    digest = wait_for_frame(
        frame, timeout_s=1.0, poll_s=0.02, baseline_sha256=baseline
    )
    thread.join()
    assert digest != baseline


def test_saved_sam_mask_preview_is_overlay_not_prompt_box(tmp_path):
    rgb = tmp_path / "RGB_000000.png"
    image = np.full((12, 16, 3), 100, dtype=np.uint8)
    assert cv2.imwrite(str(rgb), image)
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:7, 5:10] = True
    masks = tmp_path / "MASKS_000000.npz"
    np.savez_compressed(masks, white_mug=mask)

    preview = render_saved_mask_preview(rgb, masks)
    assert np.array_equal(preview[0, 0], image[0, 0])
    assert not np.array_equal(preview[4, 7], image[4, 7])
    output = tmp_path / "SAM_MASK_PREVIEW.png"
    atomic_write_png(output, preview)
    assert cv2.imread(str(output)).shape == image.shape


def test_saved_mask_preview_rejects_mismatched_geometry():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected"):
        overlay_saved_masks(image, {"white_mug": np.zeros((9, 20), dtype=bool)})
