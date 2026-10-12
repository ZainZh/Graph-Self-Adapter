import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


os.environ.setdefault("MUJOCO_GL", "egl")
MODULE_PATH = Path(__file__).parents[1] / "tools/evidence/render_twin_graph_video.py"
SPEC = importlib.util.spec_from_file_location("_render_twin_graph_video_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VIDEO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VIDEO
SPEC.loader.exec_module(VIDEO)

REPO = Path(__file__).parents[1]
TWIN_ROOT = Path("/home/zheng/github/Mujoco-RSI/runs/physics_v2/open_drawer_and_place_object")
PHYSICS = REPO / "runs/twin_rsi_v01/r3_draft01/source_snapshot/prove/task_graph/twin_physics.py"
FREEZE = REPO / "runs/twin_rsi_v01/r3_draft01/packages/twin_open_drawer_and_place_object/solution/CANDIDATE_FREEZE.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(condition):
    freeze = json.loads(FREEZE.read_text())
    return {
        "schema": VIDEO.REPORT_SCHEMA,
        "task": "open_drawer_and_place_object",
        "candidate_id": freeze["candidate_id"],
        "condition": condition,
        "bindings": {"candidate_freeze_sha256": _sha(FREEZE)},
    }


@pytest.mark.skipif(not TWIN_ROOT.is_dir(), reason="canonical Twin dependency unavailable")
def test_frozen_backend_reconstructs_hierarchical_initial_transforms_without_step():
    delta = [-.006, .004, 0.]
    moved_condition = {
        "initial_translation": {"drawer_shell": delta, "drawer": delta,
                                "red_cube": [.003, -.002, 0.]},
        "actual_initial_translation_m": {"drawer_shell": delta, "drawer": delta,
                                         "red_cube": [.003, -.002, 0.]},
    }
    zero_condition = {
        "initial_translation": {"drawer_shell": [0., 0., 0.],
                                "drawer": [0., 0., 0.], "red_cube": [0., 0., 0.]},
        "actual_initial_translation_m": {"drawer_shell": [0., 0., 0.],
                                         "drawer": [0., 0., 0.], "red_cube": [0., 0., 0.]},
    }
    moved, _, moved_data, record = VIDEO.frozen_initial_state(
        _report(moved_condition), Path("sealed.json"), TWIN_ROOT / "scene.xml",
        PHYSICS, FREEZE)
    baseline, _, base_data, _ = VIDEO.frozen_initial_state(
        _report(zero_condition), Path("sealed.json"), TWIN_ROOT / "scene.xml",
        PHYSICS, FREEZE)
    try:
        expected = np.asarray(delta)
        for role in ("drawer_shell", "drawer"):
            np.testing.assert_allclose(
                moved_data.body(f"object_{role}").xpos -
                base_data.body(f"object_{role}").xpos,
                expected, atol=1e-12)
        np.testing.assert_allclose(
            moved_data.body("object_drawer").xpos - moved_data.body("object_drawer_shell").xpos,
            base_data.body("object_drawer").xpos - base_data.body("object_drawer_shell").xpos,
            atol=1e-12)
        assert moved_data.time == base_data.time == 0.
        assert record["mj_step_executed"] is False
        assert record["physics_source"]["sha256"] == _sha(PHYSICS)
        assert record["candidate_freeze"]["sha256"] == _sha(FREEZE)
        assert len(record["initial_body_transforms_sha256"]) == 64
    finally:
        moved.close(); baseline.close()


def test_frozen_backend_rejects_unbound_source_and_holding_annotation_prefers_diagnostic(tmp_path):
    altered = tmp_path / "twin_physics.py"
    altered.write_bytes(PHYSICS.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="does not match CANDIDATE_FREEZE.code"):
        VIDEO.frozen_initial_state(
            _report({"initial_translation": {}, "actual_initial_translation_m": {}}),
            Path("sealed.json"), TWIN_ROOT / "scene.xml", altered, FREEZE)

    trace = {"bilateral_holding": np.asarray([True], dtype=bool)}
    report = {"trace": [{"controller": {"predicate_diagnostics": [
        {"predicate": "holding", "value": None}]}}]}
    label, value = VIDEO._holding_gate_annotation(report, trace, 0, "zh")
    assert label == "保持条件（夹持/目标支撑）"
    assert value == "未知"
    legacy_label, legacy_value = VIDEO._holding_gate_annotation({}, trace, 0, "zh")
    assert (legacy_label, legacy_value) == ("双指门记录", "是")


def test_v02_chinese_timing_annotations_preserve_parent_and_scaled_candidate():
    assert VIDEO.chinese_stage('c01.03.lift') == 'c01.03 · 抬升'
    assert VIDEO.v02_timing_annotation({'candidate_id':'v02_b0'},'c01.03.lift') == '1×参考速度'
    assert VIDEO.v02_timing_annotation({'candidate_id':'v02_a2'},'c01.03.lift') == '0.5×参考速度'
    assert '抓持确认≥0.2秒' in VIDEO.v02_timing_annotation({'candidate_id':'v02_a2'},'c01.02.grasp')
    assert VIDEO.v02_timing_annotation({'candidate_id':'v02_a3'},'c02.05.align') == '0.7×参考速度'
