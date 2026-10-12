import json
from pathlib import Path
import sys

import pytest
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "tools/deployment"
sys.path[:0] = [str(REPO), str(DEPLOY)]

from prove.xtrainer.io import SafetyError
from prove.xtrainer.io import RobotSample, SafetyLayer
from multitask_readonly_preflight import preflight
from multitask_dual_tool_route import TASKS, check_task, load_task_bundle, route_task
from multitask_dual_tool_one_command import main as one_command


def test_sorting_uses_selected_a3_own_model_and_four_handover_cycles():
    bundle = load_task_bundle("sorting_cube_bimanual")
    assert "twin_rsi_v02/selected/tasks/twin_sorting_cube_bimanual" in str(bundle["paths"]["model"])
    assert bundle["skill"].model.input_dim == 29
    route = route_task("sorting_cube_bimanual", bundle["graph"])
    parents = [s for s in route["stages"] if s["phase"] != "Complete"]
    assert len(parents) == 28
    assert [(s["arm"], s["object"], s["target"]) for s in parents if s["phase"] == "Grasp"] == [
        ("left", "yellow_cube", "table_handover_region"),
        ("right", "yellow_cube", "tray"),
        ("left", "red_cube", "table_handover_region"),
        ("right", "red_cube", "tray"),
    ]
    assert len([s for s in route["stages"] if s["phase"] == "Complete"]) == 4


def test_sweep_has_one_terminal_complete_no_release_and_both_tools_declared():
    route = route_task("sweep_objects")
    assert len(route["stages"]) == 10
    assert not [s for s in route["stages"] if s["phase"] == "Release"]
    completes = [s for s in route["stages"] if s["phase"] == "Complete"]
    assert len(completes) == 1
    complete = completes[0]
    assert complete["goal_relation"] == "block_inside_dustpan_with_tools_held"
    assert {(x["arm"], x["object"]) for x in complete["declared_held_tools"]} == {
        ("left", "dustpan"),
        ("right", "broom"),
    }
    right = [s for s in route["stages"] if s["arm"] == "right" and s["phase"] != "Complete"]
    assert all("passive_tools_held" in s["invariants"] for s in right)


def test_hanging_candidate_is_object_centered_and_fail_closed_but_not_executable():
    graph = route_task("hanging_mug_bimanual")
    assert graph["execution_status"] == "blocked_physics_and_model_unqualified"
    assert graph["stages"][-1]["goal_relation"] == "handle_supported_by_peg"
    assert graph["stages"][-1]["failure"] == "hanging False or UNKNOWN"
    assert "teacher_or_weld" in graph["forbidden_runtime_inputs"]
    report = check_task("hanging_mug_bimanual")
    assert report["execution_ready"] is False
    assert report["physics_task_outcome_passed"] is False
    with pytest.raises(SafetyError, match="no task-trained frozen 29D model"):
        load_task_bundle("hanging_mug_bimanual")


def test_check_is_read_only_and_never_claims_hardware_qualification():
    for task in ("sorting_cube_bimanual", "hanging_mug_bimanual", "sweep_objects"):
        report = check_task(task)
        assert report["device_connections"] == 0
        assert report["api_calls"] == 0
        assert report["episode_started"] is False
        assert report["hardware_qualified"] is False


def test_only_two_active_packages_and_portable_manifests_state_unverified_status():
    assert TASKS == ("sorting_cube_bimanual", "sweep_objects")
    for task in TASKS:
        leaf = json.loads((REPO / "task_packages/multitask_parallel_v22" / task / "TASK_LEAF.json").read_text())
        assert leaf["status"] == "portable_code_freeze_ready_not_sim_or_hardware_tested"
        assert leaf["modes"]["default"] == "check"
        assert "UNKNOWN/False never advances" in leaf["gemini_boundary"]
    archived = json.loads((REPO / "evaluations/multitask_parallel_v22/dual_tool/hanging_mug_bimanual/ARCHIVED.json").read_text())
    assert archived["status"] == "archived_not_active"
    assert archived["execution_authorized"] is False


def test_shadow_real_and_sim_require_explicit_task_material_before_io():
    with pytest.raises(SafetyError, match="explicit task config"):
        one_command(["--task", "sorting_cube_bimanual", "shadow"])
    with pytest.raises(SafetyError, match="explicit task config"):
        one_command(["--task", "sweep_objects", "real"])


def test_readonly_preflight_accepts_task_roles_without_cup_hardcode():
    class Clock:
        value = 10.0
        def __call__(self): return self.value
        def sleep(self, dt): self.value += dt
    clock = Clock()
    q = np.zeros((2, 6))
    class Robot:
        def samples(self):
            return [RobotSample(clock(), row, np.zeros(6), row[:3], np.eye(3), None, "fake") for row in q]
        def fk(self, target, gripper, arm):
            clock.sleep(.01); return target[0 if arm == "left" else 1, :3], np.eye(3)
        def command(self, *args): pytest.fail("read-only preflight commanded")
    class Vision:
        def capture(self):
            return type("Observation", (), {"stamp":clock(), "roles":{
                "yellow_cube":[0,0,0], "red_cube":[0,0,0], "tray":[0,0,0], "table_patch":[0,0,0]}})()
    safety = SafetyLayer(json.loads((REPO / "evaluations/multitask_parallel_v22/dual_tool/sorting_cube_bimanual/CONFIG_PROPOSED.json").read_text())["runtime"]["safety"], clock=clock)
    roles = ("yellow_cube", "red_cube", "tray", "table_patch")
    report = preflight(Robot(), Vision(), safety, [1., 1.], required_roles=roles, clock=clock)
    assert report["passed"] and report["required_roles"] == list(roles)
