import sys
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/deployment"))

from all_code_stage_runtime import ExactFKCacheRobot, is_code_stage
from all_code_stage_trajectory import _gripper_at, stream_stage
from all_code_stage_route import validate_route
from test_stage_trajectory import Clock, Robot as StreamRobot, setup


def stage(cycle, phase, controller="code_reference_retarget"):
    return {"cycle_id": cycle, "phase": phase, "controller": controller}


def test_exact_route_keeps_precision_nodes_on_frozen_closed_loop():
    for cycle in ("c01", "c02"):
        for phase in ("Approach", "Lift", "Move", "Release", "Wait"):
            assert is_code_stage(stage(cycle, phase))
        for phase in ("Grasp", "Align"):
            assert not is_code_stage(stage(cycle, phase, "neural_short"))
    for phase in ("Approach", "Lift", "Raise", "Move", "Lower", "Release", "Retreat"):
        assert is_code_stage(stage("white_direct", phase, "observed_white_direct_Code"))
    assert not is_code_stage(stage("white_direct", "Grasp", "observed_white_direct_Code"))
    assert not is_code_stage(stage("white_direct", "Complete", "supported_Root_completion"))


def test_exact_frozen_v19_graph_has_17_code_5_precision_3_semantic_nodes():
    path = ROOT / "evaluations/gemini_graph_v19/control/20261007_111206_7f0994a9/TASK_GRAPH.json"
    report = validate_route(json.loads(path.read_text()))
    assert report["graph_nodes"] == 25
    assert len(report["code_trajectory_stage_ids"]) == 17
    assert len(report["precision_closed_loop_stage_ids"]) == 5
    assert len(report["semantic_boundary_stage_ids"]) == 3


class Robot:
    def __init__(self):
        self.calls = []
        self.commands = 0

    def fk(self, q, g, arm):
        self.calls.append((np.asarray(q).copy(), np.asarray(g).copy(), arm))
        offset = 0 if arm == "left" else 1
        return np.asarray(q)[offset, :3] + offset, np.eye(3)


def test_precision_target_fk_prefetch_is_exact_copying_and_update_local():
    raw = Robot(); cached = ExactFKCacheRobot(raw)
    q = np.zeros((2, 6)); g = np.array([.2, .8])
    cached.prefetch_target_fk(q, g)
    assert [row[2] for row in raw.calls] == ["left", "right"]
    xyz, rotation = cached.fk(q.copy(), g.copy(), "left")
    assert len(raw.calls) == 2 and np.array_equal(xyz, np.zeros(3))
    xyz[:] = 9; rotation[:] = 9
    again, again_rotation = cached.fk(q, g, "left")
    assert np.array_equal(again, np.zeros(3)) and np.array_equal(again_rotation, np.eye(3))
    changed = q.copy(); changed[0, 0] = np.nextafter(0., 1.)
    cached.fk(changed, g, "left")
    assert len(raw.calls) == 3
    cached.clear_fk_cache(); cached.fk(q, g, "left")
    assert len(raw.calls) == 4


def test_per_sample_gripper_profile_and_constant_compatibility():
    constant = SimpleNamespace(gripper=np.array([.1, .9]), times=np.arange(3))
    assert np.array_equal(_gripper_at(constant, 2), [.1, .9])
    profile = SimpleNamespace(gripper=np.array([[0., 1.], [.5, 1.], [1., 1.]]), times=np.arange(3))
    assert np.array_equal(_gripper_at(profile, 1), [.5, 1.])


def test_stream_uses_profile_without_fk_sam_or_judge_in_servo_loop():
    limits, start, trajectory = setup()
    trajectory.gripper = np.c_[np.linspace(0., 1., len(trajectory.times)),
                               np.ones(len(trajectory.times))]
    clock = Clock(); robot = StreamRobot(start, clock); issued_g = []
    original = robot.command
    def command(q, g, flags):
        issued_g.append(np.asarray(g).copy()); original(q, g, flags)
    robot.command = command
    report = stream_stage(robot, trajectory, limits, robot.stop,
                          clock=clock, sleep=clock.sleep)
    assert report["completed"] and report["FK_calls_during_streaming"] == 0
    assert report["IK_calls_during_streaming"] == 0
    assert report["SAM_calls_during_streaming"] == 0
    assert report["judge_calls_during_streaming"] == 0
    assert np.isclose(issued_g[0][0], 0.) and np.isclose(issued_g[-1][0], 1.)
