import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/deployment"))

from prove.xtrainer.io import SafetyError
from prove.xtrainer.multitask_bundle import load_bundle
from multitask_articulated_v22 import ArticulatedExecutorV22, compile_route, load_leaf
from multitask_articulated_entry_v22 import check, real_interface


EXPECTED = {
    "open_drawer_and_place_object": [
        "Approach", "Grasp", "Pull", "Release", "Wait", "Complete",
        "Approach", "Grasp", "Lift", "Move", "Align", "Release", "Wait", "Complete",
    ],
    "hanging_mug": ["Approach", "Grasp", "Lift", "Move", "Insert", "Release", "Wait", "Complete"],
    "stack_cups": [
        "Approach", "Grasp", "Lift", "Move", "Insert", "Release", "Wait", "Complete",
        "Approach", "Grasp", "Lift", "Move", "Insert", "Release", "Wait", "Complete",
    ],
}


@pytest.mark.parametrize("task", EXPECTED)
def test_task_leaf_compiles_complete_executable_graph(task):
    leaf = load_leaf(ROOT / "task_packages/multitask_parallel_v22" / task / "TASK_LEAF.json")
    graph = compile_route(load_bundle(task)["graph"], leaf)
    assert graph["schema"].endswith("v22")
    assert [node["phase"] for node in graph["stages"]] == EXPECTED[task]
    assert all(left["next"] == right["stage_id"] for left, right in zip(graph["stages"], graph["stages"][1:]))
    assert graph["stages"][-1]["next"] is None
    assert all(node["done"] and node["invariants"] for node in graph["stages"])


@pytest.mark.parametrize("task", EXPECTED)
def test_neural_phase_and_frozen_model_are_preserved(task):
    bundle = load_bundle(task)
    leaf = load_leaf(ROOT / "task_packages/multitask_parallel_v22" / task / "TASK_LEAF.json")
    graph = compile_route(bundle["graph"], leaf)
    parent = [s["phase"] for s in bundle["graph"]["stages"] if s["controller"] == "neural_short"]
    routed = [s["phase"] for s in graph["stages"] if s["controller"] == "neural_short"]
    assert bundle["skill"].model.input_dim == 29
    assert routed == parent
    assert ("Pull" in routed) == (task == "open_drawer_and_place_object")
    assert ("Insert" in routed) == (task in ("hanging_mug", "stack_cups"))


@pytest.mark.parametrize("task,calls,relations", [
    ("open_drawer_and_place_object", 4, ["drawer_extended", "inside"]),
    ("hanging_mug", 2, ["handle_supported_by_peg"]),
    ("stack_cups", 4, ["nested_with_support", "nested_with_support"]),
])
def test_check_is_no_io_and_exact_relation_boundaries(task, calls, relations):
    report = check(task)
    assert report["status"] == "freeze_ready_not_executed"
    assert report["device_connections"] == report["api_calls"] == 0
    assert report["episode_started"] is False
    assert report["semantic_calls"] == calls
    assert [n["goal_relation"] for n in report["nodes"] if n["phase"] == "Complete"] == relations
    assert report["pull_insert_not_aliased_to_align"] is True


def test_leaf_rejects_phase_alias():
    task = "hanging_mug"
    leaf = load_leaf(ROOT / "task_packages/multitask_parallel_v22" / task / "TASK_LEAF.json")
    leaf = json.loads(json.dumps(leaf))
    next(node for node in leaf["nodes"] if node["node_type"] == "NearPegInsert")["phase"] = "Align"
    with pytest.raises(SafetyError, match="cannot alias"):
        compile_route(load_bundle(task)["graph"], leaf)


def test_executor_rejects_uncompiled_parent():
    owner = ArticulatedExecutorV22.__new__(ArticulatedExecutorV22)
    with pytest.raises(SafetyError, match="V22 route"):
        ArticulatedExecutorV22.__init__(owner, {"graph": {"schema": "old"}})


def test_real_commands_fail_before_connection_without_same_runtime_sim_receipt(tmp_path):
    task = "hanging_mug"
    scope = tmp_path / "REAL_SCOPE.json"
    scope.write_text(json.dumps({"task": task, "hardware_motion_commands_authorized": True}))
    config = ROOT / "evaluations/multitask_runtime_v20" / task / "CONFIG_PROPOSED.json"
    with pytest.raises(SafetyError, match="full SAM simulation receipt"):
        real_interface(task, "real", config, scope, tmp_path / "output")
    assert not (tmp_path / "output").exists()
