"""Task-leaf graph compiler and executor for the three articulated V22 tasks.

The compiler changes graph structure and semantic boundaries only.  Commands
still use each task's frozen reference, 29D model, and original NN phase label.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError
from multitask_special_runtime import SpecialExecutor


SCHEMA = "graph-self-adapter.multitask-articulated-route.v22"
LEAF_SCHEMA = "graph-self-adapter.multitask-articulated-leaf.v22"
SUPPORTED = {
    "open_drawer_and_place_object",
    "hanging_mug",
    "stack_cups",
}


def load_leaf(path: Path) -> dict:
    leaf = json.loads(Path(path).read_text())
    if leaf.get("schema") != LEAF_SCHEMA or leaf.get("task") not in SUPPORTED:
        raise SafetyError("qualified articulated V22 task leaf required")
    nodes = leaf.get("nodes")
    if not isinstance(nodes, list) or not nodes or len({n.get("node_id") for n in nodes}) != len(nodes):
        raise SafetyError("task leaf nodes missing or duplicated")
    return leaf


def compile_route(parent: dict, leaf: dict) -> dict:
    if parent.get("task") not in (leaf["task"], "twin_" + leaf["task"]):
        raise SafetyError("task leaf belongs to another parent graph")
    parent_by_id = {stage["stage_id"]: stage for stage in parent["stages"]}
    expected = leaf.get("expected_parent_phases")
    actual = [stage["phase"] for stage in parent["stages"]]
    if expected != actual:
        raise SafetyError("frozen parent phase sequence differs from task leaf")

    routed = []
    for node in leaf["nodes"]:
        node_id = node["node_id"]
        node_type = node["node_type"]
        if node_type == "Complete":
            source_id = node["boundary_after"]
            if source_id not in parent_by_id:
                raise SafetyError("Complete boundary source is absent")
            stage = copy.deepcopy(parent_by_id[source_id])
            stage.update(
                stage_id=node_id,
                phase="Complete",
                controller="exact_declared_relation_two_frame_judge",
                holding_contract="none",
                completion_predicates=[],
                invariant_predicates=[],
                done=["released", "placement_complete"],
                invariants=["finite"],
                semantic_facts=["released", "placement_complete"],
                reference_commands=0,
                completion_source="two current RGB frames + small proprio + exact declared relation; no evaluator truth",
                hold_at_end=False,
            )
        else:
            source_id = node["source_stage_id"]
            if source_id not in parent_by_id:
                raise SafetyError("task leaf source stage is absent")
            stage = copy.deepcopy(parent_by_id[source_id])
            if node.get("phase") != stage["phase"]:
                raise SafetyError("leaf cannot alias the frozen controller phase")
            stage.update(
                stage_id=node_id,
                completion_predicates=[],
                invariant_predicates=[],
                done=["own_pose", "own_progress"],
                invariants=["finite"],
                completion_source="own pose/progress and tentative holding state; no hidden contact/goal",
            )
            if stage["phase"] in ("Lift", "Move", "Align", "Insert", "Pull"):
                stage["invariants"].append("tentative_holding")
            if stage["phase"] == "Grasp":
                stage["done"].append("tentative_holding")
                stage["semantic_contract"] = "tentative permission only at supported boundary; UNKNOWN/False stops advancement"
            if stage["phase"] == "Release":
                stage["done"].append("opening_progress")
        stage["node_type"] = node_type
        stage["leaf_contract"] = node.get("contract")
        stage["goal_relation"] = node.get("goal_relation", stage["goal_relation"])
        routed.append(stage)

    used = [n.get("source_stage_id") for n in leaf["nodes"] if n["node_type"] != "Complete"]
    if used != [stage["stage_id"] for stage in parent["stages"]]:
        raise SafetyError("leaf must execute every frozen parent stage exactly once and in order")
    for left, right in zip(routed, routed[1:]):
        left["next"] = right["stage_id"]
    routed[-1]["next"] = None
    result = copy.deepcopy(parent)
    result.update(
        schema=SCHEMA,
        parent_graph_task=parent["task"],
        task_leaf_version=leaf["version"],
        runtime_inputs=leaf["runtime_inputs"],
        stages=routed,
    )
    return result


class ArticulatedExecutorV22(SpecialExecutor):
    """Reuse the reviewed special controller while accepting the V22 route."""

    def __init__(self, bundle, *args, **kwargs):
        if bundle["graph"].get("schema") != SCHEMA:
            raise SafetyError("explicit articulated V22 route required")
        actual_graph = bundle["graph"]
        parent_bundle = dict(bundle)
        parent_bundle["graph"] = copy.deepcopy(actual_graph)
        parent_bundle["graph"]["schema"] = "graph-self-adapter.multitask-special-route.v1"
        super().__init__(parent_bundle, *args, **kwargs)
        self.b["graph"] = actual_graph

    def run(self):
        result = super().run()
        result.update(
            schema="graph-self-adapter.multitask-articulated-run.v22",
            execution_variant="multitask_articulated_object_centered_v22",
            task_leaf_version=self.b["graph"]["task_leaf_version"],
            exact_relation_boundaries=[
                {"stage_id": stage["stage_id"], "goal_relation": stage["goal_relation"]}
                for stage in self.b["graph"]["stages"] if stage["phase"] == "Complete"
            ],
            runtime_policy_inputs=self.b["graph"]["runtime_inputs"],
            evaluator_truth_runtime_reads=0,
            mid_episode_restores=0,
        )
        write(self.output / "RESULT.json", result)
        return result


SharedCupExecutor = ArticulatedExecutorV22
