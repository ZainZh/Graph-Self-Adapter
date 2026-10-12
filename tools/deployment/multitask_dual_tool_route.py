"""Task-local routes for the dual-arm sort, sweep, and hanging-mug work.

This module is deliberately a leaf over the V20 specialist runtime.  It does
not change the frozen graphs and it does not make the unqualified hanging-mug
scene executable.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from prove.xtrainer.io import SafetyError
from prove.xtrainer.multitask_bundle import load_bundle
from multitask_special_route import role_contract, special_route


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OWNER = REPO / "evaluations/multitask_parallel_v22/dual_tool"
INVENTORY = OWNER / "TASK_ASSET_INVENTORY.json"
HANGING_GRAPH = OWNER / "hanging_mug_bimanual/TASK_GRAPH_PROPOSED.json"

TASKS = ("sorting_cube_bimanual", "sweep_objects")
ARCHIVED_TASKS = ("hanging_mug_bimanual",)


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_task_bundle(task: str):
    if task == "hanging_mug_bimanual":
        raise SafetyError(
            "hanging_mug_bimanual has no task-trained frozen 29D model and its "
            "recorded physics ends unsupported; execution remains blocked"
        )
    if task not in TASKS:
        raise SafetyError("undeclared dual-tool task: " + task)
    return load_bundle(task, inventory=INVENTORY)


def _complete_from(stage, declared_tools, *, held):
    complete = copy.deepcopy(stage)
    complete.update(
        stage_id=stage["cycle_id"] + ".complete",
        phase="Complete",
        controller="exact_supported_or_held_goal_boundary",
        holding_contract="tools_held" if held else "none",
        done=["holding", "placement_complete"] if held else ["released", "placement_complete"],
        invariants=["finite"],
        hold_at_end=held,
        declared_held_tools=copy.deepcopy(declared_tools) if held else [],
        semantic_facts=["holding", "placement_complete"] if held else ["released", "placement_complete"],
        completion_source="exact declared task relation; UNKNOWN/False cannot be replaced by an API True",
    )
    return complete


def sweep_route(graph):
    """Keep both tools held and add exactly one terminal boundary."""
    if graph.get("task") not in ("sweep_objects", "twin_sweep_objects"):
        raise SafetyError("sweep route received another task")
    result = copy.deepcopy(graph)
    routed = []
    tools = []
    for original in graph["stages"]:
        stage = copy.deepcopy(original)
        phase = stage["phase"]
        stage.update(
            completion_predicates=[],
            invariant_predicates=[],
            done=["own_pose", "own_progress"],
            invariants=["finite"],
            completion_source="own pose/progress and tentative task-local tool state",
        )
        if phase in ("Move", "Push", "Hold"):
            stage["invariants"].append("tentative_holding")
        if stage["arm"] == "right":
            stage["invariants"].append("passive_tools_held")
        if phase == "Grasp":
            stage["done"].append("tentative_holding")
            tools.append({k: stage[k] for k in ("arm", "object", "actor")})
        routed.append(stage)
    if [(s["arm"], s["object"]) for s in graph["stages"] if s["phase"] == "Grasp"] != [
        ("left", "dustpan"),
        ("right", "broom"),
    ]:
        raise SafetyError("sweep tool acquisition order changed")
    if any(s["phase"] == "Release" for s in routed):
        raise SafetyError("sweep must retain both tools")
    last = routed[-1]
    if (last["arm"], last["phase"], last["goal_relation"]) != (
        "right",
        "Hold",
        "block_inside_dustpan_with_tools_held",
    ):
        raise SafetyError("sweep terminal relation changed")
    routed.append(_complete_from(last, tools, held=True))
    for current, following in zip(routed, routed[1:]):
        current["next"] = following["stage_id"]
    routed[-1]["next"] = None
    result["stages"] = routed
    result["schema"] = "graph-self-adapter.multitask-dual-tool-route.v22"
    return result


def route_task(task: str, graph=None):
    if task == "hanging_mug_bimanual":
        value = json.loads(HANGING_GRAPH.read_text())
        if value.get("execution_status") != "blocked_physics_and_model_unqualified":
            raise SafetyError("hanging-mug uncertainty boundary missing")
        return value
    graph = graph or load_task_bundle(task)["graph"]
    if task == "sweep_objects":
        return sweep_route(graph)
    routed = special_route(graph)
    routed["schema"] = "graph-self-adapter.multitask-dual-tool-route.v22"
    return routed


def check_task(task: str):
    if task not in TASKS + ARCHIVED_TASKS:
        raise SafetyError("task is outside this leaf")
    row = next(r for r in json.loads(INVENTORY.read_text())["tasks"] if r["task"] == task)
    for binding in row["bindings"].values():
        if digest(binding["path"]) != binding["sha256"]:
            raise SafetyError("task asset changed: " + binding["path"])
    if task == "hanging_mug_bimanual":
        graph = route_task(task)
        return {
            "task": task,
            "graph_nodes": len(graph["stages"]),
            "roles": graph["roles"],
            "execution_ready": False,
            "blocked_reasons": graph["blocked_reasons"],
            "physics_task_outcome_passed": False,
            "task_trained_model_available": False,
            "device_connections": 0,
            "api_calls": 0,
            "episode_started": False,
            "hardware_qualified": False,
        }
    bundle = load_task_bundle(task)
    graph = route_task(task, bundle["graph"])
    return {
        "task": task,
        "parent_graph_stages": len(bundle["graph"]["stages"]),
        "routed_stages": len(graph["stages"]),
        "declared_model_dim": bundle["skill"].model.input_dim,
        "fixed_z": True,
        "roles": role_contract(graph, bundle["refs"]),
        "execution_ready": True,
        "formal_episode_requires_new_scope_and_result_free_freeze": True,
        "device_connections": 0,
        "api_calls": 0,
        "episode_started": False,
        "hardware_qualified": False,
    }
