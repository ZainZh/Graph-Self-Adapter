"""Portable loader for frozen V01 task bundles on the isolated V22 server."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

from prove.xtrainer.agent import digest
from prove.xtrainer.io import SafetyError


def load_server_bundle(task, *, inventory):
    """Load frozen graph/model assets while rebinding only the copied motor prior."""
    inventory = Path(inventory)
    catalog = json.loads(inventory.read_text())
    rows = [row for row in catalog["tasks"] if row["task"] == task]
    if len(rows) != 1:
        raise SafetyError("exact qualified task bundle required: " + task)
    row = rows[0]
    solution = Path(row["solution"])
    paths = {
        key: Path(row["bindings"][name]["path"])
        for key, name in (
            ("graph", "graph"),
            ("model", "model"),
            ("references", "visible_refs"),
            ("manifest", "manifest"),
            ("candidate_freeze", "candidate_freeze"),
        )
    }
    for name, binding in row["bindings"].items():
        if digest(binding["path"]) != binding["sha256"]:
            raise SafetyError("task inventory source changed: " + name)
    manifest = json.loads(paths["manifest"].read_text())
    freeze = json.loads(paths["candidate_freeze"].read_text())
    graph = json.loads(paths["graph"].read_text())
    refs = json.loads(paths["references"].read_text())
    if any(value.get("task") not in (task, "twin_" + task) for value in (manifest, freeze, graph, refs)):
        raise SafetyError("task graph/model/reference identity mismatch")
    for key, freeze_key in (("graph", "graph_sha256"), ("model", "model_sha256"), ("references", "references_sha256")):
        if digest(paths[key]) != freeze[freeze_key]:
            raise SafetyError("frozen task asset mismatch: " + key)
    source = Path(row["historical_runner"]["path"]).parents[2] / "prove/fast_adapter/twin_local_skill.py"
    if digest(source) != freeze["code"]["prove/fast_adapter/twin_local_skill.py"]:
        raise SafetyError("frozen task skill code changed")
    paths["skill_source"] = source
    if "verified_bindings" in manifest:
        motor = next(item for item in manifest["verified_bindings"]["sources"] if item["label"] == "motor_targets.npz")
    else:
        paths["external_bindings"] = solution / "EXTERNAL_BINDINGS.json"
        external = json.loads(paths["external_bindings"].read_text())
        motor = external["sources"]["motor_targets.npz"]
    declared_motor = Path(motor["path"])
    if declared_motor.name != "motor_targets.npz":
        raise SafetyError("frozen motor binding has an invalid filename")
    paths["motor_prior"] = Path(row["physics_report"]["path"]).with_name("motor_targets.npz")
    if digest(paths["motor_prior"]) != motor["sha256"]:
        raise SafetyError("portable task motor prior differs from frozen binding")
    training = manifest["model"]["training"]
    if training["input_dim"] != 29 or training["output_dim"] != 6:
        raise SafetyError("this executor requires declared frozen 29D geometry/6D residual contract")
    spec = importlib.util.spec_from_file_location("_server_multitask_skill_" + task, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    skill = module.TwinLocalSkill(paths["model"], online_z=False, feedback_contract=graph["local_feedback_contract"])
    if skill.model.input_dim != training["input_dim"]:
        raise SafetyError("loaded model differs from declared feature contract")
    with np.load(paths["motor_prior"], allow_pickle=False) as data:
        times = data["timestamps"].copy()
        joints = data["joints"].copy()
        gripper = data["gripper"].copy()
    if joints.shape != (len(times), 2, 6) or gripper.shape != (len(times), 2):
        raise SafetyError("task motor prior shape invalid")
    if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(gripper)) or not np.all(np.diff(times) > 0):
        raise SafetyError("task motor prior time/value invalid")
    if np.any(gripper < 0) or np.any(gripper > 1):
        raise SafetyError("task motor prior gripper units invalid")
    frames = sorted(refs["frames"].values(), key=lambda frame: frame["time_sec"])
    if not frames or not graph["stages"] or len({stage["stage_id"] for stage in graph["stages"]}) != len(graph["stages"]):
        raise SafetyError("task stages/reference coverage missing")
    for stage in graph["stages"]:
        start = stage["reference"]["start_time_sec"]
        end = stage["reference"]["end_time_sec"]
        if not times[0] <= start < end <= times[-1] + 1e-8:
            raise SafetyError("stage outside task motor prior")
    return {
        "task": task,
        "graph": graph,
        "refs": refs,
        "skill": skill,
        "paths": paths,
        "bindings": {str(path): digest(path) for path in paths.values()} | {str(inventory.resolve()): digest(inventory)},
        "times": times,
        "qcurve": PchipInterpolator(times, joints, axis=0),
        "gripper": gripper,
        "initial_q": joints[0],
        "initial_g": gripper[0],
        "frames": frames,
        "frame_times": np.asarray([frame["time_sec"] for frame in frames]),
        "dataset_metadata": training["dataset_metadata"],
    }
