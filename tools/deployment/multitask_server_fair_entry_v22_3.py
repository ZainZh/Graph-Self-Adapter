#!/usr/bin/env python3
"""Server-only fair-IO entry for drawer and the hanging V22.2 repair."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import multitask_articulated_entry_v22 as base
from prove.xtrainer.io import SafetyError
from multitask_hanging_repair_v22_2 import HangingExecutorV22_2, optional_table_role_contract
from multitask_server_bundle_v22_3 import load_server_bundle
from prove.xtrainer.agent import digest
from prove.xtrainer.io import load_calibration


TASKS = ("open_drawer_and_place_object", "hanging_mug")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SERVER_SHARED = REPO / "evaluations/multitask_parallel_v22/server_shared"
SERVER_INVENTORY = SERVER_SHARED / "TASK_ASSET_INVENTORY_SERVER.json"
SERVER_CATALOG = SERVER_SHARED / "EXTERNAL_ASSET_BINDINGS_V01_SERVER.json"
SERVER_MANIFEST = SERVER_SHARED / "REMOTE_DERIVATIVE_MANIFEST.json"


def validate_server_environment():
    values = {name: os.environ.get(name) for name in THREAD_ENV}
    if any(value != "1" for value in values.values()):
        raise SystemExit("server entry requires OMP/MKL/OPENBLAS/NUMEXPR thread limits all equal to 1")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise SystemExit("server entry requires an explicit CUDA_VISIBLE_DEVICES")
    egl = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if egl != "2":
        raise SystemExit("server entry requires MUJOCO_EGL_DEVICE_ID=2")
    return {"thread_limits": values, "CUDA_VISIBLE_DEVICES": visible, "MUJOCO_EGL_DEVICE_ID": egl}


def fair_backend_loader(original):
    def load(path, name):
        if Path(path).name == "sim_robot_io.py":
            import fair_sim_robot_io
            return fair_sim_robot_io
        return original(path, name)
    return load


def optional_drawer_table_contract(contract):
    result = {key: list(value) if isinstance(value, list) else value for key, value in contract.items()}
    required = result.get("SAM_roles", [])
    essential = ("drawer", "drawer_shell", "red_cube")
    if any(role not in required for role in essential):
        raise SafetyError("drawer actor/target role contract missing")
    result["SAM_roles"] = [role for role in required if role != "table_patch"]
    result["optional_SAM_roles"] = ["table_patch"]
    result["optional_role_reason"] = "auxiliary observed patch is not read by drawer/cube policy, holding, or exact relation judges"
    return result


class ServerDrawerExecutor(base.ArticulatedExecutorV22):
    def __init__(self, bundle, *args, **kwargs):
        if bundle.get("task") != "open_drawer_and_place_object":
            raise SafetyError("server drawer executor belongs to another task")
        super().__init__(bundle, *args, **kwargs)
        self.roles = optional_drawer_table_contract(self.roles)


def server_bindings(config_path, config, mode):
    task = config["task"]
    _, _, leaf_path, bundle, _, expected, _ = base.build(task)
    route = Path(config["special_route_graph"])
    actual = json.loads(route.read_text())
    if actual != expected:
        raise SafetyError("compiled server task leaf route differs")
    bundle["graph"] = actual
    bundle["paths"]["graph"] = route
    runtime = copy.deepcopy(config["runtime"])
    if runtime.get("hardware_reference_speed_scale") != 1.0 or runtime.get("vision", {}).get("role_boxes"):
        raise SafetyError("fresh initial grounding and reference clock 1.0 required")
    from multitask_special_judge import validate_config
    validate_config(runtime["stage_judge"])
    required = sum(stage["phase"] in ("Grasp", "Complete") for stage in actual["stages"])
    if runtime["stage_judge"]["maximum_calls"] != required:
        raise SafetyError("exact server task semantic call budget required")
    camera, right = load_calibration(runtime["left_calibration"], runtime["right_calibration"])
    runtime.update(T_left_from_camera=camera.tolist(), T_left_from_right=right.tolist())
    paths = list(base.SOURCES) + [leaf_path, config_path, route, Path(config["formal_run_reservation"]), Path(config["sim"]["physics_source"]), Path(runtime["sdk_file"]), Path(runtime["left_calibration"]), Path(runtime["right_calibration"]), Path(runtime["vision"]["sam_utility"]), Path(runtime["vision"]["sam_checkpoint"]), SERVER_CATALOG, SERVER_MANIFEST]
    scene = Path(config["sim"]["scene_run"])
    for name in ("scene.xml", "reconstruction.json", "motor_targets.npz", "reference.npz", "dynamics.npz", "physics_validation.json"):
        paths.append(scene / name)
    assets = json.loads(SERVER_CATALOG.read_text())
    for evaluator in assets.get("external_evaluator_sources", []):
        if digest(evaluator["path"]) != evaluator["sha256"]:
            raise SafetyError("server postseal evaluator source changed")
        paths.append(Path(evaluator["path"]))
    for path, sha in assets["task_assets"][task]["external_asset_hashes"].items():
        if digest(path) != sha:
            raise SafetyError("server task scene asset changed")
        paths.append(Path(path))
    bound = dict(bundle["bindings"])
    bound.update({str(Path(path).resolve()): digest(path) for path in paths})
    return {"dependencies": {}}, runtime, bundle, bound


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--task" not in args:
        raise SystemExit("explicit --task is required")
    index = args.index("--task")
    if index + 1 >= len(args) or args[index + 1] not in TASKS:
        raise SystemExit("server fair entry supports drawer or hanging_mug")
    task = args[index + 1]
    check_only = not any(mode in args for mode in ("prepare", "sim", "shadow", "real")) or "check" in args
    environment = None if check_only else validate_server_environment()
    if "--scope" not in args and any(mode in args for mode in ("prepare", "sim")):
        args.extend(["--scope", str(REPO / "evaluations/multitask_parallel_v22/articulated" / task / "server_fair_v22_3/TRIAL_SCOPE.json")])

    original_sources = base.SOURCES
    original_executor = base.ArticulatedExecutorV22
    original_roles = base.role_contract
    original_backend_load = base.backend.load
    original_check = base.check
    original_load_bundle = base.load_bundle
    original_bindings = base.bindings
    sources = (
        Path(__file__),
        HERE / "fair_sim_robot_io.py",
        HERE / "multitask_server_bundle_v22_3.py",
        REPO / "tests/test_fair_sim_robot_io.py",
        REPO / "tests/test_multitask_server_fair_v22_3.py",
        REPO / "task_packages/multitask_parallel_v22" / task / "run_server_fair_v22_3.sh",
    )
    if task == "hanging_mug":
        sources += (HERE / "multitask_hanging_repair_v22_2.py", REPO / "tests/test_multitask_hanging_repair_v22_2.py")
        base.ArticulatedExecutorV22 = HangingExecutorV22_2
        base.role_contract = lambda graph, refs: optional_table_role_contract(original_roles(graph, refs))
    else:
        base.ArticulatedExecutorV22 = ServerDrawerExecutor
        base.role_contract = lambda graph, refs: optional_drawer_table_contract(original_roles(graph, refs))
    base.SOURCES = original_sources + sources
    base.backend.load = fair_backend_loader(original_backend_load)
    shared_ready = all(path.is_file() for path in (SERVER_INVENTORY, SERVER_CATALOG, SERVER_MANIFEST))
    if shared_ready:
        base.load_bundle = lambda selected_task: load_server_bundle(selected_task, inventory=SERVER_INVENTORY)
        base.bindings = server_bindings
    elif not check_only:
        raise SystemExit("shared remote derivative inventory/catalog/manifest missing")
    def server_check(selected_task):
        report = original_check(selected_task)
        report.update(
            scope=str(REPO / "evaluations/multitask_parallel_v22/articulated" / selected_task / "server_fair_v22_3/TRIAL_SCOPE.json"),
            simulator_io="FairSimRobot isolated derivative",
            required_run_environment={name: "1" for name in THREAD_ENV},
            CUDA_VISIBLE_DEVICES="explicit; launcher default 2",
            MUJOCO_EGL_DEVICE_ID="2",
        )
        return report
    base.check = server_check
    try:
        result = base.main(args)
        if isinstance(result, dict) and environment is not None:
            result["server_environment"] = environment
        return result
    finally:
        base.SOURCES = original_sources
        base.ArticulatedExecutorV22 = original_executor
        base.role_contract = original_roles
        base.backend.load = original_backend_load
        base.check = original_check
        base.load_bundle = original_load_bundle
        base.bindings = original_bindings


if __name__ == "__main__":
    main()
