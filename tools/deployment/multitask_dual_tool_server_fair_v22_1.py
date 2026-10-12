#!/usr/bin/env python3
"""Server-only fair-IO candidate for sorting and sweep on GPU3."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import multitask_dual_tool_one_command as base
import multitask_dual_tool_route as routes
import multitask_dual_tool_runtime as runtime
import multitask_dual_tool_sim_entry as entry
from prove.xtrainer.agent import digest, write
from prove.xtrainer.io import SafetyError

TASKS = ("sorting_cube_bimanual", "sweep_objects")
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def validate_server_environment():
    values = {name: os.environ.get(name) for name in THREAD_ENV}
    if any(value != "1" for value in values.values()):
        raise SystemExit("server fair entry requires all CPU thread limits equal to 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise SystemExit("dual-tool server fair candidate is assigned exactly CUDA_VISIBLE_DEVICES=3")
    return {"thread_limits": values, "CUDA_VISIBLE_DEVICES": "3"}


def fair_backend_loader(original):
    def load(path, name):
        if Path(path).name == "sim_robot_io.py":
            import fair_sim_robot_io
            return fair_sim_robot_io
        return original(path, name)
    return load


def optional_table_contract(contract, task):
    result = {key: list(value) if isinstance(value, list) else value for key, value in contract.items()}
    essential = {
        "sorting_cube_bimanual": ("red_cube", "yellow_cube", "tray"),
        "sweep_objects": ("broom", "dustpan", "yellow_cube"),
    }[task]
    required = result.get("SAM_roles", [])
    if any(role not in required for role in essential):
        raise SafetyError("dual-tool actor/target role contract missing")
    result["SAM_roles"] = [role for role in required if role != "table_patch"]
    result["optional_SAM_roles"] = ["table_patch"]
    result["optional_role_reason"] = "generic observed table role is not read by Code, NN, holding, or exact task relation"
    return result


class ServerDualToolExecutor(runtime.DualToolExecutor):
    def __init__(self, bundle, *args, **kwargs):
        super().__init__(bundle, *args, **kwargs)
        self.roles = optional_table_contract(self.roles, bundle["task"])


def complete_existing_prepare(task, control, owner):
    """Finish a prepare that failed before its freeze without reserving again."""
    control = control.resolve()
    if control.parent != (owner / "control").resolve():
        raise SafetyError("resume control must belong to the exact server task owner")
    config = control / "CONFIG.json"
    reservation = control / "FORMAL_RUN_RESERVATION.json"
    first_freeze = control / "EXECUTION_FREEZE.json"
    freeze_path = control / ("EXECUTION_FREEZE_LAUNCH_FIX.json" if first_freeze.exists()
        else "EXECUTION_FREEZE.json")
    scope = owner / "TRIAL_SCOPE.json"
    source_config = owner / "CONFIG_SERVER.json"
    ledger_path = owner / "FORMAL_RUN_LEDGER.json"
    if freeze_path.exists():
        raise SafetyError("existing prepare already has a freeze")
    if not all(path.is_file() for path in (config, reservation, scope, source_config, ledger_path)):
        raise SafetyError("existing prepare material is incomplete")
    cfg = json.loads(config.read_text())
    claim = json.loads(reservation.read_text())
    ledger = json.loads(ledger_path.read_text())
    output = Path(claim["output"]).resolve()
    if cfg.get("task") != task or Path(claim["control"]).resolve() != control:
        raise SafetyError("existing reservation identity differs")
    if claim not in ledger.get("runs", []) or len(ledger["runs"]) != 1:
        raise SafetyError("exact single preserved reservation required")
    if output.exists():
        raise SafetyError("existing prepare already started an output")
    ready = routes.check_task(task)
    _, _, _, bound = entry.bindings(config, cfg, "sim")
    launcher = REPO / "task_packages/multitask_parallel_v22" / task / "run.sh"
    for path in (source_config, scope, Path(base.__file__), HERE / "macro_one_command.py", launcher):
        bound[str(path.resolve())] = digest(path)
    freeze = {
        "schema": "graph-self-adapter.shared-cup-freeze.v1",
        "status": "result_free_frozen",
        "scope_sha256": digest(scope),
        "runtime_sha256": entry.runtime_hash(),
        "bindings": [{"path": path, "sha256": sha} for path, sha in sorted(bound.items())],
        "jobs": [{"task": task, "mode": "sim", "output": str(output)}],
        "formal_reservation": str(reservation),
        "scope_not_cloned_or_reset": True,
        "generic_task": task,
        "completed_from_preserved_failed_prepare": True,
    }
    if first_freeze.exists():
        freeze["supersedes_prebackend_freeze"] = {
            "path": str(first_freeze), "sha256": digest(first_freeze),
            "reason": "server launcher lacked an existing-control execution path; no output or backend existed",
        }
    write(freeze_path, freeze)
    entry.backend.TASK = task
    entry.backend.guard(scope, freeze_path, bound, mode="sim", output=output,
        runtime_sha=entry.runtime_hash())
    report = {"task": task, "output": str(output), "freeze": str(freeze_path),
        "runtime_sha256": entry.runtime_hash(), "required_initial_SAM_roles": ready["roles"],
        "reservation_reused": True, "new_reservations": 0, "episode_started": False}
    print(json.dumps(report, indent=2), flush=True)
    return report


def run_existing_control(task, control, owner):
    control = control.resolve()
    if control.parent != (owner / "control").resolve():
        raise SafetyError("run control must belong to the exact server task owner")
    config = control / "CONFIG.json"
    scope = owner / "TRIAL_SCOPE.json"
    output = Path(json.loads((control / "FORMAL_RUN_RESERVATION.json").read_text())["output"])
    freezes = [path for path in (control / "EXECUTION_FREEZE_LAUNCH_FIX.json",
        control / "EXECUTION_FREEZE.json") if path.is_file()]
    if not freezes:
        raise SafetyError("existing control has no result-free execution freeze")
    return entry.main(["--task", task, "sim", "--config", str(config), "--scope", str(scope),
        "--execution-freeze", str(freezes[0]), "--output", str(output)])


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--task" not in args:
        raise SystemExit("explicit --task is required")
    index = args.index("--task")
    if index + 1 >= len(args) or args[index + 1] not in TASKS:
        raise SystemExit("server fair entry supports sorting_cube_bimanual or sweep_objects")
    task = args[index + 1]
    owner = REPO / "evaluations/multitask_parallel_v22/dual_tool" / task / "server_fair_v22_1"
    resume_control = None
    if "--resume-existing-control" in args:
        resume_index = args.index("--resume-existing-control")
        if resume_index + 1 >= len(args):
            raise SystemExit("--resume-existing-control requires a path")
        resume_control = Path(args[resume_index + 1])
        del args[resume_index:resume_index + 2]
    run_control = None
    if "--run-existing-control" in args:
        run_index = args.index("--run-existing-control")
        if run_index + 1 >= len(args):
            raise SystemExit("--run-existing-control requires a path")
        run_control = Path(args[run_index + 1])
        del args[run_index:run_index + 2]
    check_only = not any(mode in args for mode in ("sim", "shadow", "real")) or "check" in args
    environment = None if check_only else validate_server_environment()
    if "--config" not in args:
        args.extend(["--config", str(owner / "CONFIG_SERVER.json")])
    if "--owner" not in args:
        args.extend(["--owner", str(owner)])
    if "--scope" not in args and "sim" in args:
        args.extend(["--scope", str(owner / "TRIAL_SCOPE.json")])

    original_sources = entry.SOURCES
    original_backend_load = entry.backend.load
    original_executor = runtime.DualToolExecutor
    original_role_contract = routes.role_contract
    original_inventory = routes.INVENTORY
    extras = (
        Path(__file__), HERE / "fair_sim_robot_io.py",
        REPO / "tests/test_fair_sim_robot_io.py",
        REPO / "tests/test_multitask_dual_tool_server_fair_v22_1.py",
        REPO / "task_packages/multitask_parallel_v22" / task / "run_server_fair_v22_1.sh",
    )
    entry.SOURCES = original_sources + extras
    entry.backend.load = fair_backend_loader(original_backend_load)
    runtime.DualToolExecutor = ServerDualToolExecutor
    routes.role_contract = lambda graph, refs: optional_table_contract(original_role_contract(graph, refs), task)
    routes.INVENTORY = REPO / "evaluations/multitask_parallel_v22/dual_tool/server_shared/DUAL_TOOL_INVENTORY_SERVER.json"
    try:
        if run_control is not None:
            result = run_existing_control(task, run_control, owner)
        elif resume_control is not None:
            result = complete_existing_prepare(task, resume_control, owner)
        elif check_only:
            with contextlib.redirect_stdout(io.StringIO()):
                result = base.main(args)
        else:
            result = base.main(args)
        if isinstance(result, dict):
            result.update(server_candidate="fair_io_v22_1", simulator_io="FairSimRobot isolated derivative",
                shared_runtime_sha256=entry.runtime_hash(),
                scope=str(REPO / "evaluations/multitask_parallel_v22/dual_tool" / task / "server_fair_v22_1/TRIAL_SCOPE.json"),
                required_run_environment={name:"1" for name in THREAD_ENV}, CUDA_VISIBLE_DEVICES="explicit launcher default 3")
            if environment is not None:
                result["server_environment"] = environment
            if check_only:
                print(json.dumps(result, indent=2))
        return result
    finally:
        entry.SOURCES = original_sources
        entry.backend.load = original_backend_load
        runtime.DualToolExecutor = original_executor
        routes.role_contract = original_role_contract
        routes.INVENTORY = original_inventory


if __name__ == "__main__":
    main()
