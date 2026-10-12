#!/usr/bin/env python3
"""Prepare and optionally run one newly scoped V22 dual-arm simulator episode."""
import argparse
import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prove.xtrainer.agent import digest, write
from prove.xtrainer.io import SafetyError
from macro_one_command import reserve
from multitask_dual_tool_route import TASKS, check_task, load_task_bundle, route_task
import multitask_dual_tool_sim_entry as entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("mode", nargs="?", default="check", choices=("check", "shadow", "sim", "real"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--owner", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    default_owner = REPO / "evaluations/multitask_parallel_v22/dual_tool" / args.task
    owner = (args.owner or default_owner).resolve()
    allowed = (default_owner / "server_fair_v22_1").resolve()
    if owner not in (default_owner.resolve(), allowed):
        raise SafetyError("task owner must be the default owner or declared server fair owner")
    config_source = args.config or owner / "CONFIG_PROPOSED.json"
    scope = args.scope or owner / "TRIAL_SCOPE.json"
    ready = check_task(args.task)
    if args.mode == "check":
        ready.update(configured=config_source.is_file(), formal_scope_available=scope.is_file(),
            source_config=str(config_source), task_selected_explicitly=True)
        print(json.dumps(ready, indent=2))
        return ready
    if args.mode in ("shadow", "real"):
        if not args.config:
            raise SafetyError("portable shadow/real requires an explicit task config")
        call = ["--task", args.task, args.mode, "--config", str(args.config)]
        if not args.scope or not args.output:
            raise SafetyError(f"{args.mode} requires explicit task scope and fresh output")
        call.extend(("--scope", str(args.scope), "--output", str(args.output)))
        return entry.main(call)
    if not config_source.is_file() or not scope.is_file():
        raise SafetyError("new Root-owned per-task configuration and trial scope are required")
    cfg = copy.deepcopy(json.loads(config_source.read_text()))
    authority = json.loads(scope.read_text())
    if cfg.get("task") != args.task or authority.get("task") != args.task:
        raise SafetyError("task-specific config/scope identity mismatch")
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output = owner / "runs" / stamp
    control = owner / "control" / stamp
    if output.exists() or control.exists():
        raise SafetyError("fresh per-task output required")
    control.mkdir(parents=True)
    route = control / "TASK_GRAPH.json"
    write(route, route_task(args.task, load_task_bundle(args.task)["graph"]))
    claim = reserve(scope, owner / "FORMAL_RUN_LEDGER.json", output, control)
    reservation = control / "FORMAL_RUN_RESERVATION.json"
    write(reservation, claim)
    cfg.update(dual_tool_route_graph=str(route), formal_run_reservation=str(reservation))
    cfg["runtime"]["vision"].update(prompt_frame_output=str(output / "INITIAL_RGB.png"),
        prompt_reply_path=str(output / "INITIAL_BOXES.json"), prompt_frame_sha256=None, role_boxes={})
    config = control / "CONFIG.json"
    write(config, cfg)
    _, _, _, bound = entry.bindings(config, cfg, "sim")
    launcher = REPO / "task_packages/multitask_parallel_v22" / args.task / "run.sh"
    for path in (config_source, scope, Path(__file__), HERE / "macro_one_command.py", launcher):
        bound[str(path.resolve())] = digest(path)
    freeze = {"schema":"graph-self-adapter.shared-cup-freeze.v1", "status":"result_free_frozen",
        "scope_sha256":digest(scope), "runtime_sha256":entry.runtime_hash(),
        "bindings":[{"path":path,"sha256":sha} for path,sha in sorted(bound.items())],
        "jobs":[{"task":args.task,"mode":"sim","output":str(output)}],
        "formal_reservation":str(reservation), "scope_not_cloned_or_reset":True, "generic_task":args.task}
    freeze_path = control / "EXECUTION_FREEZE.json"
    write(freeze_path, freeze)
    entry.backend.TASK = args.task
    entry.backend.guard(scope, freeze_path, bound, mode="sim", output=output, runtime_sha=entry.runtime_hash())
    print(json.dumps({"task":args.task,"output":str(output),"freeze":str(freeze_path),
        "required_initial_SAM_roles":ready["roles"]}, indent=2), flush=True)
    if args.prepare_only:
        return freeze
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    return entry.main(["--task", args.task, "sim", "--config", str(config), "--scope", str(scope),
        "--execution-freeze", str(freeze_path), "--output", str(output)])


if __name__ == "__main__":
    main()
