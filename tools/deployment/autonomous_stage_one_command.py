#!/usr/bin/env python3
"""Reserve, freeze and run one fully autonomous V24 cup simulation."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import autonomous_stage_entry as entry
from direct_white_route import direct_route_graph
from macro_one_command import reserve
from prove.xtrainer.agent import digest, write
from prove.xtrainer.io import SafetyError


OWNER = REPO / "evaluations/cup_autonomous_stage_v24/repair_white_object_yaw_v01"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sim",), nargs="?", default="sim")
    parser.add_argument("--config", type=Path, default=REPO / "evaluations/gemini_graph_v19/CONFIG_PROPOSED.json")
    parser.add_argument("--scope", type=Path, default=OWNER / "TRIAL_SCOPE.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    config_source = args.config.resolve(); scope = args.scope.resolve()
    cfg = copy.deepcopy(json.loads(config_source.read_text()))
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    output = OWNER / "runs" / stamp; control = OWNER / "control" / stamp
    graph = json.loads((Path(cfg["portable_package"]) / "graph/task_graph.json").read_text())
    routed = direct_route_graph(graph)
    cfg.pop("qualified_handle_candidate", None); cfg.pop("handle_route_graph", None)
    cfg["direct_route_graph"] = str(control / "TASK_GRAPH.json")
    cfg["formal_run_reservation"] = str(control / "FORMAL_RUN_RESERVATION.json")
    cfg["runtime"]["vision"].update(prompt_frame_output=str(output / "INITIAL_RGB.png"),
                                     prompt_reply_path=str(output / "INITIAL_BOXES.json"),
                                     prompt_frame_sha256=None, role_boxes={},
                                     live_view_root=str(output / "LIVE"),
                                     autonomous_grounding_key_file=cfg["runtime"]["stage_judge"]["key_file"],
                                     autonomous_grounding_maximum_calls=2,
                                     autonomous_grounding_evidence_root=str(output / "AUTONOMOUS_GROUNDING"))
    cfg["runtime"]["autonomous_grounding"] = {"maximum_calls": 2}
    cfg["runtime"]["live_view"] = {"host": "127.0.0.1", "port": 18724}
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.check:
        from prove.xtrainer.portable import portable_check
        report, _, _ = portable_check(Path(cfg["portable_package"]), cfg["runtime"])
        import OpenGL.EGL as egl
        if not hasattr(egl, "EGLDeviceEXT"):
            raise SafetyError("EGL rendering dependency missing")
        declared = json.loads(scope.read_text())
        if declared.get("execution_authorized") is not True or declared.get("hardware_motion_jobs") != 0:
            raise SafetyError("formal simulator-only scope absent")
        result = {"mode": "sim", "scope": str(scope), "config_source": str(config_source),
                  "planned_fresh_output": str(output), "route_stages": len(routed["stages"]),
                  "code_trajectory_stages": 17, "precision_stages": 5, "semantic_stages": 3,
                  "runtime_sha256": entry.runtime_hash(), "hardware_connections": 0,
                  "episode_started": False, "ledger_changed": False, "manual_input": False,
                  "package_bindings_verified": True, "dependencies": report["dependencies"],
                  "EGLDeviceEXT_available": True, "scope_budget": declared["maximum_full_simulator_episodes"]}
        print(json.dumps(result, indent=2)); return result
    if control.exists() or output.exists():
        raise SafetyError("fresh formal output/control required")
    control.mkdir(parents=True); write(control / "TASK_GRAPH.json", routed)
    reservation = reserve(scope, OWNER / "FORMAL_RUN_LEDGER.json", output, control)
    write(control / "FORMAL_RUN_RESERVATION.json", reservation)
    config = control / "CONFIG.json"; write(config, cfg)
    _, _, _, bound = entry.bindings(config, cfg, "sim")
    bound.update({str(config_source): digest(config_source), str(scope): digest(scope),
                  str(Path(__file__).resolve()): digest(__file__)})
    freeze = {"schema": "graph-self-adapter.shared-cup-freeze.v1", "status": "result_free_frozen",
              "scope_sha256": digest(scope), "runtime_sha256": entry.runtime_hash(),
              "bindings": [{"path": path, "sha256": sha} for path, sha in sorted(bound.items())],
              "jobs": [{"task": entry.TASK, "mode": "sim", "output": str(output)}],
              "formal_reservation": str(control / "FORMAL_RUN_RESERVATION.json"),
              "scope_not_cloned_or_reset": True, "hardware_motion_jobs": 0}
    freeze_path = control / "EXECUTION_FREEZE.json"; write(freeze_path, freeze)
    entry.base.guard(scope, freeze_path, bound, mode="sim", output=output,
                     runtime_sha=entry.runtime_hash())
    result = {"output": str(output), "config": str(config), "scope": str(scope),
              "freeze": str(freeze_path), "formal_run_index": reservation["run_index"],
              "initial_frame": str(output / "INITIAL_RGB.png"),
              "automatic_grounding_reply": str(output / "INITIAL_BOXES.json"),
              "manual_input": False, "live_view_url": "http://127.0.0.1:18724/",
              "supported_boundaries_backend": "OpenRouter", "supported_boundary_API_budget": 6}
    print(json.dumps(result, indent=2), flush=True)
    if args.prepare_only:
        return result
    return entry.main(["sim", "--config", str(config), "--scope", str(scope),
                       "--execution-freeze", str(freeze_path), "--output", str(output)])


if __name__ == "__main__":
    main()
