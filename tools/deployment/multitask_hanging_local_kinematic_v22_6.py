#!/usr/bin/env python3
"""Dedicated one-command entry for the hanging-mug optional-table repair."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import multitask_articulated_entry_v22 as base
from multitask_hanging_repair_v22_2 import HangingExecutorV22_2, optional_table_role_contract


REPAIR_SOURCES = (Path(__file__), HERE / "multitask_hanging_repair_v22_2.py", HERE / "fair_sim_robot_io.py", HERE / "kinematic_sim_robot_io.py", REPO / "evaluations/multitask_parallel_v22/articulated/server_shared/KINEMATIC_FK_BENCHMARK_ALL.json", REPO / "tests/test_multitask_hanging_repair_v22_2.py", REPO / "tests/test_kinematic_sim_robot_io.py", REPO / "task_packages/multitask_parallel_v22/hanging_mug/run_local_kinematic_v22_6.sh")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--task" in args:
        index = args.index("--task")
        if index + 1 >= len(args) or args[index + 1] != "hanging_mug":
            raise SystemExit("V22.2 repair supports only --task hanging_mug")
    else:
        args[:0] = ["--task", "hanging_mug"]
    if "--scope" not in args and any(mode in args for mode in ("prepare", "sim")):
        args.extend(["--scope", str(REPO / "evaluations/multitask_parallel_v22/articulated/hanging_mug/local_kinematic_v22_6/TRIAL_SCOPE.json")])
    # Patch only during this synchronous invocation; importing this entry cannot
    # alter the shared drawer/stack check or runtime identity.
    original_sources = base.SOURCES
    original_executor = base.ArticulatedExecutorV22
    original_roles = base.role_contract
    original_load = base.backend.load
    base.SOURCES = original_sources + REPAIR_SOURCES
    base.ArticulatedExecutorV22 = HangingExecutorV22_2
    base.role_contract = lambda graph, refs: optional_table_role_contract(original_roles(graph, refs))
    def load(path, name):
        if Path(path).name == "sim_robot_io.py":
            import kinematic_sim_robot_io
            return kinematic_sim_robot_io
        return original_load(path, name)
    base.backend.load = load
    try:
        return base.main(args)
    finally:
        base.SOURCES = original_sources
        base.ArticulatedExecutorV22 = original_executor
        base.role_contract = original_roles
        base.backend.load = original_load


if __name__ == "__main__":
    main()
