#!/usr/bin/env python3
"""Read-only entry for V22 dual-arm task leaves; no check starts an episode."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prove.xtrainer.io import SafetyError
from multitask_dual_tool_route import TASKS, check_task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("mode", nargs="?", default="check", choices=("check", "sim", "real"))
    args = parser.parse_args(argv)
    report = check_task(args.task)
    if args.mode == "check":
        print(json.dumps(report, indent=2))
        return report
    if args.task == "hanging_mug_bimanual":
        raise SafetyError("hanging_mug_bimanual execution is blocked by its failed hanging outcome and absent task model")
    if args.mode == "real":
        raise SafetyError("no V22 task has same-runtime simulation promotion or hardware qualification")
    raise SafetyError(
        "formal simulation must be launched by the Root-owned per-task one-trial scope/freeze entry; "
        "this leaf does not reserve an episode"
    )


if __name__ == "__main__":
    main()
