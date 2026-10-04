#!/usr/bin/env python3
"""P58 four-condition held-hammer GT-teacher engineering qualification.

`freeze` reads old opened metadata and source bytes only. `run` is the sole
physical entrypoint and never reports a learned policy or adapter score.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
P58 = ROOT/"evaluations/fast_adapter_p58_functional_geometry_v01"
P30 = ROOT/"evaluations/fast_adapter_p30_online_v02/hammer-block/data"
META = P30/"meta_expert_048.json"
RAW = P30/"meta_expert_048.pt"
PLAN = P58/"PLAN.md"
SOURCE_FILES = (
    "tools/fast_adapter/p58_hammer_qualification.py",
    "prove/fast_adapter/held_hammer_env.py",
    "prove/fast_adapter/object_interaction_env.py",
)
SCHEMA = "p58.held-hammer-qualification-freeze.v1"
SOURCE_ROWS = (0, 1, 2, 3)
PREPARATION_BUDGET = 120
EXECUTION_BUDGET = 200
MASS = .35
FRICTION = 1.1
TARGET_BLOCK_HALFSIZE = (.032, .018, .032)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _opened_conditions():
    meta = read(META)
    raw = torch.load(RAW, map_location="cpu", weights_only=True)
    if len(meta["configs"]) < 4 or len(raw["reset_seeds"]) < 4:
        raise ValueError("P30v2 opened development source lacks four conditions")
    conditions = []
    for index in SOURCE_ROWS:
        original = meta["configs"][index]
        if original.get("task") != "hammer-block" or original.get("object_seed") != 3043048+index:
            raise ValueError("Changed P35/P30v2 development source row")
        seed = int(raw["reset_seeds"][index])
        if seed != 3044048+index:
            raise ValueError("Changed P30v2 reset seed")
        config = dict(original)
        config.update(mass=MASS, friction=FRICTION)
        if set(config) != {"task", "object_seed", "split", "length", "shaft", "head",
                           "head_width", "mass", "friction"}:
            raise ValueError("Unexpected hammer configuration field")
        conditions.append(dict(index=index, source_row=index,
            source_config=original, config=config, reset_seed=seed))
    return conditions


def freeze():
    """No environment construction, reset, step, rendering, or vision work."""
    return dict(schema=SCHEMA, scope="opened_P35_development_first_four_meta_expert_048",
        source_meta=dict(path=str(META), sha256=sha(META)),
        source_raw=dict(path=str(RAW), sha256=sha(RAW)),
        plan=dict(path=str(PLAN), sha256=sha(PLAN)),
        sources={name:sha(ROOT/name) for name in SOURCE_FILES},
        conditions=_opened_conditions(), fixed_mass=MASS, fixed_friction=FRICTION,
        target=dict(source="original ObjectEnv reset seed and unchanged XML",
                    block_halfsize_m=list(TARGET_BLOCK_HALFSIZE),
                    base_xy_reset_range_m=[[.15,.24],[.72,.79]],
                    dimensions_changed=False),
        preparation=dict(max_steps=PREPARATION_BUDGET, bilateral_handle=True,
                         min_lift_m=.055, stable_streak=5),
        execution=dict(max_steps=EXECUTION_BUDGET, teacher="HeldHammerEnv.teacher GT oracle",
                       grip=.9, success="head contact while held, target slide >=.045m, no direct arm contact, hold5"),
        denominator=4, automatic_retries=0, learned_policy_scored=False,
        simulator_steps_before_freeze=0)


def verify(binding):
    if binding.get("schema") != SCHEMA or binding.get("scope") != "opened_P35_development_first_four_meta_expert_048":
        raise ValueError("Wrong P58 hammer qualification scope")
    if (binding.get("fixed_mass") != MASS or binding.get("fixed_friction") != FRICTION or
            binding.get("denominator") != 4 or binding.get("automatic_retries") != 0 or
            binding.get("learned_policy_scored") is not False or
            binding.get("simulator_steps_before_freeze") != 0):
        raise ValueError("Changed qualification constants")
    for name, path in (("source_meta", META), ("source_raw", RAW), ("plan", PLAN)):
        if binding[name] != dict(path=str(path), sha256=sha(path)):
            raise ValueError("Frozen source differs: " + name)
    if binding.get("sources") != {name:sha(ROOT/name) for name in SOURCE_FILES}:
        raise ValueError("Frozen P58/P30 source changed")
    if binding.get("conditions") != _opened_conditions():
        raise ValueError("Frozen four conditions differ from old opened source")
    if binding != freeze():
        raise ValueError("Qualification budget, target or oracle contract changed")
    return binding["conditions"]


def _environment(config):
    from prove.fast_adapter.held_hammer_env import HeldHammerEnv
    return HeldHammerEnv(config, render=False, preparation_budget=PREPARATION_BUDGET)


def _state_bundle(states, actions):
    if not states:
        return None
    result = {key:np.stack([state[key] for state in states]) for key in states[0]}
    result["actions"] = np.asarray(actions, dtype=np.float32).reshape(-1,4)
    return result


def _actual_target(env):
    model = env.env.model
    xy = np.asarray(model.body("target").pos[:2], dtype=float).copy()
    size = np.asarray(model.geom("target_block").size, dtype=float).copy()
    if not (.15 <= xy[0] <= .24 and .72 <= xy[1] <= .79):
        raise ValueError("Original reset target xy outside source range")
    if not np.allclose(size, TARGET_BLOCK_HALFSIZE, rtol=0, atol=1e-12):
        raise ValueError("Original target block dimensions changed")
    if not np.isclose(float(model.body("actor").mass), MASS, atol=1e-8):
        raise ValueError("Actor mass differs from frozen .35 kg")
    if not np.isclose(float(model.geom("actor_head").friction[0]), FRICTION, atol=1e-8):
        raise ValueError("Hammer friction differs from frozen 1.1")
    return dict(base_xy_m=xy.tolist(), block_halfsize_m=size.tolist())


def run_cell(condition, output):
    """One attempt per frozen condition; preserves preparation/task failures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/"CELL.json", condition)
    env = None
    task_actions, task_states, metrics = [], [], []
    row = dict(index=condition["index"], source_row=condition["source_row"],
               reset_seed=condition["reset_seed"], attempt=1, automatic_retry=False,
               learned_policy_scored=False, oracle_only=True)
    try:
        from prove.fast_adapter.held_hammer_env import PreparationFailed
        env = _environment(condition["config"])
        try:
            env.reset(condition["reset_seed"])
        except PreparationFailed:
            row.update(status="preparation_failed", preparation_passed=False,
                       teacher_success=False, executed_task_steps=0)
        else:
            row["preparation_passed"] = bool(env.preparation["passed"])
            if not row["preparation_passed"]:
                raise ValueError("HeldHammerEnv returned an unqualified preparation")
            initial_target = _actual_target(env)
            row["target"] = initial_target
            task_states.append(env.state())
            for step in range(EXECUTION_BUDGET):
                command = np.asarray(env.teacher(), dtype=np.float32)
                if command.shape != (4,) or not np.isfinite(command).all():
                    raise ValueError("Invalid GT teacher action")
                env.step(command)
                actual = np.asarray(env.last_executed_action, dtype=np.float32)
                if actual.shape != (4,) or not np.isclose(actual[3], .9, rtol=0, atol=1e-7):
                    raise ValueError("Held hammer gripper did not remain fixed closed")
                task_actions.append(actual.copy())
                task_states.append(env.state())
                current = dict(env.metrics[-1])
                if not all(np.isfinite(v) for v in current.values()):
                    raise ValueError("Nonfinite original success metric")
                metrics.append(current)
                if current["success"]:
                    if (not current["head_contact_seen"] or current["target_displacement"] < .045
                            or current["target_direct_contact"] or current["hold_streak"] < 5):
                        raise ValueError("Old native success metric inconsistent")
                    break
            if _actual_target(env) != initial_target:
                raise ValueError("Original target base xy or dimensions changed during task")
            row.update(status="completed", teacher_success=any(bool(m["success"]) for m in metrics),
                       executed_task_steps=len(task_actions),
                       first_success_step=next((i+1 for i,m in enumerate(metrics) if m["success"]), None),
                       final_metric=metrics[-1] if metrics else None)
    except Exception as error:
        row.update(status="engineering_error", teacher_success=False,
                   executed_task_steps=len(task_actions), error=f"{type(error).__name__}: {error}")
    finally:
        if env is not None:
            prep = env.preparation
            if prep is not None:
                row.update(preparation_steps=prep["steps"],
                           preparation_passed=bool(prep["passed"]),
                           preparation_final_metric=prep["metrics"][-1] if prep["metrics"] else None)
                bundle = _state_bundle(prep["states"], prep["actions"])
                if bundle is not None:
                    np.savez_compressed(output/"PREPARATION.npz", **bundle)
                write(output/"PREPARATION_METRICS.json", prep["metrics"])
            if task_states:
                np.savez_compressed(output/"INTERACTION.npz",
                                    **_state_bundle(task_states, task_actions))
                write(output/"INTERACTION_METRICS.json", metrics)
            env.close()
    write(output/"RESULT.json", row)
    sealed = {file.name:sha(file) for file in output.iterdir() if file.is_file()}
    write(output/"SEAL.json", dict(schema="p58.hammer-qualification-cell-seal.v1",
                                   cell_index=condition["index"], files=sealed))
    return row


def run(binding_path, output):
    binding_path = Path(binding_path).resolve()
    binding = read(binding_path)
    conditions = verify(binding)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/"FREEZE.json", binding)
    rows = [run_cell(condition, output/f"condition_{condition['index']:02d}")
            for condition in conditions]
    summary = dict(schema="p58.held-hammer-qualification-result.v1",
        denominator=4, conditions=rows, preparation_passed=sum(row.get("preparation_passed") is True for row in rows),
        oracle_success=sum(row["teacher_success"] is True for row in rows),
        engineering_errors=sum(row["status"] == "engineering_error" for row in rows),
        automatic_retries=0, learned_policy_scored=False,
        qualification_passed=all(row["status"] == "completed" and row["teacher_success"] for row in rows))
    write(output/"RESULT.json", summary)
    write(output/"SEAL.json", dict(schema="p58.hammer-qualification-seal.v1",
        freeze_sha256=sha(binding_path),
        files={"FREEZE.json":sha(output/"FREEZE.json"),
               "RESULT.json":sha(output/"RESULT.json"),
               **{f"condition_{i:02d}/SEAL.json":sha(output/f"condition_{i:02d}/SEAL.json")
                  for i in SOURCE_ROWS}}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("freeze", help="Freeze four opened source conditions; no physics")
    prep.add_argument("--output", type=Path, required=True)
    launch = commands.add_parser("run", help="Run four GT-oracle engineering qualifications")
    launch.add_argument("--binding", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        write(args.output, freeze())
    else:
        result = run(args.binding, args.output)
        print(json.dumps({key:result[key] for key in (
            "denominator", "preparation_passed", "oracle_success", "engineering_errors",
            "qualification_passed")}, sort_keys=True))


if __name__ == "__main__":
    main()
