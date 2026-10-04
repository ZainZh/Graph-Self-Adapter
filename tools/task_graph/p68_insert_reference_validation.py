#!/usr/bin/env python3
"""P68 bounded insert reference validation runner.

This is a new, isolated runner for the short-insert half of P68.  It reuses the
frozen P61 task-trained visual checkpoint, the frozen P50 action head and the
unchanged P62 translation_size5 update math.  Preparing a freeze binds the
result-free INSERT_SCOPE and all source/model assets; running cells additionally
requires Root's exact freeze approval.  No simulator truth is passed to runtime.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prove.fast_adapter.p61_data import sha256
from prove.fast_adapter.p68_insert_reference_runtime import (
    P68_ARMS, PREFIX_STEPS, p68_candidate_configuration,
)
from tools.task_graph.train_precision_action_policy import dump_new

SCHEMA = "prove.p68-insert-physical-development-freeze.v1"
APPROVAL_SCHEMA = "prove.p68-root-physical-review.v1"
RESULT_SCHEMA = "prove.p68-insert-physical-result.v1"
PRE_RUN_SCHEMA = "prove.p68-insert-physical-pre-run.v1"
STEPS = 240
ENCODER_KEY = "task_trained_lila"


def binding(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha256(p))


def load_scope(path):
    scope = json.loads(Path(path).read_text())
    if scope.get("schema") != "prove.p68-insert-scope.v1":
        raise ValueError("wrong P68 insert scope schema")
    if scope.get("task") != "insert" or scope.get("cells") != 48:
        raise ValueError("P68 insert scope must describe 48 insert cells")
    if tuple(scope.get("arms", ())) != P68_ARMS:
        raise ValueError("P68 insert scope arms differ from reviewed names/order")
    if scope.get("steps_per_episode") != STEPS or scope.get("common_observation_prefix_steps") != PREFIX_STEPS:
        raise ValueError("P68 insert scope step/prefix drift")
    if scope.get("runtime_truth") is not False or scope.get("network_training_updates") != 0:
        raise ValueError("P68 insert scope must be no-truth/no-training")
    conditions = scope.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 12:
        raise ValueError("P68 insert scope must materialize 12 private configs")
    seen = set()
    for condition in conditions:
        cfg = condition.get("evaluator_configuration")
        if cfg is None or cfg.get("task") != "insert":
            raise ValueError("each P68 condition must contain an insert evaluator configuration")
        key = (condition.get("split"), condition.get("condition_seed"), condition.get("family"))
        if key in seen:
            raise ValueError("duplicate P68 condition identity")
        seen.add(key)
    return scope


def cells_from_scope(scope):
    rows = []
    videos = {(v["split"], v["seed"]) for v in scope.get("preselected_videos", [])}
    for condition_index, condition in enumerate(scope["conditions"]):
        for arm in scope["arms"]:
            rows.append(dict(
                task="insert",
                split=condition["split"],
                family=int(condition["family"]),
                condition_seed=int(condition["condition_seed"]),
                condition_index=condition_index,
                encoder=ENCODER_KEY,
                arm=arm,
                video=(condition["split"], condition["condition_seed"]) in videos,
            ))
    if len(rows) != scope["cells"]:
        raise ValueError("scope cell count mismatch")
    return rows


def cell_name(index, cell):
    return f"{index:02d}_{cell['split']}_{cell['condition_seed']}_{cell['arm']}"


def neural_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_closure():
    pending = [
        "tools/task_graph/p68_insert_reference_validation.py",
        "prove/fast_adapter/p68_insert_reference_runtime.py",
        "tests/test_p68_insert_reference.py",
    ]
    found = {}
    while pending:
        relative = pending.pop()
        if relative in found:
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        found[relative] = sha256(path)
        tree = ast.parse(path.read_text())
        modules = []
        package = list(Path(relative).with_suffix("").parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise ValueError("relative import escapes repository package")
                    base = package[: len(package) - node.level + 1]
                    module = ".".join(base + (node.module.split(".") if node.module else []))
                else:
                    module = node.module or ""
                if module:
                    modules.append(module)
                    modules.extend(module + "." + alias.name for alias in node.names)
        for module in modules:
            if not module.startswith(("prove.", "tools.")):
                continue
            parts = module.split(".")
            for i in range(1, len(parts) + 1):
                prefix = "/".join(parts[:i])
                pending.extend([prefix + ".py", prefix + "/__init__.py"])
    return dict(sorted(found.items()))


def require_egl(frozen=None):
    required = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    for key, value in required.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"{key}={value} must be set before Python starts")
    if frozen is not None:
        value = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        if value is None:
            raise RuntimeError("MUJOCO_EGL_DEVICE_ID must be set before Python starts")
        allowed = {str(x) for x in frozen.get("renderer_gpu_ids", [])}
        if value not in allowed:
            raise RuntimeError(f"MUJOCO_EGL_DEVICE_ID={value} outside frozen renderer ids {sorted(allowed)}")


def prepare(output, checkpoint, head, dino_dir, dataset_manifest, insert_scope, plan, *, model_device="cuda:0"):
    require_egl()
    from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
    from prove.task_graph.p59.core import FrozenP50Action
    from tools.task_graph.p50_precision_pilot import BACKEND, package_versions

    scope = load_scope(insert_scope)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    FrozenP50Action(head)
    if Path(plan).resolve() != (ROOT / "evaluations/fast_adapter_p68_cross_task_reference_v01/PLAN.md").resolve():
        raise ValueError("prepare must bind the adopted P68 cross-task plan")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema") != "prove.p61-relation-checkpoint.v1" or payload["training"]["arm"] != ENCODER_KEY:
        raise ValueError("wrong P61 task-trained LiLa checkpoint identity")
    files = dict(
        action_head=binding(head),
        plan=binding(plan),
        insert_scope=binding(insert_scope),
        dataset_manifest=binding(dataset_manifest),
        task_trained_lila=binding(checkpoint),
    )
    for key in ("source", "checkpoint"):
        files["lila_" + key] = binding(payload["lila_binding"][key])
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        files["dino_" + name] = binding(Path(dino_dir) / name)
    manifest = json.loads(Path(dataset_manifest).read_text())
    if files["dino_model.safetensors"]["sha256"] != manifest["dino_weights_sha256"]:
        raise ValueError("runtime DINO differs from P61/P62 training cache")
    frozen = dict(
        schema=SCHEMA,
        status="frozen_before_results",
        phase="bounded_development",
        fresh_test_claim=False,
        task="insert",
        cells=cells_from_scope(scope),
        steps_per_episode=STEPS,
        automatic_retries=int(scope["automatic_retries"]),
        model_threads=int(scope["cpu_threads"]),
        max_workers_per_gpu=int(scope["max_workers_per_gpu"]),
        resource_assignment=scope["resource_assignment"],
        first_cell_alone=bool(scope["first_cell_alone"]),
        renderer_gpu_ids=[2, 3],
        model_device=model_device,
        dino_dir=str(Path(dino_dir).resolve()),
        files=files,
        sources=source_closure(),
        robot_assets=source_assets(),
        packages=package_versions(),
        backend=BACKEND,
        insert_scope=scope,
        conditions=scope["conditions"],
        candidate_configuration=p68_candidate_configuration(),
        runtime_contract=dict(
            inputs="448 RGB, visible role masks, calibrated camera and own pose only; no seed/split/config/truth/corners/PnP",
            common_prefix_steps=PREFIX_STEPS,
            arms=list(P68_ARMS),
            frame_fixed="current visual estimate, z=0 every frame",
            reference_fixed="first public finite reference geometry held fixed after prefix, z=0",
            entry_once="same reference geometry; one post-prefix P62 translation_size5 update at first eligible new feedback with S>=3",
            continuous="same reference geometry; P62 translation_size5 update at every later eligible feedback with S>=3",
            update="unchanged P62 translation_size5, active indices [0,1,2,6,7], 3 Gauss-Newton steps; rotation latent strictly zero",
            orientation="runtime retains P61 learned rotation output; no GT or nominal rotation replacement",
            end_of_skill="runtime exposes reset_skill_state for queue cleanup; final controller target remains in saved trace until episode end",
        ),
        evaluation_contract=dict(
            method="unchanged P47/P62 evaluate_sealed after trajectory seal",
            success="native insert success plus strict mechanical checks as in P62",
            denominator="all48 prespecified cells including aborts; no automatic retries",
            reporting="descriptive bounded development only; not fresh heldout confirmation",
        ),
    )
    dump_new(output / "PHYSICAL_FREEZE.json", frozen)
    for name, digest in frozen["sources"].items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        if sha256(target) != digest:
            raise ValueError("source changed during snapshot")
    return frozen


def validate_freeze(path, approval=None, check_runtime=True):
    frozen = json.loads(Path(path).read_text())
    if frozen.get("schema") != SCHEMA or frozen.get("status") != "frozen_before_results":
        raise ValueError("wrong P68 insert freeze")
    scope = frozen.get("insert_scope")
    if frozen.get("cells") != cells_from_scope(load_scope(frozen["files"]["insert_scope"]["path"])):
        raise ValueError("P68 cells drift from bound scope")
    if (frozen.get("steps_per_episode") != STEPS or frozen.get("candidate_configuration") != p68_candidate_configuration()
            or tuple(scope.get("arms", ())) != P68_ARMS):
        raise ValueError("P68 freeze contract drift")
    if approval is not None:
        reviewed = json.loads(Path(approval).read_text())
        if (reviewed.get("schema") != APPROVAL_SCHEMA or reviewed.get("freeze_sha256") != sha256(path)
                or reviewed.get("physical_execution_authorized") is not True or reviewed.get("root_reviewed") is not True):
            raise ValueError("missing exact-freeze root physical review")
    if check_runtime:
        require_egl(frozen)
        from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
        from tools.task_graph.p50_precision_pilot import package_versions
        if source_closure() != frozen["sources"] or source_assets() != frozen["robot_assets"] or package_versions() != frozen["packages"]:
            raise ValueError("P68 runtime source/assets/packages changed")
        for key, item in frozen["files"].items():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"changed frozen file: {key}")
    return frozen


def condition_for_cell(frozen, cell):
    rows = [c for c in frozen["conditions"] if c["split"] == cell["split"] and int(c["condition_seed"]) == int(cell["condition_seed"])]
    if len(rows) != 1:
        raise ValueError("cell condition missing from frozen scope")
    return copy.deepcopy(rows[0]["evaluator_configuration"])


def run_cell(freeze_path, approval_path, index):
    if approval_path is None:
        raise ValueError("exact root physical review required before simulator construction")
    frozen = validate_freeze(freeze_path, approval_path)
    if not 0 <= index < len(frozen["cells"]):
        raise ValueError("cell outside P68 bounded scope")
    torch.set_num_threads(int(frozen["model_threads"]))
    from prove.fast_adapter.p62_runtime import PUBLIC_POSE_KEYS
    from prove.fast_adapter.p68_insert_reference_runtime import P68InsertReferenceRuntime
    from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
    from prove.task_graph.precision_action_policy import PrecisionActionController, load_frozen_head, encode_state
    from tools.fast_adapter.p45_common import observe_roles
    from tools.fast_adapter.p47_evaluate import snapshot, evaluate_sealed
    from tools.task_graph.p50_precision_pilot import controller_state, domain_excursions

    cell = frozen["cells"][index]
    output = Path(freeze_path).parent / "cells" / cell_name(index, cell)
    output.mkdir(parents=True, exist_ok=False)
    dump_new(output / "PRE_RUN.json", dict(schema=PRE_RUN_SCHEMA, cell=cell, cell_index=index,
                                           freeze=binding(freeze_path), approval=binding(approval_path),
                                           steps=STEPS, automatic_retry=False))
    env = None
    traces, observations, rgb_frames, mask_frames, videos, details = [], [], [], [], [], []
    started = time.monotonic()
    try:
        runtime = P68InsertReferenceRuntime(
            frozen["files"][cell["encoder"]]["path"], frozen["dino_dir"], cell["arm"], device=frozen["model_device"])
        head = load_frozen_head(frozen["files"]["action_head"]["path"])
        models = dict(encoder=runtime.encoder, dino=runtime.vision.model, action_head=head)
        before = {k: neural_state_sha256(v) for k, v in models.items()}
        cfg = condition_for_cell(frozen, cell)
        env = RobotGeometrySkill(cfg, render=True)
        if env.model.nmocap != 0 or env.model.nu < 7:
            raise ValueError("articulated Panda required")
        calibration = env.calibrate_held_geometry()
        if calibration.get("source") != "RGB visible-mask bbox plus known own-tool pose":
            raise ValueError("public held-length calibration required")
        length = float(calibration["length"])
        controller = PrecisionActionController(head, env.start_pos, env.start_R, length=length)
        p, R = env.proprio()
        previous = np.r_[p, R.ravel(), 1.]
        for t in range(STEPS):
            observed, rgb, masks = observe_roles(env)
            public = {key: copy.deepcopy(observed[key]) for key in PUBLIC_POSE_KEYS}
            admitted = runtime.update(public, rgb, masks, t)
            estimate = runtime.estimate()
            proprio = env.proprio()
            observations.append(dict(public, t=t, previous_action=previous.copy(), contour_admitted=admitted))
            if cell["video"] or t == 0:
                rgb_frames.append(rgb.copy())
                mask_frames.append(masks.copy())
            pre_state = controller_state(controller)
            phase = "align" if controller.phase == "observe" else controller.phase
            domain = domain_excursions(encode_state(estimate, proprio, length, phase)) if t >= PREFIX_STEPS and estimate is not None else None
            p, R, closed, stage = controller.command(t, estimate, proprio)
            row = dict(t=t, state=snapshot(env), command_p=p.copy(), command_R=R.copy(), closed=bool(closed), stage=stage,
                       estimate=copy.deepcopy(estimate), proprio=copy.deepcopy(proprio), adaptation=copy.deepcopy(runtime.records[-1]),
                       controller_before=pre_state, controller_after=controller_state(controller), head_domain=domain,
                       neural_action=copy.deepcopy(controller.records[-1]) if controller.records and controller.records[-1]["t"] == t else None,
                       controls_before=env.data.ctrl.copy())
            if cell["video"]:
                videos.append(np.concatenate((env.render_global(), rgb), axis=1))
                details.append(env.render_detail())
            env.step(p, R, closed)
            row.update(controls_after=env.data.ctrl.copy(), robot_tracking=env.robot_tracking_diagnostics())
            traces.append(row)
            previous = np.r_[p, R.ravel(), float(closed)]
        terminal = snapshot(env)
        for name, data in [("TRACE.npy", traces), ("OBS.npy", observations), ("PREDICTIONS.npy", runtime.records)]:
            np.save(output / name, np.array(data, dtype=object), allow_pickle=True)
        np.save(output / "TERMINAL.npy", terminal, allow_pickle=True)
        np.savez_compressed(output / "PUBLIC_PIXELS.npz", rgb=np.asarray(rgb_frames), visible_masks=np.asarray(mask_frames))
        filenames = ["TRACE.npy", "OBS.npy", "PREDICTIONS.npy", "TERMINAL.npy", "PRE_RUN.json", "PUBLIC_PIXELS.npz"]
        if cell["video"]:
            for name, data in [("FRAMES.npz", videos), ("DETAIL_FRAMES.npz", details)]:
                np.savez_compressed(output / name, frames=np.asarray(data))
                filenames.append(name)
        after = {k: neural_state_sha256(v) for k, v in models.items()}
        if before != after:
            raise RuntimeError("frozen weights changed before evaluator")
        validate_freeze(freeze_path, approval_path)
        dump_new(output / "SEAL.json", dict(schema="prove.p68-insert-trajectory-seal.v1", freeze=binding(freeze_path),
                                            files={name: sha256(output / name) for name in filenames},
                                            neural_parameters_before=before, neural_parameters_after=after))
        facts = evaluate_sealed(env, traces, terminal, output)
        result = dict(schema=RESULT_SCHEMA, cell=cell, cell_index=index, **facts, steps=STEPS,
                      seal_sha256=sha256(output / "SEAL.json"), frozen_neural_weights_unchanged=True,
                      gradient_steps=sum(r["U"] for r in runtime.records), update_events=runtime.update_events,
                      admitted_contour_frames=sum(r["admitted"] for r in runtime.records),
                      inactive_rotation_z_exact_zero=all(np.array_equal(r["z_after"][3:6], np.zeros(3)) for r in runtime.records),
                      reference_t=runtime.reference_t, runtime_public_fields=list(PUBLIC_POSE_KEYS),
                      runtime_pixel_inputs=["rgb", "actor_mask", "held_object_mask", "goal_mask"],
                      neural_precision_commands=sum(r["neural_action"] is not None for r in traces),
                      final_controller_phase=controller.phase,
                      tool_calibration={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in calibration.items()},
                      elapsed_seconds=time.monotonic() - started, fresh_test_claim=False)
        runtime.reset_skill_state()
        dump_new(output / "RESULT.json", result)
        return result
    except Exception as exc:
        for name, data in [("TRACE_PARTIAL.npy", traces), ("OBS_PARTIAL.npy", observations)]:
            np.save(output / name, np.array(data, dtype=object), allow_pickle=True)
        partial = {name: sha256(output / name) for name in ("TRACE_PARTIAL.npy", "OBS_PARTIAL.npy")}
        if rgb_frames:
            np.savez_compressed(output / "PUBLIC_PIXELS_PARTIAL.npz", rgb=np.asarray(rgb_frames), visible_masks=np.asarray(mask_frames))
            partial["PUBLIC_PIXELS_PARTIAL.npz"] = sha256(output / "PUBLIC_PIXELS_PARTIAL.npz")
        if env is not None:
            np.save(output / "STATE_AT_ABORT.npy", snapshot(env), allow_pickle=True)
            partial["STATE_AT_ABORT.npy"] = sha256(output / "STATE_AT_ABORT.npy")
        dump_new(output / "ABORT.json", dict(cell=cell, type=type(exc).__name__, reason=str(exc),
                                             completed_steps=len(traces), partial_files=partial,
                                             automatic_retry=False, task_success_denominator_included=True))
        raise
    finally:
        if env is not None:
            env.close()


def run_shard(freeze_path, approval_path, start, stop, progress):
    frozen = validate_freeze(freeze_path, approval_path)
    if not (0 <= start <= stop <= len(frozen["cells"])):
        raise ValueError("invalid shard bounds")
    progress_path = Path(progress)
    rows = []
    stop_path = Path(freeze_path).parent / "STOP_PROGRAM_ERROR.json"
    for index in range(start, stop):
        if stop_path.exists():
            raise RuntimeError(f"P68 insert task stop file already exists: {stop_path}")
        stamp = dict(index=index, cell=frozen["cells"][index], status="started", t=time.time())
        rows.append(stamp)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(rows, indent=2) + "\n")
        try:
            result = run_cell(freeze_path, approval_path, index)
        except Exception as exc:
            rows[-1] = dict(index=index, cell=frozen["cells"][index], status="program_error",
                            type=type(exc).__name__, reason=str(exc), t=time.time())
            progress_path.write_text(json.dumps(rows, indent=2) + "\n")
            dump_new(stop_path, dict(schema="prove.p68-insert-stop-program-error.v1",
                                     freeze=binding(freeze_path), approval=binding(approval_path),
                                     index=index, cell=frozen["cells"][index], type=type(exc).__name__,
                                     reason=str(exc), progress=str(progress_path),
                                     policy="stop all new P68 insert submissions; keep already running cells to landing"))
            raise
        rows[-1] = dict(index=index, cell=frozen["cells"][index], status="complete", result_sha256=sha256(Path(freeze_path).parent / "cells" / cell_name(index, frozen["cells"][index]) / "RESULT.json"), success=bool(result.get("success", False)), t=time.time())
        progress_path.write_text(json.dumps(rows, indent=2) + "\n")
    return dict(schema="prove.p68-insert-shard-progress.v1", start=start, stop=stop, completed=len(rows), progress=str(progress_path))


def summarize(path):
    frozen = validate_freeze(path, check_runtime=False)
    rows = []
    for i, cell in enumerate(frozen["cells"]):
        directory = Path(path).parent / "cells" / cell_name(i, cell)
        result = directory / "RESULT.json"
        abort = directory / "ABORT.json"
        if result.exists():
            record = json.loads(result.read_text())
            seal = json.loads((directory / "SEAL.json").read_text())
            if record["cell"] != cell or record["seal_sha256"] != sha256(directory / "SEAL.json") or seal["freeze"]["sha256"] != sha256(path):
                raise ValueError("result seal drift")
            if seal["neural_parameters_before"] != seal["neural_parameters_after"]:
                raise ValueError("weight drift")
            for name, digest in seal["files"].items():
                if sha256(directory / name) != digest:
                    raise ValueError("trajectory changed")
            rows.append(dict(cell=cell, status="complete", result=record))
        elif abort.exists():
            rows.append(dict(cell=cell, status="abort", abort=json.loads(abort.read_text())))
        else:
            rows.append(dict(cell=cell, status="not_complete"))
    return dict(schema="prove.p68-insert-physical-summary.v1", expected=len(frozen["cells"]),
                completed=sum(r["status"] == "complete" for r in rows),
                aborted=sum(r["status"] == "abort" for r in rows), cells=rows, fresh_test_claim=False)


def main():
    p = argparse.ArgumentParser()
    subs = p.add_subparsers(dest="mode", required=True)
    prep = subs.add_parser("prepare")
    for name in ("output", "checkpoint", "head", "dino-dir", "dataset-manifest", "insert-scope", "plan"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--model-device", default="cuda:0")
    run = subs.add_parser("run-cell")
    run.add_argument("--freeze", required=True)
    run.add_argument("--approval", required=True)
    run.add_argument("--index", required=True, type=int)
    shard = subs.add_parser("run-shard")
    shard.add_argument("--freeze", required=True)
    shard.add_argument("--approval", required=True)
    shard.add_argument("--start", required=True, type=int)
    shard.add_argument("--stop", required=True, type=int)
    shard.add_argument("--progress", required=True)
    summary = subs.add_parser("summarize")
    summary.add_argument("--freeze", required=True)
    args = p.parse_args()
    if args.mode == "prepare":
        value = prepare(args.output, args.checkpoint, args.head, args.dino_dir, args.dataset_manifest, args.insert_scope, args.plan, model_device=args.model_device)
    elif args.mode == "run-cell":
        value = run_cell(args.freeze, args.approval, args.index)
    elif args.mode == "run-shard":
        value = run_shard(args.freeze, args.approval, args.start, args.stop, args.progress)
    else:
        value = summarize(args.freeze)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
