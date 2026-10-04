#!/usr/bin/env python3
"""Result-free P60-B2 validation of public outer-contour latent updates.

The frozen P45 insert/G/seed11 model still supplies corner/PnP initialization,
current-frame encoding and its learned rank-eight decoder. P60-B2 keeps B1's
unordered outer-mask contour and changes only the optimizer's curvature units. Both arms use the same
frozen P50 action head and 240-step Panda task.  Physics requires a root review
receipt bound to the exact freeze produced by ``prepare``.

The shell must set MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl and
MUJOCO_EGL_DEVICE_ID=0 before starting this program.  This module checks those
values before any helper is allowed to import MuJoCo; it never sets them late.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from prove.task_graph.precision_action_policy import (
    PrecisionActionController, encode_state, load_frozen_head,
)
from tools.task_graph.p50_precision_pilot import (
    ADAPTER_SHA, BACKEND, HEAD_SHA, controller_state, domain_excursions,
    package_versions, shadow_command, validate_offline_evidence,
)
from tools.task_graph.p60_precision_validation import (
    _binding, _bound_path, _validate_p50_freeze,
    _validate_p59_contact_diagnostic, source_closure as base_source_closure,
)
from tools.task_graph.train_precision_action_policy import dump_new, sha

SCHEMA = "prove.p60-contour-scaled-development-freeze.v1"
MODEL_CONTRACT_SCHEMA = "prove.p60-contour-scaled-model-contract.v1"
APPROVAL_SCHEMA = "prove.p60-contour-scaled-root-physical-review.v1"
RESULT_SCHEMA = "prove.p60-contour-scaled-cell-result.v1"
SEAL_SCHEMA = "prove.p60-contour-scaled-trajectory-seal.v1"
SUMMARY_SCHEMA = "prove.p60-contour-scaled-development-summary.v1"
RUNNER = "tools/task_graph/p60_contour_scaled_validation.py"
TEST = "tests/test_p60_contour_scaled_validation.py"
CONTOUR_SOURCE = "prove/fast_adapter/p60_contour_scaled.py"
CONTOUR_TEST = "tests/test_p60_contour_scaled.py"
BASE_CONTOUR_SOURCE = "prove/fast_adapter/p60_contour.py"
BASE_CONTOUR_TEST = "tests/test_p60_contour.py"
OFFLINE_REPLAY_SOURCE = "tools/fast_adapter/p60_contour_scaled_replay.py"
OFFLINE_AUDIT_SOURCE = "tools/fast_adapter/p60_contour_scaled_p50_audit.py"
PLAN_RELATIVE = "evaluations/fast_adapter_p60_observation_adaptation_v01/PLAN.md"
AMENDMENT_RELATIVE = "evaluations/fast_adapter_p60_observation_adaptation_v01/AMENDMENT_B2_CURVATURE_SCALE.md"
STEPS = 240
ARMS = ("contour_scaled_fixed", "contour_scaled_continuous")
ARM_MODES = {"contour_scaled_fixed": "fixed", "contour_scaled_continuous": "continuous"}
SPLITS = (("id", 605100), ("geometry", 605200))
_EGL_VALIDATED_BEFORE_MUJOCO = False


def _require_egl_environment(gpu=0):
    global _EGL_VALIDATED_BEFORE_MUJOCO
    required = {
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
    }
    wrong = {name: os.environ.get(name) for name, expected in required.items()
             if os.environ.get(name) != expected}
    if wrong:
        raise RuntimeError(
            "P60-B2 requires shell EGL variables before Python starts: "
            "MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0")
    if "mujoco" in sys.modules and not _EGL_VALIDATED_BEFORE_MUJOCO:
        raise RuntimeError("MuJoCo was imported before P60-B2 EGL validation")
    _EGL_VALIDATED_BEFORE_MUJOCO = True
    return required


def cells():
    """Eight prespecified development conditions x two contour arms."""
    return [dict(task="insert", split=split, condition_seed=base + offset,
                 controller="p50_frozen_neural", arm=arm,
                 adapter_mode=ARM_MODES[arm], rep="geometry", model_seed=11,
                 video=offset == 0)
            for split, base in SPLITS for offset in range(4) for arm in ARMS]


def cell_name(index, cell):
    return f"{index:02d}_{cell['split']}_{cell['condition_seed']}_{cell['arm']}"


def source_closure():
    result = base_source_closure()
    for relative in (RUNNER, TEST, CONTOUR_SOURCE, CONTOUR_TEST,
                     BASE_CONTOUR_SOURCE, BASE_CONTOUR_TEST,
                     OFFLINE_REPLAY_SOURCE, OFFLINE_AUDIT_SOURCE):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required P60-B2 source missing: {relative}")
        result[relative] = sha(path)
    return dict(sorted(result.items()))


def _model_contract(adapter, head, plan, amendment, b2_offline_audit):
    return {
        "schema": MODEL_CONTRACT_SCHEMA,
        "status": "frozen_before_results",
        "candidate": "B1 outer-contour candidate with one curvature-scale optimizer correction",
        "scope": "Panda insert development on eight declared P60 conditions",
        "plan": _binding(plan),
        "amendment": _binding(amendment),
        "offline_qualification": _binding(b2_offline_audit),
        "models": {"p45_geometry": _binding(adapter), "p50_action_head": _binding(head)},
        "arms": ARM_MODES,
        "runtime_input": {
            "geometry": "P45 public observation plus current rendered slot mask",
            "contour": "all visible mask contours -> outer convex hull; closed opening required",
            "truth": "no target pose, size, simulator success, force, or future observation",
        },
        "adaptation": {
            "latent": "persistent z8 only",
            "history": "all through 32, then first 12 plus latest 20 admitted observations",
            "continuous": "three damped autograd Gauss-Newton steps per admitted observation",
            "fixed": "z fixed; current P45 visual encoding remains active",
            "loss": "bidirectional point-to-segment Chamfer on projected decoded perimeter",
            "local_linearization": "nearest cyclic segment assignment fixed locally",
            "solve": "H=J^T J/N, g=autograd; s=max(trace(H)/8,1e-12); solve(H/s+1e-4 I,g/s)",
            "damping_after_curvature_normalization": 1e-4,
            "step_scale": .65, "latent_step_norm_limit": 1.0,
        },
        "control": "unchanged frozen P50 PrecisionActionController for every valid estimate",
        "budget": {"steps_per_episode": STEPS, "automatic_retries": 0, "cells": 16},
        "unchanged_from_b1": ["residual", "measurements", "encoder", "decoder", "history",
                              "three inner steps", "step scale", "step clip", "frames", "P50 head"],
        "excluded": ["old ray metric at runtime", "ground truth", "batch_fit physical arm",
                     "pure-contour initialization", "model training", "parameter search",
                     "fresh-test claim"],
    }


def _validate_b2_offline_audit(path):
    record = json.loads(Path(path).read_text())
    rows = record.get("summary", [])
    seal = Path(path).parent / "PREDICTION_SEAL.json"
    if (record.get("schema") != "prove.p60-contour-scaled-p50-postseal-audit.v1"
            or record.get("physics_integration_steps") != 0
            or record.get("physical_success_claim") is not False
            or record.get("historical_trajectories") != 4
            or record.get("underlying_object_conditions") != 2
            or len(rows) != 4
            or any(row.get("direction_cells") != 12
                   or row.get("online_better_world_position_cells") != 12
                   or row.get("positive_projection_cells") != 12 for row in rows)
            or not seal.is_file()
            or record.get("prediction_seal", {}).get("sha256") != sha(seal)):
        raise ValueError("P60-B2 requires the sealed four-trajectory zero-physics qualification")
    return record


def freeze_pilot(output, adapter, head, offline_query, p50_freeze,
                 p59_contact_diagnostic, plan, amendment, b2_offline_audit):
    """Create the result-free B2 contract/freeze; run no simulator."""
    egl = _require_egl_environment(0)
    from prove.fast_adapter.geometry3d_env import configuration
    from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("P60-B2 freeze output must be empty or new")
    if Path(plan).resolve() != (ROOT / PLAN_RELATIVE).resolve():
        raise ValueError("P60-B2 must bind the adopted P60 plan")
    if Path(amendment).resolve() != (ROOT / AMENDMENT_RELATIVE).resolve():
        raise ValueError("P60-B2 must bind the adopted curvature-scale amendment")
    _validate_p50_freeze(p50_freeze, adapter, head)
    _validate_p59_contact_diagnostic(p59_contact_diagnostic)
    _validate_b2_offline_audit(b2_offline_audit)
    offline_paths = validate_offline_evidence(head, offline_query)
    contract = _model_contract(adapter, head, plan, amendment, b2_offline_audit)
    dump_new(output / "MODEL_CONTRACT.json", contract)
    artifacts = dict(adapter=adapter, action_head=head, p50_physical_freeze=p50_freeze,
                     p59_contact_diagnostic=p59_contact_diagnostic, plan=plan,
                     amendment=amendment,
                     b2_offline_audit=b2_offline_audit,
                     model_contract=output / "MODEL_CONTRACT.json", **offline_paths)
    frozen = {
        "schema": SCHEMA, "status": "frozen", "phase": "development",
        "fresh_test_claim": False, "root_review_required": True,
        "task": "insert", "cells": cells(), "steps_per_episode": STEPS,
        "automatic_retries": 0, "backend": BACKEND, "model_device": "cpu",
        "model_threads": 1, "renderer_gpu_ids": [0], "physics_workers_maximum": 1,
        "required_shell_environment": egl, "sources": source_closure(),
        "packages": package_versions(), "robot_assets": source_assets(),
        "files": {name: _binding(path) for name, path in artifacts.items()},
        "conditions": [dict(split=split, condition_seed=seed,
                            evaluator_configuration=configuration("insert", seed, split))
                       for split, base in SPLITS for seed in range(base, base + 4)],
        "input_contract": {
            "update_call": "adapter.update(obs, t, masks[2]) on every frame",
            "slot_mask": "actual public rendered role mask; outer hull only",
            "held_length": "env.calibrate_held_geometry public RGB visible-mask calibration",
            "evaluator_truth": "sealed separately and never supplied to runtime",
        },
        "control_contract": {
            "command_budget": STEPS, "gripper": "closed preheld peg",
            "missing_geometry": "P50 holds measured EEF pose", "no_force_change": True,
            "action": "same frozen P50 neural action head in both arms",
        },
        "evaluation_contract": {
            "method": "unchanged P47 evaluate_sealed after trajectory/model seal",
            "denominator": "all 16 prespecified development cells including aborts",
            "videos": "all 240 frames for both arms at first seed of each split only",
            "retries": "none", "independent_success_rate_claim": False,
        },
    }
    dump_new(output / "PHYSICAL_FREEZE.json", frozen)
    snapshot_dir = output / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=False)
    for name, digest in frozen["sources"].items():
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        if sha(target) != digest:
            raise ValueError("Source changed while creating P60-B2 snapshot")
    return frozen


def validate_freeze(path, approval=None, check_runtime=True):
    frozen = json.loads(Path(path).read_text())
    if (frozen.get("schema") != SCHEMA or frozen.get("status") != "frozen"
            or frozen.get("phase") != "development" or frozen.get("fresh_test_claim") is not False
            or frozen.get("cells") != cells() or frozen.get("backend") != BACKEND
            or frozen.get("steps_per_episode") != STEPS or frozen.get("automatic_retries") != 0
            or frozen.get("model_device") != "cpu" or frozen.get("model_threads") != 1
            or frozen.get("renderer_gpu_ids") != [0]
            or frozen.get("physics_workers_maximum") != 1
            or frozen.get("required_shell_environment") != {
                "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "MUJOCO_EGL_DEVICE_ID": "0"}):
        raise ValueError("P60-B2 scope differs from the fixed 16-cell contour design")
    if approval is not None:
        receipt = json.loads(Path(approval).read_text())
        if (receipt.get("schema") != APPROVAL_SCHEMA
                or receipt.get("root_reviewed") is not True
                or receipt.get("physical_execution_authorized") is not True
                or receipt.get("freeze_sha256") != sha(path)):
            raise ValueError("P60-B2 physics requires root review bound to this exact freeze")
    if check_runtime:
        _require_egl_environment(0)
        if frozen["sources"] != source_closure() or frozen["packages"] != package_versions():
            raise ValueError("P60-B2 runtime source or package closure changed")
        from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
        if frozen["robot_assets"] != source_assets():
            raise ValueError("P60-B2 Panda assets changed")
        for name, item in frozen["files"].items():
            if sha(_bound_path(item)) != item["sha256"]:
                raise ValueError(f"Frozen P60-B2 artifact changed: {name}")
        contract = json.loads(_bound_path(frozen["files"]["model_contract"]).read_text())
        if contract != _model_contract(_bound_path(frozen["files"]["adapter"]),
                                       _bound_path(frozen["files"]["action_head"]),
                                       _bound_path(frozen["files"]["plan"]),
                                       _bound_path(frozen["files"]["amendment"]),
                                       _bound_path(frozen["files"]["b2_offline_audit"])):
            raise ValueError("P60-B2 model contract changed")
        _validate_b2_offline_audit(_bound_path(frozen["files"]["b2_offline_audit"]))
        validate_offline_evidence(_bound_path(frozen["files"]["action_head"]),
                                  _bound_path(frozen["files"]["offline_query"]))
        _validate_p50_freeze(_bound_path(frozen["files"]["p50_physical_freeze"]),
                             _bound_path(frozen["files"]["adapter"]),
                             _bound_path(frozen["files"]["action_head"]))
        _validate_p59_contact_diagnostic(_bound_path(frozen["files"]["p59_contact_diagnostic"]))
    return frozen


def _seal(output, filenames, before, after, freeze_binding):
    seal = {"schema": SEAL_SCHEMA, "freeze": freeze_binding,
            "files": {name: sha(Path(output) / name) for name in filenames},
            "neural_parameters_before": before, "neural_parameters_after": after}
    dump_new(Path(output) / "SEAL.json", seal)
    if before != after:
        raise RuntimeError("Frozen neural weights changed; evaluator must not run")
    return seal


def run_cell(freeze_path, approval_path, index, gpu):
    """Execute one create-once cell after review of the exact result-free freeze."""
    _require_egl_environment(gpu)
    if approval_path is None:
        raise ValueError("P60-B2 root review receipt required before simulator construction")
    torch.set_num_threads(1)
    frozen = validate_freeze(freeze_path, approval_path, check_runtime=False)
    if gpu != 0 or gpu not in frozen["renderer_gpu_ids"]:
        raise ValueError("P60-B2 renderer is frozen to physical GPU 0")
    if not 0 <= index < len(frozen["cells"]):
        raise ValueError("Cell index outside frozen P60-B2 scope")
    frozen = validate_freeze(freeze_path, approval_path)
    cell = frozen["cells"][index]
    output = Path(freeze_path).parent / "cells" / cell_name(index, cell)
    output.mkdir(parents=True, exist_ok=False)
    freeze_binding = {"path": str(Path(freeze_path)), "sha256": sha(freeze_path)}
    dump_new(output / "PRE_RUN.json", {
        "schema": "prove.p60-contour-scaled-pre-run.v1", "cell_index": index,
        "cell": cell, "freeze": freeze_binding, "approval_sha256": sha(approval_path),
        "renderer_gpu": gpu, "steps": STEPS, "automatic_retry": False,
        "evaluator_only_state": "saved but never supplied to runtime",
    })

    from prove.fast_adapter.geometry3d_adapter import pose
    from prove.fast_adapter.geometry3d_env import configuration
    from prove.fast_adapter.p45_neural import load_model
    from prove.fast_adapter.p60_contour_scaled import P60ContourScaledLatent
    from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
    from tools.fast_adapter.p45_common import observe_roles
    from tools.fast_adapter.p47_evaluate import PUBLIC_KEYS, evaluate_sealed, parameters_sha, snapshot

    env = None
    traces, observations, rgb_frames, mask_frames, video_frames, detail_frames = [], [], [], [], [], []
    started = time.monotonic()
    try:
        model, package = load_model(_bound_path(frozen["files"]["adapter"]), "cpu")
        if model.task != "insert" or model.representation != "geometry" or package["seed"] != 11:
            raise ValueError("Wrong frozen P60-B2 geometry adapter identity")
        model.eval().requires_grad_(False)
        head = load_frozen_head(_bound_path(frozen["files"]["action_head"]))
        before_hash = {"geometry_model": parameters_sha(model), "action_head": parameters_sha(head)}
        cfg = configuration("insert", cell["condition_seed"], cell["split"])
        expected = next(row["evaluator_configuration"] for row in frozen["conditions"]
                        if row["split"] == cell["split"]
                        and row["condition_seed"] == cell["condition_seed"])
        if cfg != expected:
            raise ValueError("P60-B2 environment configuration changed after freeze")
        env = RobotGeometrySkill(cfg, render=True)
        if env.model.nmocap != 0 or env.model.nu < 7:
            raise ValueError("P60-B2 requires articulated actuated Panda without mocap")
        calibration = env.calibrate_held_geometry()
        if calibration.get("source") != "RGB visible-mask bbox plus known own-tool pose":
            raise ValueError("P60-B2 held length must use public visual calibration")
        length = float(calibration["length"])
        controller = PrecisionActionController(head, env.start_pos, env.start_R, length=length)
        adapter = P60ContourScaledLatent(model, cell["adapter_mode"], device="cpu")
        p, rotation = env.proprio()
        previous_action = np.r_[p, rotation.ravel(), 1.]
        admitted_frames = 0
        for t in range(STEPS):
            observed, rgb, masks = observe_roles(env)
            obs = {key: copy.deepcopy(observed[key]) for key in PUBLIC_KEYS}
            obs["local"] = np.zeros(48, np.float32)
            admitted = adapter.update(obs, t, masks[2])
            admitted_frames += int(admitted)
            estimate, proprio = adapter.estimate(), env.proprio()
            observations.append(dict(obs, t=t, previous_action=previous_action.copy(),
                                     geometry_source=cell["arm"], contour_admitted=admitted))
            if cell["video"] or t == 0:
                rgb_frames.append(rgb.copy())
                mask_frames.append(masks.copy())
            before_state = controller_state(controller)
            update = (copy.deepcopy(adapter.records[-1])
                      if admitted and adapter.records and adapter.records[-1]["t"] == t else None)
            shadow = None
            if update is not None:
                shadow = shadow_command(controller, t, pose(np.asarray(update["pre_update_points"])), proprio)
            phase = "align" if controller.phase == "observe" else controller.phase
            domain = (domain_excursions(encode_state(estimate, proprio, length, phase))
                      if t >= 12 and estimate is not None else None)
            p, rotation, closed, stage = controller.command(t, estimate, proprio)
            if update is not None:
                p0, r0, closed0, stage0 = shadow
                update.update(
                    same_state_action_position_delta=float(np.linalg.norm(p - p0)),
                    same_state_action_rotation_delta=float(np.linalg.norm(
                        Rotation.from_matrix(rotation @ r0.T).as_rotvec())),
                    same_state_closed_changed=bool(closed != closed0),
                    same_state_stage_changed=bool(stage != stage0),
                    shadow_command_position=np.asarray(p0).copy(),
                    shadow_command_rotation=np.asarray(r0).copy())
            neural_record = (copy.deepcopy(controller.records[-1])
                             if controller.records and controller.records[-1]["t"] == t else None)
            row = dict(t=t, state=snapshot(env), command_p=p.copy(), command_R=rotation.copy(),
                       closed=bool(closed), stage=stage, estimate=copy.deepcopy(estimate),
                       proprio=copy.deepcopy(proprio), controller_before=before_state,
                       controller_after=controller_state(controller), adaptation=update,
                       head_domain=domain, neural_action=neural_record,
                       controls_before=env.data.ctrl.copy())
            if cell["video"]:
                video_frames.append(np.concatenate([env.render_global(), rgb], axis=1))
                detail_frames.append(env.render_detail())
            env.step(p, rotation, closed)
            row["controls_after"] = env.data.ctrl.copy()
            row["robot_tracking"] = env.robot_tracking_diagnostics()
            traces.append(row)
            previous_action = np.r_[p, rotation.ravel(), float(closed)]
        terminal = snapshot(env)
        if cell["video"]:
            _, rgb, _ = observe_roles(env)
            video_frames.append(np.concatenate([env.render_global(), rgb], axis=1))
            detail_frames.append(env.render_detail())
        np.save(output / "TRACE.npy", np.array(traces, dtype=object), allow_pickle=True)
        np.save(output / "OBS.npy", np.array(observations, dtype=object), allow_pickle=True)
        np.save(output / "TERMINAL.npy", terminal, allow_pickle=True)
        np.savez_compressed(output / "PUBLIC_PIXELS.npz", rgb=np.asarray(rgb_frames),
                            visible_masks=np.asarray(mask_frames))
        filenames = ["TRACE.npy", "OBS.npy", "TERMINAL.npy", "PRE_RUN.json", "PUBLIC_PIXELS.npz"]
        if cell["video"]:
            np.savez_compressed(output / "FRAMES.npz", frames=np.asarray(video_frames))
            np.savez_compressed(output / "DETAIL_FRAMES.npz", frames=np.asarray(detail_frames))
            filenames += ["FRAMES.npz", "DETAIL_FRAMES.npz"]
        validate_freeze(freeze_path, approval_path)
        after_hash = {"geometry_model": parameters_sha(model), "action_head": parameters_sha(head)}
        _seal(output, filenames, before_hash, after_hash, freeze_binding)
        facts = evaluate_sealed(env, traces, terminal, output)
        changes = [row["adaptation"] for row in traces if row["adaptation"] is not None]
        precision_changes = [row["adaptation"] for row in traces
                             if row["t"] >= 12 and row["adaptation"] is not None]
        update_events = sum(bool(row["inner"]) for row in changes)
        result = dict(
            schema=RESULT_SCHEMA, cell=cell, cell_index=index, phase="development",
            fresh_test_claim=False, **facts, steps=STEPS,
            seal_sha256=sha(output / "SEAL.json"), frozen_neural_weights_unchanged=True,
            gradient_steps=sum(int(row["U"]) for row in changes), update_events=update_events,
            admitted_contour_frames=admitted_frames,
            valid_corner_frames=sum(row["corners"] is not None for row in observations),
            neural_precision_commands=sum(row["neural_action"] is not None for row in traces),
            head_domain_excursion_steps=sum(bool(row["head_domain"] and row["head_domain"]["outside_training_domain"])
                                            for row in traces),
            max_precision_same_state_position_delta=max(
                [row["same_state_action_position_delta"] for row in precision_changes] + [0.]),
            max_precision_same_state_rotation_delta=max(
                [row["same_state_action_rotation_delta"] for row in precision_changes] + [0.]),
            final_controller_phase=controller.phase,
            tool_calibration={key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in calibration.items()},
            elapsed_seconds=time.monotonic() - started,
            interpretation="bounded P60-B2 development; contour arms share one frozen P50 head")
        dump_new(output / "RESULT.json", result)
        return result
    except Exception as exc:
        partial = {}
        for name, values in (("TRACE_PARTIAL.npy", traces), ("OBS_PARTIAL.npy", observations)):
            np.save(output / name, np.array(values, dtype=object), allow_pickle=True)
            partial[name] = sha(output / name)
        if rgb_frames:
            np.savez_compressed(output / "PUBLIC_PIXELS_PARTIAL.npz", rgb=np.asarray(rgb_frames),
                                visible_masks=np.asarray(mask_frames))
            partial["PUBLIC_PIXELS_PARTIAL.npz"] = sha(output / "PUBLIC_PIXELS_PARTIAL.npz")
        if env is not None:
            try:
                np.save(output / "STATE_AT_ABORT.npy", snapshot(env), allow_pickle=True)
                partial["STATE_AT_ABORT.npy"] = sha(output / "STATE_AT_ABORT.npy")
            except Exception:
                pass
        dump_new(output / "ABORT.json", dict(
            schema="prove.p60-contour-scaled-cell-abort.v1", cell=cell,
            type=type(exc).__name__, reason=str(exc), completed_steps=len(traces),
            partial_files=partial, automatic_retry=False,
            interpretation="retained execution abort; never excluded or automatically retried"))
        raise
    finally:
        if env is not None:
            env.close()


def summarize(freeze_path):
    frozen = validate_freeze(freeze_path, check_runtime=False)
    rows = []
    for index, cell in enumerate(frozen["cells"]):
        output = Path(freeze_path).parent / "cells" / cell_name(index, cell)
        result, abort = output / "RESULT.json", output / "ABORT.json"
        if result.exists():
            record = json.loads(result.read_text())
            if (record.get("schema") != RESULT_SCHEMA or record.get("cell_index") != index
                    or record.get("cell") != cell
                    or record["seal_sha256"] != sha(output / "SEAL.json")):
                raise ValueError("P60-B2 result or trajectory seal changed")
            seal = json.loads((output / "SEAL.json").read_text())
            if (seal.get("schema") != SEAL_SCHEMA
                    or seal.get("freeze", {}).get("sha256") != sha(freeze_path)
                    or seal["neural_parameters_before"] != seal["neural_parameters_after"]):
                raise ValueError("Changed model in completed P60-B2 cell")
            for name, digest in seal["files"].items():
                if sha(output / name) != digest:
                    raise ValueError(f"Sealed P60-B2 trace changed: {output / name}")
            rows.append(dict(cell=cell, status="complete", result=record))
        elif abort.exists():
            record = json.loads(abort.read_text())
            if record.get("schema") != "prove.p60-contour-scaled-cell-abort.v1" or record.get("cell") != cell:
                raise ValueError("P60-B2 abort does not match frozen arm and condition")
            rows.append(dict(cell=cell, status="abort", abort=record))
        else:
            rows.append(dict(cell=cell, status="not_complete"))
    return dict(schema=SUMMARY_SCHEMA, expected_cells=len(cells()),
                completed=sum(row["status"] == "complete" for row in rows),
                aborted=sum(row["status"] == "abort" for row in rows), cells=rows,
                fresh_test_claim=False, independent_success_rate_claim=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    for name in ("output", "adapter", "head", "offline-query", "p50-freeze",
                 "p59-contact-diagnostic", "plan", "amendment", "b2-offline-audit"):
        prep.add_argument("--" + name, type=Path, required=True)
    run = sub.add_parser("run-cell")
    run.add_argument("--freeze", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True)
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--gpu", type=int, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        value = freeze_pilot(args.output, args.adapter, args.head, args.offline_query,
                             args.p50_freeze, args.p59_contact_diagnostic, args.plan,
                             args.amendment, args.b2_offline_audit)
    elif args.mode == "run-cell":
        value = run_cell(args.freeze, args.approval, args.index, args.gpu)
    else:
        value = summarize(args.freeze)
    print(json.dumps(value, indent=2, allow_nan=False))
