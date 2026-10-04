#!/usr/bin/env python3
"""Result-free freeze and bounded P58 held-hammer development evaluator.

Runtime actions consume rendered public observations only. Simulator metrics are
written after each step for evaluation and never passed to a policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools.fast_adapter import p58_hammer_qualification as qualification


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "evaluations/fast_adapter_p58_functional_geometry_v01/PLAN.md"
MOUNT = ROOT / "evaluations/fast_adapter_p35_resolution_views_v01/preflight/WRIST_MOUNT.json"
ARMS = ("absolute_positions_fixed", "absolute_positions_online", "absolute_positions_direct",
        "role_relative_fixed", "role_relative_online", "role_relative_direct",
        "p30v2_visual_bc", "train_mean_geometry")
SCHEMA = "p58.held-hammer-development-freeze.v1"
SOURCES = (
    "tools/fast_adapter/p58_hammer_runner.py",
    "tools/fast_adapter/p58_hammer_qualification.py",
    "tools/fast_adapter/p58_hammer_export.py",
    "tools/fast_adapter/p35_camera.py",
    "tools/fast_adapter/p30v2_common.py",
    "tools/fast_adapter/p29_common.py",
    "tools/fast_adapter/p22_data.py",
    "prove/fast_adapter/p58_hammer_observation.py",
    "prove/fast_adapter/p58_hammer_readout.py",
    "prove/fast_adapter/p58_hammer_runtime.py",
    "prove/fast_adapter/p58_geometry_model.py",
    "prove/task_graph/p57_public_geometry.py",
    "prove/fast_adapter/continuous_adapter.py",
    "prove/fast_adapter/visual_alignment.py",
    "prove/fast_adapter/held_hammer_env.py",
    "prove/fast_adapter/object_interaction_env.py",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def jsonable(value):
    """Turn detached runtime diagnostics into finite, auditable JSON values."""
    if isinstance(value, dict):
        return {str(key):jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Nonfinite runtime telemetry")
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(f"Unsupported telemetry type: {type(value).__name__}")


def _value_fingerprint(value):
    """Hash frozen in-memory arrays, excluding each episode's mutable latent z."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode()); visit(item[key])
        elif torch.is_tensor(item):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        else:
            raise TypeError(f"Non-array frozen parameter: {type(item).__name__}")
    visit(value)
    return digest.hexdigest()


def frozen_fingerprints(runtime, arm):
    if arm == "p30v2_visual_bc":
        return dict(policy_state_sha256=_value_fingerprint(runtime.base.state_dict()),
                    visual_package_sha256=None)
    return dict(policy_state_sha256=_value_fingerprint(runtime.model.state_dict()),
                visual_package_sha256=_value_fingerprint(runtime.readout))


def cells():
    return [dict(condition_index=i, arm=arm) for i in range(4) for arm in ARMS]


def freeze(output, *, absolute_model, relative_model, absolute_readout,
           relative_readout, bc_model, bc_pca, qualification_freeze,
           qualification_result, dino_dir=None):
    """Bind every development input without constructing or stepping a simulator."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    qf = read(qualification_freeze)
    conditions = qualification.verify(qf)
    qr = read(qualification_result)
    qualification_seal_path = Path(qualification_result).parent/"SEAL.json"
    qualification_seal = read(qualification_seal_path)
    if (qualification_seal.get("schema") != "p58.hammer-qualification-seal.v1"
            or qualification_seal.get("freeze_sha256") != sha(qualification_freeze)
            or qualification_seal.get("files", {}).get("RESULT.json") != sha(qualification_result)):
        raise ValueError("Prior Oracle qualification seal mismatch")
    if (qr.get("schema") != "p58.held-hammer-qualification-result.v1"
            or qr.get("qualification_passed") is not True
            or qr.get("denominator") != 4 or qr.get("oracle_success") != 4):
        raise ValueError("Four-condition held-hammer engineering qualification failed")
    model_paths = dict(absolute_positions=absolute_model, role_relative=relative_model,
                       absolute_readout=absolute_readout, relative_readout=relative_readout,
                       p30v2_visual_bc=bc_model, p30v2_visual_pca=bc_pca)
    from prove.fast_adapter.p58_hammer_runtime import make_runtime
    for representation, checkpoint, readout_path in (
            ("absolute_positions", absolute_model, absolute_readout),
            ("role_relative", relative_model, relative_readout)):
        # Decode and bind both packages before any physical result is opened.
        make_runtime(checkpoint, readout_path, representation, "fixed")
    from prove.fast_adapter.visual_alignment import DEFAULT_MODEL, DINO_SHA256
    dino = Path(dino_dir or DEFAULT_MODEL)
    if sha(dino/"model.safetensors") != DINO_SHA256:
        raise ValueError("Historical P30v2 DINO weights changed")
    result = dict(schema=SCHEMA, status="frozen", result_unopened=True,
        physics_steps_before_freeze=0, denominator=32, automatic_retries=0,
        preparation_budget=120, execution_budget=200, grip=.9,
        fixed_physics=dict(mass=.35, friction=1.1),
        source_scope="opened_P35_development_first_four_meta_expert_048",
        conditions=conditions, cells=cells(),
        success="native HeldHammerEnv.metrics: held head contact, slide >=.045m, no direct contact, hold5",
        runtime_input="public rendered role masks, RGB only for old P30v2 BC, camera calibration and robot proprioception",
        public_gate="estimated head/target distance only; freeze target anchor when entering push",
        learned_controller="shared frozen relation controller: estimated pre-y alignment then estimated target push; not exact HeldHammerEnv GT teacher",
        plan=dict(path=str(PLAN), sha256=sha(PLAN)),
        qualification_freeze=dict(path=str(qualification_freeze), sha256=sha(qualification_freeze)),
        qualification_result=dict(path=str(qualification_result), sha256=sha(qualification_result)),
        qualification_seal=dict(path=str(qualification_seal_path), sha256=sha(qualification_seal_path)),
        mount=dict(path=str(MOUNT), sha256=sha(MOUNT)),
        historical_visual=dict(model_dir=str(dino), files={name:sha(dino/name)
            for name in ("model.safetensors", "config.json", "preprocessor_config.json")}),
        models={name:dict(path=str(path), sha256=sha(path)) for name, path in model_paths.items()},
        sources={name:sha(ROOT/name) for name in SOURCES})
    write(output, result)
    return result


def verify(freeze_path):
    frozen = read(freeze_path)
    if (frozen.get("schema") != SCHEMA or frozen.get("status") != "frozen"
            or frozen.get("result_unopened") is not True
            or frozen.get("physics_steps_before_freeze") != 0
            or frozen.get("denominator") != 32 or frozen.get("automatic_retries") != 0
            or frozen.get("preparation_budget") != 120
            or frozen.get("execution_budget") != 200 or frozen.get("grip") != .9
            or frozen.get("fixed_physics") != dict(mass=.35, friction=1.1)
            or frozen.get("cells") != cells()):
        raise ValueError("P58 held-hammer development contract changed")
    for ref in (frozen["plan"], frozen["qualification_freeze"],
                frozen["qualification_result"], frozen["qualification_seal"],
                frozen["mount"], *frozen["models"].values()):
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Frozen input changed: " + ref["path"])
    from prove.fast_adapter.visual_alignment import DINO_SHA256
    visual = frozen["historical_visual"]
    if visual["files"].get("model.safetensors") != DINO_SHA256:
        raise ValueError("Historical P30v2 visual model changed")
    for name, digest in visual["files"].items():
        if sha(Path(visual["model_dir"])/name) != digest:
            raise ValueError("Historical P30v2 visual asset changed: " + name)
    if frozen["sources"] != {name:sha(ROOT/name) for name in SOURCES}:
        raise ValueError("Frozen execution source changed")
    conditions = qualification.verify(read(frozen["qualification_freeze"]["path"]))
    if frozen["conditions"] != conditions:
        raise ValueError("Frozen conditions changed")
    old_seal = read(frozen["qualification_seal"]["path"])
    if (old_seal.get("freeze_sha256") != frozen["qualification_freeze"]["sha256"]
            or old_seal.get("files", {}).get("RESULT.json") != frozen["qualification_result"]["sha256"]):
        raise ValueError("Prior qualification evidence chain changed")
    return frozen


def verify_review(freeze_path, review_path):
    """Require a post-freeze root review before any learned physical cell."""
    review = read(review_path)
    if (review.get("schema") != "p58.hammer-development-root-review.v1"
            or review.get("status") != "approved"
            or review.get("freeze_sha256") != sha(freeze_path)
            or review.get("cells_authorized") != 32):
        raise ValueError("P58 32-cell physical execution has no matching root review")
    return review


def _render_public(env, rig):
    from prove.fast_adapter.object_interaction_env import mujoco
    from prove.fast_adapter.p58_hammer_observation import (
        RESOLUTION, ROLE_GEOMS, camera_matrices, eef_transform,
        public_frame, visible_masks)
    from tools.fast_adapter.p58_hammer_export import render
    geom_ids = [int(env.env.model.geom(name).id) for name in ROLE_GEOMS]
    images, masks, matrices, poses = [], [], [], []
    for camera in (env.cam, "gripperPOV"):
        rgb, segmentation, calibration = render(rig.renderers[RESOLUTION], env.env.data, camera)
        K, pose = camera_matrices(calibration)
        images.append(rgb)
        masks.append(visible_masks(segmentation, geom_ids, mujoco.mjtObj.mjOBJ_GEOM))
        matrices.append(K)
        poses.append(pose)
    public = public_frame(np.stack(masks), np.stack(matrices), np.stack(poses),
                          env.public(), eef_transform(env))
    public["masks"] = np.stack(masks)
    public["rgb"] = np.stack(images)
    return public


class OldVisualBC:
    def __init__(self, frozen):
        from tools.fast_adapter.p29_common import Visual
        from prove.fast_adapter.continuous_adapter import ObjectSkill
        model = frozen["models"]
        pca = torch.load(model["p30v2_visual_pca"]["path"], map_location="cpu", weights_only=True)
        self.visual = Visual(frozen["historical_visual"]["model_dir"], "hammer-block", pca)
        self.base = ObjectSkill().cuda()
        blob = torch.load(model["p30v2_visual_bc"]["path"], map_location="cpu", weights_only=True)
        self.base.load_state_dict(blob["state"])
        self.base.eval().requires_grad_(False)
        self.previous = torch.zeros(1, 4, device="cuda")

    def step(self, env, public):
        # This is the historical RGB-only P30 visual path, not the P58 mask head.
        rgb = env.rgb()[None]
        with torch.no_grad():
            vision = self.visual.encode(rgb)
            proprio = torch.as_tensor(env.public()[None], device="cuda", dtype=torch.float32)
            action = self.base(vision, proprio, self.previous).detach().clone()
        action[:, 3] = .9
        self.previous = action
        return action[0].cpu().numpy(), dict(mode="frozen_P30v2_visual_BC")


def _new_runtime(frozen, arm):
    """The six learned arms and train-mean share one public-only runtime API."""
    from prove.fast_adapter.p58_hammer_runtime import make_runtime
    representation, mode = arm.rsplit("_", 1)
    if arm == "train_mean_geometry":
        representation, mode = "absolute_positions", "train_mean"
    readout_key = {"absolute_positions":"absolute_readout",
                   "role_relative":"relative_readout"}[representation]
    return make_runtime(
        frozen["models"][representation]["path"],
        frozen["models"][readout_key]["path"],
        representation, mode)


def rollout(freeze_path, review_path, index, output):
    """One physical cell. Caller must first obtain root's result-free review."""
    frozen = verify(freeze_path)
    verify_review(freeze_path, review_path)
    if not 0 <= index < 32:
        raise IndexError(index)
    cell = frozen["cells"][index]
    condition = frozen["conditions"][cell["condition_index"]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/"PRE_RUN.json", dict(schema="p58.hammer-prerun.v1", index=index,
        cell=cell, condition=condition, freeze_sha256=sha(freeze_path),
        root_review=dict(path=str(review_path), sha256=sha(review_path)),
        frozen_model=frozen["models"], started_unix=time.time(), attempt=1))
    from prove.fast_adapter.held_hammer_env import HeldHammerEnv, PreparationFailed
    from tools.fast_adapter.p35_camera import CameraRig
    env = None
    rig = None
    runtime = None
    actions, metrics, public_rows, telemetry, states = [], [], [], [], []
    result = dict(schema="p58.hammer-development-cell.v1", index=index, cell=cell,
                  preparation_passed=False, executed_task_steps=0,
                  validated_success=False, attempt=1, automatic_retry=False)
    try:
        env = HeldHammerEnv(condition["config"], render=cell["arm"] == "p30v2_visual_bc",
                            preparation_budget=120)
        try:
            env.reset(condition["reset_seed"])
        except PreparationFailed:
            result["status"] = "preparation_failed"
        else:
            result["preparation_passed"] = bool(env.preparation["passed"])
            result["target_configuration_audit"] = qualification._actual_target(env)
            states.append(env.state())
            rig = CameraRig(env, read(frozen["mount"]["path"]))
            runtime = (OldVisualBC(frozen) if cell["arm"] == "p30v2_visual_bc"
                       else _new_runtime(frozen, cell["arm"]))
            result["frozen_fingerprints_before"] = frozen_fingerprints(runtime, cell["arm"])
            for t in range(200):
                public = _render_public(env, rig)
                if cell["arm"] == "p30v2_visual_bc":
                    action, info = runtime.step(env, public)
                else:
                    # P58 runtime receives exactly the allowlisted public frame.
                    action, info = runtime.step({k:public[k] for k in
                        ("mask_tokens", "mask_valid", "camera_K", "camera_pose", "proprio", "eef_transform")})
                action = np.asarray(action, np.float32)
                if action.shape != (4,) or not np.isfinite(action).all():
                    raise ValueError("Invalid public-runtime action")
                if not np.isclose(action[3], .9, rtol=0, atol=1e-7):
                    raise ValueError("P58 hammer gripper must remain closed")
                env.step(action)
                actual = np.asarray(env.last_executed_action, np.float32).copy()
                states.append(env.state())
                metric = dict(env.metrics[-1])
                if not all(np.isfinite(value) for value in metric.values()):
                    raise ValueError("Nonfinite native metric")
                actions.append(actual); metrics.append(metric); telemetry.append(info)
                public_rows.append({k:public[k] for k in
                    ("masks", "mask_tokens", "mask_valid", "camera_K", "camera_pose", "proprio", "eef_transform")})
                if metric["success"]:
                    if not (metric["head_contact_seen"] and metric["target_displacement"] >= .045
                            and not metric["target_direct_contact"] and metric["hold_streak"] >= 5):
                        raise ValueError("Native success contract inconsistent")
                    break
            result.update(status="completed", executed_task_steps=len(actions),
                          validated_success=any(bool(m["success"]) for m in metrics),
                          final_metric=metrics[-1] if metrics else None,
                          first_success_step=next((t+1 for t,m in enumerate(metrics) if m["success"]), None))
    except Exception as error:
        result.update(status="engineering_error", error=f"{type(error).__name__}: {error}",
                      executed_task_steps=len(actions))
    finally:
        if runtime is not None:
            try:
                result["frozen_fingerprints_after"] = frozen_fingerprints(runtime, cell["arm"])
                same = result["frozen_fingerprints_before"] == result["frozen_fingerprints_after"]
                result["frozen_parameters_unchanged"] = bool(same)
                if not same:
                    result.update(status="engineering_error", validated_success=False,
                                  error="Frozen model/readout parameters changed during rollout")
            except Exception as error:
                result.update(status="engineering_error", validated_success=False,
                              error=f"Frozen parameter audit failed: {type(error).__name__}: {error}")
        if rig is not None:
            rig.close()
        if env is not None:
            prep = env.preparation
            if prep is not None:
                result.update(preparation_passed=bool(prep["passed"]), preparation_steps=prep["steps"],
                              preparation_final_metric=prep["metrics"][-1] if prep["metrics"] else None)
                np.savez_compressed(output/"PREPARATION.npz",
                    actions=np.asarray(prep["actions"], np.float32).reshape(-1,4))
                write(output/"PREPARATION_METRICS.json", prep["metrics"])
            env.close()
    if public_rows:
        np.savez_compressed(output/"PUBLIC.npz", **{key:np.stack([row[key] for row in public_rows])
                                                 for key in public_rows[0]})
    if states:
        np.savez_compressed(output/"INTERACTION_STATE.npz",
                            **{key:np.stack([state[key] for state in states]) for key in states[0]})
    np.savez_compressed(output/"ACTIONS.npz", executed_actions=np.asarray(actions, np.float32).reshape(-1,4))
    write(output/"METRICS.json", metrics)
    write(output/"TELEMETRY.json", telemetry)
    write(output/"RESULT.json", result)
    files = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    write(output/"SEAL.json", dict(schema="p58.hammer-development-cell-seal.v1",
                                    freeze_sha256=sha(freeze_path), index=index, files=files))
    return result


def batch(freeze_path, review_path, output, python, workers):
    """Finite queue: at most `workers` independent cells; stop after any error."""
    frozen = verify(freeze_path)
    verify_review(freeze_path, review_path)
    if not 1 <= workers <= 2:
        raise ValueError("Held-hammer evaluation permits one or two concurrent cells")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output/"QUEUE.json", dict(schema="p58.hammer-development-queue.v1",
        freeze_sha256=sha(freeze_path), cells=frozen["cells"], workers=workers,
        root_review=dict(path=str(review_path), sha256=sha(review_path)),
        python=str(python), automatic_retries=0))
    results = []
    for start in range(0, 32, workers):
        processes = []
        for index in range(start, min(start+workers, 32)):
            cmd = [str(python), "-m", "tools.fast_adapter.p58_hammer_runner", "run",
                   "--freeze", str(freeze_path), "--review", str(review_path), "--index", str(index),
                   "--output", str(output/f"cell_{index:02d}")]
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            env.setdefault("MUJOCO_GL", "egl")
            env.setdefault("PYOPENGL_PLATFORM", "egl")
            env["PYTHONPATH"] = os.pathsep.join((str(ROOT/"data/runtime/fast-adapter-p28-deps"),
                                                  str(ROOT), env.get("PYTHONPATH", "")))
            log = (output/f"cell_{index:02d}.log").open("x")
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            processes.append((index, proc, log, time.monotonic(), cmd))
        for index, proc, log, launched, cmd in processes:
            try:
                code = proc.wait(timeout=max(.001, 1800.-(time.monotonic()-launched)))
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = -9
            finally:
                log.close()
            cell_path = output/f"cell_{index:02d}"
            result = read(cell_path/"RESULT.json") if (cell_path/"RESULT.json").exists() else None
            results.append(dict(index=index, command=cmd, returncode=code, result=result,
                                log_sha256=sha(output/f"cell_{index:02d}.log"),
                                cell_seal_sha256=sha(cell_path/"SEAL.json") if (cell_path/"SEAL.json").exists() else None))
        if any(row["returncode"] or row["result"] is None or row["result"]["status"] == "engineering_error"
               for row in results[-len(processes):]):
            break
    write(output/"QUEUE_RESULT.json", dict(schema="p58.hammer-development-queue-result.v1",
        freeze_sha256=sha(freeze_path), launched=len(results), results=results,
        stopped_on_engineering_error=len(results) < 32 or any(row["returncode"] for row in results)))
    return results


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    for name in ("output", "absolute-model", "relative-model", "absolute-readout",
                 "relative-readout", "bc-model", "bc-pca", "qualification-freeze",
                 "qualification-result"):
        f.add_argument("--"+name, required=True)
    f.add_argument("--dino-dir")
    r = sub.add_parser("run")
    r.add_argument("--freeze", required=True); r.add_argument("--index", type=int, required=True)
    r.add_argument("--review", required=True)
    r.add_argument("--output", required=True)
    q = sub.add_parser("batch")
    q.add_argument("--freeze", required=True); q.add_argument("--output", required=True)
    q.add_argument("--review", required=True)
    q.add_argument("--python", default=sys.executable); q.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "freeze":
        value = freeze(args.output, absolute_model=args.absolute_model,
            relative_model=args.relative_model, absolute_readout=args.absolute_readout,
            relative_readout=args.relative_readout, bc_model=args.bc_model, bc_pca=args.bc_pca,
            qualification_freeze=args.qualification_freeze,
            qualification_result=args.qualification_result, dino_dir=args.dino_dir)
        print(json.dumps(dict(path=args.output, sha256=sha(args.output), cells=len(value["cells"]))))
    elif args.command == "run":
        print(json.dumps(rollout(args.freeze, args.review, args.index, args.output)))
    else:
        print(json.dumps(dict(results=len(batch(args.freeze, args.review, args.output, args.python, args.workers)))))


if __name__ == "__main__":
    main()
