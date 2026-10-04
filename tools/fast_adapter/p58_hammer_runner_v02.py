#!/usr/bin/env python3
"""P58 hammer V2 measurement wrapper over the immutable V1 physical loop.

Only public-mask cleanup, revised model/readout loading and result-free source
bindings change. Each cell still uses the same four conditions, eight arms,
preparation budget, execution budget and native success metric.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from tools.fast_adapter import p58_hammer_runner as base
from tools.fast_adapter import p58_hammer_qualification as qualification


ROOT = base.ROOT
SCHEMA = "p58.held-hammer-development-freeze.v2"
REVIEW_SCHEMA = "p58.hammer-development-root-review.v2"
REVISION_NOTE = ROOT/"evaluations/fast_adapter_p58_functional_geometry_v01/hammer/MEASUREMENT_REVISION_v02.md"
SOURCES = tuple(dict.fromkeys(base.SOURCES + (
    "tools/fast_adapter/p58_hammer_runner_v02.py",
    "tools/fast_adapter/p58_hammer_measurement_v02.py",
    "tools/fast_adapter/p58_hammer_readout_v02.py",
    "tools/fast_adapter/p58_hammer_train_v02.py",
    "tools/fast_adapter/p58_hammer_freeze_v02.py",
    "prove/fast_adapter/p58_hammer_observation_v02.py",
    "prove/fast_adapter/p58_hammer_readout_v02.py",
    "prove/fast_adapter/p58_hammer_runtime_v02.py",
)))
_BASE_VERIFY = base.verify


def _render_public(env, rig):
    """Render V2 visible LCC masks and their matching public features."""
    from prove.fast_adapter.object_interaction_env import mujoco
    from prove.fast_adapter.p58_hammer_observation_v02 import (
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
    clean = np.stack(masks)
    public = public_frame(clean, np.stack(matrices), np.stack(poses),
                          env.public(), eef_transform(env))
    public["masks"] = clean
    public["rgb"] = np.stack(images)
    return public


def _new_runtime(frozen, arm):
    from prove.fast_adapter.p58_hammer_runtime_v02 import make_runtime
    representation, mode = arm.rsplit("_", 1)
    if arm == "train_mean_geometry":
        representation, mode = "absolute_positions", "train_mean"
    readout = {"absolute_positions":"absolute_readout",
               "role_relative":"relative_readout"}[representation]
    return make_runtime(frozen["models"][representation]["path"],
                        frozen["models"][readout]["path"], representation, mode)


def verify_review(freeze_path, review_path):
    review = base.read(review_path)
    if (review.get("schema") != REVIEW_SCHEMA or review.get("status") != "approved"
            or review.get("freeze_sha256") != base.sha(freeze_path)
            or review.get("cells_authorized") != 32):
        raise ValueError("P58 V2 physical execution has no matching root review")
    return review


@contextmanager
def _patched_base():
    """One-cell process-local dispatch into V1's byte-frozen rollout loop."""
    names = ("SCHEMA", "SOURCES", "verify", "verify_review", "_render_public", "_new_runtime")
    old = {name:getattr(base, name) for name in names}
    try:
        base.SCHEMA, base.SOURCES = SCHEMA, SOURCES
        base.verify, base.verify_review = verify, verify_review
        base._render_public, base._new_runtime = _render_public, _new_runtime
        yield
    finally:
        for name, value in old.items():
            setattr(base, name, value)


def freeze(output, *, absolute_model, relative_model, absolute_readout,
           relative_readout, bc_model, bc_pca, qualification_freeze,
           qualification_result, geometry_manifest, readout_report,
           training_freeze, dino_dir=None):
    """Generate the V2 result-free contract; never construct a simulator."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    qf = base.read(qualification_freeze)
    conditions = qualification.verify(qf)
    qr = base.read(qualification_result)
    seal_path = Path(qualification_result).parent/"SEAL.json"
    old_seal = base.read(seal_path)
    if (qr.get("schema") != "p58.held-hammer-qualification-result.v1"
            or qr.get("qualification_passed") is not True or qr.get("denominator") != 4
            or qr.get("oracle_success") != 4
            or old_seal.get("freeze_sha256") != base.sha(qualification_freeze)
            or old_seal.get("files", {}).get("RESULT.json") != base.sha(qualification_result)):
        raise ValueError("Prior four-condition Oracle qualification failed or changed")
    geometry = base.read(geometry_manifest)
    if (geometry.get("schema") != "p58.hammer-geometry-export.v2"
            or geometry.get("derived_without_render_or_physics") is not True
            or geometry.get("episodes") != 64
            or geometry.get("train_objects") != 24
            or geometry.get("development_objects") != 8
            or not isinstance(geometry.get("measurement_revision"), dict)
            or "uv_valid_counts" not in geometry["measurement_revision"]):
        raise ValueError("P58 V2 geometry derivation does not match the frozen 64 episodes")
    trained = base.read(training_freeze)
    readout_summary = base.read(readout_report)
    if (trained.get("schema") != "p58.hammer-training-freeze.v2"
            or trained.get("manifest", {}).get("sha256") != base.sha(geometry_manifest)
            or trained.get("measurement_revision", {}).get("sha256") != base.sha(REVISION_NOTE)
            or trained.get("plan", {}).get("sha256") != base.sha(base.PLAN)):
        raise ValueError("P58 V2 training freeze has different geometry, revision or plan bytes")
    if (readout_summary.get("schema") != "p58.hammer-ridge-report.v2"
            or readout_summary.get("manifest_sha256") != base.sha(geometry_manifest)
            or readout_summary.get("freeze_sha256") != base.sha(training_freeze)
            or readout_summary.get("training_objects") != 24
            or readout_summary.get("development_objects") != 8):
        raise ValueError("P58 V2 readout report is not bound to this training freeze")
    from prove.fast_adapter.p58_hammer_runtime_v02 import make_runtime
    for representation, checkpoint, readout in (
            ("absolute_positions", absolute_model, absolute_readout),
            ("role_relative", relative_model, relative_readout)):
        package = readout_summary.get("packages", {}).get(representation, {})
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (package.get("sha256") != base.sha(readout)
                or payload.get("schema") != "p58.hammer-geometry-model.v2"
                or payload.get("manifest_sha256") != base.sha(geometry_manifest)
                or payload.get("freeze_sha256") != base.sha(training_freeze)
                or payload.get("readout_sha256") != base.sha(readout)):
            raise ValueError("P58 V2 model/readout/geometry metadata mismatch")
        make_runtime(checkpoint, readout, representation, "fixed")
    from prove.fast_adapter.visual_alignment import DEFAULT_MODEL, DINO_SHA256
    dino = Path(dino_dir or DEFAULT_MODEL)
    if base.sha(dino/"model.safetensors") != DINO_SHA256:
        raise ValueError("Historical P30v2 DINO bytes changed")
    models = dict(absolute_positions=absolute_model, role_relative=relative_model,
                  absolute_readout=absolute_readout, relative_readout=relative_readout,
                  p30v2_visual_bc=bc_model, p30v2_visual_pca=bc_pca)
    ref = lambda path:dict(path=str(path), sha256=base.sha(path))
    value = dict(schema=SCHEMA, status="frozen", result_unopened=True,
        physics_steps_before_freeze=0, denominator=32, automatic_retries=0,
        preparation_budget=120, execution_budget=200, grip=.9,
        fixed_physics=dict(mass=.35, friction=1.1),
        source_scope="opened_P35_development_first_four_meta_expert_048",
        conditions=conditions, cells=base.cells(),
        success="native HeldHammerEnv.metrics: held head contact, slide >=.045m, no direct contact, hold5",
        runtime_input="V2 cleaned public role masks, RGB only for old P30v2 BC, camera calibration and robot proprioception",
        public_gate="estimated head/target distance only; freeze target anchor when entering push",
        learned_controller="shared frozen relation controller; not exact HeldHammerEnv GT teacher",
        measurement_revision="largest 8-connected component and image-valid semantic UV; V1 measurement did not qualify",
        plan=ref(base.PLAN), qualification_freeze=ref(qualification_freeze),
        qualification_result=ref(qualification_result), qualification_seal=ref(seal_path),
        mount=ref(base.MOUNT), geometry_manifest=ref(geometry_manifest),
        measurement_revision_note=ref(REVISION_NOTE), readout_report=ref(readout_report),
        training_freeze=ref(training_freeze),
        historical_visual=dict(model_dir=str(dino), files={name:base.sha(dino/name)
            for name in ("model.safetensors", "config.json", "preprocessor_config.json")}),
        models={name:ref(path) for name,path in models.items()},
        sources={name:base.sha(ROOT/name) for name in SOURCES})
    base.write(output, value)
    verify(output)
    return value


def verify(freeze_path):
    with _patched_base():
        frozen = _BASE_VERIFY(freeze_path)
    if (frozen.get("measurement_revision") !=
            "largest 8-connected component and image-valid semantic UV; V1 measurement did not qualify"):
        raise ValueError("P58 V2 public-mask revision changed")
    for name in ("geometry_manifest", "measurement_revision_note", "readout_report", "training_freeze"):
        ref = frozen[name]
        if base.sha(ref["path"]) != ref["sha256"]:
            raise ValueError("P58 V2 frozen evidence changed: " + name)
    geometry = base.read(frozen["geometry_manifest"]["path"])
    if geometry.get("schema") != "p58.hammer-geometry-export.v2":
        raise ValueError("P58 V2 geometry manifest changed")
    return frozen


def _rewrite(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".revision_tmp")
    temporary.write_text(json.dumps(base.jsonable(value), ensure_ascii=False,
                                    indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def rollout(freeze_path, review_path, index, output):
    verify(freeze_path)
    verify_review(freeze_path, review_path)
    with _patched_base():
        result = base.rollout(freeze_path, review_path, index, output)
    out = Path(output)
    pre = base.read(out/"PRE_RUN.json")
    pre.update(schema="p58.hammer-prerun.v2", measurement_revision="public_mask_LCC_v02")
    _rewrite(out/"PRE_RUN.json", pre)
    result.update(schema="p58.hammer-development-cell.v2",
                  measurement_revision="public_mask_LCC_v02")
    _rewrite(out/"RESULT.json", result)
    files = {path.name:base.sha(path) for path in out.iterdir()
             if path.is_file() and path.name != "SEAL.json"}
    _rewrite(out/"SEAL.json", dict(schema="p58.hammer-development-cell-seal.v2",
        freeze_sha256=base.sha(freeze_path), root_review_sha256=base.sha(review_path),
        index=index, files=files))
    return result


def batch(freeze_path, review_path, output, python, workers):
    frozen = verify(freeze_path)
    verify_review(freeze_path, review_path)
    if not 1 <= workers <= 2:
        raise ValueError("P58 V2 development permits one or two active cells")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base.write(output/"QUEUE.json", dict(schema="p58.hammer-development-queue.v2",
        freeze_sha256=base.sha(freeze_path), root_review_sha256=base.sha(review_path),
        cells=frozen["cells"], workers=workers, python=str(python), automatic_retries=0))
    rows = []
    for start in range(0, 32, workers):
        launched = []
        for index in range(start, min(start+workers, 32)):
            command = [str(python), "-m", "tools.fast_adapter.p58_hammer_runner_v02", "run",
                "--freeze", str(freeze_path), "--review", str(review_path),
                "--index", str(index), "--output", str(output/f"cell_{index:02d}")]
            environment = os.environ.copy()
            environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            environment.setdefault("MUJOCO_GL", "egl")
            environment.setdefault("PYOPENGL_PLATFORM", "egl")
            environment["PYTHONPATH"] = os.pathsep.join((
                str(ROOT/"data/runtime/fast-adapter-p28-deps"), str(ROOT), environment.get("PYTHONPATH", "")))
            log = (output/f"cell_{index:02d}.log").open("x")
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            launched.append((index, command, process, log, time.monotonic()))
        for index, command, process, log, begun in launched:
            try:
                code = process.wait(timeout=max(.001, 1800.-(time.monotonic()-begun)))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = -9
            finally:
                log.close()
            cell = output/f"cell_{index:02d}"
            result = base.read(cell/"RESULT.json") if (cell/"RESULT.json").exists() else None
            rows.append(dict(index=index, command=command, returncode=code, result=result,
                log_sha256=base.sha(output/f"cell_{index:02d}.log"),
                cell_seal_sha256=base.sha(cell/"SEAL.json") if (cell/"SEAL.json").exists() else None))
        if any(row["returncode"] or row["result"] is None
               or row["result"].get("status") == "engineering_error" for row in rows[-len(launched):]):
            break
    base.write(output/"QUEUE_RESULT.json", dict(schema="p58.hammer-development-queue-result.v2",
        freeze_sha256=base.sha(freeze_path), launched=len(rows), results=rows,
        stopped_on_engineering_error=len(rows) < 32 or any(row["returncode"] for row in rows)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    for name in ("output", "absolute-model", "relative-model", "absolute-readout",
                 "relative-readout", "bc-model", "bc-pca", "qualification-freeze",
                 "qualification-result", "geometry-manifest",
                 "readout-report", "training-freeze"):
        f.add_argument("--"+name, required=True)
    f.add_argument("--dino-dir")
    r = sub.add_parser("run")
    r.add_argument("--freeze", required=True); r.add_argument("--review", required=True)
    r.add_argument("--index", type=int, required=True); r.add_argument("--output", required=True)
    q = sub.add_parser("batch")
    q.add_argument("--freeze", required=True); q.add_argument("--review", required=True)
    q.add_argument("--output", required=True); q.add_argument("--python", default=sys.executable)
    q.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "freeze":
        value = freeze(args.output, absolute_model=args.absolute_model,
            relative_model=args.relative_model, absolute_readout=args.absolute_readout,
            relative_readout=args.relative_readout, bc_model=args.bc_model, bc_pca=args.bc_pca,
            qualification_freeze=args.qualification_freeze,
            qualification_result=args.qualification_result,
            geometry_manifest=args.geometry_manifest,
            readout_report=args.readout_report,
            training_freeze=args.training_freeze, dino_dir=args.dino_dir)
        print(json.dumps(dict(path=args.output, sha256=base.sha(args.output), cells=len(value["cells"]))))
    elif args.command == "run":
        print(json.dumps(rollout(args.freeze, args.review, args.index, args.output)))
    else:
        print(json.dumps(dict(results=len(batch(args.freeze, args.review, args.output,
                                                args.python, args.workers)))))


if __name__ == "__main__":
    main()
