#!/usr/bin/env python3
"""P57 B frozen five-arm, four-entry local s9 development evaluation.

Only original P56 validation expert near-entry states are restored. No P56
private teacher labels enter either policy; judgments remain evaluator-only.
Preparing the binding does not create an environment or run physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from prove.fast_adapter.neural_geometry_visual import mask_geometry
from prove.task_graph.p52_stage_policy import Config, StageRuntime
from prove.task_graph.p57_public_geometry import public_geometry
from tools.task_graph.collect_libero_demo_gt import dump
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.p52_full_task import InsertJudge, structural_measurement
from tools.task_graph.p52_policy_pilot import Vision, validate_features
from tools.task_graph.p53_train_short import sha
from tools.task_graph import p56_collection as collection
from tools.task_graph import p56_evaluate as p56

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("strong_bc", "geometry_prior", "geometry_fixed", "geometry_online", "geometry_direct")
SOURCE_INDICES = (54, 56, 58, 60)
SCHEMA = "p57.geometry-control-eval.v1"


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def check_ref(value):
    if sha(value["path"]) != value["sha256"]:
        raise ValueError("Frozen reference changed: " + value["path"])


def verify_source_entry(entry, binding, collection_binding):
    """Verify original validation expert state without reading LABELS/private GT."""
    root = Path(entry["directory"]).resolve()
    cell = collection_binding["cells"][entry["source_cell_index"]]
    if cell["index"] not in SOURCE_INDICES or cell["split"] != "validation" or cell["source_policy"] != "expert":
        raise ValueError("P57 B must use original four validation expert entries")
    if read(root/"CELL.json") != cell or read(root/"BINDING.json") != collection_binding:
        raise ValueError("P56 validation cell or binding changed")
    if sha(root/"BINDING.json") != binding["collection_binding"]["sha256"]:
        raise ValueError("P56 entry collection binding hash differs")
    result = read(root/"RESULT.json")
    seal = read(root/"SEAL.json")
    if sha(root/"RESULT.json") != entry["result_sha256"] or sha(root/"SEAL.json") != entry["seal_sha256"]:
        raise ValueError("P56 entry result or seal changed")
    if result.get("seal_sha256") != entry["seal_sha256"] or seal.get("binding_sha256") != binding["collection_binding"]["sha256"]:
        raise ValueError("P56 entry seal chain differs")
    if (result.get("condition_id") != cell["condition_id"] or result.get("source_policy") != "expert"
            or result.get("near_entry_qualified") is not True):
        raise ValueError("Original expert did not qualify at near entry")
    near = read(root/"NEAR_ENTRY.json")
    if (near.get("schema") != collection.ENTRY_SCHEMA or near.get("source_cell_index") != cell["index"]
            or near.get("condition_id") != cell["condition_id"] or
            near.get("geometry_id") != cell["geometry_id"] or
            near.get("scale_xyz") != cell["scale_xyz"] or
            near.get("frame") != "private/near_entry_state.npz"):
        raise ValueError("Original near-entry state identity differs")
    state = root/near["frame"]
    if (sha(root/"NEAR_ENTRY.json") != entry["near_entry_sha256"] or
            sha(root/"NEAR_ENTRY.json") != seal["files"].get("NEAR_ENTRY.json") or
            sha(state) != near["frame_sha256"] or sha(state) != seal["files"].get(near["frame"])):
        raise ValueError("Original near-entry state is not sealed")
    with np.load(state, allow_pickle=False) as frame:
        if not all(key in frame for key in ("supervisor_sim_state", "supervisor_ctrl")):
            raise ValueError("Near-entry full simulator state is missing")
        if hashlib.sha256(frame["supervisor_sim_state"].tobytes()).hexdigest() != near["state_sha256"]:
            raise ValueError("Near-entry simulator state content changed")
    # Hash the remaining sealed files, including private files, without loading
    # their data. The policy never receives these paths or contents.
    for relative, expected in seal["files"].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("P56 sealed entry file changed: " + relative)
    return cell, near


def _expected_cells():
    return [dict(index=i*len(ARMS)+j, entry=i, arm=arm)
            for i in range(4) for j, arm in enumerate(ARMS)]


def prepare(train_manifest_path, geometry_manifest_path, a2_analysis_path,
            plan_path, a2_head_path, training_freeze_path,
            action_checkpoint_path, strong_bc_path):
    """Result-free 20-cell source/model freeze; performs no simulator call."""
    train_manifest_path = Path(train_manifest_path).resolve()
    geometry_manifest_path = Path(geometry_manifest_path).resolve()
    a2_analysis_path = Path(a2_analysis_path).resolve()
    plan_path = Path(plan_path).resolve()
    if plan_path.name != "PLAN_B.md":
        raise ValueError("P57 B plan required")
    train = read(train_manifest_path)
    parent = train["collection_binding"]
    check_ref(parent)
    collection_binding = read(parent["path"])
    collection.verify(collection_binding)
    source = {row["cell_index"]: row for row in train["episodes"]}
    if len(source) != len(train["episodes"]) or not set(SOURCE_INDICES).issubset(source):
        raise ValueError("P56 manifest lacks exact four validation expert cells")
    entries = []
    for index in SOURCE_INDICES:
        root = Path(source[index]["directory"]).resolve()
        entries.append(dict(source_cell_index=index, directory=str(root),
                            result_sha256=sha(root/"RESULT.json"),
                            seal_sha256=sha(root/"SEAL.json"),
                            near_entry_sha256=sha(root/"NEAR_ENTRY.json")))
    geometry = read(geometry_manifest_path)
    if geometry.get("schema") != "p57.geometry-pack-manifest.v1" or geometry.get("cohort_counts") != dict(
            train=34, validation=8, transitions=573, frames=615):
        raise ValueError("Wrong P57 geometry training bundle")
    a2 = torch.load(a2_head_path, map_location="cpu", weights_only=True)
    if (a2.get("schema") != "p57.a2-translation-readout.v1" or a2.get("mode") != "mask_geometry"
            or a2.get("pack_manifest_sha256") != sha(geometry_manifest_path)):
        raise ValueError("A2 mask readout differs from qualified development choice")
    analysis = read(a2_analysis_path)
    mask_run = next((row for row in analysis.get("runs", []) if row.get("mode") == "mask_geometry"), None)
    if (analysis.get("schema") != "p57.a2-translation-analysis.v1" or
            analysis.get("pack_manifest_sha256") != sha(geometry_manifest_path) or
            mask_run is None or mask_run.get("head_sha256") != sha(a2_head_path) or
            mask_run.get("gate", {}).get("qualified") is not True):
        raise ValueError("Frozen A2 mask qualification evidence differs")
    p56.load_skill(strong_bc_path, "teacher_only_bc")
    # Action checkpoint validation is shared with the runtime loader; this
    # also checks its A2 head binding before any evaluation state is opened.
    from prove.task_graph.p57_geometry_policy import load_geometry_policy
    _, _, action_package = load_geometry_policy(action_checkpoint_path, a2_head_path, device="cpu")
    if action_package.get("schema") != "p57.geometry-policy.v1":
        raise ValueError("Wrong P57 B action checkpoint")
    from tools.task_graph.p57_train_geometry_policy import verify as verify_training
    train_freeze = verify_training(training_freeze_path, geometry_manifest_path,
                                   a2_head_path, action_checkpoint_path)
    if (action_package.get("freeze_sha256") != sha(training_freeze_path) or
            train_freeze["plan"]["sha256"] != sha(plan_path)):
        raise ValueError("Action checkpoint differs from frozen P57 B plan/training")
    binding = dict(schema=SCHEMA, scope="opened_four_validation_near_entries_s9_h24",
        collection_binding=parent, train_manifest=ref(train_manifest_path),
        geometry_manifest=ref(geometry_manifest_path), a2_analysis=ref(a2_analysis_path),
        plan=ref(plan_path), training_freeze=ref(training_freeze_path),
        a2_head=ref(a2_head_path), action_checkpoint=ref(action_checkpoint_path),
        strong_bc=ref(strong_bc_path), entries=entries, cells=_expected_cells(),
        arms=list(ARMS), max_actions=24, automatic_retries=0,
        sources={path: sha(ROOT/path) for path in source_closure(ROOT,
            ["tools/task_graph/p57_evaluate_geometry.py", "tools/task_graph/p52_vision_worker.py",
             "prove/task_graph/p57_geometry_policy.py"])},
        primary_success="official native goal AND at least one finger touching book jointly hold3",
        denominator="all 4 opened validation near-entry states for each of 5 arms; no retry")
    verify(binding)
    return binding


def verify(binding):
    if (binding.get("schema") != SCHEMA or binding.get("arms") != list(ARMS) or
            binding.get("cells") != _expected_cells() or binding.get("max_actions") != 24 or
            binding.get("automatic_retries") != 0 or len(binding.get("entries", [])) != 4):
        raise ValueError("Changed P57 B 20-cell execution scope")
    for key in ("collection_binding", "train_manifest", "geometry_manifest", "a2_analysis",
                "plan", "training_freeze",
                "a2_head", "action_checkpoint", "strong_bc"):
        check_ref(binding[key])
    for source, expected in binding["sources"].items():
        if sha(ROOT/source) != expected:
            raise ValueError("Frozen evaluator source changed: " + source)
    geometry = read(binding["geometry_manifest"]["path"])
    if geometry.get("schema") != "p57.geometry-pack-manifest.v1" or geometry.get("cohort_counts") != dict(
            train=34, validation=8, transitions=573, frames=615):
        raise ValueError("Frozen P57 geometry bundle scope changed")
    analysis = read(binding["a2_analysis"]["path"])
    mask_run = next((row for row in analysis.get("runs", []) if row.get("mode") == "mask_geometry"), None)
    if (analysis.get("schema") != "p57.a2-translation-analysis.v1" or
            analysis.get("pack_manifest_sha256") != binding["geometry_manifest"]["sha256"] or
            mask_run is None or mask_run.get("head_sha256") != binding["a2_head"]["sha256"] or
            mask_run.get("gate", {}).get("qualified") is not True):
        raise ValueError("A2 mask qualification changed")
    from tools.task_graph.p57_train_geometry_policy import verify as verify_training
    freeze = verify_training(binding["training_freeze"]["path"],
        binding["geometry_manifest"]["path"], binding["a2_head"]["path"],
        binding["action_checkpoint"]["path"])
    if freeze["plan"]["sha256"] != binding["plan"]["sha256"]:
        raise ValueError("P57 B training and evaluation plans differ")
    from prove.task_graph.p57_geometry_policy import load_geometry_policy
    _, _, action_package = load_geometry_policy(
        binding["action_checkpoint"]["path"], binding["a2_head"]["path"], device="cpu")
    if action_package.get("freeze_sha256") != binding["training_freeze"]["sha256"]:
        raise ValueError("Action checkpoint training freeze changed")
    p56.load_skill(binding["strong_bc"]["path"], "teacher_only_bc")
    train = read(binding["train_manifest"]["path"])
    if train["collection_binding"] != binding["collection_binding"]:
        raise ValueError("P56 train manifest collection binding differs")
    collection_binding = read(binding["collection_binding"]["path"])
    collection.verify(collection_binding)
    if tuple(entry["source_cell_index"] for entry in binding["entries"]) != SOURCE_INDICES:
        raise ValueError("Changed four original P56 validation expert entries")
    for entry in binding["entries"]:
        verify_source_entry(entry, binding, collection_binding)
    return collection_binding


def mask_public_features(frame):
    """Exact P52 mask boxes/proprio plus A2 tokens; no vision extraction."""
    masks = np.asarray(frame["visible_masks"])
    geometry = public_geometry(frame)
    boxes, visible = zip(*(mask_geometry(view) for view in masks))
    visible = np.asarray(visible, dtype=bool)
    if not np.array_equal(visible, geometry["mask_valid"]):
        raise ValueError("P52 bbox and P57 mask visibility differ")
    R = geometry["eef_transform"][:3, :3]
    proprio = np.concatenate((np.asarray(frame["eef_pos"]), R[:, :2].T.reshape(-1),
                              np.asarray(frame["gripper_qpos"]), np.asarray(frame["joint_pos"])))
    result = dict(raw=np.zeros((2, 3, 3072), dtype=np.float32),
                  boxes=np.asarray(boxes, dtype=np.float32)/448.,
                  visible=visible.astype(np.float32), proprio=proprio.astype(np.float32),
                  **geometry)
    shapes = dict(raw=(2,3,3072), boxes=(2,3,4), visible=(2,3), proprio=(18,),
                  mask_tokens=(2,3,55), mask_valid=(2,3), camera_K=(2,3,3),
                  camera_pose=(2,4,4), eef_transform=(4,4))
    if set(result) != set(shapes):
        raise ValueError("Unknown or missing public runtime feature")
    for key, shape in shapes.items():
        if result[key].shape != shape or (key != "mask_valid" and not np.isfinite(result[key]).all()):
            raise ValueError("Invalid public runtime feature: " + key)
    if not np.all((result["boxes"] >= 0) & (result["boxes"] <= 1)):
        raise ValueError("Public bbox outside normalized image")
    return {key: torch.as_tensor(value, dtype=(torch.bool if key == "mask_valid" else torch.float32))
            for key, value in result.items()}


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Nonfinite runtime telemetry")
        return value
    raise TypeError("Unknown runtime telemetry type: " + str(type(value)))


def _append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")


def _weights_unchanged(model, before):
    if any(not torch.equal(before[key], value) for key, value in model.state_dict().items()):
        raise RuntimeError("Frozen deployment network weights mutated")


def run(binding_path, index, output):
    """Execute one predeclared cell; no retry and no private policy input."""
    binding = read(binding_path)
    collection_binding = verify(binding)
    if not isinstance(index, int) or not 0 <= index < 20:
        raise ValueError("Cell outside frozen five-arm/four-entry scope")
    cell = binding["cells"][index]
    entry = binding["entries"][cell["entry"]]
    source_cell = collection_binding["cells"][entry["source_cell_index"]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump(output/"BINDING.json", binding)
    dump(output/"CELL.json", cell)
    started = time.time()
    dump(output/"EXECUTION.json", dict(schema="p57.geometry-control-execution.v1",
        binding_sha256=sha(binding_path), cell=cell, run_started_unix=started,
        command=["run", "--binding", str(Path(binding_path).resolve()),
                 "--index", str(index), "--output", str(output.resolve())],
        automatic_retry=False))
    (output/"public").mkdir()
    (output/"private").mkdir()
    arm = cell["arm"]
    env = renderer = vision = runtime = None
    steps = 0
    try:
        if arm == "strong_bc":
            model, _ = p56.load_skill(binding["strong_bc"]["path"], "teacher_only_bc")
            runtime = StageRuntime(model, online=False)
            readout = None
        else:
            from prove.task_graph.p57_geometry_policy import (
                GeometryPolicyRuntime, load_geometry_policy)
            model, readout, _ = load_geometry_policy(
                binding["action_checkpoint"]["path"], binding["a2_head"]["path"],
                device="cpu")
            runtime = GeometryPolicyRuntime(model, readout, arm=arm, device="cpu")
            runtime.reset()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        _, _, original = collection.verify(collection_binding)
        child, geometry, props = collection.build_child(collection_binding, source_cell, output, original)
        near = read(Path(entry["directory"])/"NEAR_ENTRY.json")
        scene_sha = hashlib.sha256(child["environment"]["relocated_xml"].encode()).hexdigest()
        if scene_sha != near["scene_xml_sha256"]:
            raise ValueError("Restored geometry XML differs from sealed near-entry scene")
        env, _ = collection.initialize_child(child, output, geometry["axes"], props["halfsize"],
                                             near_entry=Path(entry["directory"])/near["frame"])
        if arm == "strong_bc":
            vision = Vision(original["vision"])
            dump(output/"VISUAL_BINDING.json", vision.binding)
        else:
            dump(output/"VISUAL_BINDING.json", dict(
                mode="mask_geometry_without_dino_reextraction",
                a2_head=binding["a2_head"], action_checkpoint=binding["action_checkpoint"],
                public_sources=["visible_masks", "camera_K", "camera_pose", "eef_pos",
                                "eef_quat_xyzw", "gripper_qpos", "joint_pos"],
                raw="zero_placeholder_not_consumed_by_mask_geometry"))
        renderer = collection.PublicRenderer(env, original["public"])
        names = collection._names(env)
        judge = InsertJudge()
        obs = env._get_observations(force_update=True)
        previous = previous_action = None
        official_hold = 0
        official_ever = official_instant_ever = False
        while True:
            frame = output/"public"/f"frame_{steps:04d}.npz"
            np.savez_compressed(frame, **renderer.observe(obs))
            if arm == "strong_bc":
                features = vision.encode(frame)
                validate_features(features, model.config)
                if previous is None:
                    runtime.enter(features)
                z_before = runtime.z.detach().clone()
                feedback_loss = None
                if previous is not None:
                    feedback_loss = runtime.feedback(previous, previous_action, features)
                action = runtime.command(features).detach().cpu().numpy()
                telemetry = dict(z_before=z_before, z_after=runtime.z,
                    feedback_loss=feedback_loss, z_updates=runtime.updates,
                    public_source="rgb_masks_proprio_frozen_DINO_LiLa")
            else:
                with np.load(frame, allow_pickle=False) as public_frame:
                    features = mask_public_features(public_frame)
                action, telemetry = runtime.step(features)
                action = np.asarray(action, dtype=np.float64)
                telemetry = dict(telemetry, public_source="masks_boxes_proprio_camera_eef_no_DINO")
            torch.save(features, output/"public"/f"features_{steps:04d}.pt")
            if action.shape != (7,) or not np.isfinite(action).all() or action[-1] != 1:
                raise ValueError("Invalid closed-gripper OSC7 command")
            official, fingers, wrong = structural_measurement(env, names)
            judgment = judge.observe(official, fingers, wrong)
            official_hold = official_hold + 1 if official else 0
            official_ever = official_ever or official_hold >= 3
            official_instant_ever = official_instant_ever or bool(official)
            stop = judgment["stop"] or ("micro_budget24" if steps >= 24 else None)
            _append(output/"public/ACTIONS.jsonl", dict(t=steps,
                observation=frame.name, observation_sha256=sha(frame),
                features_sha256=sha(output/"public"/f"features_{steps:04d}.pt"),
                telemetry=telemetry, proposed_action=action.tolist(),
                executed=stop is None, executed_physical_steps=steps))
            _append(output/"private/JUDGMENTS.jsonl", dict(t=steps,
                official_success=official, official_hold=official_hold,
                finger_contacts=fingers, wrong_target=wrong, **judgment))
            if stop:
                break
            if arm == "strong_bc":
                previous, previous_action = features, torch.tensor(action[:6], dtype=torch.float32)
            obs, _, _, _ = env.step(action.copy())
            steps += 1
        _weights_unchanged(model, before)
        verify(binding)
        files = {str(path.relative_to(output)): sha(path) for name in ("public", "private")
                 for path in (output/name).iterdir()}
        for relative in ("CELL.json", "EXECUTION.json", "VISUAL_BINDING.json",
                         "SUPERVISOR_CALIBRATION.json"):
            files[relative] = sha(output/relative)
        dump(output/"SEAL.json", dict(schema="p57.geometry-control-seal.v1",
            binding_sha256=sha(output/"BINDING.json"), cell=cell,
            files=files, executed_physical_steps=steps, automatic_retries=0))
        result = dict(schema="p57.geometry-control-result.v1", cell=cell,
            condition_id=source_cell["condition_id"], source_cell_index=source_cell["index"],
            near_entry_qualified=True, micro_executed=True,
            system_success=bool(judgment["complete"]), micro_success=bool(judgment["complete"]),
            official_only_hold3=official_ever, official_success=bool(official),
            official_instant_ever=official_instant_ever,
            executed_physical_steps=steps, reason=stop, automatic_retries=0,
            full_task_success_evidence=False, seal_sha256=sha(output/"SEAL.json"))
        dump(output/"RESULT.json", result)
        return result
    except Exception as exc:
        dump(output/"ABORT.json", dict(cell=cell, reason=str(exc),
            executed_physical_steps=steps, automatic_retry=False))
        raise
    finally:
        if runtime is not None and hasattr(runtime, "exit"):
            runtime.exit()
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()
        if vision is not None:
            vision.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze source/model/entry chain; no physics")
    for name in ("train_manifest", "geometry_manifest", "a2_analysis", "plan",
                 "a2_head", "training_freeze", "action_checkpoint", "strong_bc", "output"):
        prep.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    cell = commands.add_parser("run", help="Execute one predeclared H24 evaluation cell")
    cell.add_argument("--binding", type=Path, required=True)
    cell.add_argument("--index", type=int, required=True)
    cell.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        if args.output.exists():
            raise FileExistsError(args.output)
        dump(args.output, prepare(args.train_manifest, args.geometry_manifest,
                                  args.a2_analysis, args.plan, args.a2_head,
                                  args.training_freeze, args.action_checkpoint, args.strong_bc))
    else:
        print(json.dumps(run(args.binding, args.index, args.output), allow_nan=False))


if __name__ == "__main__":
    main()
