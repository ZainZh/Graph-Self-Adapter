#!/usr/bin/env python3
"""P57 Stage A offline geometry readout; never a physical or Adapter rollout."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from prove.task_graph.p57_ray_geometry import _frame, fit_virtual_anchor_frames
from tools.task_graph.p57_train_measurement import load_bundle

MODES = ("mask_geometry", "dino_lila", "fusion")
SEEDS = (11, 22)
METHODS = ("instant", "history", "pose_prior")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1.) / 2., -1., 1.)))


def pose_error(estimated, truth):
    relative_est = np.linalg.inv(estimated[1]) @ estimated[0]
    relative_gt = np.linalg.inv(truth[1]) @ truth[0]
    return float(np.linalg.norm(relative_est[:3, 3] - relative_gt[:3, 3])), angle(
        relative_est[:3, :3].T @ relative_gt[:3, :3])


def _prior_frames(prior, eef):
    held = array(prior["book_in_eef_anchors"]).astype(float)
    goal = array(prior["desired_world_anchors"]).astype(float)
    book = held @ eef[:3, :3].T + eef[:3, 3]
    book_frame = _frame(book)[0]
    goal_frame = _frame(goal)[0]
    if book_frame is None or goal_frame is None:
        return None
    return book_frame, goal_frame


def _motion_diagnostic(labels):
    book = array(labels["book_in_eef"]).astype(float)
    goal = array(labels["desired_book_world"]).astype(float)
    initial_book = book[0]
    initial_goal = goal[0]
    shifts = np.linalg.norm(book[:, :3, 3] - initial_book[:3, 3], axis=1)
    turns = np.array([angle(initial_book[:3, :3].T @ b[:3, :3]) for b in book])
    step_shifts = np.linalg.norm(np.diff(book[:, :3, 3], axis=0), axis=1)
    step_turns = np.array([angle(a[:3, :3].T @ b[:3, :3])
                           for a, b in zip(book[:-1], book[1:])])
    goal_shifts = np.linalg.norm(goal[:, :3, 3] - initial_goal[:3, 3], axis=1)
    goal_turns = np.array([angle(initial_goal[:3, :3].T @ g[:3, :3]) for g in goal])
    rms = lambda x: float(np.sqrt(np.mean(x**2))) if len(x) else 0.
    return dict(book_in_eef_max_translation_m=float(shifts.max()),
                book_in_eef_max_rotation_rad=float(turns.max()),
                book_in_eef_step_rms_translation_m=rms(step_shifts),
                book_in_eef_step_rms_rotation_rad=rms(step_turns),
                fixed_goal_max_translation_drift_m=float(goal_shifts.max()),
                fixed_goal_max_rotation_drift_rad=float(goal_turns.max()))


def analyze_episode(prediction, episode, prior, *, history=8):
    """Fit public rays first; access private labels only for scored diagnostics."""
    if any(prediction[k] != episode[k] for k in ("condition_id", "source_policy", "source_seal_sha256")):
        raise ValueError("Prediction and bundle identity mismatch")
    public = episode["input"]
    expected_hashes = [frame["sha256"] for frame in public["public_frames"]]
    if prediction["public_frame_sha256"] != expected_hashes:
        raise ValueError("Public frame identity mismatch")
    for key, pred_key in (("camera_K", "camera_K"), ("camera_pose", "camera_pose"),
                          ("eef_transform", "eef_transform")):
        if not np.allclose(array(prediction[pred_key]), array(public[key]), atol=1e-4):
            raise ValueError(f"Prediction public {key} differs from sealed bundle")
    uv = array(prediction["pred_uv"])
    valid = array(prediction["role_valid"])
    K = array(prediction["camera_K"])
    camera = array(prediction["camera_pose"])
    eef = array(prediction["eef_transform"])
    fits = fit_virtual_anchor_frames(uv, valid, K, camera, eef, history=history)
    if len(fits) != len(array(episode["labels"]["world_book"])):
        raise ValueError("Prediction/label frame count differs")
    scores = {name: [] for name in METHODS}
    missing = {name: 0 for name in METHODS}
    ray_rank_failures = {name: 0 for name in ("instant", "history")}
    frame_collinear = {name: 0 for name in ("instant", "history")}
    ray_conditions = {name: [] for name in ("instant", "history")}
    deformations = {name: [] for name in ("instant", "history")}
    for t, result in enumerate(fits):
        labels = episode["labels"]
        truth = (array(labels["world_book"])[t], array(labels["desired_book_world"])[t])
        for name in ("instant", "history"):
            branch = result[name]
            ray_conditions[name].extend(
                value for role in ("book", "desired") for value in branch["condition"][role]
                if value is not None and np.isfinite(value))
            deformations[name].extend(
                branch["deformation_rms_m"][role] for role in ("book", "desired")
                if branch["deformation_rms_m"][role] is not None)
            if any("rank_deficient" in branch["reason"][role] or
                   "ill_conditioned" in branch["reason"][role]
                   for role in ("book", "desired")):
                ray_rank_failures[name] += 1
            if any(branch["frame_reason"][role] == "frame_collinear"
                   for role in ("book", "desired")):
                frame_collinear[name] += 1
            if all(branch["valid"].values()):
                scores[name].append(pose_error(
                    (branch["book_frame"], branch["desired_frame"]), truth))
            else:
                missing[name] += 1
        pose = _prior_frames(prior, eef[t])
        if pose is None:
            missing["pose_prior"] += 1
        else:
            scores["pose_prior"].append(pose_error(pose, truth))
    frames = len(fits)
    methods = {}
    for name in METHODS:
        value = np.asarray(scores[name], dtype=float).reshape(-1, 2)
        methods[name] = dict(valid_frames=len(value), total_frames=frames,
                             coverage=len(value) / frames,
                             relative_translation_m=float(value[:, 0].mean()) if len(value) else None,
                             relative_rotation_rad=float(value[:, 1].mean()) if len(value) else None,
                             missing_frames=missing[name])
        for column, label in ((0, "relative_translation"), (1, "relative_rotation")):
            for quantile, suffix in ((.5, "p50"), (.9, "p90"), (1., "max")):
                methods[name][f"{label}_{suffix}_{'m' if column == 0 else 'rad'}"] = (
                    float(np.quantile(value[:, column], quantile)) if len(value) else None)
        if name in ray_rank_failures:
            methods[name].update(rank_or_condition_failed_frames=ray_rank_failures[name],
                                 frame_collinear_frames=frame_collinear[name],
                                 max_ray_condition=max(ray_conditions[name]) if ray_conditions[name] else None,
                                 max_frame_deformation_m=max(deformations[name]) if deformations[name] else None)
    return dict(condition_id=episode["condition_id"], source_policy=episode["source_policy"],
                frames=frames, public_entry_valid=bool(valid[0].all()),
                entry_fit_valid=bool(all(fits[0]["instant"]["valid"].values())),
                methods=methods, rigid_assumption=_motion_diagnostic(episode["labels"]))


def macro(rows):
    """One vote per trajectory; missing episodes remain visible in coverage."""
    result = {}
    for name in METHODS:
        entries = [row["methods"][name] for row in rows]
        scored = [entry for entry in entries if entry["valid_frames"]]
        by_condition = {}
        for row in rows:
            entry = row["methods"][name]
            if entry["valid_frames"]:
                by_condition.setdefault(row["condition_id"], []).append(entry)
        def worst_condition(field):
            if not by_condition:
                return None
            condition, value = max(((condition, float(np.mean([entry[field] for entry in group])))
                                    for condition, group in by_condition.items()), key=lambda pair: pair[1])
            return dict(condition_id=condition, value=value)
        result[name] = dict(episodes=len(entries), scored_episodes=len(scored),
                            mean_coverage=float(np.mean([e["coverage"] for e in entries])),
                            relative_translation_m=float(np.mean(
                                [e["relative_translation_m"] for e in scored])) if scored else None,
                            relative_rotation_rad=float(np.mean(
                                [e["relative_rotation_rad"] for e in scored])) if scored else None,
                            worst_condition_translation_m=worst_condition("relative_translation_m"),
                            worst_condition_rotation_rad=worst_condition("relative_rotation_rad"),
                            max_episode_translation_m=max((e["relative_translation_max_m"] for e in scored),
                                                          default=None),
                            max_episode_rotation_rad=max((e["relative_rotation_max_rad"] for e in scored),
                                                         default=None))
    result["public_valid_entries"] = sum(row["public_entry_valid"] for row in rows)
    result["distinct_public_valid_conditions"] = len({
        row["condition_id"] for row in rows if row["public_entry_valid"]})
    result["fit_valid_entries"] = sum(row["entry_fit_valid"] for row in rows)
    result["distinct_fit_valid_conditions"] = len({
        row["condition_id"] for row in rows if row["entry_fit_valid"]})
    result["rigid_assumption_worst"] = {
        key: max(row["rigid_assumption"][key] for row in rows)
        for key in rows[0]["rigid_assumption"]}
    return result


def head_gate(summary):
    current = summary["history"]
    prior = summary["pose_prior"]
    public_ok = summary["distinct_public_valid_conditions"] == 4 and summary["public_valid_entries"] == 8
    entry_fit_ok = summary["distinct_fit_valid_conditions"] == 4 and summary["fit_valid_entries"] == 8
    coverage_ok = current["scored_episodes"] == 8 and current["mean_coverage"] == 1.
    accuracy = coverage_ok and current["relative_translation_m"] <= .010 and (
        current["relative_rotation_rad"] <= np.deg2rad(15))
    better_prior = coverage_ok and prior["scored_episodes"] == 8 and prior["mean_coverage"] == 1. and (
        current["relative_translation_m"] < prior["relative_translation_m"] and
        current["relative_rotation_rad"] < prior["relative_rotation_rad"])
    return dict(public_entries=bool(public_ok), geometric_entries=bool(entry_fit_ok),
                full_development_frame_coverage=bool(coverage_ok),
                geometry_thresholds=bool(accuracy), beats_train_pose_prior=bool(better_prior),
                qualified=bool(public_ok and entry_fit_ok and coverage_ok and accuracy and better_prior))


def _read_predictions(path, manifest_hash):
    path = Path(path).resolve()
    head_path = path.with_name("HEAD.pt")
    public = torch.load(path, map_location="cpu", weights_only=True)
    head = torch.load(head_path, map_location="cpu", weights_only=True)
    if public.get("schema") != "p57.public-anchor-predictions.v1" or public.get("labels_included") is not False:
        raise ValueError("Unexpected or private P57 predictions")
    if head.get("schema") != "p57.geometry-measurement-checkpoint.v1":
        raise ValueError("Unexpected P57 head schema")
    if head.get("pack_manifest_sha256") != manifest_hash:
        raise ValueError("P57 head was trained on a different pack")
    if head.get("public_predictions_sha256") != digest(path):
        raise ValueError("P57 public predictions changed")
    mode, seed = head["mode"], head["seed"]
    if mode not in MODES or seed not in SEEDS or len(public["predictions"]) != 8:
        raise ValueError("Unexpected P57 head or development episodes")
    return (mode, seed), head["baseline_pose_prior"], public["predictions"], head_path


def analyze(manifest_path, prediction_paths):
    manifest_path = Path(manifest_path).resolve()
    bundle = load_bundle(manifest_path)
    dev = {(ep["condition_id"], ep["source_policy"]): ep
           for ep in bundle["episodes"] if ep["split"] == "validation"}
    if len(dev) != 8 or len(prediction_paths) != 6:
        raise ValueError("Expected eight development trajectories and six P57 heads")
    runs = []
    seen = set()
    for path in prediction_paths:
        identity, prior, predictions, head_path = _read_predictions(path, digest(manifest_path))
        if identity in seen:
            raise ValueError("Repeated P57 mode/seed")
        seen.add(identity)
        rows = []
        used = set()
        for prediction in predictions:
            key = prediction["condition_id"], prediction["source_policy"]
            if key not in dev or key in used:
                raise ValueError("Missing or repeated development identity")
            used.add(key)
            rows.append(analyze_episode(prediction, dev[key], prior))
        if used != set(dev):
            raise ValueError("Incomplete development predictions")
        summary = macro(rows)
        runs.append(dict(mode=identity[0], seed=identity[1],
                         head_sha256=digest(head_path), predictions_sha256=digest(path),
                         per_episode=rows, macro=summary, gate=head_gate(summary)))
    if seen != {(mode, seed) for mode in MODES for seed in SEEDS}:
        raise ValueError("Incomplete six-head comparison")
    by_mode = {mode: dict(seeds=list(SEEDS), qualified=all(
        next(run for run in runs if run["mode"] == mode and run["seed"] == seed)["gate"]["qualified"]
        for seed in SEEDS)) for mode in MODES}
    return dict(schema="p57.ray-geometry-analysis.v1", pack_manifest_sha256=digest(manifest_path),
                source_sha256={str(Path(__file__).resolve()): digest(__file__),
                    str(Path(__file__).resolve().parents[2] / "prove/task_graph/p57_ray_geometry.py"):
                        digest(Path(__file__).resolve().parents[2] / "prove/task_graph/p57_ray_geometry.py"),
                    str(Path(__file__).resolve().parents[2] / "prove/task_graph/p57_public_geometry.py"):
                        digest(Path(__file__).resolve().parents[2] / "prove/task_graph/p57_public_geometry.py")},
                history_max_frames=8, ray_condition_limit=1e5,
                scope="Offline development only; private labels used for scoring, never ray fitting",
                runs=runs, by_mode=by_mode,
                any_mode_qualified=any(item["qualified"] for item in by_mode.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.manifest, args.prediction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
