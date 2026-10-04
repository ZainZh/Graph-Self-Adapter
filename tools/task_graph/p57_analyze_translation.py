#!/usr/bin/env python3
"""P57 A2 offline development score; private labels never enter ray fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from prove.task_graph.p57_ray_geometry import _frame
from prove.task_graph.p57_translation_geometry import (
    fit_origin_translations, frames_with_orientation_prior)
from tools.task_graph.p57_train_measurement import load_bundle

MODES = ("mask_geometry", "dino_lila", "fusion")
METHODS = ("instant", "window8", "pose_prior")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _prior_frames(prior, eef):
    held = array(prior["book_in_eef_anchors"]).astype(float)
    goal = array(prior["desired_world_anchors"]).astype(float)
    if held.shape != (3, 3) or goal.shape != (3, 3):
        raise ValueError("Training pose prior must contain two sets of three anchors")
    held_frame = _frame(held)[0]
    goal_frame = _frame(goal)[0]
    if held_frame is None or goal_frame is None:
        raise ValueError("Degenerate training pose prior")
    book = eef @ held_frame
    return book, goal_frame, held_frame[:3, :3], goal_frame[:3, :3]


def _relative_error(estimated, truth):
    estimated_relative = np.linalg.inv(estimated[1]) @ estimated[0]
    true_relative = np.linalg.inv(truth[1]) @ truth[0]
    vector_mm = 1000. * (estimated_relative[:3, 3] - true_relative[:3, 3])
    R = estimated_relative[:3, :3].T @ true_relative[:3, :3]
    degrees = float(np.rad2deg(np.arccos(np.clip((np.trace(R)-1.)/2., -1., 1.))))
    return vector_mm, float(np.linalg.norm(vector_mm)), degrees


def _summary(frames, total):
    scored = [frame for frame in frames if frame["valid"]]
    lengths = np.array([frame["relative_translation_norm_mm"] for frame in scored])
    angles = np.array([frame["relative_rotation_deg"] for frame in scored])
    result = dict(valid_frames=len(scored), total_frames=total,
                  coverage=float(len(scored)/total),
                  relative_translation_mean_mm=float(lengths.mean()) if len(lengths) else None,
                  relative_rotation_mean_deg=float(angles.mean()) if len(angles) else None,
                  frames=frames)
    for name, values in (("relative_translation", lengths), ("relative_rotation", angles)):
        unit = "mm" if name == "relative_translation" else "deg"
        for quantile, suffix in ((.5, "p50"), (.9, "p90"), (1., "max")):
            result[f"{name}_{suffix}_{unit}"] = (
                float(np.quantile(values, quantile)) if len(values) else None)
    return result


def analyze_episode(prediction, episode, prior, *, history=8):
    """Fit only from public prediction; use private poses afterwards to score."""
    for key in ("condition_id", "split", "source_policy", "source_seal_sha256"):
        if prediction.get(key) != episode.get(key):
            raise ValueError(f"Prediction and sealed episode {key} differ")
    public = episode["input"]
    if prediction.get("public_frame_sha256") != [f["sha256"] for f in public["public_frames"]]:
        raise ValueError("Public frame hash identity differs")
    for key in ("camera_K", "camera_pose", "eef_transform"):
        if not np.allclose(array(prediction[key]), array(public[key]), rtol=0, atol=1e-4):
            raise ValueError(f"Prediction public {key} differs from sealed bundle")
    uv = array(prediction["pred_uv"])
    valid = array(prediction["role_valid"])
    K = array(prediction["camera_K"])
    camera = array(prediction["camera_pose"])
    eef = array(prediction["eef_transform"])
    fits = fit_origin_translations(uv, valid, K, camera, eef, history=history)
    labels = episode["labels"]
    truth_book = array(labels["world_book"])
    truth_goal = array(labels["desired_book_world"])
    if len(fits) != len(truth_book) or truth_goal.shape != truth_book.shape:
        raise ValueError("Prediction/private score frame count differs")
    scored = {method: [] for method in METHODS}
    for t, fit in enumerate(fits):
        pose_book, pose_goal, held_R, goal_R = _prior_frames(prior, eef[t])
        truth = (truth_book[t], truth_goal[t])
        for method in METHODS:
            if method == "pose_prior":
                estimated, diagnostic = (pose_book, pose_goal), None
            else:
                diagnostic = fit[method]
                estimated = frames_with_orientation_prior(
                    diagnostic, eef[t], held_R, goal_R)
            if estimated is None:
                scored[method].append(dict(t=t, valid=False, reason=diagnostic["reason"],
                                           rank=diagnostic["rank"],
                                           condition=diagnostic["condition"],
                                           rays=diagnostic["rays"]))
                continue
            vector, length, degrees = _relative_error(estimated, truth)
            scored[method].append(dict(t=t, valid=True,
                relative_translation_vector_mm=vector.tolist(),
                relative_translation_norm_mm=length, relative_rotation_deg=degrees,
                rank=None if diagnostic is None else diagnostic["rank"],
                condition=None if diagnostic is None else diagnostic["condition"],
                rms_ray_m=None if diagnostic is None else diagnostic["rms_ray_m"],
                rays=None if diagnostic is None else diagnostic["rays"]))
    return dict(condition_id=episode["condition_id"], source_policy=episode["source_policy"],
                frames=len(fits), public_entry_valid=bool(valid[0].all()),
                instant_entry_valid=bool(all(fits[0]["instant"]["valid"].values())),
                window_entry_valid=bool(all(fits[0]["window8"]["valid"].values())),
                methods={method: _summary(scored[method], len(fits)) for method in METHODS})


def macro(rows):
    """Give each trajectory one vote; retain every individual and case score."""
    methods = {}
    for method in METHODS:
        entries = [row["methods"][method] for row in rows]
        valid = [entry for entry in entries if entry["valid_frames"]]
        methods[method] = dict(episodes=len(entries), scored_episodes=len(valid),
            mean_coverage=float(np.mean([entry["coverage"] for entry in entries])) if entries else 0.,
            relative_translation_macro_mm=float(np.mean([
                entry["relative_translation_mean_mm"] for entry in valid])) if valid else None,
            relative_rotation_macro_deg=float(np.mean([
                entry["relative_rotation_mean_deg"] for entry in valid])) if valid else None,
            max_frame_translation_mm=max((entry["relative_translation_max_mm"] for entry in valid),default=None),
            max_frame_rotation_deg=max((entry["relative_rotation_max_deg"] for entry in valid),default=None))
    condition_sources = {}
    for row in rows:
        condition_sources.setdefault(row["condition_id"], set()).add(row["source_policy"])
    return dict(trajectories=len(rows), distinct_conditions=len(condition_sources),
                paired_conditions=sum(sources == {"expert", "policy_fixed"}
                                      for sources in condition_sources.values()),
                public_valid_entries=sum(row["public_entry_valid"] for row in rows),
                instant_valid_entries=sum(row["instant_entry_valid"] for row in rows),
                window_valid_entries=sum(row["window_entry_valid"] for row in rows),
                methods=methods)


def a2_gate(summary):
    current = summary["methods"]["window8"]
    prior = summary["methods"]["pose_prior"]
    entry = (summary["trajectories"] == 8 and summary["distinct_conditions"] == 4 and
        summary["paired_conditions"] == 4 and summary["public_valid_entries"] == 8 and
        summary["instant_valid_entries"] == 8 and summary["window_valid_entries"] == 8)
    coverage = current["scored_episodes"] == 8 and current["mean_coverage"] == 1.
    translation = coverage and current["relative_translation_macro_mm"] <= 5.
    better_prior = (coverage and prior["scored_episodes"] == 8 and
        prior["mean_coverage"] == 1. and
        current["relative_translation_macro_mm"] <= .8 * prior["relative_translation_macro_mm"])
    orientation = (coverage and current["relative_rotation_macro_deg"] <= 2. and
                   current["max_frame_rotation_deg"] <= 5.)
    return dict(all_entries_valid=bool(entry), full_frame_coverage=bool(coverage),
                relative_translation_at_most_5mm=bool(translation),
                translation_improves_prior_at_least_20pct=bool(better_prior),
                fixed_orientation_macro_at_most_2deg_and_max_frame_at_most_5deg=bool(orientation),
                qualified=bool(entry and coverage and translation and better_prior and orientation))


def _predictions(path, manifest_hash):
    path = Path(path).resolve()
    public = torch.load(path, map_location="cpu", weights_only=True)
    head_path = path.with_name("HEAD.pt")
    head = torch.load(head_path, map_location="cpu", weights_only=True)
    if (public.get("schema") != "p57.a2-public-origin-predictions.v1" or
            public.get("role_order") != ["book", "desired_book_frame"] or
            public.get("labels_included") is not False):
        raise ValueError("Unexpected or private P57 A2 predictions")
    if head.get("schema") != "p57.a2-translation-readout.v1" or head.get("pack_manifest_sha256") != manifest_hash:
        raise ValueError("Wrong P57 A2 checkpoint or pack binding")
    if head.get("public_predictions_sha256") != digest(path):
        raise ValueError("P57 A2 prediction file changed after checkpoint")
    mode = head["mode"]
    if mode not in MODES or len(public.get("predictions", [])) != 8:
        raise ValueError("Expected one A2 mode and eight development predictions")
    return (mode, head["baseline_pose_prior"], public["predictions"], head_path,
            head["selected"], head["freeze_sha256"])


def analyze(manifest_path, prediction_paths, training_freeze_path):
    manifest_path = Path(manifest_path).resolve()
    training_freeze_path = Path(training_freeze_path).resolve()
    freeze = json.loads(training_freeze_path.read_text())
    root = Path(__file__).resolve().parents[2]
    plan = root/'evaluations/fast_adapter_p57_geometry_tokens_v01/PLAN_A2.md'
    if freeze.get("schema") != "p57.a2-translation-training-freeze.v1":
        raise ValueError("Wrong A2 result-free training freeze")
    if freeze["pack_manifest"]["sha256"] != digest(manifest_path):
        raise ValueError("A2 training freeze and pack differ")
    if freeze["plan"]["sha256"] != digest(plan):
        raise ValueError("A2 plan changed after training freeze")
    for source, expected in freeze["sources"].items():
        if digest(source) != expected:
            raise ValueError(f"A2 training source changed: {source}")
    bundle = load_bundle(manifest_path)
    dev = {(ep["condition_id"], ep["source_policy"]): ep
           for ep in bundle["episodes"] if ep["split"] == "validation"}
    if len(dev) != 8 or len({key[0] for key in dev}) != 4 or len(prediction_paths) != 3:
        raise ValueError("Expected eight development trajectories and three A2 modes")
    if any({source for condition, source in dev if condition == case} !=
           {"expert", "policy_fixed"} for case in {condition for condition, _ in dev}):
        raise ValueError("A2 development conditions must be paired expert/policy")
    runs, seen = [], set()
    for path in prediction_paths:
        mode, prior, predictions, head_path, selected, freeze_hash = _predictions(
            path, digest(manifest_path))
        if freeze_hash != digest(training_freeze_path):
            raise ValueError("A2 head and result-free training freeze differ")
        if mode in seen:
            raise ValueError("Duplicate A2 mode")
        seen.add(mode)
        identities = [(p["condition_id"], p["source_policy"]) for p in predictions]
        if set(identities) != set(dev) or len(set(identities)) != 8:
            raise ValueError("Development predictions do not cover the sealed eight episodes")
        rows = [analyze_episode(p, dev[key], prior) for p, key in zip(predictions, identities)]
        summary = macro(rows)
        by_source = {source: macro([r for r in rows if r["source_policy"] == source])
                     for source in ("expert", "policy_fixed")}
        by_case = {condition: macro([r for r in rows if r["condition_id"] == condition])
                   for condition in sorted({r["condition_id"] for r in rows})}
        runs.append(dict(mode=mode, selected=selected,
                         representation_claim=("train_bbox_relative_mean_only" if selected["dim"] == 0
                                               else "origin_readout_with_selected_pca"),
                         prediction_path=str(Path(path).resolve()),
                         prediction_sha256=digest(path), head_path=str(head_path),
                         head_sha256=digest(head_path), trajectories=rows,
                         training_freeze_sha256=freeze_hash,
                         macro=summary, by_source=by_source, by_case=by_case,
                         gate=a2_gate(summary)))
    if seen != set(MODES):
        raise ValueError("Missing A2 mode")
    sources = [root/'prove/task_graph/p57_translation_geometry.py',
               root/'prove/task_graph/p57_ray_geometry.py', Path(__file__).resolve()]
    return dict(schema="p57.a2-translation-analysis.v1",
                pack_manifest_sha256=digest(manifest_path),
                plan_a2_sha256=digest(plan),
                training_freeze_sha256=digest(training_freeze_path),
                sources={str(p): digest(p) for p in sources},
                scope="Offline development scoring only; no physics or private runtime input",
                history_max_frames=8, ray_condition_limit=1e5, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--training-freeze", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path, nargs=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze(args.manifest, args.predictions, args.training_freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({run["mode"]: run["gate"] for run in report["runs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
