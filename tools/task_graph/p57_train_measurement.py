#!/usr/bin/env python3
"""Train one P57 Stage A public measurement head; never launches physics.

One invocation produces one final checkpoint and readout for one mode/seed.
The caller freezes its bundle, mode, seed and output path before running.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from prove.task_graph.p57_measurement_head import (GeometryMeasurementHead,
                                                   MeasurementConfig, MODES, INPUT_KEYS)
from prove.task_graph.p57_public_geometry import project_world_points


LABEL_KEYS = frozenset({"book_world_anchors", "desired_book_world_anchors", "book_uv",
                        "desired_book_uv", "book_uv_valid", "desired_book_uv_valid"})


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_bundle(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "p57.geometry-pack-manifest.v1" or manifest.get("cohort_counts") != dict(
            train=34, validation=8, transitions=573, frames=615):
        raise ValueError("Unexpected P57 Stage A pack manifest")
    if len(manifest.get("episodes", [])) != 42:
        raise ValueError("P57 Stage A requires all 42 sealed trajectories")
    episodes = []
    for record in manifest["episodes"]:
        ref = record["bundle"]
        source = Path(ref["path"])
        path = source.resolve()
        if not path.is_relative_to(manifest_path.parent) or not path.is_file():
            # A copied sealed pack retains its original absolute manifest refs.
            # Only the identical basename inside this manifest directory qualifies.
            path = (manifest_path.parent / source.name).resolve()
        if not path.is_relative_to(manifest_path.parent) or sha256(path) != ref["sha256"]:
            raise ValueError("P57 episode file is outside copied pack or changed")
        episode = torch.load(path, map_location="cpu", weights_only=True)
        if episode.get("schema") != "p57.geometry-bundle.v1":
            raise ValueError("Unexpected per-episode P57 geometry schema")
        if any(episode.get(key) != record.get(key) for key in ("condition_id", "split", "source_policy")):
            raise ValueError("P57 episode/manifest identity mismatch")
        episodes.append(episode)
    return dict(schema="p57.geometry-bundle.v1", episodes=episodes)


def public_input(episode, frame=None, *, device="cpu"):
    """Explicit allowlist: private labels cannot reach the model."""
    source = episode["input"]
    features = source["features"]
    if set(features) != {"raw", "boxes", "visible", "proprio"}:
        raise ValueError("Unexpected cached public feature keys")
    value = {**features, **{key: source[key] for key in INPUT_KEYS - set(features)}}
    if set(value) != INPUT_KEYS:
        raise ValueError("Missing public P57 inputs")
    if frame is not None:
        value = {key: item[frame] for key, item in value.items()}
    return {key: item.to(device=device, dtype=(torch.bool if key == "mask_valid" else torch.float32))
            for key, item in value.items()}


def targets(episode, frame=None, *, device="cpu"):
    labels = episode["labels"]
    if not LABEL_KEYS.issubset(labels):
        raise ValueError("Missing isolated geometry labels")
    select = lambda key: labels[key] if frame is None else labels[key][frame]
    uv = torch.stack((select("book_uv"), select("desired_book_uv")), -3).to(device=device, dtype=torch.float32)
    valid = torch.stack((select("book_uv_valid"), select("desired_book_uv_valid")), -2).to(device=device, dtype=torch.bool)
    return uv, valid


def validate_bundle(bundle):
    if bundle.get("schema") != "p57.geometry-bundle.v1":
        raise ValueError("Wrong P57 geometry bundle schema")
    episodes = bundle.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Empty geometry bundle")
    train = []; dev = []
    seen = set()
    for index, episode in enumerate(episodes):
        identity = episode.get("condition_id"), episode.get("source_policy")
        if identity in seen:
            raise ValueError("Duplicate episode identity")
        seen.add(identity)
        obs = public_input(episode)
        length = obs["proprio"].shape[0]
        if not 2 <= length <= 25:
            raise ValueError("P57 expects actual T+1 short-stage frames")
        uv, valid = targets(episode)
        if uv.shape != (length, 2, 2, 3, 2) or valid.shape != (length, 2, 2, 3):
            raise ValueError("Invalid virtual-anchor label shapes")
        if not torch.isfinite(uv).all():
            raise ValueError("Nonfinite virtual-anchor label")
        for key, item in obs.items():
            if item.shape[0] != length:
                raise ValueError(f"Public {key} frame count differs")
        if len(episode["input"].get("public_frames", [])) != length:
            raise ValueError("P57 public frame provenance missing")
        split = episode.get("split")
        if split == "train": train.append(index)
        elif split == "validation": dev.append(index)
        else: raise ValueError("P57 Stage A accepts train/validation only")
    if not train or not dev:
        raise ValueError("Both train and validation trajectories required")
    return train, dev


def fit_normalization(model, episodes, train_indices):
    with torch.no_grad():
        rows = [public_input(episodes[i]) for i in train_indices]
        raw = torch.cat([r["raw"] for r in rows], 0)
        mask = torch.cat([r["mask_tokens"] for r in rows], 0)
        aux = torch.cat([model._auxiliary(r) for r in rows], 0)
        for name, x in (("raw", raw), ("mask", mask), ("aux", aux)):
            mean = x.mean(0)
            scale = x.std(0, unbiased=False).clamp_min(1e-3)
            getattr(model, name + "_mean").copy_(mean)
            getattr(model, name + "_scale").copy_(scale)


def fit_bbox_relative_mean(episodes, train_indices, image_size):
    total = torch.zeros(2, 2, 3, 2)
    count = torch.zeros(2, 2, 3, 1)
    for i in train_indices:
        episode = episodes[i]
        obs = public_input(episode)
        uv, valid = targets(episode)
        box = obs["boxes"][..., 1:3, :] * image_size
        center = (box[..., :2] + box[..., 2:]) * .5
        extent = (box[..., 2:] - box[..., :2]).clamp_min(1.)
        relative = (uv - center[..., None, :]) / extent[..., None, :]
        mask = valid & obs["mask_valid"][..., 1:3, None]
        total += (relative * mask[..., None]).sum(0)
        count += mask[..., None].sum(0)
    return total / count.clamp_min(1)


def prior_pixels(obs, prior, image_size):
    box = obs["boxes"][..., 1:3, :] * image_size
    center = (box[..., :2] + box[..., 2:]) * .5
    extent = (box[..., 2:] - box[..., :2]).clamp_min(1.)
    return center[..., None, :] + extent[..., None, :] * prior


def fit_pose_prior(episodes, train_indices):
    """Train-only book-in-EEF and static desired-world anchor means.

    This intentionally strong scene-memory diagnostic uses private training
    labels offline, then only EEF pose and camera calibration at readout.
    """
    held = []; goal = []
    anchor = np.array([[0., 0., 0., 1.], [.03, 0., 0., 1.], [0., 0., .03, 1.]])
    for i in train_indices:
        label = episodes[i]["labels"]
        transform = label["book_in_eef"].numpy()
        held.append((transform @ anchor.T).transpose(0, 2, 1)[..., :3])
        goal.append(label["desired_book_world_anchors"].numpy())
    return dict(book_in_eef_anchors=np.concatenate(held).mean(0),
                desired_world_anchors=np.concatenate(goal).mean(0))


def pose_prior_pixels(obs, prior):
    eef = obs["eef_transform"].numpy()
    k = obs["camera_K"].numpy()
    camera = obs["camera_pose"].numpy()
    held = (eef[:, :3, :3] @ prior["book_in_eef_anchors"].T).transpose(0, 2, 1) + eef[:, None, :3, 3]
    goal = np.broadcast_to(prior["desired_world_anchors"], held.shape)
    result = np.zeros((len(held), 2, 2, 3, 2), np.float32)
    for t in range(len(held)):
        for view in range(2):
            result[t, view, 0] = project_world_points(held[t], k[t, view], camera[t, view])[0]
            result[t, view, 1] = project_world_points(goal[t], k[t, view], camera[t, view])[0]
    return torch.from_numpy(result)


def supervised_loss(model, obs, target, valid):
    pred, role_valid = model(obs)
    admissible = valid & role_valid[..., None]
    if not admissible.any():
        raise ValueError("Batch has no admissible projected anchors")
    residual = (pred - target) / model.config.image_size
    point = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=.01, reduction="none").mean(-1)
    return (point * admissible).sum() / admissible.sum(), pred, admissible


def _triangulate(uv, K, pose):
    """Two calibrated public rays. Return None for degenerate viewing geometry."""
    camera = np.asarray(pose, np.float64)
    intrinsics = np.asarray(K, np.float64)
    uv = np.asarray(uv, np.float64)
    axes = np.linalg.solve(intrinsics, np.c_[uv, np.ones(2)][:, :, None]).squeeze(-1)
    # MuJoCo camera looks along -Z and image Y points down.
    axes *= np.array([1., -1., -1.])
    axes = np.einsum("vij,vj->vi", camera[:, :3, :3], axes)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    projector = np.eye(3)[None] - axes[:, :, None] * axes[:, None, :]
    system = projector.sum(0)
    singular = np.linalg.svd(system, compute_uv=False)
    if singular[-1] < 1e-5 or singular[0] / singular[-1] > 1e5:
        return None
    origin = camera[:, :3, 3]
    point = np.linalg.solve(system, np.einsum("vij,vj->i", projector, origin))
    if not np.isfinite(point).all():
        return None
    return point, float(singular[-1]), float(singular[0] / singular[-1])


def _frame_from_anchors(points):
    points = np.asarray(points, np.float64)
    x = points[1] - points[0]
    z = points[2] - points[0]
    nx = np.linalg.norm(x)
    if not np.isfinite(points).all() or nx < 1e-5:
        return None
    x /= nx
    z -= x * np.dot(x, z)
    nz = np.linalg.norm(z)
    if nz < 1e-5:
        return None
    z /= nz
    y = np.cross(z, x)
    return points[0], np.stack((x, y, z), -1)


def _relative_frame_errors(estimated, truth):
    estimate = [_frame_from_anchors(estimated[role]) for role in range(2)]
    actual = [_frame_from_anchors(truth[role]) for role in range(2)]
    if any(value is None for value in estimate + actual):
        return None
    eb, eg = estimate
    tb, tg = actual
    estimated_delta = eb[1].T @ (eg[0] - eb[0])
    true_delta = tb[1].T @ (tg[0] - tb[0])
    estimated_rotation = eb[1].T @ eg[1]
    true_rotation = tb[1].T @ tg[1]
    difference = estimated_rotation @ true_rotation.T
    angle = np.arccos(np.clip((np.trace(difference) - 1.) / 2., -1., 1.))
    return float(np.linalg.norm(estimated_delta - true_delta)), float(angle)


def evaluate(model, episodes, indices, bbox_prior, pose_prior):
    rows = []
    model.eval()
    for index in indices:
        episode = episodes[index]
        obs = public_input(episode)
        target, target_valid = targets(episode)
        with torch.no_grad():
            predicted, role_valid = model(obs)
        comparator = prior_pixels(obs, bbox_prior, model.config.image_size)
        pose_comparator = pose_prior_pixels(obs, pose_prior)
        valid = target_valid & role_valid[..., None]
        error = torch.linalg.vector_norm(predicted - target, dim=-1)
        prior_error = torch.linalg.vector_norm(comparator - target, dim=-1)
        pose_error = torch.linalg.vector_norm(pose_comparator - target, dim=-1)
        item = dict(condition_id=episode.get("condition_id"), source_policy=episode.get("source_policy"),
                    frames=int(len(obs["proprio"])), valid_view_points=int(valid.sum()),
                    pixel_mean=float(error[valid].mean()) if valid.any() else None,
                    pixel_median=float(error[valid].median()) if valid.any() else None,
                    bbox_relative_mean_pixel_mean=float(prior_error[valid].mean()) if valid.any() else None,
                    pose_prior_pixel_mean=float(pose_error[valid].mean()) if valid.any() else None,
                    book_pixel_mean=float(error[:, :, 0][valid[:, :, 0]].mean()) if valid[:, :, 0].any() else None,
                    desired_pixel_mean=float(error[:, :, 1][valid[:, :, 1]].mean()) if valid[:, :, 1].any() else None)
        tri_errors = [[], [], []]
        relation_errors = [[], [], []]
        rank_min = []
        condition_max = []
        two_view_candidates = 0
        degenerate = [0, 0, 0]
        world = torch.stack((episode["labels"]["book_world_anchors"],
                             episode["labels"]["desired_book_world_anchors"]), -3).numpy()
        for t in range(len(obs["proprio"])):
            reconstructed = [np.full((2, 3, 3), np.nan) for _ in range(3)]
            for role in range(2):
                for anchor in range(3):
                    if not valid[t, :, role, anchor].all():
                        continue
                    two_view_candidates += 1
                    for which, uv in enumerate((predicted, comparator, pose_comparator)):
                        solution = _triangulate(uv[t, :, role, anchor].numpy(),
                                                obs["camera_K"][t].numpy(), obs["camera_pose"][t].numpy())
                        if solution is None:
                            degenerate[which] += 1
                            continue
                        point, minimum, condition = solution
                        reconstructed[which][role, anchor] = point
                        tri_errors[which].append(float(np.linalg.norm(point - world[t, role, anchor])))
                        if which == 0:
                            rank_min.append(minimum)
                            condition_max.append(condition)
            for which in range(3):
                relation = _relative_frame_errors(reconstructed[which], world[t])
                if relation is not None:
                    relation_errors[which].append(relation)
        def metric(which, component):
            return (float(np.mean([pair[component] for pair in relation_errors[which]]))
                    if relation_errors[which] else None)
        item.update(two_view_candidates=two_view_candidates,
                    triangulated_points=len(tri_errors[0]),
                    triangulation_degenerate_count=degenerate[0],
                    bbox_relative_mean_degenerate_count=degenerate[1],
                    pose_prior_degenerate_count=degenerate[2],
                    triangulated_3d_mean_m=float(np.mean(tri_errors[0])) if tri_errors[0] else None,
                    bbox_relative_mean_3d_mean_m=float(np.mean(tri_errors[1])) if tri_errors[1] else None,
                    pose_prior_3d_mean_m=float(np.mean(tri_errors[2])) if tri_errors[2] else None,
                    relation_frames=len(relation_errors[0]),
                    relation_translation_mean_m=metric(0, 0),
                    relation_rotation_mean_rad=metric(0, 1),
                    bbox_relative_mean_relation_translation_m=metric(1, 0),
                    bbox_relative_mean_relation_rotation_rad=metric(1, 1),
                    pose_prior_relation_translation_m=metric(2, 0),
                    pose_prior_relation_rotation_rad=metric(2, 1),
                    ray_system_min_singular=float(min(rank_min)) if rank_min else None,
                    ray_system_max_condition=float(max(condition_max)) if condition_max else None)
        rows.append(item)
    return rows


def macro_report(rows):
    """Each development trajectory has one vote, including source-policy cuts."""
    fields = ("pixel_mean", "book_pixel_mean", "desired_pixel_mean", "bbox_relative_mean_pixel_mean",
              "pose_prior_pixel_mean", "triangulated_3d_mean_m", "bbox_relative_mean_3d_mean_m",
              "pose_prior_3d_mean_m", "relation_translation_mean_m", "relation_rotation_mean_rad",
              "bbox_relative_mean_relation_translation_m", "bbox_relative_mean_relation_rotation_rad",
              "pose_prior_relation_translation_m", "pose_prior_relation_rotation_rad")
    def group(items):
        return dict(episodes=len(items), **{
            field: (float(np.mean([row[field] for row in items if row[field] is not None]))
                    if any(row[field] is not None for row in items) else None)
            for field in fields})
    return dict(all=group(rows), by_source_policy={source: group([row for row in rows if row["source_policy"] == source])
                                                   for source in sorted({row["source_policy"] for row in rows})})


def public_predictions(model, episodes, indices):
    """Save a label-free public readout for later ray consistency diagnostics."""
    model.eval()
    rows = []
    for index in indices:
        episode = episodes[index]
        obs = public_input(episode)
        with torch.no_grad():
            uv, role_valid = model(obs)
        rows.append(dict(condition_id=episode["condition_id"], split=episode["split"],
                         source_policy=episode["source_policy"],
                         source_seal_sha256=episode["source_seal_sha256"],
                         public_frame_sha256=[item["sha256"] for item in episode["input"]["public_frames"]],
                         pred_uv=uv.detach().cpu(), role_valid=role_valid.detach().cpu(),
                         camera_K=obs["camera_K"].cpu(), camera_pose=obs["camera_pose"].cpu(),
                         eef_transform=obs["eef_transform"].cpu()))
    return dict(schema="p57.public-anchor-predictions.v1", role_order=["book", "desired_book_frame"],
                frame_order=["origin", "positive_x_0.03m", "positive_z_0.03m"],
                predictions=rows, labels_included=False)


def train(manifest_path, output, mode, seed, *, freeze_path, device="cpu", updates=1500, batch=64):
    from tools.task_graph.p57_freeze_training import verify
    verify(freeze_path, manifest_path, mode, seed, output)
    prediction_path = output.parent / "PREDICTIONS.pt"
    report_path = output.parent / "REPORT.json"
    for path in (output, prediction_path, report_path):
        if path.exists():
            raise FileExistsError(path)
    if mode not in MODES or seed not in (11, 22) or updates != 1500 or batch != 64:
        raise ValueError("Stage A frozen mode, seeds and budget required")
    bundle = load_bundle(manifest_path)
    train_indices, dev_indices = validate_bundle(bundle)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    config = MeasurementConfig(mode=mode)
    model = GeometryMeasurementHead(config)
    fit_normalization(model, bundle["episodes"], train_indices)
    bbox_prior = fit_bbox_relative_mean(bundle["episodes"], train_indices, config.image_size)
    pose_prior = fit_pose_prior(bundle["episodes"], train_indices)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    episodes = bundle["episodes"]
    loss_trace = []
    for step in range(updates):
        # Uniform trajectory first, then uniform actual frame; no padded windows.
        chosen = [train_indices[int(torch.randint(len(train_indices), (), generator=rng))] for _ in range(batch)]
        pairs = [(i, int(torch.randint(len(episodes[i]["input"]["features"]["proprio"]), (), generator=rng))) for i in chosen]
        samples = [public_input(episodes[i], t) for i, t in pairs]
        obs = {key: torch.stack([sample[key] for sample in samples]).to(device)
               for key in INPUT_KEYS}
        labels = [targets(episodes[i], t) for i, t in pairs]
        uv = torch.stack([item[0] for item in labels]).to(device)
        valid = torch.stack([item[1] for item in labels]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = supervised_loss(model, obs, uv, valid)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            entry = dict(update=step + 1, loss=float(loss.detach()))
            loss_trace.append(entry)
            print(json.dumps(entry), flush=True)
    model.to("cpu")
    report = evaluate(model, episodes, dev_indices, bbox_prior, pose_prior)
    public_readout = public_predictions(model, episodes, dev_indices)
    sources = {str(Path(__file__).resolve()): sha256(__file__),
               str(Path(__file__).resolve().parents[2] / "prove/task_graph/p57_measurement_head.py"):
                   sha256(Path(__file__).resolve().parents[2] / "prove/task_graph/p57_measurement_head.py")}
    payload = dict(schema="p57.geometry-measurement-checkpoint.v1", mode=mode, seed=seed,
                   config=asdict(config), state_dict=model.state_dict(), parameter_count=model.parameter_count(),
                   pack_manifest_sha256=sha256(manifest_path), source_sha256=sources,
                   training=dict(updates=updates, batch=batch, optimizer="Adam", learning_rate=3e-4,
                                 sampling="uniform train trajectory then uniform actual frame",
                                 normalization="train trajectories only", selection="final checkpoint only"),
                   training_loss_trace=loss_trace,
                   baseline_bbox_relative_mean=bbox_prior,
                   baseline_pose_prior={key: torch.from_numpy(value) for key, value in pose_prior.items()},
                   validation_per_episode=report, validation_macro=macro_report(report),
                   public_predictions_path=str(prediction_path.resolve()),
                   claim="offline virtual-anchor measurement qualification only; no physics or adapter success")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(public_readout, prediction_path)
    payload["public_predictions_sha256"] = sha256(prediction_path)
    torch.save(payload, output)
    report_json = dict(schema="p57.geometry-measurement-report.v1", mode=mode, seed=seed,
                       checkpoint=dict(path=str(output.resolve()), sha256=sha256(output)),
                       public_predictions=dict(path=str(prediction_path.resolve()),
                                               sha256=payload["public_predictions_sha256"]),
                       freeze_sha256=sha256(freeze_path), pack_manifest_sha256=sha256(manifest_path),
                       source_sha256=sources, config=asdict(config), parameters=payload["parameter_count"],
                       training=payload["training"], training_loss_trace=loss_trace,
                       validation_macro=payload["validation_macro"], validation_per_episode=report,
                       limitation="Opened development trajectories; no physics or Fast Adapter comparison")
    report_path.write_text(json.dumps(report_json, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(output=str(output), mode=mode, seed=seed,
                          parameters=payload["parameter_count"], validation_episodes=len(report),
                          report=str(report_path))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, choices=(11, 22), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args.manifest, args.output, args.mode, args.seed, freeze_path=args.freeze, device=args.device)


if __name__ == "__main__":
    main()
