#!/usr/bin/env python3
"""Re-render the frozen P35 held-hammer subset without task integration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from prove.fast_adapter.held_hammer_env import HeldHammerEnv
from prove.fast_adapter.object_interaction_env import mujoco
from prove.fast_adapter.p58_hammer_observation import (
    PRIVATE_KEYS, PUBLIC_KEYS, RESOLUTION, ROLE_GEOMS, camera_matrices, eef_transform,
    mask_projection_distances, private_labels, public_frame, visible_masks,
)
from tools.fast_adapter.p35_camera import CameraRig


ROOT = Path("evaluations/fast_adapter_p58_functional_geometry_v01/hammer")
SOURCE = Path("evaluations/fast_adapter_p30_online_v02/hammer-block/data")
MOUNT = Path("evaluations/fast_adapter_p35_resolution_views_v01/preflight/WRIST_MOUNT.json")
TIMES = (0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48)
OFFSETS = (0, 8, 16, 48)
ROLES = ("expert", "base")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def render(renderer, data, camera):
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    cs = renderer.scene.camera
    calibration = {key: np.mean([np.asarray(getattr(item, key)) for item in cs], axis=0).tolist()
                   for key in ("pos", "forward", "up", "frustum_bottom", "frustum_top",
                               "frustum_center", "frustum_near")}
    renderer.enable_segmentation_rendering()
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return rgb, segmentation, calibration


def restore_reset_model_parameters_without_integration(env, seed):
    """Replay P30 reset RNG only; trajectory qpos overwrites actor state later."""
    env.env.seed(int(seed))
    rng = env.env.np_random
    # Meta-World's reset path consumes four generator values before this
    # task's reset_model samples actor/target positions.  This offset is
    # guarded against the recorded t0 RGB for every exported episode below.
    rng.random(4)
    rng.uniform(-.12, .025)  # actor x, stored qpos is authoritative
    rng.uniform(.52, .63)   # actor y, stored qpos is authoritative
    target_xy = [rng.uniform(.15, .24), rng.uniform(.72, .79)]
    env.env.model.body("target").pos[:2] = target_xy
    return np.asarray(target_xy, dtype=np.float32)


def export(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    mount = json.loads(MOUNT.read_text())
    rows, sources = [], []
    for role in ROLES:
        for offset in OFFSETS:
            source = SOURCE / f"meta_{role}_{offset:03d}.pt"
            sidecar = source.with_suffix(".json")
            raw = torch.load(source, map_location="cpu", weights_only=True)
            meta = json.loads(sidecar.read_text())
            sources.extend((dict(path=str(source), sha256=sha(source)),
                            dict(path=str(sidecar), sha256=sha(sidecar))))
            for local, (config, reset_seed) in enumerate(zip(meta["configs"], raw["reset_seeds"])):
                # Historical mass/friction are image-supervision provenance only.
                # They are never copied into the P58 fixed-physics behavior grid.
                env = HeldHammerEnv(config)
                target_xy = restore_reset_model_parameters_without_integration(env, int(reset_seed))
                try:
                    for key, value in raw["states"].items():
                        getattr(env.env.data, key)[:] = value[local, 0].numpy()
                    mujoco.mj_forward(env.env.model, env.env.data)
                    restored_t0 = env.rgb()
                    recorded_t0 = raw["audit_rgb"][0, local].numpy()
                    pixel_delta = np.abs(restored_t0.astype(np.int16)-recorded_t0.astype(np.int16))
                    audit_max_pixel = int(pixel_delta.max())
                    audit_changed_pixels = int(np.any(pixel_delta != 0, axis=-1).sum())
                    if audit_max_pixel != 0:
                        raise RuntimeError(
                            f"t0 render mismatch object={offset+local} role={role} "
                            f"max_pixel={audit_max_pixel} changed_pixels={audit_changed_pixels}")
                    rig = CameraRig(env, mount)
                    geom_ids = [int(env.env.model.geom(name).id) for name in ROLE_GEOMS]
                    frames = []
                    for t in TIMES:
                        for key, value in raw["states"].items():
                            getattr(env.env.data, key)[:] = value[local, t].numpy()
                        mujoco.mj_forward(env.env.model, env.env.data)
                        images, masks, Ks, poses = [], [], [], []
                        for renderer, camera in ((rig.renderers[RESOLUTION], env.cam),
                                                 (rig.renderers[RESOLUTION], "gripperPOV")):
                            rgb, seg, cal = render(renderer, env.env.data, camera)
                            K, pose = camera_matrices(cal)
                            images.append(rgb); masks.append(visible_masks(
                                seg, geom_ids, mujoco.mjtObj.mjOBJ_GEOM))
                            Ks.append(K); poses.append(pose)
                        public = public_frame(np.stack(masks), np.stack(Ks), np.stack(poses),
                                              raw["proprio"][local, t].numpy(), eef_transform(env))
                        label = private_labels(env, public["camera_K"], public["camera_pose"])
                        label["semantic_mask_distance_px"] = mask_projection_distances(
                            np.stack(masks), label["semantic_uv"])
                        if np.any(public["mask_valid"] & ~label["projected_valid"]):
                            raise ValueError("Visible role center projects behind camera")
                        public.update(rgb=np.stack(images), masks=np.stack(masks))
                        if set(public) != PUBLIC_KEYS or set(label) != PRIVATE_KEYS:
                            raise ValueError("P58 public/private key isolation changed")
                        frames.append((t, public, label))
                finally:
                    if "rig" in locals():
                        rig.close(); del rig
                    env.close()
                object_id = offset + local
                split = "train" if object_id < 24 else "development"
                stem = f"{split}_{object_id:02d}_{role}"
                public_name, private_name = stem+"_PUBLIC.npz", stem+"_PRIVATE.npz"
                np.savez_compressed(output/public_name, times=np.asarray(TIMES, np.int16),
                    **{key: np.stack([frame[1][key] for frame in frames]) for key in sorted(PUBLIC_KEYS)})
                np.savez_compressed(output/private_name, times=np.asarray(TIMES, np.int16),
                    **{key: np.stack([frame[2][key] for frame in frames]) for key in sorted(PRIVATE_KEYS)})
                rows.append(dict(public_path=public_name, public_sha256=sha(output/public_name),
                                 private_path=private_name, private_sha256=sha(output/private_name), split=split,
                                 object_id=object_id, role=role, reset_seed=int(reset_seed),
                                 frames=len(TIMES), source=str(source), source_row=local,
                                 target_xy=target_xy.tolist(), t0_rgb_exact=True,
                                 t0_max_pixel=audit_max_pixel, t0_changed_pixels=audit_changed_pixels,
                                 historical_visual_config=config))
                print("P58_HAMMER_EXPORT", stem, flush=True)
    manifest = dict(schema="p58.hammer-geometry-export.v1", task="held-hammer-block",
        rows=rows, times=list(TIMES), episodes=len(rows), frames=len(rows)*len(TIMES),
        train_objects=24, development_objects=8, paired_roles=list(ROLES),
        public_keys=sorted(PUBLIC_KEYS), private_keys=sorted(PRIVATE_KEYS),
        explicit_mj_step_calls=0, reset_or_teacher_calls=0, mj_forward_only=True,
        dino_recomputed=False,
        historical_mass_friction_for_visual_supervision_only=True,
        restored_t0_rgb_exact_all=True,
        behavior_physics=dict(mass=.35, friction=1.1),
        sources=sources + [dict(path=str(MOUNT), sha256=sha(MOUNT)),
            dict(path="prove/fast_adapter/p58_hammer_observation.py",
                 sha256=sha("prove/fast_adapter/p58_hammer_observation.py")),
            dict(path=str(Path(__file__)), sha256=sha(Path(__file__)))])
    dump(output/"MANIFEST.json", manifest)
    print(json.dumps(dict(output=str(output), episodes=len(rows), frames=manifest["frames"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"geometry_export_v01")
    export(parser.parse_args().output)
