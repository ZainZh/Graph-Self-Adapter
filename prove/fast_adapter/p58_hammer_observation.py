"""P58 public observation and private labels for the held-hammer task.

The public path contains only rendered visible-part masks, fixed camera
calibration and robot proprioception.  Functional-point labels are produced by
separate functions so callers cannot accidentally pass simulator sites to a
runtime policy.
"""
from __future__ import annotations

import numpy as np

from prove.task_graph.p57_public_geometry import mask_token, project_world_points


ROLES = ("hammer_head", "target_block")
ROLE_GEOMS = ("actor_head", "target_block")
RESOLUTION = 448
PUBLIC_KEYS = frozenset({"rgb", "masks", "mask_tokens", "mask_valid", "camera_K",
                         "camera_pose", "proprio", "eef_transform"})
PRIVATE_KEYS = frozenset({"world_positions", "absolute_positions", "role_relative",
                          "semantic_uv", "semantic_depth", "projected_valid",
                          "semantic_mask_distance_px", "target_displacement",
                          "head_target_contact"})


def camera_matrices(calibration, resolution=RESOLUTION):
    """Convert P35's renderer camera record to K and camera-to-world pose."""
    cal = {key: np.asarray(value, dtype=np.float64) for key, value in calibration.items()}
    position = cal["pos"].reshape(3)
    forward = cal["forward"].reshape(3)
    up = cal["up"].reshape(3)
    forward /= np.linalg.norm(forward)
    up -= forward * float(up @ forward)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = np.stack((right, up, -forward), axis=1)
    scalar = lambda key: float(np.asarray(cal[key]).reshape(-1)[0])
    near = scalar("frustum_near")
    bottom = scalar("frustum_bottom")
    top = scalar("frustum_top")
    if not np.isfinite([near, bottom, top]).all() or near <= 0 or top <= bottom:
        raise ValueError("Invalid renderer frustum")
    focal = float(resolution) * near / (top - bottom)
    center = scalar("frustum_center")
    principal_x = resolution / 2. - focal * center / near
    K = np.array([[focal, 0., principal_x],
                  [0., focal, resolution / 2.],
                  [0., 0., 1.]], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3], pose[:3, 3] = rotation, position
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("Non-orthonormal renderer camera")
    return K.astype(np.float32), pose.astype(np.float32)


def visible_masks(segmentation, geom_ids, geom_object_type):
    """Return visible geom masks; segmentation decides occlusion."""
    seg = np.asarray(segmentation)
    if seg.ndim != 3 or seg.shape[-1] < 2:
        raise ValueError("Expected MuJoCo segmentation image [H,W,2]")
    rendered = (seg[..., 0] >= 0) & (seg[..., 1] == int(geom_object_type))
    result = np.stack([rendered & (seg[..., 0] == int(index)) for index in geom_ids])
    return result.astype(np.uint8)


def public_frame(masks, camera_K, camera_pose, proprio, eef_transform):
    """Validate and encode one two-view runtime observation."""
    masks = np.asarray(masks, dtype=np.uint8)
    K = np.asarray(camera_K, dtype=np.float32)
    camera = np.asarray(camera_pose, dtype=np.float32)
    proprio = np.asarray(proprio, dtype=np.float32)
    eef = np.asarray(eef_transform, dtype=np.float32)
    if masks.shape != (2, 2, RESOLUTION, RESOLUTION):
        raise ValueError("Expected two views and head/target masks at 448")
    if K.shape != (2, 3, 3) or camera.shape != (2, 4, 4):
        raise ValueError("Invalid two-view calibration")
    if proprio.shape != (4,) or eef.shape != (4, 4):
        raise ValueError("Invalid held-hammer robot proprioception")
    tokens = np.zeros((2, 2, 55), dtype=np.float32)
    valid = np.zeros((2, 2), dtype=bool)
    for view in range(2):
        for role in range(2):
            tokens[view, role], valid[view, role] = mask_token(masks[view, role])
    arrays = (K, camera, proprio, eef, tokens)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Nonfinite P58 public observation")
    return dict(mask_tokens=tokens, mask_valid=valid, camera_K=K,
                camera_pose=camera, proprio=proprio, eef_transform=eef)


def private_labels(env, camera_K, camera_pose):
    """Evaluator/training-only functional centers and their image projections."""
    data = env.env.data
    head = np.asarray(data.site("tool_site").xpos, dtype=np.float64).copy()
    target = np.asarray(data.site("target_site").xpos, dtype=np.float64).copy()
    world = np.stack((head, target))
    uv, depth, projected = [], [], []
    for view in range(2):
        local_uv, in_front = project_world_points(world, camera_K[view], camera_pose[view])
        # Depth is retained only in the private label for audit, never runtime.
        inv = np.linalg.inv(camera_pose[view].astype(np.float64))
        xyz = (inv @ np.c_[world, np.ones(2)].T).T[:, :3]
        uv.append(local_uv); depth.append(-xyz[:, 2]); projected.append(in_front)
    eef = eef_transform(env).astype(np.float64)
    head_in_eef = eef[:3, :3].T @ (head - eef[:3, 3])
    displacement = float(data.joint("target_slide").qpos[0])
    head_contact = bool(env.env.contacts()[3])
    return dict(world_positions=world.astype(np.float32),
                role_relative=np.concatenate((head_in_eef, target)).astype(np.float32),
                absolute_positions=world.reshape(-1).astype(np.float32),
                semantic_uv=np.asarray(uv, dtype=np.float32),
                semantic_depth=np.asarray(depth, dtype=np.float32),
                projected_valid=np.asarray(projected, dtype=bool),
                target_displacement=np.float32(displacement),
                head_target_contact=np.bool_(head_contact))


def mask_projection_distances(masks, semantic_uv):
    """Evaluator-only distance from each projected semantic center to its mask."""
    masks = np.asarray(masks, dtype=bool)
    uv = np.asarray(semantic_uv, dtype=np.float64)
    if masks.shape != (2, 2, RESOLUTION, RESOLUTION) or uv.shape != (2, 2, 2):
        raise ValueError("Invalid mask/projection audit shapes")
    result = np.full((2, 2), np.inf, dtype=np.float32)
    for view in range(2):
        for role in range(2):
            yy, xx = np.nonzero(masks[view, role])
            if len(xx):
                delta = np.stack((xx - uv[view, role, 0], yy - uv[view, role, 1]), axis=1)
                result[view, role] = float(np.sqrt((delta * delta).sum(1).min()))
    return result


def eef_transform(env):
    """World-from-EEF from public robot pose; translation matches tcp_center."""
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.asarray(env.env.tcp_center, dtype=np.float64)
    # Sawyer position control keeps orientation fixed.  The hand rotation is
    # robot proprioception and does not expose object state.
    result[:3, :3] = np.asarray(env.env.data.body("hand").xmat, dtype=np.float64).reshape(3, 3)
    return result.astype(np.float32)
