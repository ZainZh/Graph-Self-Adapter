"""P59 public observation schema for the Panda pre-held insertion task.

This module accepts rendered RGB, visible role masks, calibration, robot pose,
and already-issued actions.  It deliberately has no simulator/configuration or
private-label dependency.  The compact mask representation is an engineering
reference; the separately validated 3072-value cache is the frozen
DINOv3/LiLa-fusion representation and must retain that explicit name.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from prove.task_graph.p57_public_geometry import mask_token


RESOLUTION = 448
ROLES = ("robot", "held_peg", "slot")
MASK_TOKEN_DIM = 55
MASK_REPRESENTATION_DIM = 196
DINO_LILA_ROLE_DIM = 3072
PUBLIC_KEYS = frozenset({
    "rgb", "role_masks", "mask_tokens", "mask_valid", "camera_K",
    "camera_pose", "eef_transform", "previous_action", "frame_index",
})


def _finite(value, shape, name, dtype=np.float32):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _pose(position, rotation, name):
    position = _finite(position, (3,), f"{name}_position")
    rotation = _finite(rotation, (3, 3), f"{name}_rotation")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError(f"{name}_rotation must be a proper rotation")
    result = np.eye(4, dtype=np.float32)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def unpack_role_masks(packed):
    """Unpack P45's last-axis packbits without admitting padded pixels."""
    value = np.asarray(packed, dtype=np.uint8)
    if value.shape != (len(ROLES), RESOLUTION, RESOLUTION // 8):
        raise ValueError("packed role masks must have shape [3,448,56]")
    return np.unpackbits(value, axis=-1, count=RESOLUTION).astype(np.uint8)


def public_frame(rgb, role_masks, camera_K, camera_position, camera_rotation,
                 eef_position, eef_rotation, previous_action, frame_index):
    """Build one runtime-legal P59 frame from P45 public fields."""
    image = np.asarray(rgb, dtype=np.uint8)
    masks = np.asarray(role_masks, dtype=np.uint8)
    if image.shape != (RESOLUTION, RESOLUTION, 3):
        raise ValueError("rgb must retain native 448x448 resolution")
    if masks.shape != (len(ROLES), RESOLUTION, RESOLUTION):
        raise ValueError("role_masks must have shape [3,448,448]")
    masks = masks.astype(bool).astype(np.uint8)
    K = _finite(camera_K, (3, 3), "camera_K")
    if abs(float(np.linalg.det(K))) < 1e-8:
        raise ValueError("camera_K must be nonsingular")
    camera = _pose(camera_position, camera_rotation, "camera")
    eef = _pose(eef_position, eef_rotation, "eef")
    action = _finite(previous_action, (13,), "previous_action")
    if isinstance(frame_index, bool) or int(frame_index) != frame_index or frame_index < 0:
        raise ValueError("frame_index must be a nonnegative integer")
    tokens = np.zeros((len(ROLES), MASK_TOKEN_DIM), dtype=np.float32)
    valid = np.zeros(len(ROLES), dtype=bool)
    for role in range(len(ROLES)):
        tokens[role], valid[role] = mask_token(masks[role])
    result = dict(rgb=image, role_masks=masks, mask_tokens=tokens,
                  mask_valid=valid, camera_K=K, camera_pose=camera,
                  eef_transform=eef, previous_action=action,
                  frame_index=np.int64(frame_index))
    if set(result) != PUBLIC_KEYS:
        raise RuntimeError("P59 public schema drift")
    return result


def mask_representation(frame):
    """Return the declared 196D mask/calibration/EEF engineering reference."""
    if set(frame) != PUBLIC_KEYS:
        raise ValueError("P59 public frame keys differ from the allowlist")
    token = _finite(frame["mask_tokens"], (3, 55), "mask_tokens")
    valid = np.asarray(frame["mask_valid"])
    if valid.shape != (3,) or valid.dtype != np.bool_:
        raise ValueError("mask_valid must be boolean [3]")
    K = _finite(frame["camera_K"], (3, 3), "camera_K")
    camera = _finite(frame["camera_pose"], (4, 4), "camera_pose")
    eef = _finite(frame["eef_transform"], (4, 4), "eef_transform")
    calibration = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
                           dtype=np.float32) / RESOLUTION
    value = np.concatenate((token.reshape(-1), valid.astype(np.float32),
                            calibration, camera[:3].reshape(-1),
                            eef[:3].reshape(-1))).astype(np.float32)
    if value.shape != (MASK_REPRESENTATION_DIM,) or not np.isfinite(value).all():
        raise RuntimeError("P59 mask representation dimension drift")
    return value


def validate_dino_lila_cache(value):
    """Validate, without renaming, P45's frozen DINOv3/LiLa role moments."""
    result = _finite(value, (len(ROLES), DINO_LILA_ROLE_DIM),
                     "dino_lila_role_moments")
    return result


def observed_intervals(frame_indices):
    """Return endpoints, time deltas, and a strict one-step mask."""
    indices = np.asarray(frame_indices)
    if indices.ndim != 1 or len(indices) < 2 or indices.dtype.kind not in "iu":
        raise ValueError("frame_indices must be a one-dimensional integer array")
    delta = np.diff(indices)
    if np.any(delta <= 0):
        raise ValueError("frame_indices must increase strictly")
    return dict(start=np.arange(len(indices) - 1, dtype=np.int64),
                stop=np.arange(1, len(indices), dtype=np.int64),
                delta_steps=delta.astype(np.int64), one_step=(delta == 1))


def executed_world_delta6(eef_position, eef_rotation,
                          issued_position, issued_rotation):
    """Encode a public issued pose relative to the actual pre-action EEF.

    This geometry-independent feedback action is distinct from P50's native
    slot-frame network output.  It uses only actual proprioception and the
    issued absolute command, so it neither requires nor leaks slot geometry.
    """
    q = _finite(eef_position, (3,), "eef_position", dtype=np.float64)
    Q = _finite(eef_rotation, (3, 3), "eef_rotation", dtype=np.float64)
    p = _finite(issued_position, (3,), "issued_position", dtype=np.float64)
    R = _finite(issued_rotation, (3, 3), "issued_rotation", dtype=np.float64)
    for value, name in ((Q, "eef_rotation"), (R, "issued_rotation")):
        if (not np.allclose(value.T @ value, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(value), 1., atol=1e-5)):
            raise ValueError(f"{name} must be a proper rotation")
    return np.r_[(p - q) / .2,
                 Rotation.from_matrix(R @ Q.T).as_rotvec() / .5].astype(np.float32)
