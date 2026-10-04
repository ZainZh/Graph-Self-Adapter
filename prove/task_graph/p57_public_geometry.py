"""P57 geometry from P52/P56 public frames only.

Roles are gripper, book, and visible caddy, in the renderer's fixed order.
The caddy mask is its visible geometry, not its hidden insertion site. No
object pose, size, depth, contact, or success signal enters this module.

Each view/role has a 55-value token, in this order:
  bbox xyxy (4), centroid xy (2), covariance xx/xy/yy (3), area (1),
  covariance eigenvalue ratio (1), silhouette projection q10/q50/q90 along
  image directions 0/45/90/135 degrees (12), occupied-pixel histograms in
  image x and y with 16 bins each (32).
Pixel coordinates use pixel centers divided by width/height. Covariance uses
the population denominator. Histograms each sum to one for a present mask.
The eigenvalue ratio is capped at 100; it is a confidence cue, not an angle.
Absent masks have an all-zero token and a separate false validity flag.
"""

from __future__ import annotations

import numpy as np


ROLES = ("gripper", "book", "caddy")
TOKEN_DIM = 55
_DIRECTIONS = np.array(
    [[1.0, 0.0], [2**-0.5, 2**-0.5], [0.0, 1.0],
     [-2**-0.5, 2**-0.5]], dtype=np.float64)


def _finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name}: expected finite {shape}")
    return array


def mask_token(mask):
    """Return (55-value token, present) for one visible binary image mask."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind not in "bu":
        raise ValueError("mask must be a 2D boolean or unsigned-integer array")
    height, width = mask.shape
    if height == 0 or width == 0:
        raise ValueError("mask image dimensions must be positive")
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return np.zeros(TOKEN_DIM, dtype=np.float32), False
    x = (xx.astype(np.float64) + 0.5) / width
    y = (yy.astype(np.float64) + 0.5) / height
    points = np.stack((x, y), axis=-1)
    center = points.mean(axis=0)
    residual = points - center
    covariance = residual.T @ residual / len(points)
    eigenvalues = np.linalg.eigvalsh(covariance)
    ratio = min(float(eigenvalues[-1] / max(eigenvalues[0], 1e-8)), 100.0)
    projected = points @ _DIRECTIONS.T
    quantiles = np.quantile(projected, (0.1, 0.5, 0.9), axis=0).T.ravel()
    hx = np.bincount(np.minimum(15, (x * 16).astype(int)), minlength=16) / len(x)
    hy = np.bincount(np.minimum(15, (y * 16).astype(int)), minlength=16) / len(y)
    token = np.concatenate((
        [(xx.min() + 0.5) / width, (yy.min() + 0.5) / height,
         (xx.max() + 0.5) / width, (yy.max() + 0.5) / height],
        center, [covariance[0, 0], covariance[0, 1], covariance[1, 1]],
        [len(x) / (width * height), ratio], quantiles, hx, hy,
    )).astype(np.float32)
    assert token.shape == (TOKEN_DIM,)
    return token, True


def eef_transform(position, quaternion_xyzw):
    """World-from-EEF transform; robosuite's robot0_eef_quat is xyzw."""
    p = _finite_array(position, (3,), "eef_pos")
    x, y, z, w = _finite_array(quaternion_xyzw, (4,), "eef_quat_xyzw")
    norm = np.linalg.norm((x, y, z, w))
    if norm < 1e-8:
        raise ValueError("EEF quaternion has zero norm")
    x, y, z, w = np.array((x, y, z, w)) / norm
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = p
    return transform


def _calibration(camera_K, camera_pose):
    K = _finite_array(camera_K, (3, 3), "camera_K")
    pose = _finite_array(camera_pose, (4, 4), "camera_pose")
    if abs(np.linalg.det(K)) < 1e-12 or not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("Singular intrinsic matrix or invalid camera transform")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Camera rotation must be orthonormal")
    return K, pose


def project_world_points(points_world, camera_K, camera_pose):
    """Project world points to top-left image pixels; return (uv, in_front).

    MuJoCo camera coordinates look along -Z with +Y upward. camera_pose maps
    camera to world, as saved by PublicRenderer. Points behind the camera get
    zero uv and false validity, so callers never receive infinities or NaNs.
    """
    K, pose = _calibration(camera_K, camera_pose)
    points = np.asarray(points_world, dtype=np.float64)
    if points.shape[-1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("points_world must have trailing xyz and be finite")
    shape = points.shape[:-1]
    local = (points.reshape(-1, 3) - pose[:3, 3]) @ pose[:3, :3]
    optical = local * [1, -1, -1]
    depth = optical[:, 2]
    valid = depth > 1e-8
    uv = np.zeros((len(local), 2), dtype=np.float64)
    homogeneous = optical[valid] @ K.T
    uv[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return uv.reshape(*shape, 2), valid.reshape(shape)


def rays(pixels, camera_K, camera_pose):
    """Return world ray origins and unit directions for top-left image pixels."""
    K, pose = _calibration(camera_K, camera_pose)
    uv = np.asarray(pixels, dtype=np.float64)
    if uv.shape[-1:] != (2,) or not np.isfinite(uv).all():
        raise ValueError("pixels must have trailing uv and be finite")
    shape = uv.shape[:-1]
    flat = uv.reshape(-1, 2)
    homogeneous = np.column_stack((flat, np.ones(len(flat))))
    optical = homogeneous @ np.linalg.inv(K).T
    local = optical * [1, -1, -1]
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    directions = local @ pose[:3, :3].T
    origins = np.broadcast_to(pose[:3, 3], directions.shape).copy()
    return origins.reshape(*shape, 3), directions.reshape(*shape, 3)


def public_geometry(frame):
    """Encode an NPZ-like public frame; no training/evaluator keys are read."""
    masks = np.asarray(frame["visible_masks"])
    if masks.ndim != 4 or masks.shape[0:2] != (2, 3):
        raise ValueError("Expected public visible_masks [2,3,H,W]")
    tokens = np.zeros((2, 3, TOKEN_DIM), dtype=np.float32)
    valid = np.zeros((2, 3), dtype=bool)
    for view in range(2):
        for role in range(3):
            tokens[view, role], valid[view, role] = mask_token(masks[view, role])
    camera_K = _finite_array(frame["camera_K"], (2, 3, 3), "camera_K")
    camera_pose = _finite_array(frame["camera_pose"], (2, 4, 4), "camera_pose")
    for view in range(2):
        _calibration(camera_K[view], camera_pose[view])
    return dict(mask_tokens=tokens, mask_valid=valid, camera_K=camera_K.copy(),
                camera_pose=camera_pose.copy(),
                eef_transform=eef_transform(frame["eef_pos"], frame["eef_quat_xyzw"]))
