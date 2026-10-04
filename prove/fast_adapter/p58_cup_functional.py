"""P58 cup functional-frame geometry from public masks and calibration.

The three points are semantic training targets, not visible mask landmarks.
Runtime adaptation consumes only predicted image rays, calibration and EEF state.
"""
from __future__ import annotations

import numpy as np

from prove.fast_adapter.p58_geometry_model import GeometryAdapterConfig, LowRankRayAdapter


REPRESENTATIONS = ("position_only", "functional_frame")
GEOMETRY_SCALE = .01


def functional_points(center, rotation, size):
    """Return grasp center and two axis points from isolated training truth."""
    center = np.asarray(center, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    if (center.shape[-1:] != (3,) or rotation.shape[-2:] != (3, 3) or size.shape[-1:] != (2,)
            or not np.isfinite(center).all() or not np.isfinite(rotation).all() or not np.isfinite(size).all()):
        raise ValueError("Invalid cup label shapes")
    grasp = center + rotation[..., :, 0] * (size[..., 0, None] - .003) - rotation[..., :, 2] * .004
    return np.stack((grasp, grasp + .03 * rotation[..., :, 0],
                     grasp + .03 * rotation[..., :, 1]), axis=-2)


def camera_pose(position, rotation):
    """Build world-from-camera transforms from P45 camera fields."""
    position = np.asarray(position, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if (position.shape[-1:] != (3,) or rotation.shape[-2:] != (3,3)
            or position.shape[:-1] != rotation.shape[:-2]
            or not np.isfinite(position).all() or not np.isfinite(rotation).all()
            or not np.allclose(rotation.swapaxes(-1,-2)@rotation,np.eye(3),atol=1e-5)
            or np.any(np.linalg.det(rotation) < .999)):
        raise ValueError("Invalid camera/EEF pose fields")
    pose = np.broadcast_to(np.eye(4), (*position.shape[:-1], 4, 4)).copy()
    pose[..., :3, :3] = rotation
    pose[..., :3, 3] = position
    return pose


def world_to_camera(points, pose):
    points, pose = np.asarray(points), np.asarray(pose)
    if not np.isfinite(points).all() or not np.isfinite(pose).all(): raise ValueError("Nonfinite pose transform")
    return np.einsum("...ij,...kj->...ki", pose[..., :3, :3].swapaxes(-1, -2),
                     points - pose[..., None, :3, 3])


def camera_to_world(points, pose):
    points, pose = np.asarray(points), np.asarray(pose)
    if not np.isfinite(points).all() or not np.isfinite(pose).all(): raise ValueError("Nonfinite pose transform")
    return np.einsum("...ij,...kj->...ki", pose[..., :3, :3], points) + pose[..., None, :3, 3]


def project(points_world, K, pose):
    if not np.isfinite(points_world).all() or not np.isfinite(K).all() or not np.isfinite(pose).all():
        raise ValueError("Nonfinite projection input")
    local = world_to_camera(points_world, pose)
    optical = local * np.array([1., -1., -1.])
    valid = optical[..., 2] > 1e-8
    homogeneous = np.einsum("...ij,...kj->...ki", K, optical)
    uv = homogeneous[..., :2] / np.maximum(homogeneous[..., 2:3], 1e-8)
    return uv, valid


def points_to_pose(points):
    """Recover the nearest proper rigid functional frame by Procrustes."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-2:] != (3, 3) or not np.isfinite(points).all():
        raise ValueError("Functional frame requires three finite points")
    nominal = np.array([[0., 0., 0.], [.03, 0., 0.], [0., .03, 0.]])
    src = nominal - nominal.mean(0)
    dst = points - points.mean(-2, keepdims=True)
    cross = np.einsum('ki,...kj->...ij', src, dst)
    u, _, vt = np.linalg.svd(cross)
    correction = np.broadcast_to(np.eye(3), u.shape).copy()
    correction[..., 2, 2] = np.linalg.det(vt.swapaxes(-1, -2) @ u.swapaxes(-1, -2))
    rotation = vt.swapaxes(-1, -2) @ correction @ u.swapaxes(-1, -2)
    translation = points.mean(-2) - np.einsum('...ij,j->...i', rotation, nominal.mean(0))
    pose = np.broadcast_to(np.eye(4), (*points.shape[:-2], 4, 4)).copy()
    pose[..., :3, :3], pose[..., :3, 3] = rotation, translation
    return pose


def mean_rotation(rotations):
    """Nearest proper SO(3) matrix to the arithmetic training rotation mean."""
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.shape[-2:] != (3,3) or not np.isfinite(rotations).all():
        raise ValueError("Invalid rotations")
    matrix = rotations.reshape(-1,3,3).mean(0)
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3); correction[2,2] = np.linalg.det(u@vt)
    return u@correction@vt


def public_features(mask_tokens, mask_valid, K, camera, eef):
    """Flatten the declared public mask, calibration and proprioceptive fields."""
    token = np.asarray(mask_tokens, dtype=np.float32)
    valid = np.asarray(mask_valid, dtype=np.float32)
    K, camera, eef = map(lambda x: np.asarray(x, dtype=np.float32), (K, camera, eef))
    if token.shape[-2:] != (3, 55) or valid.shape[-1:] != (3,):
        raise ValueError("Expected three P57 mask tokens and validity flags")
    if K.shape[-2:] != (3, 3) or camera.shape[-2:] != (4, 4) or eef.shape[-2:] != (4, 4):
        raise ValueError("Invalid public calibration or EEF transform")
    lead = token.shape[:-2]
    if valid.shape[:-1] != lead or K.shape[:-2] != lead or camera.shape[:-2] != lead or eef.shape[:-2] != lead:
        raise ValueError("Public feature leading dimensions differ")
    calib = np.stack((K[..., 0, 0], K[..., 1, 1], K[..., 0, 2], K[..., 1, 2]), -1) / 448.
    pieces = (token.reshape(*lead, -1), valid, calib,
              camera[..., :3, :].reshape(*lead, 12), eef[..., :3, :].reshape(*lead, 12))
    value = np.concatenate(pieces, -1).astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite public feature")
    return value


def ray_system(uv, K, pose):
    """Return point-to-ray projectors A and ray origins C."""
    uv, K, pose = map(lambda x: np.asarray(x, dtype=np.float64), (uv, K, pose))
    homogeneous = np.concatenate((uv, np.ones((*uv.shape[:-1], 1))), -1)
    optical = np.einsum("...ij,...kj->...ki", np.linalg.inv(K), homogeneous)
    local = optical * np.array([1., -1., -1.])
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    direction = np.einsum("...ij,...kj->...ki", pose[..., :3, :3], local)
    A = np.eye(3) - direction[..., :, None] * direction[..., None, :]
    C = np.broadcast_to(pose[..., None, :3, 3], direction.shape).copy()
    return A.astype(np.float32), C.astype(np.float32)


class AdapterConfig(GeometryAdapterConfig):
    def __init__(self, representation="functional_frame", **kwargs):
        if representation not in REPRESENTATIONS:
            raise ValueError("Unknown P58 representation")
        object.__setattr__(self, "representation", representation)
        super().__init__(point_count=1 if representation == "position_only" else 3, **kwargs)


class FunctionalFrameAdapter(LowRankRayAdapter):
    pass


def support_window(history):
    """P47 causal support contract: all <=32, otherwise first12 + latest20."""
    return history if len(history) <= 32 else history[:12] + history[-20:]
