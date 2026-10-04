"""P59 training-only functional geometry labels and projection helpers.

The semantic points are defined by isolated training truth.  They are not
visible corners and this module is not imported by the runtime observation
path.  Runtime size and held-length values must come from a frozen public
readout or calibration, never from these labels.
"""
from __future__ import annotations

import numpy as np

from prove.task_graph.p57_public_geometry import project_world_points


AXIS_POINT_OFFSET_M = .03
CONTROL_GEOMETRY_DIM = 12
PRIVATE_KEYS = frozenset({
    "slot_points_world", "held_tip_world", "slot_half_size",
    "held_length", "control_geometry", "semantic_uv",
    "semantic_depth", "projected_valid", "role_visible_valid",
})


def _finite(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def rotation_6d(rotation):
    """Return world x-axis then world y-axis, matching semantic axis points."""
    value = _finite(rotation, (3, 3), "slot_rotation")
    if (not np.allclose(value.T @ value, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value), 1., atol=1e-5)):
        raise ValueError("slot_rotation must be a proper rotation")
    return np.concatenate((value[:, 0], value[:, 1])).astype(np.float32)


def rotation_from_6d(value):
    """Gram-Schmidt a six-value x/y-axis prediction into SO(3)."""
    raw = _finite(value, (6,), "rotation_6d")
    x = raw[:3]
    if np.linalg.norm(x) < 1e-8:
        raise ValueError("rotation_6d x axis is degenerate")
    x = x / np.linalg.norm(x)
    y = raw[3:] - x * float(x @ raw[3:])
    if np.linalg.norm(y) < 1e-8:
        raise ValueError("rotation_6d axes are collinear")
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    return np.stack((x, y, z), axis=1).astype(np.float32)


def functional_points(slot_position, slot_rotation, eef_position,
                      eef_rotation, held_length):
    """Define slot origin/two axes and the physical tip of the held peg."""
    center = _finite(slot_position, (3,), "slot_position")
    rotation = rotation_from_6d(rotation_6d(slot_rotation)).astype(np.float64)
    eef_p = _finite(eef_position, (3,), "eef_position")
    eef_R = rotation_from_6d(rotation_6d(eef_rotation)).astype(np.float64)
    length = float(held_length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("held_length must be positive")
    slot = np.stack((center,
                     center + AXIS_POINT_OFFSET_M * rotation[:, 0],
                     center + AXIS_POINT_OFFSET_M * rotation[:, 1]))
    # The P45 tool +Z axis points away from the peg tip.
    tip = eef_p - eef_R[:, 2] * length
    return slot.astype(np.float32), tip.astype(np.float32)


def private_labels(slot_position, slot_rotation, slot_half_size,
                   eef_position, eef_rotation, held_length,
                   camera_K, camera_pose, role_visible):
    """Build one isolated outer-supervision/evaluation label record."""
    size = _finite(slot_half_size, (2,), "slot_half_size")
    if np.any(size <= 0):
        raise ValueError("slot_half_size must be positive")
    K = _finite(camera_K, (3, 3), "camera_K")
    pose = _finite(camera_pose, (4, 4), "camera_pose")
    visible = np.asarray(role_visible)
    if visible.shape != (3,) or visible.dtype != np.bool_:
        raise ValueError("role_visible must be boolean [3]")
    points, tip = functional_points(slot_position, slot_rotation, eef_position,
                                    eef_rotation, held_length)
    all_points = np.concatenate((points, tip[None]), axis=0)
    uv, in_front = project_world_points(all_points, K, pose)
    local = (all_points.astype(np.float64) - pose[:3, 3]) @ pose[:3, :3]
    depth = -local[:, 2]
    inside = ((uv[:, 0] >= 0.) & (uv[:, 0] < 448.) &
              (uv[:, 1] >= 0.) & (uv[:, 1] < 448.))
    role_visibility = np.r_[np.repeat(visible[2], 3), visible[1]]
    control = np.concatenate((np.asarray(slot_position, dtype=np.float32),
                              rotation_6d(slot_rotation), size.astype(np.float32),
                              np.array([held_length], dtype=np.float32)))
    result = dict(slot_points_world=points, held_tip_world=tip,
                  slot_half_size=size.astype(np.float32),
                  held_length=np.float32(held_length),
                  control_geometry=control.astype(np.float32),
                  semantic_uv=uv.astype(np.float32),
                  semantic_depth=depth.astype(np.float32),
                  projected_valid=(in_front & inside),
                  role_visible_valid=(role_visibility & in_front & inside))
    if set(result) != PRIVATE_KEYS or control.shape != (CONTROL_GEOMETRY_DIM,):
        raise RuntimeError("P59 private label schema drift")
    return result
