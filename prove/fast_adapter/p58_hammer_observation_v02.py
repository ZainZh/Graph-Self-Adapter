"""P58 hammer measurement revision: deterministic role-mask cleanup.

V1 renderer calibration, private geometry labels, and robot observations remain
unchanged.  This revision removes disconnected segmentation speckles by keeping
the largest 8-connected component of each public role mask before computing the
55-value mask token.  The same function is used by offline derivation and the
runtime renderer.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from prove.fast_adapter.p58_hammer_observation import (
    PRIVATE_KEYS, PUBLIC_KEYS, RESOLUTION, ROLE_GEOMS, camera_matrices,
    eef_transform, mask_projection_distances, private_labels,
    public_frame as _public_frame_v1, visible_masks as _visible_masks_v1,
)


def largest_component(mask):
    """Return the largest 8-connected component; ties use the first pixel."""
    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError("Role mask must be two dimensional")
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros(foreground.shape, dtype=np.uint8)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    # scipy labels components in row-major discovery order; np.argmax makes
    # an equal-size tie deterministic by retaining the earliest component.
    keep = int(np.argmax(sizes))
    return (labels == keep).astype(np.uint8)


def clean_role_masks(masks):
    """Apply largest-component cleanup to any leading view/role dimensions."""
    value = np.asarray(masks)
    if value.ndim < 2 or value.shape[-2:] != (RESOLUTION, RESOLUTION):
        raise ValueError("Expected role masks with trailing 448x448 dimensions")
    flat = value.reshape(-1, RESOLUTION, RESOLUTION)
    return np.stack([largest_component(mask) for mask in flat]).reshape(value.shape)


def visible_masks(segmentation, geom_ids, geom_object_type):
    """Extract geom masks and remove renderer segmentation speckles."""
    return clean_role_masks(_visible_masks_v1(segmentation, geom_ids, geom_object_type))


def public_frame(masks, camera_K, camera_pose, proprio, eef_transform):
    """Encode a public frame after the V2 largest-component transform."""
    return _public_frame_v1(clean_role_masks(masks), camera_K, camera_pose,
                            proprio, eef_transform)


def semantic_uv_valid(semantic_uv, semantic_depth, mask_valid):
    """Training-only label validity: visible role, positive depth, and in image."""
    uv = np.asarray(semantic_uv, dtype=np.float64)
    depth = np.asarray(semantic_depth, dtype=np.float64)
    visible = np.asarray(mask_valid, dtype=bool)
    if uv.shape != (2, 2, 2) or depth.shape != (2, 2) or visible.shape != (2, 2):
        raise ValueError("Invalid two-view semantic label shapes")
    inside = ((uv[..., 0] >= 0.) & (uv[..., 0] < RESOLUTION) &
              (uv[..., 1] >= 0.) & (uv[..., 1] < RESOLUTION))
    return visible & (depth > 0.) & inside
