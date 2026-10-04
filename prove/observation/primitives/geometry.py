"""Dependency-light RGB-D and coordinate-frame geometry primitives."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _intrinsics(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        return tuple(float(value[key]) for key in ("fx", "fy", "cx", "cy"))
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("intrinsics must be fx/fy/cx/cy or a 3x3 matrix")
    return (
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    )


def _transform(value: Any) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("coordinate transform must be finite 4x4")
    return transform


def depth_to_world_points(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    depth = np.asarray(inputs["depth"], dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    valid = np.isfinite(depth) & (depth > 0)
    support = np.ones(depth.shape, dtype=bool)
    return _masked_depth_points(
        support,
        valid,
        depth,
        inputs["intrinsics"],
        inputs["camera_to_world"],
    )


def mask_to_world_points(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    mask = np.asarray(inputs["mask"], dtype=bool)
    depth = np.asarray(inputs["depth"], dtype=np.float64)
    if mask.shape != depth.shape or depth.ndim != 2:
        raise ValueError("mask and depth must be aligned HxW arrays")
    valid = mask & np.isfinite(depth) & (depth > 0)
    return _masked_depth_points(
        mask,
        valid,
        depth,
        inputs["intrinsics"],
        inputs["camera_to_world"],
    )


def _masked_depth_points(
    support: np.ndarray,
    valid: np.ndarray,
    depth: np.ndarray,
    intrinsics: Any,
    camera_to_world: Any,
) -> Mapping[str, Any]:
    fx, fy, cx, cy = _intrinsics(intrinsics)
    if fx <= 0 or fy <= 0:
        raise ValueError("focal lengths must be positive")
    transform = _transform(camera_to_world)
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    camera = np.column_stack(
        ((cols - cx) * z / fx, (rows - cy) * z / fy, z)
    )
    if camera.size:
        homogeneous = np.column_stack((camera, np.ones(len(camera))))
        world = (transform @ homogeneous.T).T[:, :3]
    else:
        world = np.empty((0, 3), dtype=np.float64)
    support_count = int(np.count_nonzero(support))
    return {
        "world_points": world,
        "valid_ratio": (
            float(np.count_nonzero(valid) / support_count)
            if support_count
            else 0.0
        ),
    }


def transform_points(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    points = np.asarray(inputs["points"], dtype=np.float64)
    transform = _transform(inputs["transform"])
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return {"transformed_points": (transform @ homogeneous.T).T[:, :3]}


def robust_centroid(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    points = np.asarray(inputs["points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return {"centroid": None, "dispersion": None}
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return {
        "centroid": center,
        "dispersion": float(np.median(distances)),
    }


def project_world_points(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    points = np.asarray(inputs["world_points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world_points must be Nx3")
    world_to_camera = _transform(inputs["world_to_camera"])
    fx, fy, cx, cy = _intrinsics(inputs["intrinsics"])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    valid = camera[:, 2] > 0
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    pixels[valid, 0] = fx * camera[valid, 0] / camera[valid, 2] + cx
    pixels[valid, 1] = fy * camera[valid, 1] / camera[valid, 2] + cy
    return {"pixels": pixels, "in_front": valid}
