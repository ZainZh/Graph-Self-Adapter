"""Atomic geometry measurements for upright open-container candidates.

These primitives know nothing about PLACE_IN, success, residuals, or templates.
They only summarize a measured point cloud and relate a measured point to that
summary. Observer gating and an independent evaluator decide whether those
measurements are sufficient evidence for a physical transition.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


_LOWER_QUANTILE = 0.05
_UPPER_QUANTILE = 0.95
_MINIMUM_POINTS = 16
_MINIMUM_EXTENT_M = 1e-4


def _finite_points(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be Nx3")
    return points[np.all(np.isfinite(points), axis=1)]


def _canonical_xy_axes(centered_xy: np.ndarray) -> np.ndarray:
    covariance = centered_xy.T @ centered_xy / max(len(centered_xy), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].T
    primary = axes[0]
    dominant = int(np.argmax(np.abs(primary)))
    if primary[dominant] < 0:
        axes[0] *= -1.0
    if np.linalg.det(axes) < 0:
        axes[1] *= -1.0
    return axes


def upright_pointcloud_envelope(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a robust world-z-aligned envelope of measured surface points."""

    points = _finite_points(inputs["points"])
    if len(points) < _MINIMUM_POINTS:
        return {
            "envelope": None,
            "valid": False,
            "support_count": int(len(points)),
            "reason": "insufficient_finite_points",
        }
    origin_xy = np.median(points[:, :2], axis=0)
    centered_xy = points[:, :2] - origin_xy
    axes_xy = _canonical_xy_axes(centered_xy)
    local_xy = centered_xy @ axes_xy.T
    lower_xy = np.quantile(local_xy, _LOWER_QUANTILE, axis=0)
    upper_xy = np.quantile(local_xy, _UPPER_QUANTILE, axis=0)
    z_low, z_high = np.quantile(
        points[:, 2], [_LOWER_QUANTILE, _UPPER_QUANTILE]
    )
    spans = np.concatenate((upper_xy - lower_xy, [z_high - z_low]))
    if not np.all(np.isfinite(spans)) or np.any(spans <= _MINIMUM_EXTENT_M):
        return {
            "envelope": None,
            "valid": False,
            "support_count": int(len(points)),
            "reason": "degenerate_robust_extent",
        }
    envelope = {
        "origin_xy_world": origin_xy,
        "axes_xy_world": axes_xy,
        "lower_xy_local": lower_xy,
        "upper_xy_local": upper_xy,
        "z_low_world": float(z_low),
        "z_high_world": float(z_high),
        "support_count": int(len(points)),
        "lower_quantile": _LOWER_QUANTILE,
        "upper_quantile": _UPPER_QUANTILE,
    }
    return {
        "envelope": envelope,
        "valid": True,
        "support_count": int(len(points)),
        "reason": "robust_upright_envelope_available",
    }


def point_in_upright_envelope(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Measure a point's signed clearances to an upright envelope."""

    point = np.asarray(inputs["point"], dtype=np.float64).reshape(-1)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point must be a finite vector3")
    envelope = inputs["envelope"]
    if not isinstance(envelope, Mapping):
        raise ValueError("envelope must be a mapping")
    origin_xy = np.asarray(envelope["origin_xy_world"], dtype=np.float64)
    axes_xy = np.asarray(envelope["axes_xy_world"], dtype=np.float64)
    lower_xy = np.asarray(envelope["lower_xy_local"], dtype=np.float64)
    upper_xy = np.asarray(envelope["upper_xy_local"], dtype=np.float64)
    if (
        origin_xy.shape != (2,)
        or axes_xy.shape != (2, 2)
        or lower_xy.shape != (2,)
        or upper_xy.shape != (2,)
        or not np.all(np.isfinite(origin_xy))
        or not np.all(np.isfinite(axes_xy))
        or not np.all(np.isfinite(lower_xy))
        or not np.all(np.isfinite(upper_xy))
    ):
        raise ValueError("malformed upright envelope")
    z_low = float(envelope["z_low_world"])
    z_high = float(envelope["z_high_world"])
    if not np.isfinite(z_low) or not np.isfinite(z_high) or z_high <= z_low:
        raise ValueError("malformed upright envelope height")
    local_xy = (point[:2] - origin_xy) @ axes_xy.T
    horizontal_clearances = np.concatenate(
        (local_xy - lower_xy, upper_xy - local_xy)
    )
    horizontal_margin = float(np.min(horizontal_clearances))
    above_bottom = float(point[2] - z_low)
    below_rim = float(z_high - point[2])
    spans = upper_xy - lower_xy
    horizontal_normalized = float(
        np.min(horizontal_clearances / np.tile(spans, 2))
    )
    vertical_span = z_high - z_low
    vertical_normalized = float(min(above_bottom, below_rim) / vertical_span)
    return {
        "inside_envelope": bool(
            horizontal_margin >= 0.0 and above_bottom >= 0.0 and below_rim >= 0.0
        ),
        "horizontal_margin_m": horizontal_margin,
        "above_bottom_m": above_bottom,
        "below_rim_m": below_rim,
        "normalized_margin": min(horizontal_normalized, vertical_normalized),
        "point_local_xy": local_xy,
    }
