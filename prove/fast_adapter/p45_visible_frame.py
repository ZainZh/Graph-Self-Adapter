"""Visible-pixel rectangular-frame corners for P45 candidate engineering.

The four box members of a handle leave bevels in their combined silhouette.
Approximating that silhouette by four *vertices* mistakes bevel endpoints for
the functional rectangle. Here the four long exterior edges are fit as image
lines, and adjacent line intersections define the rectangle's outer corners.

Inputs are visible mask pixels only. No calibration, geometry dimensions,
simulator state, target projections, depth or success is accepted. A real
closed visible opening remains necessary; an interrupted frame yields None.
"""
from __future__ import annotations

import cv2
import numpy as np


def _ordered(points):
    center = points.mean(0)
    order = np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    points = points[order]
    return np.roll(points, -np.argmin(points.sum(1)), axis=0)


def _edge_line(contour, start, end):
    """Fit a silhouette edge from its central pixel support, omitting bevels."""
    delta = end - start
    length = np.linalg.norm(delta)
    direction = delta / length
    along = (contour - start) @ direction
    normal = np.array([-direction[1], direction[0]])
    across = (contour - start) @ normal
    support = contour[(along > .08 * length) & (along < .92 * length)
                      & (np.abs(across) <= 1.25)]
    if len(support) < 5:
        return None
    for _ in range(3):
        center = support.mean(0)
        _, _, vectors = np.linalg.svd(support - center, full_matrices=False)
        normal = vectors[-1]
        residual = (support - center) @ normal
        keep = np.abs(residual) <= 1.
        if keep.all() or keep.sum() < 5:
            break
        support = support[keep]
    spread = np.ptp((support - center) @ vectors[0])
    if spread < .4 * length:
        return None
    return np.r_[normal, -np.dot(normal, center)]


def visible_frame_corners(mask):
    """Return ``(corners_or_none, diagnostics)`` from a 2-D visible-part mask.

    Four dominant silhouette lines need distinct orientations, useful pixel
    support and convex bounded intersections. These are public image checks,
    rather than tests against privileged target pose or future performance.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError('Expected a two-dimensional visible frame mask')
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    info = dict(opening_visible=False, corner_method='visible_exterior_line_intersections',
                fitted_edges=0, reason='closed visible opening unavailable')
    if hierarchy is None:
        return None, info
    holes = [(float(cv2.contourArea(c)), int(h[3]))
             for c, h in zip(contours, hierarchy[0]) if h[3] >= 0 and cv2.contourArea(c) > 25]
    if not holes:
        return None, info
    hole_area, parent = max(holes)
    info['opening_visible'] = True
    contour = contours[parent].reshape(-1, 2).astype(float)
    hull = cv2.convexHull(contours[parent])
    perimeter = cv2.arcLength(hull, True)
    vertices = cv2.approxPolyDP(hull, max(.7, .005 * perimeter), True).reshape(-1, 2).astype(float)
    lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    if len(vertices) < 4:
        return None, dict(info, reason='fewer than four silhouette edges')
    edge_ids = sorted(np.argsort(lengths)[-4:].tolist())
    if min(lengths[edge_ids]) < 6.:
        return None, dict(info, reason='insufficient image extent')
    lines = []
    for i in edge_ids:
        line = _edge_line(contour, vertices[i], vertices[(i + 1) % len(vertices)])
        if line is None:
            return None, dict(info, fitted_edges=len(lines), reason='insufficient straight-edge support')
        lines.append(line)
    corners = []
    for i, line in enumerate(lines):
        previous = lines[(i - 1) % 4]
        cross = np.cross(previous, line)
        if abs(cross[2]) < .2:
            return None, dict(info, fitted_edges=4, reason='adjacent edges nearly parallel')
        corners.append(cross[:2] / cross[2])
    corners = np.asarray(corners)
    # Reject unstable extensions or an edge selection that does not enclose the
    # observed opening. A small pixel allowance covers line-fit quantization.
    xmin, ymin = contour.min(0) - .15 * perimeter
    xmax, ymax = contour.max(0) + .15 * perimeter
    bounded = ((corners[:, 0] >= xmin) & (corners[:, 0] <= xmax)
               & (corners[:, 1] >= ymin) & (corners[:, 1] <= ymax)).all()
    polygon = corners.astype(np.float32)
    if not bounded or not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) <= hole_area:
        return None, dict(info, fitted_edges=4, reason='unstable exterior-line intersections')
    return _ordered(corners), dict(info, fitted_edges=4, reason='valid visible exterior frame')
