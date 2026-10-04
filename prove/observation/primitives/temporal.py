"""Small temporal statistics; no interaction success semantics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def change_point_score(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    values = np.asarray(inputs["values"], dtype=np.float64).reshape(-1)
    if len(values) < 3 or not np.all(np.isfinite(values)):
        return {"change_score": None}
    reference = values[:-1]
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = max(1.4826 * mad, float(inputs["minimum_scale"]))
    return {"change_score": float(abs(values[-1] - median) / scale)}


def motion_consistency(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    first = np.asarray(inputs["first_motion"], dtype=np.float64).reshape(-1)
    second = np.asarray(inputs["second_motion"], dtype=np.float64).reshape(-1)
    if first.shape != second.shape or not len(first):
        raise ValueError("motion vectors must have equal non-empty shape")
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    similarity = None if norms <= 1e-12 else float(np.dot(first, second) / norms)
    return {"cosine_similarity": similarity}
