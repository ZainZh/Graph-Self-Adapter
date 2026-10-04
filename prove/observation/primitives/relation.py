"""Interaction-agnostic spatial relation measurements."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite XYZ")
    return vector


def relative_vector(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    target = _vector(inputs["target_position"], "target_position")
    reference = _vector(inputs["reference_position"], "reference_position")
    return {"relative_vector": target - reference}


def relative_drift(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    current = _vector(inputs["current_relative_vector"], "current_relative_vector")
    baseline = _vector(inputs["baseline_relative_vector"], "baseline_relative_vector")
    return {"relative_drift": float(np.linalg.norm(current - baseline))}
