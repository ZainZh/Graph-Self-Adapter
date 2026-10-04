"""Model-independent feature comparison primitives."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def cosine_feature_match(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    source = np.asarray(inputs["source_features"], dtype=np.float64)
    target = np.asarray(inputs["target_features"], dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("feature arrays must be NxD and MxD with common D")
    source_norm = np.linalg.norm(source, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=1, keepdims=True)
    valid_source = source_norm[:, 0] > 1e-12
    valid_target = target_norm[:, 0] > 1e-12
    normalized_source = np.zeros_like(source)
    normalized_target = np.zeros_like(target)
    normalized_source[valid_source] = source[valid_source] / source_norm[valid_source]
    normalized_target[valid_target] = target[valid_target] / target_norm[valid_target]
    similarities = normalized_source @ normalized_target.T
    if not source.shape[0] or not target.shape[0]:
        mutual = np.empty((0, 2), dtype=np.int64)
    else:
        source_best = np.argmax(similarities, axis=1)
        target_best = np.argmax(similarities, axis=0)
        pairs = [
            (source_index, int(target_index))
            for source_index, target_index in enumerate(source_best)
            if target_best[target_index] == source_index
            and valid_source[source_index]
            and valid_target[target_index]
        ]
        mutual = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return {
        "similarity_matrix": similarities,
        "mutual_matches": mutual,
    }
