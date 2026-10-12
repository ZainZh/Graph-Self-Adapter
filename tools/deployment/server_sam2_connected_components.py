"""Server ABI-compatible SAM2 connected-components backend.

This module implements the exact callable exposed by the upstream CUDA
extension.  It is deployed as ``sam2/_C.py`` only in the isolated server SAM2
tree when that tree's copied binary is incompatible with the server glibc and
no local CUDA compiler is available.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def get_connected_componnets(mask):
    """Return 8-connected foreground labels and per-pixel component areas."""
    if not isinstance(mask, torch.Tensor) or mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be a torch tensor shaped (N, 1, H, W)")
    binary = mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy()
    labels = np.zeros(binary.shape, dtype=np.int32)
    areas = np.zeros(binary.shape, dtype=np.int32)
    for batch in range(binary.shape[0]):
        count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            binary[batch, 0], connectivity=8, ltype=cv2.CV_32S
        )
        labels[batch, 0] = component_labels
        component_areas = stats[:, cv2.CC_STAT_AREA].astype(np.int32, copy=False)
        component_areas[0] = 0
        areas[batch, 0] = component_areas[component_labels]
        if count != len(component_areas):
            raise RuntimeError("OpenCV connected-component count differs from stats")
    return (
        torch.from_numpy(labels).to(device=mask.device),
        torch.from_numpy(areas).to(device=mask.device),
    )
