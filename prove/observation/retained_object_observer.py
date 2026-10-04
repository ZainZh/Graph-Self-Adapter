"""Minimal visual check that a grasped target remains with the gripper.

This observer deliberately answers one question only: after the gripper enters
its engaged range, does the measured target keep the same target-to-EEF
relation? Target tracking and RGB-D geometry are upstream primitives. This
module does not perform semantic re-identification, use DINO features, inspect
task outcome, or read evaluator-only state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT = "prove.retained-object-observer-evidence.v1"
HELD = "HELD"
LOST = "LOST"
UNKNOWN = "UNKNOWN"


class RetainedObjectObserverError(ValueError):
    """Raised when the minimal observer receives an invalid runtime trace."""


@dataclass(frozen=True)
class RetainedObjectObserverConfig:
    """The three physical limits used by the minimal state machine."""

    engaged_aperture_m: float = 0.030
    maximum_initial_target_eef_distance_m: float = 0.100
    maximum_retained_relation_drift_m: float = 0.050

    def __post_init__(self) -> None:
        for name in (
            "engaged_aperture_m",
            "maximum_initial_target_eef_distance_m",
            "maximum_retained_relation_drift_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise RetainedObjectObserverError(f"{name} must be finite and positive")


def _xyz(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise RetainedObjectObserverError(f"{label} must be finite XYZ")
    return result


def derive_retained_object_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: RetainedObjectObserverConfig = RetainedObjectObserverConfig(),
) -> dict[str, Any]:
    """Derive causal HELD / LOST / UNKNOWN evidence from arrived rows."""

    if not isinstance(config, RetainedObjectObserverConfig):
        raise TypeError("config must be RetainedObjectObserverConfig")
    if not rows:
        raise RetainedObjectObserverError("trace must not be empty")

    baseline_relation: np.ndarray | None = None
    baseline_frame: int | None = None
    lost_frame: int | None = None
    frames: list[dict[str, Any]] = []

    for frame_index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or raw.get("frame_index") != frame_index:
            raise RetainedObjectObserverError("frames must be ordered and zero-based")
        available = raw.get("measurement_available")
        if not isinstance(available, bool):
            raise RetainedObjectObserverError("measurement_available must be bool")
        aperture = float(raw["gripper_aperture_m"])
        if not np.isfinite(aperture) or aperture < 0.0:
            raise RetainedObjectObserverError(
                "gripper_aperture_m must be finite and non-negative"
            )

        relation: np.ndarray | None = None
        target_distance: float | None = None
        drift: float | None = None
        if available:
            target = _xyz(raw["target_world_m"], "target_world_m")
            eef = _xyz(raw["eef_world_m"], "eef_world_m")
            relation = target - eef
            target_distance = float(np.linalg.norm(relation))

        engaged = aperture <= config.engaged_aperture_m
        state = UNKNOWN
        status = "unknown"
        reason = "gripper_not_engaged"

        if lost_frame is not None:
            state = LOST
            status = "trusted"
            reason = "previous_measured_relation_loss"
        elif not engaged:
            reason = "gripper_not_engaged"
        elif relation is None:
            reason = "target_measurement_unavailable"
        elif baseline_relation is None:
            if target_distance > config.maximum_initial_target_eef_distance_m:
                reason = "no_target_near_gripper_at_engagement"
            else:
                baseline_relation = relation.copy()
                baseline_frame = frame_index
                drift = 0.0
                state = HELD
                status = "trusted"
                reason = "target_near_gripper_at_engagement"
        else:
            drift = float(np.linalg.norm(relation - baseline_relation))
            if drift > config.maximum_retained_relation_drift_m:
                lost_frame = frame_index
                state = LOST
                status = "trusted"
                reason = "target_no_longer_retains_gripper_relation"
            else:
                state = HELD
                status = "trusted"
                reason = "target_retains_gripper_relation"

        frames.append(
            {
                "frame_index": frame_index,
                "measurement": {
                    "target_eef_relation_available": relation is not None,
                    "target_eef_distance_m": target_distance,
                    "relation_drift_from_engagement_m": drift,
                    "gripper_aperture_m": aperture,
                    "gripper_engaged": engaged,
                },
                "hypothesis": {
                    "semantic_target_identity_verified": False,
                },
                "verified_evidence": {
                    "retention_state": {
                        "value": state,
                        "status": status,
                        "reason": reason,
                    }
                },
            }
        )

    return {
        "format": FORMAT,
        "observer_scope": (
            "post-engagement target retention under an upstream tracked-target "
            "measurement"
        ),
        "config": asdict(config),
        "baseline_frame": baseline_frame,
        "first_lost_frame": lost_frame,
        "slip_detected": lost_frame is not None,
        "decision_frame": lost_frame,
        "frames": frames,
        "epistemic_invariants": {
            "arrived_measurements_only": True,
            "missing_measurement_yields_unknown": True,
            "semantic_identity_is_not_claimed": True,
            "dinov3_used": False,
            "runtime_gt_access": False,
            "control_authority": False,
        },
    }
