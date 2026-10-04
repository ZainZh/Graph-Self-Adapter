"""Phase-local physical-state readout for sparse trusted measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class PhaseStateError(ValueError):
    """Raised when trusted metric evidence is malformed."""


@dataclass(frozen=True)
class PhaseStateConfig:
    minimum_lift_m: float = 0.05
    early_lift_offsets: tuple[int, int] = (0, 25)
    post_lift_offsets: tuple[int, int] = (30, 55)
    late_transport_start_offset: int = 90
    maximum_segment_gap_frames: int = 20
    early_attachment_error_scales: float = 0.375
    decoupling_error_scales: float = 0.40
    sustained_drift_scales: float = 0.40

    def __post_init__(self) -> None:
        if self.minimum_lift_m <= 0:
            raise PhaseStateError("minimum_lift_m must be positive")
        if self.early_lift_offsets[0] > self.early_lift_offsets[1]:
            raise PhaseStateError("early lift offsets are reversed")
        if self.post_lift_offsets[0] > self.post_lift_offsets[1]:
            raise PhaseStateError("post lift offsets are reversed")
        if self.maximum_segment_gap_frames <= 0:
            raise PhaseStateError("maximum segment gap must be positive")
        for value in (
            self.early_attachment_error_scales,
            self.decoupling_error_scales,
            self.sustained_drift_scales,
        ):
            if value <= 0:
                raise PhaseStateError("state thresholds must be positive")


def derive_lift_onset(
    trace_records: Sequence[Mapping[str, Any]],
    *,
    actor: str,
    minimum_lift_m: float,
) -> tuple[int, float]:
    """Derive lift onset from runtime EEF motion only."""

    frames: list[int] = []
    positions: list[Any] = []
    for record in trace_records:
        try:
            frames.append(int(record["relative_frame"]))
            positions.append(record["arms"][actor]["eef_position_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PhaseStateError("execution trace lacks EEF state") from exc
    xyz = np.asarray(positions, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(frames) < 3:
        raise PhaseStateError("execution trace must contain finite XYZ samples")
    if not np.all(np.isfinite(xyz)) or any(
        later <= earlier for earlier, later in zip(frames, frames[1:])
    ):
        raise PhaseStateError("execution trace frames/positions are invalid")
    candidates = [
        offset
        for offset in range(len(xyz) - 1)
        if float(np.max(xyz[offset:, 2]) - xyz[offset, 2]) >= minimum_lift_m
    ]
    if not candidates:
        raise PhaseStateError("no EEF lift satisfying minimum_lift_m")
    offset = min(candidates, key=lambda item: (xyz[item, 2], frames[item]))
    rise = float(np.max(xyz[offset:, 2]) - xyz[offset, 2])
    return frames[offset], rise


def _phase(frame: int, lift_onset: int, config: PhaseStateConfig) -> str:
    relative = frame - lift_onset
    if config.early_lift_offsets[0] <= relative <= config.early_lift_offsets[1]:
        return "early_lift"
    if config.post_lift_offsets[0] <= relative <= config.post_lift_offsets[1]:
        return "post_lift"
    if relative >= config.late_transport_start_offset:
        return "late_transport"
    return "outside_decision_phase"


def _optional_state(value: bool | None, reason: str) -> dict[str, Any]:
    return {
        "value": value,
        "status": "verified" if value is not None else "unknown",
        "reason": reason,
    }


def derive_phase_aware_state(
    *,
    evidence_rows: Sequence[Mapping[str, Any]],
    trace_records: Sequence[Mapping[str, Any]],
    actor: str,
    target_scale_m: float,
    config: PhaseStateConfig = PhaseStateConfig(),
) -> dict[str, Any]:
    """Derive GRASP physical states without joining across long sparse gaps."""

    if not np.isfinite(target_scale_m) or target_scale_m <= 0:
        raise PhaseStateError("target_scale_m must be finite and positive")
    lift_onset, lift_rise = derive_lift_onset(
        trace_records,
        actor=actor,
        minimum_lift_m=config.minimum_lift_m,
    )
    rows: list[dict[str, Any]] = []
    for raw in evidence_rows:
        frame = int(raw["frame_index"])
        target = np.asarray(raw["target_world_m"], dtype=np.float64)
        eef = np.asarray(raw["eef_world_m"], dtype=np.float64)
        if target.shape != (3,) or eef.shape != (3,):
            raise PhaseStateError("target/EEF measurements must be XYZ")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(eef)):
            raise PhaseStateError("target/EEF measurements must be finite")
        rows.append(
            {
                "frame_index": frame,
                "phase": _phase(frame, lift_onset, config),
                "target": target,
                "eef": eef,
                "relative": target - eef,
            }
        )
    rows.sort(key=lambda row: row["frame_index"])
    if len({row["frame_index"] for row in rows}) != len(rows):
        raise PhaseStateError("duplicate evidence frame")

    segment_rows: list[dict[str, Any]] = []
    errors: dict[str, list[float]] = {
        "early_lift": [],
        "post_lift": [],
        "late_transport": [],
    }
    for previous, current in zip(rows, rows[1:]):
        gap = int(current["frame_index"] - previous["frame_index"])
        same_phase = previous["phase"] == current["phase"]
        valid = (
            same_phase
            and current["phase"] in errors
            and gap <= config.maximum_segment_gap_frames
        )
        coupling_error = float(
            np.linalg.norm(
                (current["target"] - previous["target"])
                - (current["eef"] - previous["eef"])
            )
            / target_scale_m
        )
        segment_rows.append(
            {
                "start_frame": int(previous["frame_index"]),
                "end_frame": int(current["frame_index"]),
                "frame_gap": gap,
                "start_phase": previous["phase"],
                "end_phase": current["phase"],
                "temporally_valid": valid,
                "coupling_error_object_scales": coupling_error,
            }
        )
        if valid:
            errors[current["phase"]].append(coupling_error)

    early_errors = errors["early_lift"]
    if early_errors:
        early_attached = bool(
            float(np.median(early_errors))
            < config.early_attachment_error_scales
        )
        early_reason = "phase_local_coupling_measured"
    else:
        early_attached = None
        early_reason = "no_temporally_valid_early_lift_segment"

    early_relative = [
        row["relative"] for row in rows if row["phase"] == "early_lift"
    ]
    after_early = [
        row
        for row in rows
        if row["phase"] in {"post_lift", "late_transport"}
    ]
    if early_relative and after_early:
        anchor = np.median(np.asarray(early_relative), axis=0)
        drifts = [
            float(np.linalg.norm(row["relative"] - anchor) / target_scale_m)
            for row in after_early
        ]
        sustained_drift = float(np.median(drifts[-3:]))
    else:
        drifts = []
        sustained_drift = None
    post_errors = errors["post_lift"]
    if early_attached is True and post_errors and sustained_drift is not None:
        later_decoupled = bool(
            max(post_errors) >= config.decoupling_error_scales
            and sustained_drift >= config.sustained_drift_scales
        )
        decoupling_reason = "phase_local_post_lift_measurement"
    else:
        later_decoupled = None
        decoupling_reason = (
            "early_attachment_not_verified"
            if early_attached is not True
            else "post_lift_decoupling_evidence_unavailable"
        )

    late_errors = errors["late_transport"]
    if late_errors:
        late_coupled = bool(
            float(np.median(late_errors)) < config.decoupling_error_scales
        )
        late_reason = "phase_local_late_transport_measurement"
    else:
        late_coupled = None
        late_reason = "no_temporally_valid_late_transport_segment"

    if early_attached is False:
        held: bool | None = False
        follows: bool | None = False
        residual = "grasp_not_established"
    elif early_attached is True and later_decoupled is True:
        held, follows, residual = True, False, "slip_during_lift"
    elif (
        early_attached is True
        and later_decoupled is False
        and late_coupled is True
    ):
        held, follows, residual = True, True, "none"
    else:
        held, follows, residual = None, None, "unknown"

    return {
        "format": "prove.phase-aware-physical-state.v1",
        "actor": actor,
        "lift_onset_relative_frame": lift_onset,
        "lift_rise_m": lift_rise,
        "target_scale_m": float(target_scale_m),
        "config": asdict(config),
        "states": {
            "early_attached": _optional_state(early_attached, early_reason),
            "later_decoupled": _optional_state(
                later_decoupled, decoupling_reason
            ),
            "late_coupled": _optional_state(late_coupled, late_reason),
        },
        "observed_effect": {
            "held_by_actor": {"value": held, "source": "phase_aware_physical_state"},
            "follows_actor_during_lift": {
                "value": follows,
                "source": "phase_aware_physical_state",
            },
            "object_shifted_during_closure": {
                "value": None,
                "source": "unobserved_by_this_candidate",
            },
        },
        "interaction_residual": {
            "value": residual,
            "source": "frozen_grasp_effect_mapping",
        },
        "phase_measurement_counts": {
            phase: sum(row["phase"] == phase for row in rows)
            for phase in ("early_lift", "post_lift", "late_transport")
        },
        "phase_segment_errors_object_scales": errors,
        "sustained_relative_drift_object_scales": sustained_drift,
        "relative_drifts_object_scales": drifts,
        "segments": segment_rows,
        "epistemic_invariants": {
            "long_gap_segments_rejected": True,
            "missing_early_attachment_mapped_to_false": False,
            "evaluator_phase_labels_used": False,
            "runtime_gt_access": False,
        },
    }
