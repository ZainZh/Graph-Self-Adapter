"""Runtime-safe event trigger for bounded Observer re-grounding.

The trigger consumes only robot kinematics, frame time, and feedback from the
previous ObservationEvidencePacket.  It never receives contact, held-state,
interaction outcome, residual, target masks, or evaluator annotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence


class EventTriggerError(ValueError):
    """Raised when trigger inputs or state violate the online contract."""


_FEEDBACK = frozenset({"trusted", "ambiguous", "unavailable"})


@dataclass(frozen=True)
class EventTriggerConfig:
    """Time/metric constants frozen before development evaluation."""

    frame_stride: int = 5
    motion_step_threshold_m: float = 0.03
    active_refresh_frames: int = 15
    active_grace_frames: int = 15
    maximum_staleness_frames: int = 45
    uncertainty_retry_frames: int = 5
    maximum_consecutive_retries: int = 2

    def __post_init__(self) -> None:
        positive_ints = (
            self.frame_stride,
            self.active_refresh_frames,
            self.active_grace_frames,
            self.maximum_staleness_frames,
            self.uncertainty_retry_frames,
        )
        if any(value <= 0 for value in positive_ints):
            raise EventTriggerError("trigger frame intervals must be positive")
        if not math.isfinite(self.motion_step_threshold_m) or self.motion_step_threshold_m <= 0:
            raise EventTriggerError("motion threshold must be finite and positive")
        if self.maximum_consecutive_retries < 0:
            raise EventTriggerError("maximum retries cannot be negative")
        if self.uncertainty_retry_frames % self.frame_stride:
            raise EventTriggerError("retry interval must align with frame stride")


@dataclass(frozen=True)
class EventTriggerState:
    previous_frame: int | None = None
    previous_eef_position_m: tuple[float, float, float] | None = None
    last_trigger_frame: int | None = None
    last_motion_frame: int | None = None
    last_feedback: str | None = None
    consecutive_retries: int = 0
    awaiting_feedback: bool = False


@dataclass(frozen=True)
class EventTriggerDecision:
    frame_id: int
    requested: bool
    reasons: tuple[str, ...]
    eef_step_m: float | None
    state: EventTriggerState


class EventReobservationTrigger:
    """Callable state machine for sparse, event-focused re-grounding."""

    trigger_id = "prove.observer_trigger.eef_event_reobservation"
    trigger_version = "1.0.0"

    def __init__(self, config: EventTriggerConfig = EventTriggerConfig()) -> None:
        self.config = config
        self.state = EventTriggerState()

    def reset(self) -> EventTriggerState:
        self.state = EventTriggerState()
        return self.state

    def decide(
        self,
        *,
        frame_id: int,
        eef_position_m: Sequence[float],
        terminal: bool = False,
    ) -> EventTriggerDecision:
        if self.state.awaiting_feedback:
            raise EventTriggerError("feedback is required before the next frame")
        if frame_id < 0 or frame_id % self.config.frame_stride:
            raise EventTriggerError("frame must be non-negative and stride-aligned")
        if self.state.previous_frame is not None and frame_id <= self.state.previous_frame:
            raise EventTriggerError("trigger frames must be strictly increasing")
        position = _vector3(eef_position_m)
        step = (
            None
            if self.state.previous_eef_position_m is None
            else math.dist(position, self.state.previous_eef_position_m)
        )
        motion = step is not None and step >= self.config.motion_step_threshold_m
        motion_was_active = (
            self.state.last_motion_frame is not None
            and frame_id - self.state.last_motion_frame <= self.config.active_grace_frames
        )
        last_motion = frame_id if motion else self.state.last_motion_frame

        reasons: list[str] = []
        if self.state.last_trigger_frame is None:
            reasons.append("bootstrap")
        if terminal:
            reasons.append("terminal_verification")
        if motion and not motion_was_active:
            reasons.append("eef_motion_onset")

        since_trigger = (
            None
            if self.state.last_trigger_frame is None
            else frame_id - self.state.last_trigger_frame
        )
        active_now = (
            last_motion is not None
            and frame_id - last_motion <= self.config.active_grace_frames
        )
        if (
            since_trigger is not None
            and since_trigger >= self.config.active_refresh_frames
            and active_now
        ):
            reasons.append("active_motion_refresh")
        retry = (
            since_trigger is not None
            and since_trigger >= self.config.uncertainty_retry_frames
            and self.state.last_feedback in {"ambiguous", "unavailable"}
            and self.state.consecutive_retries < self.config.maximum_consecutive_retries
        )
        if retry:
            reasons.append("uncertainty_retry")
        if (
            since_trigger is not None
            and since_trigger >= self.config.maximum_staleness_frames
        ):
            reasons.append("maximum_staleness")

        requested = bool(reasons)
        retries = self.state.consecutive_retries
        if requested:
            retries = retries + 1 if reasons == ["uncertainty_retry"] else 0
        new_state = replace(
            self.state,
            previous_frame=frame_id,
            previous_eef_position_m=position,
            last_trigger_frame=frame_id if requested else self.state.last_trigger_frame,
            last_motion_frame=last_motion,
            consecutive_retries=retries,
            awaiting_feedback=requested,
        )
        self.state = new_state
        return EventTriggerDecision(
            frame_id=frame_id,
            requested=requested,
            reasons=tuple(dict.fromkeys(reasons)),
            eef_step_m=step,
            state=new_state,
        )

    def feedback(self, status: str) -> EventTriggerState:
        if not self.state.awaiting_feedback:
            raise EventTriggerError("feedback is accepted only after a trigger")
        if status not in _FEEDBACK:
            raise EventTriggerError("unsupported Observer feedback status")
        self.state = replace(
            self.state,
            last_feedback=status,
            consecutive_retries=0 if status == "trusted" else self.state.consecutive_retries,
            awaiting_feedback=False,
        )
        return self.state


def _vector3(value: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise EventTriggerError("EEF position must be a length-three vector")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise EventTriggerError("EEF position must be finite")
    return result  # type: ignore[return-value]
