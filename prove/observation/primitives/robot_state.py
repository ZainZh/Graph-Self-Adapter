"""Typed accessors over sanitized robot state."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def get_eef_pose(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    actor = str(inputs["actor"])
    robot_state = inputs["robot_state"]
    try:
        pose = np.asarray(robot_state["arms"][actor]["eef_pose_world"], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing EEF pose for actor {actor}") from exc
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("eef_pose_world must be finite 4x4")
    return {"eef_pose_world": pose}


def get_gripper_state(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    actor = str(inputs["actor"])
    robot_state = inputs["robot_state"]
    try:
        state = robot_state["arms"][actor]["gripper_state"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing gripper state for actor {actor}") from exc
    if not isinstance(state, Mapping):
        raise ValueError("gripper_state must be a mapping")
    return {"gripper_state": dict(state)}
