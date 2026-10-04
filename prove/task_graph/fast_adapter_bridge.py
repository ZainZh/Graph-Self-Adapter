"""Stage-local reuse of P45/P47 estimators; no LIBERO controller is implied.

A profile is a caller attestation, not certification that pixels actually meet
it. New environments still require separately scoped observer/control checks.
Only the legacy visible observation contract enters the frozen estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class AdapterProfile:
    task: str
    interaction: str
    controller_format: str
    wrist_rgb_shape: tuple[int, int, int]
    calibrated_wrist: bool
    four_visible_coplanar_corners: bool
    visible_role_masks: bool
    eef_pose: bool
    rigid_preheld_object: bool = False


INTERACTIONS = {'cup': 'cup_handle_grasp', 'insert': 'preheld_peg_slot_insertion',
                'slider': 'sliding_handle_pull'}
CONTROLLER_FORMAT = 'world_position_rotation_matrix_closed'
PUBLIC_SHAPES = {'camera_position': (3,), 'camera_rotation': (3, 3), 'K': (3, 3),
                 'position': (3,), 'rotation': (3, 3), 'role_bbox': (3, 4),
                 'role_visible': (3,), 'local': (48,)}


def compatibility_issues(profile: AdapterProfile, model_task: str) -> tuple[str, ...]:
    issues = []
    if model_task not in INTERACTIONS or profile.task != model_task:
        issues.append('checkpoint task does not match a supported stage task')
    if profile.interaction != INTERACTIONS.get(model_task):
        issues.append('interaction is outside the frozen task domain')
    if profile.controller_format != CONTROLLER_FORMAT:
        issues.append('legacy absolute SE(3)/gripper control requires a qualified controller bridge')
    if profile.wrist_rgb_shape != (448, 448, 3):
        issues.append('wrist RGB must use the calibrated 448x448 feature coordinates')
    for field in ('calibrated_wrist', 'four_visible_coplanar_corners', 'visible_role_masks', 'eef_pose'):
        if getattr(profile, field) is not True:
            issues.append(f'missing required observable: {field}')
    if model_task == 'insert' and profile.rigid_preheld_object is not True:
        issues.append('Insert requires a rigid preheld peg; placement is not peg insertion')
    return tuple(issues)


def public_observation(observation: Mapping[str, Any], representation: str) -> dict:
    """Reject extra fields, including truth; copy arrays across the boundary."""
    required = set(PUBLIC_SHAPES) - {'local'} | {'corners'}
    allowed = required | {'local'}
    if set(observation) - allowed or required - set(observation):
        raise ValueError('Adapter observations must contain only the legacy public fields')
    result = {}
    for key, shape in PUBLIC_SHAPES.items():
        if key == 'local' and representation == 'geometry':
            result[key] = np.zeros(48, dtype=np.float32)
            continue
        if key not in observation:
            raise ValueError('Fusion requires current frozen visual features')
        value = np.asarray(observation[key], dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid public field {key}: expected finite {shape}')
        result[key] = value.copy()
    corners = observation['corners']
    if corners is not None:
        corners = np.asarray(corners, dtype=np.float32)
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            raise ValueError('corners must be four visible image points or None')
        if np.any(corners < 0) or np.any(corners >= 448):
            raise ValueError('corners lie outside the calibrated image')
        corners = corners.copy()
    if result['K'][0, 0] <= 0 or result['K'][1, 1] <= 0:
        raise ValueError('Camera focal lengths must be positive')
    result['corners'] = corners
    return result


class FastAdapterStageBridge:
    """New learner per stage; current observations for both matched arms.

    This emits an estimate, never an actuator command or completion verdict.
    The graph supervisor owns completion and calls finish_stage()/stop().
    """
    def __init__(self, model, *, arm='continuous', family='p47', device='cpu',
                 learner_factory: Callable | None = None):
        if arm not in ('fixed', 'continuous') or family not in ('p45', 'p47'):
            raise ValueError('Stage comparison requires fixed/continuous and p45/p47')
        if model.representation not in ('geometry', 'relative_fusion', 'absolute_fusion'):
            raise ValueError('Unsupported frozen representation')
        if family == 'p47' and model.representation == 'absolute_fusion':
            raise ValueError('P47 excludes absolute_fusion')
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.arm, self.family, self.device = arm, family, device
        self._factory = learner_factory
        self._learner = None
        self.stage_id = None
        self._last_t = -1
        self._adaptation_frozen = False

    def start_stage(self, stage_id: str, profile: AdapterProfile):
        if self.stage_id is not None:
            raise RuntimeError('Finish the active stage before starting another')
        if not isinstance(stage_id, str) or not stage_id.strip():
            raise ValueError('A nonempty stage ID is required')
        issues = compatibility_issues(profile, self.model.task)
        if issues:
            raise ValueError('; '.join(issues))
        factory = self._factory
        if factory is None:
            if self.family == 'p47':
                from prove.fast_adapter.p47_neural import P47Latent
                factory = P47Latent
            else:
                from prove.fast_adapter.p45_neural import P45Latent
                factory = P45Latent
        self._learner = factory(self.model, arm=self.arm, device=self.device)
        self.stage_id, self._last_t, self._adaptation_frozen = stage_id, -1, False

    def freeze_adaptation(self):
        """Retain the estimate when the existing controller locks a moving target."""
        if self.stage_id is None:
            raise RuntimeError('No active stage')
        self._adaptation_frozen = True

    def step(self, observation: Mapping[str, Any], t: int):
        if self.stage_id is None:
            raise RuntimeError('No active stage')
        if isinstance(t, bool) or not isinstance(t, int) or t <= self._last_t:
            raise ValueError('Stage step indices must increase monotonically')
        public = public_observation(observation, self.model.representation)
        if not self._adaptation_frozen:
            self._learner.update(public, t)
        self._last_t = t
        return deepcopy(self._learner.estimate())

    def finish_stage(self):
        if self.stage_id is None:
            raise RuntimeError('No active stage')
        summary = dict(stage_id=self.stage_id, arm=self.arm,
                       records=deepcopy(self._learner.records))
        self.stop()
        return summary

    def stop(self):
        """Discard z, support views and reference on completion or cancellation."""
        self._learner = None
        self.stage_id, self._last_t, self._adaptation_frozen = None, -1, False


def paired_bridges(model, **kwargs):
    """Bind matched arms to the identical frozen model object."""
    return {arm: FastAdapterStageBridge(model, arm=arm, **kwargs)
            for arm in ('fixed', 'continuous')}
