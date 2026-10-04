"""Frozen one-shot EEF feedback skills for same-initial-state qualification.

This is neither a pretrained policy nor Fast Adapter. Source world-coordinate
poses restrict the claimed scope. No simulator object state enters this module.
LIBERO's exporter stores post-action EEF observations and axis-angle rotation.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from .executor import RuleRouter
from .schema import SKILLS


@dataclass(frozen=True)
class DemoSkillConfig:
    position_tolerance: float = 0.006
    orientation_tolerance: float = 0.10
    translation_scale: float = 0.05
    rotation_scale: float = 0.5
    gain: float = 1.0
    max_action: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('controller values must be positive and finite')
        if self.max_action > 1:
            raise ValueError('OSC input bounds are [-1, 1]')


class QualifierRuleRouter(RuleRouter):
    """Explicit P2 mode; never substitutes a demo for an Adapter-labelled call."""
    @staticmethod
    def required_kind(skill):
        if skill not in SKILLS:
            raise ValueError('unknown skill')
        return 'code' if skill in ('Move', 'Approach') else 'fixed_demo'

    def resolve(self, grounded):
        backend = self._backends.get(grounded.skill)
        return backend if backend and backend.compatible(grounded) else None


class FrozenDemoSkill:
    revision = 'p49-frozen-demo-eef-feedback-v1'

    def __init__(self, positions, rotations, grippers, segments, config=DemoSkillConfig()):
        self.positions = np.array(positions, dtype=float, copy=True)
        self.rotations = np.array(rotations, dtype=float, copy=True)
        self.grippers = np.array(grippers, dtype=float, copy=True)
        n = len(self.positions)
        if self.positions.shape != (n, 3) or self.rotations.shape != (n, 3) or self.grippers.shape != (n,):
            raise ValueError('malformed source trajectory')
        if not all(np.isfinite(a).all() for a in (self.positions, self.rotations, self.grippers)):
            raise ValueError('nonfinite trajectory')
        for a in (self.positions, self.rotations, self.grippers):
            a.flags.writeable = False
        self.segments = dict(segments)
        if not self.segments or any(not 0 <= start < end <= n for start, end in self.segments.values()):
            raise ValueError('invalid frozen stage intervals')
        self.config = config
        self.stage_id = None
        self.index = None
        self._robot = None
        self.diagnostics = {}

    @classmethod
    def from_source(cls, path, graph, config=None):
        with h5py.File(path, 'r') as f:
            d = f['data/demo_0']
            if config is None:
                increments = np.linalg.norm(np.diff(d['obs/ee_pos'][()], axis=0), axis=1)
                nonzero = increments[increments > 1e-9]
                tolerance = float(np.clip(np.median(nonzero) if len(nonzero) else .002, .002, .008))
                config = DemoSkillConfig(position_tolerance=tolerance)
            result = cls(d['obs/ee_pos'][()], d['obs/ee_ori'][()], d['actions'][:, 6],
                         {s.id: (s.evidence.start_step, s.evidence.end_step) for s in graph.nodes}, config)
        result.source_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return result

    def set_observation(self, observation):
        # Copy only robot proprioception: object truth and supervisor facts are discarded.
        pos = np.asarray(observation['robot0_eef_pos'], dtype=float)
        quat = np.asarray(observation['robot0_eef_quat'], dtype=float)
        if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
            raise ValueError('invalid robot EEF pose')
        self._robot = (pos.copy(), Rotation.from_quat(quat).as_matrix())

    def on_enter(self, stage_id):
        if stage_id not in self.segments:
            raise ValueError('unbound stage')
        self.stage_id = stage_id
        self.index = self.segments[stage_id][0]

    def on_exit(self):
        self.stage_id = self.index = None

    def _error(self, index):
        pos, rotation = self._robot
        target = Rotation.from_rotvec(self.rotations[index].copy()).as_matrix()
        return self.positions[index] - pos, Rotation.from_matrix(target @ rotation.T).as_rotvec()

    def __call__(self, payload):
        if payload['stage_id'] != self.stage_id or self._robot is None:
            raise ValueError('backend requires stage entry and robot observation')
        dp, dr = self._error(self.index)
        reached = np.linalg.norm(dp) <= self.config.position_tolerance and np.linalg.norm(dr) <= self.config.orientation_tolerance
        # Repeated calls with a stationary off-target robot cannot run the trajectory.
        while reached and self.index + 1 < self.segments[self.stage_id][1]:
            self.index += 1
            dp, dr = self._error(self.index)
            reached = np.linalg.norm(dp) <= self.config.position_tolerance and np.linalg.norm(dr) <= self.config.orientation_tolerance
        # Stage-specific fixed grip: open before Grasp, closed from Grasp onward.
        grip = -1.0 if self.segments[self.stage_id][0] < 57 else 1.0
        action = np.r_[self.config.gain * dp / self.config.translation_scale,
                       self.config.gain * dr / self.config.rotation_scale, grip]
        action = np.clip(action, -self.config.max_action, self.config.max_action)
        self.diagnostics = dict(source_index=int(self.index), position_error=float(np.linalg.norm(dp)),
                                orientation_error=float(np.linalg.norm(dr)), waypoint_reached=bool(reached))
        return action
