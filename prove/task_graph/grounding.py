"""Role grounding with explicit absent roles and observation provenance.

References identify permitted observations, not simulator poses. Supervisor
facts never enter this contract or the backend call.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
from .schema import Stage, Predicate

PERMITTED_SOURCES = frozenset(('vision', 'robot_state', 'public', 'simulator_visible_mask'))


@dataclass(frozen=True)
class InputReference:
    uri: str
    source_kind: str
    provenance: str

    def __post_init__(self):
        if self.source_kind not in PERMITTED_SOURCES:
            raise ValueError(f'forbidden runtime input source: {self.source_kind}')
        if not self.uri or not self.provenance:
            raise ValueError('input reference requires URI and provenance')


@dataclass(frozen=True)
class RoleBinding:
    semantic_ref: str | None
    present: bool
    roi: InputReference | None = None
    mask: InputReference | None = None
    visible: bool | None = None

    def __post_init__(self):
        if self.present != (self.semantic_ref is not None):
            raise ValueError('present denotes a known semantic role, not current visibility')
        if self.visible is not None and not isinstance(self.visible, bool):
            raise ValueError('visible must be true, false or unknown (None)')
        if not self.present and self.visible is True:
            raise ValueError('absent role cannot be visible')
        if self.visible is True and self.roi is None and self.mask is None:
            raise ValueError('visible role requires observed ROI or mask reference')
        if not self.present and (self.roi is not None or self.mask is not None):
            raise ValueError('absent role cannot have ROI or mask')


@dataclass(frozen=True)
class GroundedSkill:
    graph_id: str
    task_id: str
    instance_id: str
    stage_id: str
    skill: str
    actor: RoleBinding
    object: RoleBinding
    object_part: RoleBinding
    target: RoleBinding
    goal_relation: str
    termination: tuple[Predicate, ...]
    input_references: tuple[InputReference, ...]

    @classmethod
    def bind(cls, stage: Stage, roles: Mapping[str, RoleBinding],
             input_references: tuple[InputReference, ...] = (), *,
             graph_id: str, task_id: str, instance_id: str) -> 'GroundedSkill':
        if not graph_id or not task_id or not instance_id:
            raise ValueError('grounding requires graph/task/instance identity')
        expected = ('actor', 'object', 'object_part', 'target')
        if set(roles) != set(expected):
            raise ValueError('grounding must explicitly bind all four roles')
        for role in expected:
            if roles[role].semantic_ref != getattr(stage, role):
                raise ValueError(f'grounding semantic mismatch: {role}')
        return cls(graph_id, task_id, instance_id, stage.id, stage.skill, **roles, goal_relation=stage.goal_relation,
                   termination=stage.termination, input_references=tuple(input_references))

    def runtime_payload(self) -> dict:
        """Only goals, roles and allowlisted observation references; no supervisor values."""
        return asdict(self)
