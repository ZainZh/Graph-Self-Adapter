"""Strict, declarative evidence-backed stage graphs; no executable predicates."""
from __future__ import annotations

from dataclasses import dataclass
import json
import operator
from pathlib import Path
from typing import Any, Mapping

import jsonschema

SKILLS = frozenset(('Move', 'Approach', 'Grasp', 'Lift', 'Align', 'Insert', 'Push', 'Pull', 'Rotate', 'Release'))
SCHEMA_PATH = Path(__file__).resolve().parents[2] / 'schemas/task_graph_v1.schema.json'


@dataclass(frozen=True)
class Predicate:
    signal: str
    op: str
    value: bool | float | str

    def evaluate(self, supervisor: Mapping[str, Any]) -> bool | None:
        actual = supervisor.get(self.signal)
        if actual is None:
            return None
        # Boolean is not a numerical measurement, despite Python's bool/int subtype.
        if isinstance(self.value, bool):
            if not isinstance(actual, bool):
                return None
        elif isinstance(self.value, (int, float)):
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                return None
            import math
            if not math.isfinite(actual) or not math.isfinite(self.value):
                return None
        elif not isinstance(actual, str):
            return None
        return {'eq': operator.eq, 'ne': operator.ne, 'lt': operator.lt,
                'le': operator.le, 'gt': operator.gt, 'ge': operator.ge}[self.op](actual, self.value)


@dataclass(frozen=True)
class Evidence:
    source: str
    start_step: int
    end_step: int
    provenance: str


@dataclass(frozen=True)
class Stage:
    id: str
    skill: str
    actor: str
    object: str | None
    object_part: str | None
    target: str | None
    goal_relation: str
    preconditions: tuple[Predicate, ...]
    invariants: tuple[Predicate, ...]
    termination: tuple[Predicate, ...]
    next_stage: str | None
    evidence: Evidence
    step_budget: int
    no_progress_budget: int


@dataclass(frozen=True)
class TaskGraph:
    graph_id: str
    task_id: str
    entry_stage: str
    supervisor_inputs: tuple[str, ...]
    nodes: tuple[Stage, ...]

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> 'TaskGraph':
        try:
            jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text())).validate(document)
        except jsonschema.ValidationError as exc:
            raise ValueError(f'invalid task graph: {exc.message}') from exc
        nodes = []
        for raw in document['nodes']:
            fields = dict(raw)
            for name in ('preconditions', 'invariants', 'termination'):
                fields[name] = tuple(Predicate(**item) for item in raw[name])
            fields['evidence'] = Evidence(**raw['evidence'])
            if fields['evidence'].end_step < fields['evidence'].start_step:
                raise ValueError('evidence interval is reversed')
            if fields['no_progress_budget'] > fields['step_budget']:
                raise ValueError('no-progress budget exceeds stage step budget')
            for predicate in fields['preconditions'] + fields['invariants'] + fields['termination']:
                if predicate.signal not in document['supervisor_inputs']:
                    raise ValueError(f'undeclared supervisor input: {predicate.signal}')
                if isinstance(predicate.value, float):
                    import math
                    if not math.isfinite(predicate.value):
                        raise ValueError('predicate value must be finite')
            nodes.append(Stage(**fields))
        graph = cls(document['graph_id'], document['task_id'], document['entry_stage'],
                    tuple(document['supervisor_inputs']), tuple(nodes))
        graph.validate_topology()
        return graph

    @classmethod
    def load(cls, path: str | Path) -> 'TaskGraph':
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate_topology(self) -> None:
        by_id = {node.id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError('duplicate stage ID')
        if self.entry_stage not in by_id:
            raise ValueError('entry stage does not exist')
        for node in self.nodes:
            if node.next_stage is not None and node.next_stage not in by_id:
                raise ValueError(f'dangling edge: {node.next_stage}')
        # Single explicit successor is intentional in v1; decisions stop for replanning.
        for start in by_id:
            seen = set()
            cursor = start
            while cursor is not None:
                if cursor in seen:
                    raise ValueError('cycle in task graph')
                seen.add(cursor)
                cursor = by_id[cursor].next_stage
        reached = set()
        cursor = self.entry_stage
        while cursor is not None:
            reached.add(cursor)
            cursor = by_id[cursor].next_stage
        if reached != set(by_id):
            raise ValueError('unreachable stage')

    def stage(self, stage_id: str) -> Stage:
        return next(node for node in self.nodes if node.id == stage_id)
