"""Bounded deterministic stage orchestration. This module starts no simulator."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from .grounding import GroundedSkill
from .schema import SKILLS, Stage, TaskGraph

CODE_SKILLS = frozenset(('Move', 'Approach', 'Release'))
ADAPTER_SKILLS = frozenset(('Align', 'Insert'))


@dataclass(frozen=True)
class Backend:
    kind: str
    revision: str
    call: Callable[[dict], Any]
    compatible: Callable[[GroundedSkill], bool]


class RuleRouter:
    def __init__(self):
        self._backends: dict[str, Backend] = {}

    @staticmethod
    def required_kind(skill: str) -> str:
        if skill not in SKILLS:
            raise ValueError(f'unknown skill: {skill}')
        return 'code' if skill in CODE_SKILLS else 'fast_adapter' if skill in ADAPTER_SKILLS else 'fixed_policy'

    def register(self, skill: str, *, kind: str, revision: str,
                 call: Callable[[dict], Any], compatible: Callable[[GroundedSkill], bool]) -> None:
        if kind != self.required_kind(skill) or not revision:
            raise ValueError('backend kind/revision does not meet fixed routing contract')
        if skill in self._backends:
            raise ValueError('backend already registered')
        self._backends[skill] = Backend(kind, revision, call, compatible)

    def resolve(self, grounded: GroundedSkill) -> Backend | None:
        if grounded.skill in ADAPTER_SKILLS:
            # Actor may be known from proprioception; manipulated visual entities
            # require actual visible evidence, never semantic existence alone.
            for name in ("object", "object_part", "target"):
                role = getattr(grounded, name)
                if role.present and role.visible is not True:
                    return None
        backend = self._backends.get(grounded.skill)
        return backend if backend is not None and backend.compatible(grounded) else None


@dataclass(frozen=True)
class JudgeResult:
    state: str  # continue / complete / local_failure / structural_failure
    stage_id: str
    reason: str
    action: Any = None
    next_stage: str | None = None
    graph_complete: bool = False
    replan_requested: bool = False
    backend_revision: str | None = None


class GraphExecutor:
    def __init__(self, graph: TaskGraph, router: RuleRouter, *, instance_id: str):
        if not instance_id:
            raise ValueError("executor requires instance identity")
        self.instance_id = instance_id
        self.graph, self.router = graph, router
        self.current_stage = graph.entry_stage
        self.steps = self.stalled = 0
        self.entered = False
        self.stopped = False
        self.last_result: JudgeResult | None = None

    def _stop(self, state: str, reason: str) -> JudgeResult:
        self.stopped = True
        self.last_result = JudgeResult(state, self.current_stage, reason,
                                       replan_requested=True)
        return self.last_result

    def tick(self, grounded: GroundedSkill, supervisor: Mapping[str, Any], *,
             progress: bool = False) -> JudgeResult:
        """Judge current observations, then issue at most one backend action.

        ``progress`` is an explicit supervisor report; no backend prediction or
        adapter loss is interpreted as progress or task completion. Budget counts
        tick opportunities, including waiting for missing supervisor measurements.
        """
        if self.stopped:
            return self.last_result
        if (grounded.graph_id, grounded.task_id, grounded.instance_id) != (
                self.graph.graph_id, self.graph.task_id, self.instance_id):
            return self._stop("structural_failure", "grounding_identity_mismatch")
        stage = self.graph.stage(self.current_stage)
        if grounded.stage_id != stage.id or grounded.skill != stage.skill:
            return self._stop('structural_failure', 'grounding_stage_mismatch')
        for role in ('actor', 'object', 'object_part', 'target'):
            if getattr(grounded, role).semantic_ref != getattr(stage, role):
                return self._stop('structural_failure', 'grounding_role_mismatch')
        if grounded.goal_relation != stage.goal_relation or grounded.termination != stage.termination:
            return self._stop('structural_failure', 'grounding_contract_mismatch')
        facts = {key: supervisor[key] for key in self.graph.supervisor_inputs if key in supervisor}
        for signal in ('object_lost', 'wrong_target'):
            if facts.get(signal) is True:
                return self._stop('structural_failure', signal)
        invariants = [predicate.evaluate(facts) for predicate in stage.invariants]
        if False in invariants:
            return self._stop('structural_failure', 'invariant_violated')
        # Entry conditions are checked once, never confused with ongoing invariants.
        if not self.entered:
            conditions = [predicate.evaluate(facts) for predicate in stage.preconditions]
            if False in conditions:
                return self._stop('structural_failure', 'precondition_failed')
            if None in conditions:
                return self._wait(stage, progress=False, reason='precondition_unknown')
            self.entered = True
        if None in invariants:
            return self._wait(stage, progress=False, reason='invariant_unknown')
        if all(predicate.evaluate(facts) is True for predicate in stage.termination):
            finished = stage.next_stage is None
            result = JudgeResult('complete', stage.id, 'termination_verified',
                                 next_stage=stage.next_stage, graph_complete=finished)
            self.last_result = result
            if finished:
                self.stopped = True
            else:
                self.current_stage = stage.next_stage
                self.steps = self.stalled = 0
                self.entered = False
            return result
        if progress is True:
            self.stalled = 0
        if self.stalled >= stage.no_progress_budget:
            return self._stop('structural_failure', 'repeated_no_progress')
        if self.steps >= stage.step_budget:
            return self._stop('structural_failure', 'stage_step_budget_exhausted')
        backend = self.router.resolve(grounded)
        if backend is None:
            return self._stop('structural_failure', 'capability_gap')
        self.steps += 1
        self.stalled = 0 if progress else self.stalled + 1
        action = backend.call(grounded.runtime_payload())
        local_error = backend.kind == 'fast_adapter' and any(
            facts.get(name) is True for name in ('local_error', 'local_failure'))
        self.last_result = JudgeResult('local_failure' if local_error else 'continue', stage.id,
                                       'local_correction_issued' if local_error else 'action_issued', action=action,
                                       backend_revision=backend.revision)
        return self.last_result

    def _wait(self, stage: Stage, *, progress: bool, reason: str) -> JudgeResult:
        if self.stalled >= stage.no_progress_budget:
            return self._stop('structural_failure', 'repeated_no_progress')
        if self.steps >= stage.step_budget:
            return self._stop('structural_failure', 'stage_step_budget_exhausted')
        self.steps += 1
        self.stalled += 1
        self.last_result = JudgeResult('continue', stage.id, reason)
        return self.last_result
