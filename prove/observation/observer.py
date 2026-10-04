"""Restricted ObserverSpec graph compiler and GT-free runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import (
    EvidenceDatum,
    EvidenceKind,
    EvidenceProvenance,
    EvidenceStatus,
    ObservationEvidencePacket,
)
from .primitive import (
    EpistemicRole,
    Primitive,
    PrimitiveContractError,
    PrimitiveRegistry,
)


class ObserverContractError(ValueError):
    """Raised when a graph is untyped, unsafe, or outside runtime authority."""


_FORBIDDEN_INPUT_TOKENS = (
    "ground_truth",
    "simulator_gt",
    "oracle",
    "evaluator",
    "object_pose_gt",
    "residual_gt",
)


@dataclass(frozen=True)
class ObserverInputSpec:
    name: str
    type_name: str
    kind: EvidenceKind
    independence_group: str

    def __post_init__(self) -> None:
        lowered = self.name.lower()
        if any(token in lowered for token in _FORBIDDEN_INPUT_TOKENS):
            raise ObserverContractError(
                f"observer runtime input is evaluator-only: {self.name}"
            )
        if self.kind is EvidenceKind.VERIFIED:
            raise ObserverContractError(
                "runtime inputs are measurements or hypotheses, never pre-verified GT"
            )
        if not self.name or not self.type_name or not self.independence_group:
            raise ObserverContractError("observer input contract is incomplete")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObserverInputSpec":
        return cls(
            name=str(value["name"]),
            type_name=str(value["type"]),
            kind=EvidenceKind(str(value["kind"])),
            independence_group=str(value["independence_group"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type_name,
            "kind": self.kind.value,
            "independence_group": self.independence_group,
        }


@dataclass(frozen=True)
class ObserverNode:
    node_id: str
    primitive_ref: str
    bindings: Mapping[str, str]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObserverNode":
        return cls(
            node_id=str(value["node_id"]),
            primitive_ref=str(value["primitive"]),
            bindings=MappingProxyType(
                {str(key): str(item) for key, item in value["bindings"].items()}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "primitive": self.primitive_ref,
            "bindings": dict(self.bindings),
        }


@dataclass(frozen=True)
class EvidenceOutputSpec:
    name: str
    source: str
    kind: EvidenceKind
    confidence: float | str | None = None
    supporting_measurements: tuple[str, ...] = ()
    value_rule: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceOutputSpec":
        rule = value.get("value_rule")
        return cls(
            name=str(value["name"]),
            source=str(value["source"]),
            kind=EvidenceKind(str(value["kind"])),
            confidence=value.get("confidence"),
            supporting_measurements=tuple(
                str(item) for item in value.get("supporting_measurements", ())
            ),
            value_rule=None if rule is None else MappingProxyType(dict(rule)),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "kind": self.kind.value,
            "supporting_measurements": list(self.supporting_measurements),
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.value_rule is not None:
            result["value_rule"] = dict(self.value_rule)
        return result


@dataclass(frozen=True)
class Predicate:
    source: str
    op: str
    value: Any = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Predicate":
        return cls(
            source=str(value["source"]),
            op=str(value["op"]),
            value=value.get("value"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {"source": self.source, "op": self.op}
        if self.op not in {"truthy", "falsy"}:
            result["value"] = self.value
        return result


@dataclass(frozen=True)
class GateSpec:
    target: str
    predicate: Predicate
    on_fail: EvidenceStatus

    def __post_init__(self) -> None:
        if self.on_fail not in {
            EvidenceStatus.AMBIGUOUS,
            EvidenceStatus.UNAVAILABLE,
        }:
            raise ObserverContractError("failed gates may only abstain")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateSpec":
        return cls(
            target=str(value["target"]),
            predicate=Predicate.from_dict(value["predicate"]),
            on_fail=EvidenceStatus(str(value["on_fail"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "predicate": self.predicate.to_dict(),
            "on_fail": self.on_fail.value,
        }


@dataclass(frozen=True)
class AbstentionRule:
    target: str
    when: Predicate
    status: EvidenceStatus

    def __post_init__(self) -> None:
        if self.status not in {
            EvidenceStatus.AMBIGUOUS,
            EvidenceStatus.UNAVAILABLE,
        }:
            raise ObserverContractError("abstention cannot create trusted evidence")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AbstentionRule":
        return cls(
            target=str(value["target"]),
            when=Predicate.from_dict(value["when"]),
            status=EvidenceStatus(str(value["status"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "when": self.when.to_dict(),
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ObserverSpec:
    observer_id: str
    version: str
    parent_id: str | None
    primary_modification: str
    inputs: tuple[ObserverInputSpec, ...]
    state: tuple[ObserverInputSpec, ...]
    nodes: tuple[ObserverNode, ...]
    evidence_outputs: tuple[EvidenceOutputSpec, ...]
    gates: tuple[GateSpec, ...] = ()
    abstention_rules: tuple[AbstentionRule, ...] = ()
    state_updates: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObserverSpec":
        return cls(
            observer_id=str(value["observer_id"]),
            version=str(value["version"]),
            parent_id=(
                None if value.get("parent_id") is None else str(value["parent_id"])
            ),
            primary_modification=str(value["primary_modification"]),
            inputs=tuple(
                ObserverInputSpec.from_dict(item) for item in value["inputs"]
            ),
            state=tuple(
                ObserverInputSpec.from_dict(item) for item in value.get("state", ())
            ),
            nodes=tuple(ObserverNode.from_dict(item) for item in value["nodes"]),
            evidence_outputs=tuple(
                EvidenceOutputSpec.from_dict(item)
                for item in value["evidence_outputs"]
            ),
            gates=tuple(GateSpec.from_dict(item) for item in value.get("gates", ())),
            abstention_rules=tuple(
                AbstentionRule.from_dict(item)
                for item in value.get("abstention_rules", ())
            ),
            state_updates=MappingProxyType(
                {
                    str(key): str(item)
                    for key, item in value.get("state_updates", {}).items()
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "prove.observer-spec.v1",
            "observer_id": self.observer_id,
            "version": self.version,
            "parent_id": self.parent_id,
            "primary_modification": self.primary_modification,
            "inputs": [item.to_dict() for item in self.inputs],
            "state": [item.to_dict() for item in self.state],
            "nodes": [item.to_dict() for item in self.nodes],
            "evidence_outputs": [
                item.to_dict() for item in self.evidence_outputs
            ],
            "gates": [item.to_dict() for item in self.gates],
            "abstention_rules": [
                item.to_dict() for item in self.abstention_rules
            ],
            "state_updates": dict(self.state_updates),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ObserverState:
    values: Mapping[str, Any]
    runtime_errors: tuple[Mapping[str, str], ...] = ()
    source_artifact_sha256: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _SignalSpec:
    type_name: str
    kind: EvidenceKind
    independence_groups: frozenset[str]


@dataclass(frozen=True)
class _RuntimeSignal:
    value: Any
    spec: _SignalSpec
    primitive_ref: str
    input_refs: tuple[str, ...]
    artifact_hashes: tuple[str, ...]


@dataclass(frozen=True)
class _CompiledNode:
    spec: ObserverNode
    primitive: Primitive
    outputs: Mapping[str, _SignalSpec]


class ObserverCompiler:
    """Compile one declarative graph after type and evidence-lineage checks."""

    def __init__(self, registry: PrimitiveRegistry) -> None:
        if not registry.frozen:
            raise ObserverContractError("observer compilation requires frozen registry")
        self.registry = registry

    def compile(self, spec: ObserverSpec) -> "CompiledObserver":
        self._validate_spec_identity(spec)
        signals: dict[str, _SignalSpec] = {}
        for input_spec in spec.inputs:
            signals[f"$input.{input_spec.name}"] = _SignalSpec(
                type_name=input_spec.type_name,
                kind=input_spec.kind,
                independence_groups=frozenset({input_spec.independence_group}),
            )
        for state_spec in spec.state:
            signals[f"$state.{state_spec.name}"] = _SignalSpec(
                type_name=state_spec.type_name,
                kind=state_spec.kind,
                independence_groups=frozenset({state_spec.independence_group}),
            )

        remaining = list(spec.nodes)
        compiled: list[_CompiledNode] = []
        while remaining:
            progressed = False
            for node in list(remaining):
                if not all(binding in signals for binding in node.bindings.values()):
                    continue
                primitive = self.registry.resolve(node.primitive_ref)
                expected = {port.name: port for port in primitive.meta.input_ports}
                if set(node.bindings) != set(expected):
                    raise ObserverContractError(
                        f"node {node.node_id} bindings must be {sorted(expected)}"
                    )
                bound_specs = {
                    name: signals[reference]
                    for name, reference in node.bindings.items()
                }
                for name, port in expected.items():
                    if bound_specs[name].type_name != port.type_name:
                        raise ObserverContractError(
                            f"node {node.node_id}.{name} expects {port.type_name}, "
                            f"got {bound_specs[name].type_name}"
                        )
                output_specs: dict[str, _SignalSpec] = {}
                input_kinds = {item.kind for item in bound_specs.values()}
                groups = frozenset(
                    group
                    for item in bound_specs.values()
                    for group in item.independence_groups
                )
                for port in primitive.meta.output_ports:
                    if port.epistemic_role is EpistemicRole.MEASUREMENT:
                        if EvidenceKind.HYPOTHESIS in input_kinds:
                            raise ObserverContractError(
                                f"{node.node_id}.{port.name} claims measurement "
                                "while consuming a hypothesis"
                            )
                        kind = EvidenceKind.MEASUREMENT
                    elif port.epistemic_role is EpistemicRole.HYPOTHESIS:
                        kind = EvidenceKind.HYPOTHESIS
                    elif port.epistemic_role is EpistemicRole.PROPAGATE:
                        kind = (
                            EvidenceKind.HYPOTHESIS
                            if EvidenceKind.HYPOTHESIS in input_kinds
                            else EvidenceKind.MEASUREMENT
                        )
                    else:  # pragma: no cover - PrimitiveMeta already checks this
                        raise ObserverContractError("unknown epistemic role")
                    output = _SignalSpec(
                        type_name=port.type_name,
                        kind=kind,
                        independence_groups=groups,
                    )
                    reference = f"$node.{node.node_id}.{port.name}"
                    if reference in signals:
                        raise ObserverContractError(
                            f"duplicate node output reference: {reference}"
                        )
                    signals[reference] = output
                    output_specs[port.name] = output
                compiled.append(
                    _CompiledNode(
                        spec=node,
                        primitive=primitive,
                        outputs=MappingProxyType(output_specs),
                    )
                )
                remaining.remove(node)
                progressed = True
            if not progressed:
                unresolved = [node.node_id for node in remaining]
                raise ObserverContractError(
                    f"observer graph has missing references or a cycle: {unresolved}"
                )

        evidence_by_name = self._validate_outputs(spec, signals)
        for gate in spec.gates:
            self._validate_rule_target(gate.target, gate.predicate.source, evidence_by_name, signals)
        for rule in spec.abstention_rules:
            self._validate_rule_target(rule.target, rule.when.source, evidence_by_name, signals)
        state_names = {item.name for item in spec.state}
        if set(spec.state_updates) - state_names:
            raise ObserverContractError("state_updates target undeclared state")
        for state_name, source in spec.state_updates.items():
            if source not in signals:
                raise ObserverContractError(
                    f"state update {state_name} references missing signal {source}"
                )
            expected_type = next(
                item.type_name for item in spec.state if item.name == state_name
            )
            if signals[source].type_name != expected_type:
                raise ObserverContractError(
                    f"state update {state_name} has wrong type"
                )
        return CompiledObserver(
            spec=spec,
            nodes=tuple(compiled),
            signal_specs=MappingProxyType(signals),
        )

    @staticmethod
    def _validate_spec_identity(spec: ObserverSpec) -> None:
        if not spec.observer_id.startswith("prove.observer."):
            raise ObserverContractError("observer_id must use prove.observer.*")
        if not spec.primary_modification:
            raise ObserverContractError("one primary_modification is required")
        input_names = [item.name for item in spec.inputs]
        state_names = [item.name for item in spec.state]
        node_names = [item.node_id for item in spec.nodes]
        output_names = [item.name for item in spec.evidence_outputs]
        for label, names in (
            ("inputs", input_names),
            ("state", state_names),
            ("nodes", node_names),
            ("evidence outputs", output_names),
        ):
            if len(names) != len(set(names)):
                raise ObserverContractError(f"duplicate {label}")
        if set(input_names) & set(state_names):
            raise ObserverContractError("input and state names must be disjoint")

    @staticmethod
    def _validate_outputs(
        spec: ObserverSpec,
        signals: Mapping[str, _SignalSpec],
    ) -> Mapping[str, EvidenceOutputSpec]:
        outputs = {item.name: item for item in spec.evidence_outputs}
        for item in spec.evidence_outputs:
            if item.source not in signals:
                raise ObserverContractError(
                    f"evidence {item.name} references missing signal {item.source}"
                )
            source = signals[item.source]
            if item.kind is EvidenceKind.MEASUREMENT:
                if source.kind is not EvidenceKind.MEASUREMENT:
                    raise ObserverContractError(
                        f"measurement {item.name} is tainted by a hypothesis"
                    )
                if item.supporting_measurements:
                    raise ObserverContractError(
                        "measurement output cannot declare verification supports"
                    )
            elif item.kind is EvidenceKind.HYPOTHESIS:
                if item.supporting_measurements:
                    raise ObserverContractError(
                        "hypothesis output cannot declare verification supports"
                    )
            else:
                if not item.supporting_measurements:
                    raise ObserverContractError(
                        f"verified evidence {item.name} needs measurement support"
                    )
                support_specs: list[_SignalSpec] = []
                for support_name in item.supporting_measurements:
                    support = outputs.get(support_name)
                    if support is None or support.kind is not EvidenceKind.MEASUREMENT:
                        raise ObserverContractError(
                            f"{item.name} support {support_name} is not a measurement"
                        )
                    support_specs.append(signals[support.source])
                if source.kind is EvidenceKind.HYPOTHESIS:
                    independent = any(
                        support.independence_groups.isdisjoint(
                            source.independence_groups
                        )
                        for support in support_specs
                    )
                    if not independent:
                        raise ObserverContractError(
                            f"{item.name} hypothesis has no independent measurement support"
                        )
            if isinstance(item.confidence, str):
                if item.confidence not in signals:
                    raise ObserverContractError(
                        f"confidence source is missing: {item.confidence}"
                    )
            elif item.confidence is not None:
                confidence = float(item.confidence)
                if not 0.0 <= confidence <= 1.0:
                    raise ObserverContractError("confidence must be within [0,1]")
            _validate_value_rule(item.value_rule)
        return MappingProxyType(outputs)

    @staticmethod
    def _validate_rule_target(
        target: str,
        source: str,
        outputs: Mapping[str, EvidenceOutputSpec],
        signals: Mapping[str, _SignalSpec],
    ) -> None:
        if target not in outputs:
            raise ObserverContractError(f"gate targets missing evidence {target}")
        if source not in signals:
            raise ObserverContractError(f"gate references missing signal {source}")


class CompiledObserver:
    """Runnable observer graph that can only consume its declared inputs."""

    def __init__(
        self,
        *,
        spec: ObserverSpec,
        nodes: tuple[_CompiledNode, ...],
        signal_specs: Mapping[str, _SignalSpec],
    ) -> None:
        self.spec = spec
        self.nodes = nodes
        self.signal_specs = signal_specs

    def reset(self) -> ObserverState:
        return ObserverState(
            values=MappingProxyType({item.name: None for item in self.spec.state}),
            source_artifact_sha256=MappingProxyType(
                {item.name: () for item in self.spec.state}
            ),
        )

    def observe(
        self,
        *,
        frame_id: int | str,
        inputs: Mapping[str, Any],
        state: ObserverState | None = None,
        input_artifact_sha256: Mapping[str, tuple[str, ...]] | None = None,
    ) -> tuple[ObservationEvidencePacket, ObserverState]:
        expected_inputs = {item.name for item in self.spec.inputs}
        if set(inputs) != expected_inputs:
            raise ObserverContractError(
                f"observer inputs must be {sorted(expected_inputs)}"
            )
        current_state = self.reset() if state is None else state
        expected_state = {item.name for item in self.spec.state}
        if set(current_state.values) != expected_state:
            raise ObserverContractError(
                f"observer state must be {sorted(expected_state)}"
            )
        state_artifacts = dict(current_state.source_artifact_sha256)
        if state_artifacts and set(state_artifacts) != expected_state:
            raise ObserverContractError(
                f"observer state provenance must be {sorted(expected_state)}"
            )
        input_artifacts = dict(input_artifact_sha256 or {})
        if set(input_artifacts) - expected_inputs:
            raise ObserverContractError(
                "input provenance references undeclared observer inputs"
            )
        for name, digests in input_artifacts.items():
            _validate_artifact_digests(name, digests)

        signals: dict[str, _RuntimeSignal] = {}
        for item in self.spec.inputs:
            reference = f"$input.{item.name}"
            signals[reference] = _RuntimeSignal(
                value=inputs[item.name],
                spec=self.signal_specs[reference],
                primitive_ref=f"prove.input.{item.name}@1.0.0",
                input_refs=(),
                artifact_hashes=tuple(input_artifacts.get(item.name, ())),
            )
        for item in self.spec.state:
            reference = f"$state.{item.name}"
            signals[reference] = _RuntimeSignal(
                value=current_state.values[item.name],
                spec=self.signal_specs[reference],
                primitive_ref=f"prove.state.{item.name}@1.0.0",
                input_refs=(),
                artifact_hashes=tuple(state_artifacts.get(item.name, ())),
            )

        runtime_errors = list(current_state.runtime_errors)
        for node in self.nodes:
            bound = {
                name: signals[reference].value
                for name, reference in node.spec.bindings.items()
            }
            try:
                result = node.primitive.run(bound)
            except Exception as exc:  # safety boundary: fail closed and record
                result = {
                    port.name: None for port in node.primitive.meta.output_ports
                }
                runtime_errors.append(
                    MappingProxyType(
                        {
                            "node_id": node.spec.node_id,
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                )
            refs = tuple(node.spec.bindings.values())
            artifacts = tuple(
                sorted(
                    {
                        item.sha256
                        for item in node.primitive.meta.sealed_artifacts
                    }
                    | {
                        digest
                        for reference in refs
                        for digest in signals[reference].artifact_hashes
                    }
                )
            )
            for name, value in result.items():
                reference = f"$node.{node.spec.node_id}.{name}"
                signals[reference] = _RuntimeSignal(
                    value=value,
                    spec=node.outputs[name],
                    primitive_ref=node.primitive.meta.ref,
                    input_refs=refs,
                    artifact_hashes=artifacts,
                )

        evidence = self._materialize_evidence(signals)
        next_values = dict(current_state.values)
        next_artifacts = {
            name: tuple(state_artifacts.get(name, ())) for name in expected_state
        }
        for state_name, source in self.spec.state_updates.items():
            next_values[state_name] = signals[source].value
            next_artifacts[state_name] = signals[source].artifact_hashes
        next_state = ObserverState(
            values=MappingProxyType(next_values),
            runtime_errors=tuple(runtime_errors),
            source_artifact_sha256=MappingProxyType(next_artifacts),
        )
        packet = ObservationEvidencePacket(
            observer_id=self.spec.observer_id,
            observer_version=self.spec.version,
            frame_id=frame_id,
            measurements=evidence[EvidenceKind.MEASUREMENT],
            hypotheses=evidence[EvidenceKind.HYPOTHESIS],
            verified=evidence[EvidenceKind.VERIFIED],
            status=_packet_status(evidence),
        )
        return packet, next_state
    def _materialize_evidence(
        self,
        signals: Mapping[str, _RuntimeSignal],
    ) -> dict[EvidenceKind, dict[str, EvidenceDatum]]:
        by_kind: dict[EvidenceKind, dict[str, EvidenceDatum]] = {
            EvidenceKind.MEASUREMENT: {},
            EvidenceKind.HYPOTHESIS: {},
            EvidenceKind.VERIFIED: {},
        }
        output_specs = {item.name: item for item in self.spec.evidence_outputs}
        pending_verified: list[EvidenceOutputSpec] = []
        for output in self.spec.evidence_outputs:
            if output.kind is EvidenceKind.VERIFIED:
                pending_verified.append(output)
                continue
            signal = signals[output.source]
            value = _apply_value_rule(signal.value, output.value_rule)
            status = (
                EvidenceStatus.UNAVAILABLE
                if value is None
                else EvidenceStatus.AVAILABLE
            )
            datum = EvidenceDatum(
                name=output.name,
                kind=output.kind,
                value=value,
                status=status,
                confidence=_confidence(output.confidence, signals, status),
                provenance=_provenance(signal),
            )
            by_kind[output.kind][output.name] = datum

        for output in pending_verified:
            signal = signals[output.source]
            value = _apply_value_rule(signal.value, output.value_rule)
            supports = [
                by_kind[EvidenceKind.MEASUREMENT][name]
                for name in output.supporting_measurements
            ]
            available = value is not None and all(
                item.status is EvidenceStatus.AVAILABLE for item in supports
            )
            status = (
                EvidenceStatus.TRUSTED
                if available
                else EvidenceStatus.UNAVAILABLE
            )
            if not available:
                value = None
            datum = EvidenceDatum(
                name=output.name,
                kind=EvidenceKind.VERIFIED,
                value=value,
                status=status,
                confidence=_confidence(output.confidence, signals, status),
                provenance=_provenance(signal),
                derived_from=tuple(
                    f"measurement:{name}"
                    for name in output.supporting_measurements
                ),
            )
            by_kind[EvidenceKind.VERIFIED][output.name] = datum

        for gate in self.spec.gates:
            if not _predicate(gate.predicate, signals):
                _abstain(
                    by_kind,
                    output_specs[gate.target].kind,
                    gate.target,
                    gate.on_fail,
                )
        for rule in self.spec.abstention_rules:
            if _predicate(rule.when, signals):
                _abstain(
                    by_kind,
                    output_specs[rule.target].kind,
                    rule.target,
                    rule.status,
                )
        return by_kind


def _validate_artifact_digests(name: str, digests: tuple[str, ...]) -> None:
    if len(digests) != len(set(digests)):
        raise ObserverContractError(
            f"input {name} provenance contains duplicate artifact hashes"
        )
    for digest in digests:
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            raise ObserverContractError(
                f"input {name} provenance requires lowercase SHA-256 digests"
            )


def _provenance(signal: _RuntimeSignal) -> EvidenceProvenance:
    return EvidenceProvenance(
        primitive_ref=signal.primitive_ref,
        input_refs=signal.input_refs,
        source_artifact_sha256=signal.artifact_hashes,
        independence_groups=tuple(sorted(signal.spec.independence_groups)),
    )


def _confidence(
    source: float | str | None,
    signals: Mapping[str, _RuntimeSignal],
    status: EvidenceStatus,
) -> float | None:
    if status is EvidenceStatus.UNAVAILABLE or source is None:
        return None
    value = signals[source].value if isinstance(source, str) else source
    if value is None:
        return None
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ObserverContractError("runtime confidence must be within [0,1]")
    return confidence


def _validate_value_rule(rule: Mapping[str, Any] | None) -> None:
    if rule is None:
        return
    if set(rule) != {"op", "value"}:
        raise ObserverContractError("value_rule must contain op and value")
    if rule["op"] not in {"gte", "gt", "lte", "lt", "eq", "neq"}:
        raise ObserverContractError("unsupported value_rule operator")


def _apply_value_rule(value: Any, rule: Mapping[str, Any] | None) -> Any:
    if value is None or rule is None:
        return value
    return _compare(value, str(rule["op"]), rule["value"])


def _predicate(
    predicate: Predicate,
    signals: Mapping[str, _RuntimeSignal],
) -> bool:
    value = signals[predicate.source].value
    if predicate.op == "truthy":
        return bool(value)
    if predicate.op == "falsy":
        return not bool(value)
    if value is None:
        return False
    return _compare(value, predicate.op, predicate.value)


def _compare(left: Any, op: str, right: Any) -> bool:
    if op == "gte":
        return bool(left >= right)
    if op == "gt":
        return bool(left > right)
    if op == "lte":
        return bool(left <= right)
    if op == "lt":
        return bool(left < right)
    if op == "eq":
        return bool(left == right)
    if op == "neq":
        return bool(left != right)
    raise ObserverContractError(f"unsupported predicate operator: {op}")


def _abstain(
    evidence: dict[EvidenceKind, dict[str, EvidenceDatum]],
    kind: EvidenceKind,
    name: str,
    status: EvidenceStatus,
) -> None:
    current = evidence[kind][name]
    replacement_status = status
    if current.value is None:
        replacement_status = EvidenceStatus.UNAVAILABLE
    elif kind is not EvidenceKind.VERIFIED and status is EvidenceStatus.UNAVAILABLE:
        replacement_status = EvidenceStatus.UNAVAILABLE
    evidence[kind][name] = EvidenceDatum(
        name=current.name,
        kind=current.kind,
        value=None if replacement_status is EvidenceStatus.UNAVAILABLE else current.value,
        status=replacement_status,
        confidence=None if replacement_status is EvidenceStatus.UNAVAILABLE else current.confidence,
        provenance=current.provenance,
        derived_from=current.derived_from,
    )


def _packet_status(
    evidence: Mapping[EvidenceKind, Mapping[str, EvidenceDatum]],
) -> EvidenceStatus:
    if any(
        item.status is EvidenceStatus.TRUSTED
        for item in evidence[EvidenceKind.VERIFIED].values()
    ):
        return EvidenceStatus.TRUSTED
    if any(
        item.status in {EvidenceStatus.AVAILABLE, EvidenceStatus.AMBIGUOUS}
        for domain in evidence.values()
        for item in domain.values()
    ):
        return EvidenceStatus.AMBIGUOUS
    return EvidenceStatus.UNAVAILABLE
