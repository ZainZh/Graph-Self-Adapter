"""Atomic primitive metadata, wrappers, and stable PROVE indexing."""

from __future__ import annotations

import hashlib
import inspect
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol


class PrimitiveContractError(ValueError):
    """Raised when a primitive or registry violates the frozen contract."""


class PrimitiveLifecycle(str, Enum):
    PROPOSED = "proposed"
    AVAILABLE_UNQUALIFIED = "available_unqualified"
    QUALIFIED_BASELINE = "qualified_baseline"
    FROZEN_DEFAULT = "frozen_default"
    BLOCKED = "blocked"
    RETIRED = "retired"


class EpistemicRole(str, Enum):
    MEASUREMENT = "measurement"
    HYPOTHESIS = "hypothesis"
    PROPAGATE = "propagate"


_ID_RE = re.compile(r"^prove(?:\.[a-z][a-z0-9_]*){2,}$")
_VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_OUTPUTS = {
    "expected_effect",
    "observed_effect",
    "interaction_residual",
    "grasp_success",
    "interaction_success",
}
_SEARCH_ELIGIBLE = {
    PrimitiveLifecycle.QUALIFIED_BASELINE,
    PrimitiveLifecycle.FROZEN_DEFAULT,
}


@dataclass(frozen=True)
class PortSpec:
    name: str
    type_name: str
    epistemic_role: EpistemicRole | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.type_name:
            raise PrimitiveContractError("port name and type are required")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PortSpec":
        role = value.get("epistemic_role")
        return cls(
            name=str(value["name"]),
            type_name=str(value["type"]),
            epistemic_role=None if role is None else EpistemicRole(str(role)),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {"name": self.name, "type": self.type_name}
        if self.epistemic_role is not None:
            result["epistemic_role"] = self.epistemic_role.value
        return result


@dataclass(frozen=True)
class ArtifactSpec:
    role: str
    uri: str
    sha256: str
    byte_size: int

    def __post_init__(self) -> None:
        if not self.role or not self.uri:
            raise PrimitiveContractError("artifact role and URI are required")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise PrimitiveContractError("artifact sha256 must be a lowercase digest")
        if self.byte_size <= 0:
            raise PrimitiveContractError("artifact byte_size must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactSpec":
        return cls(
            role=str(value["role"]),
            uri=str(value["uri"]),
            sha256=str(value["sha256"]),
            byte_size=int(value["byte_size"]),
        )


@dataclass(frozen=True)
class PrimitiveMeta:
    primitive_id: str
    version: str
    category: str
    lifecycle: PrimitiveLifecycle
    input_ports: tuple[PortSpec, ...]
    output_ports: tuple[PortSpec, ...]
    implementation_hash: str
    implementation_entrypoint: str
    source_repository: str
    source_revision: str
    source_path: str
    deterministic: bool
    sealed_artifacts: tuple[ArtifactSpec, ...] = ()
    runtime_reads_evaluator_gt: bool = False

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.primitive_id):
            raise PrimitiveContractError(
                "primitive_id must use the prove.category.name namespace"
            )
        if not _VERSION_RE.fullmatch(self.version):
            raise PrimitiveContractError("primitive version must be semantic x.y.z")
        if not _SHA_RE.fullmatch(self.implementation_hash):
            raise PrimitiveContractError(
                "implementation_hash must be sha256:<lowercase digest>"
            )
        if self.runtime_reads_evaluator_gt:
            raise PrimitiveContractError("primitive runtime may not read evaluator GT")
        if not self.input_ports or not self.output_ports:
            raise PrimitiveContractError("primitive ports may not be empty")
        input_names = [port.name for port in self.input_ports]
        output_names = [port.name for port in self.output_ports]
        if len(set(input_names)) != len(input_names):
            raise PrimitiveContractError("primitive input names must be unique")
        if len(set(output_names)) != len(output_names):
            raise PrimitiveContractError("primitive output names must be unique")
        if _FORBIDDEN_OUTPUTS.intersection(output_names):
            raise PrimitiveContractError(
                "atomic primitives may not emit interaction interpretation fields"
            )
        if any(port.epistemic_role is not None for port in self.input_ports):
            raise PrimitiveContractError(
                "epistemic_role is declared only on primitive outputs"
            )
        if any(port.epistemic_role is None for port in self.output_ports):
            raise PrimitiveContractError(
                "every primitive output requires an epistemic_role"
            )
        required_source = (
            self.implementation_entrypoint,
            self.source_repository,
            self.source_revision,
            self.source_path,
        )
        if any(not item for item in required_source):
            raise PrimitiveContractError("implementation provenance is incomplete")

    @property
    def ref(self) -> str:
        return f"{self.primitive_id}@{self.version}"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrimitiveMeta":
        source = value["source"]
        return cls(
            primitive_id=str(value["primitive_id"]),
            version=str(value["version"]),
            category=str(value["category"]),
            lifecycle=PrimitiveLifecycle(str(value["lifecycle"])),
            input_ports=tuple(PortSpec.from_dict(port) for port in value["inputs"]),
            output_ports=tuple(PortSpec.from_dict(port) for port in value["outputs"]),
            implementation_hash=str(value["implementation_hash"]),
            implementation_entrypoint=str(value["entrypoint"]),
            source_repository=str(source["repository"]),
            source_revision=str(source["revision"]),
            source_path=str(source["source_path"]),
            deterministic=bool(value["deterministic"]),
            sealed_artifacts=tuple(
                ArtifactSpec.from_dict(item)
                for item in value.get("sealed_artifacts", ())
            ),
            runtime_reads_evaluator_gt=bool(
                value.get("runtime_reads_evaluator_gt", False)
            ),
        )


class Primitive(Protocol):
    @property
    def meta(self) -> PrimitiveMeta: ...

    @property
    def runtime_adapter_hash(self) -> str: ...

    @property
    def runtime_config_hash(self) -> str: ...

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]: ...


class CallablePrimitive:
    """Validate a plain callable against sealed PrimitiveMeta."""

    def __init__(
        self,
        meta: PrimitiveMeta,
        implementation: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        self._meta = meta
        self._implementation = implementation
        self._runtime_adapter_hash = _callable_hash(implementation)
        self._runtime_config_hash = getattr(
            implementation,
            "adapter_config_hash",
            "sha256:" + hashlib.sha256(b"{}").hexdigest(),
        )
        if not _SHA_RE.fullmatch(self._runtime_config_hash):
            raise PrimitiveContractError(
                "runtime adapter configuration hash must be sha256:<digest>"
            )

    @property
    def meta(self) -> PrimitiveMeta:
        return self._meta

    @property
    def runtime_adapter_hash(self) -> str:
        return self._runtime_adapter_hash

    @property
    def runtime_config_hash(self) -> str:
        return self._runtime_config_hash

    def run(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        expected_inputs = {port.name for port in self.meta.input_ports}
        if set(inputs) != expected_inputs:
            raise PrimitiveContractError(
                f"{self.meta.ref} inputs must be {sorted(expected_inputs)}"
            )
        outputs = dict(self._implementation(MappingProxyType(dict(inputs))))
        expected_outputs = {port.name for port in self.meta.output_ports}
        if set(outputs) != expected_outputs:
            raise PrimitiveContractError(
                f"{self.meta.ref} outputs must be {sorted(expected_outputs)}"
            )
        if _FORBIDDEN_OUTPUTS.intersection(outputs):
            raise PrimitiveContractError(
                "primitive implementation emitted an interaction interpretation"
            )
        return MappingProxyType(outputs)


class PrimitiveRegistry:
    """Manual, append-only construction followed by an immutable lookup index."""

    def __init__(self, *, registry_id: str, version: str) -> None:
        self.registry_id = registry_id
        self.version = version
        self._meta: dict[str, PrimitiveMeta] = {}
        self._implementations: dict[str, Primitive] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def register(
        self,
        meta: PrimitiveMeta,
        implementation: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        if self._frozen:
            raise PrimitiveContractError("foundation registry is frozen")
        if meta.ref in self._meta:
            raise PrimitiveContractError(f"duplicate primitive ref: {meta.ref}")
        self._meta[meta.ref] = meta
        if implementation is not None:
            self._implementations[meta.ref] = CallablePrimitive(meta, implementation)

    def freeze(self) -> None:
        if not self._meta:
            raise PrimitiveContractError("cannot freeze an empty registry")
        self._frozen = True

    def describe(self, ref: str) -> PrimitiveMeta:
        try:
            return self._meta[ref]
        except KeyError as exc:
            raise PrimitiveContractError(f"unknown PROVE primitive: {ref}") from exc

    def resolve(self, ref: str, *, search_eligible: bool = True) -> Primitive:
        meta = self.describe(ref)
        if search_eligible and meta.lifecycle not in _SEARCH_ELIGIBLE:
            raise PrimitiveContractError(
                f"{ref} is {meta.lifecycle.value}, not search eligible"
            )
        try:
            return self._implementations[ref]
        except KeyError as exc:
            raise PrimitiveContractError(
                f"{ref} has no qualified runtime adapter"
            ) from exc

    def refs(self, *, category: str | None = None) -> tuple[str, ...]:
        return tuple(
            sorted(
                ref
                for ref, meta in self._meta.items()
                if category is None or meta.category == category
            )
        )

    def metadata(self) -> Mapping[str, PrimitiveMeta]:
        return MappingProxyType(dict(self._meta))


def _callable_hash(implementation: Any) -> str:
    declared = getattr(implementation, "adapter_hash", None)
    if declared is not None:
        value = str(declared)
        if not _SHA_RE.fullmatch(value):
            raise PrimitiveContractError(
                "runtime adapter hash must be sha256:<digest>"
            )
        return value
    target = implementation if inspect.isfunction(implementation) else type(implementation)
    source_path = inspect.getsourcefile(target)
    if source_path is None:
        raise PrimitiveContractError(
            "runtime adapter must expose adapter_hash or have inspectable source"
        )
    payload = Path(source_path).read_bytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()
