"""User-facing indexed calls over the frozen Foundation Primitive Catalog."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from .primitive import Primitive, PrimitiveContractError, PrimitiveMeta, PrimitiveRegistry


@dataclass(frozen=True)
class PrimitiveCallAudit:
    """Content-addressed provenance emitted for every public tool invocation."""

    primitive_ref: str
    registry_id: str
    registry_version: str
    implementation_hash: str
    runtime_adapter_hash: str
    runtime_config_hash: str
    sealed_artifact_sha256: tuple[str, ...]
    input_sha256: str
    output_sha256: str
    invocation_sha256: str
    duration_seconds: float
    runtime_reads_evaluator_gt: bool = False
    format: str = "prove.primitive-call-audit.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrimitiveCallResult(Mapping[str, Any]):
    """Primitive outputs plus the immutable call record that produced them."""

    outputs: Mapping[str, Any]
    audit: PrimitiveCallAudit

    def __getitem__(self, key: str) -> Any:
        return self.outputs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.outputs)

    def __len__(self) -> int:
        return len(self.outputs)


class IndexedPrimitiveTool:
    """One callable primitive addressed by its stable ``prove.*`` reference."""

    def __init__(
        self,
        *,
        primitive: Primitive,
        registry_id: str,
        registry_version: str,
    ) -> None:
        self._primitive = primitive
        self._registry_id = registry_id
        self._registry_version = registry_version

    @property
    def meta(self) -> PrimitiveMeta:
        return self._primitive.meta

    def __call__(self, **inputs: Any) -> PrimitiveCallResult:
        return self.invoke(inputs)

    def invoke(self, inputs: Mapping[str, Any]) -> PrimitiveCallResult:
        frozen_inputs = MappingProxyType(dict(inputs))
        input_sha256 = fingerprint(frozen_inputs)
        started = time.perf_counter()
        outputs = self._primitive.run(frozen_inputs)
        duration = time.perf_counter() - started
        output_sha256 = fingerprint(outputs)
        invocation_payload = {
            "format": "prove.primitive-invocation.v1",
            "primitive_ref": self.meta.ref,
            "registry_id": self._registry_id,
            "registry_version": self._registry_version,
            "implementation_hash": self.meta.implementation_hash,
            "runtime_adapter_hash": self._primitive.runtime_adapter_hash,
            "runtime_config_hash": self._primitive.runtime_config_hash,
            "sealed_artifact_sha256": [
                item.sha256 for item in self.meta.sealed_artifacts
            ],
            "input_sha256": input_sha256,
            "output_sha256": output_sha256,
            "runtime_reads_evaluator_gt": False,
        }
        invocation_sha256 = hashlib.sha256(
            _canonical_json(invocation_payload)
        ).hexdigest()
        audit = PrimitiveCallAudit(
            primitive_ref=self.meta.ref,
            registry_id=self._registry_id,
            registry_version=self._registry_version,
            implementation_hash=self.meta.implementation_hash,
            runtime_adapter_hash=self._primitive.runtime_adapter_hash,
            runtime_config_hash=self._primitive.runtime_config_hash,
            sealed_artifact_sha256=tuple(
                item.sha256 for item in self.meta.sealed_artifacts
            ),
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            invocation_sha256=invocation_sha256,
            duration_seconds=duration,
        )
        return PrimitiveCallResult(MappingProxyType(dict(outputs)), audit)


class ProveToolIndex:
    """Immutable, ergonomic facade over a :class:`PrimitiveRegistry`."""

    def __init__(self, registry: PrimitiveRegistry) -> None:
        if not registry.frozen:
            raise PrimitiveContractError("PROVE tool index requires a frozen registry")
        self._registry = registry

    @property
    def registry_id(self) -> str:
        return self._registry.registry_id

    @property
    def version(self) -> str:
        return self._registry.version

    def canonical_ref(self, name_or_ref: str) -> str:
        if "@" in name_or_ref:
            self._registry.describe(name_or_ref)
            return name_or_ref
        matches = tuple(
            ref for ref in self._registry.refs() if ref.rsplit("@", 1)[0] == name_or_ref
        )
        if not matches:
            raise PrimitiveContractError(f"unknown PROVE primitive: {name_or_ref}")
        if len(matches) != 1:
            raise PrimitiveContractError(
                f"ambiguous PROVE primitive version for {name_or_ref}: {list(matches)}"
            )
        return matches[0]

    def names(self, *, category: str | None = None) -> tuple[str, ...]:
        return self._registry.refs(category=category)

    def describe(self, name_or_ref: str) -> PrimitiveMeta:
        return self._registry.describe(self.canonical_ref(name_or_ref))

    def get(
        self,
        name_or_ref: str,
        *,
        allow_unqualified: bool = False,
    ) -> IndexedPrimitiveTool:
        ref = self.canonical_ref(name_or_ref)
        primitive = self._registry.resolve(
            ref,
            search_eligible=not allow_unqualified,
        )
        return IndexedPrimitiveTool(
            primitive=primitive,
            registry_id=self.registry_id,
            registry_version=self.version,
        )

    def __getitem__(self, name_or_ref: str) -> IndexedPrimitiveTool:
        return self.get(name_or_ref)

    def call(
        self,
        name_or_ref: str,
        inputs: Mapping[str, Any] | None = None,
        *,
        allow_unqualified: bool = False,
        **keyword_inputs: Any,
    ) -> PrimitiveCallResult:
        if inputs is not None and keyword_inputs:
            raise PrimitiveContractError(
                "pass primitive inputs either as a mapping or keyword arguments"
            )
        values = dict(inputs or keyword_inputs)
        return self.get(
            name_or_ref,
            allow_unqualified=allow_unqualified,
        ).invoke(values)

    def inventory(self) -> tuple[dict[str, Any], ...]:
        rows = []
        for ref in self.names():
            meta = self._registry.describe(ref)
            try:
                self._registry.resolve(ref, search_eligible=False)
                runnable = True
            except PrimitiveContractError:
                runnable = False
            rows.append(
                {
                    "ref": ref,
                    "category": meta.category,
                    "lifecycle": meta.lifecycle.value,
                    "runnable": runnable,
                    "search_eligible": meta.lifecycle.value
                    in {"qualified_baseline", "frozen_default"},
                    "inputs": [port.to_dict() for port in meta.input_ports],
                    "outputs": [port.to_dict() for port in meta.output_ports],
                }
            )
        return tuple(rows)


def fingerprint(value: Any) -> str:
    """Hash values without storing raw sensor data in the audit record."""

    return hashlib.sha256(_canonical_json(_describe_value(value))).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _describe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PrimitiveContractError("cannot fingerprint non-finite float")
        return {"kind": "float", "value": value.hex()}
    if isinstance(value, Enum):
        return _describe_value(value.value)
    if isinstance(value, Path):
        return _describe_path(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "kind": "bytes",
            "byte_size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if is_dataclass(value):
        return _describe_value(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PrimitiveContractError("fingerprinted mapping keys must be strings")
        return {
            "kind": "mapping",
            "items": [
                [key, _describe_value(value[key])] for key in sorted(value)
            ],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "sequence",
            "items": [_describe_value(item) for item in value],
        }
    if isinstance(value, (set, frozenset)):
        items = [_describe_value(item) for item in value]
        return {
            "kind": "set",
            "items": sorted(items, key=lambda item: _canonical_json(item)),
        }
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if hasattr(value, "contiguous"):
            value = value.contiguous()
        if hasattr(value, "numpy"):
            return _describe_array(value.numpy())
    if hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"):
        return _describe_array(value)
    if value.__class__.__module__.startswith("PIL."):
        try:
            import numpy as np

            return _describe_array(np.asarray(value))
        except Exception as exc:  # pragma: no cover - optional dependency failure
            raise PrimitiveContractError("cannot fingerprint PIL image") from exc
    if hasattr(value, "to_dict"):
        return _describe_value(value.to_dict())
    raise PrimitiveContractError(
        f"unsupported value for content-addressed tool call: {type(value).__name__}"
    )


def _describe_array(value: Any) -> dict[str, Any]:
    try:
        payload = value.tobytes(order="C")
    except TypeError:
        payload = value.tobytes()
    return {
        "kind": "array",
        "dtype": str(value.dtype),
        "shape": [int(item) for item in value.shape],
        "byte_size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _describe_path(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        digest = hashlib.sha256()
        byte_size = 0
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                byte_size += len(chunk)
        return {
            "kind": "file",
            "path": str(resolved),
            "byte_size": byte_size,
            "sha256": digest.hexdigest(),
        }
    if resolved.is_dir():
        files = []
        for child in sorted(item for item in resolved.rglob("*") if item.is_file()):
            item = _describe_path(child)
            item["relative_path"] = str(child.relative_to(resolved))
            item.pop("path", None)
            files.append(item)
        return {"kind": "directory", "path": str(resolved), "files": files}
    raise PrimitiveContractError(f"path input does not exist: {resolved}")
