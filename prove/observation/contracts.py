"""Epistemic contracts shared by every PROVE observer.

There are exactly three evidence domains: measurement, hypothesis, and
verified_evidence. The packet constructor is an enforcement boundary:
hypotheses cannot be labelled trusted, and trusted verified evidence cannot
be created from hypotheses alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class EvidenceContractError(ValueError):
    """Raised when runtime evidence violates the epistemic boundary."""


class EvidenceKind(str, Enum):
    MEASUREMENT = "measurement"
    HYPOTHESIS = "hypothesis"
    VERIFIED = "verified_evidence"


class EvidenceStatus(str, Enum):
    AVAILABLE = "available"
    TRUSTED = "trusted"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


_ALLOWED_STATUS = {
    EvidenceKind.MEASUREMENT: {
        EvidenceStatus.AVAILABLE,
        EvidenceStatus.AMBIGUOUS,
        EvidenceStatus.UNAVAILABLE,
    },
    EvidenceKind.HYPOTHESIS: {
        EvidenceStatus.AVAILABLE,
        EvidenceStatus.AMBIGUOUS,
        EvidenceStatus.UNAVAILABLE,
    },
    EvidenceKind.VERIFIED: {
        EvidenceStatus.TRUSTED,
        EvidenceStatus.AMBIGUOUS,
        EvidenceStatus.UNAVAILABLE,
    },
}


@dataclass(frozen=True)
class EvidenceProvenance:
    """Trace one value to a sealed primitive call and independent sources."""

    primitive_ref: str
    input_refs: tuple[str, ...] = ()
    source_artifact_sha256: tuple[str, ...] = ()
    independence_groups: tuple[str, ...] = ()
    uses_evaluator_gt: bool = False

    def __post_init__(self) -> None:
        if not self.primitive_ref:
            raise EvidenceContractError("primitive_ref must be non-empty")
        if self.uses_evaluator_gt:
            raise EvidenceContractError(
                "runtime evidence provenance may not use evaluator-only GT"
            )
        for digest in self.source_artifact_sha256:
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise EvidenceContractError(
                    "source artifact digests must be lowercase SHA-256"
                )
        if len(set(self.independence_groups)) != len(self.independence_groups):
            raise EvidenceContractError("independence_groups must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_ref": self.primitive_ref,
            "input_refs": list(self.input_refs),
            "source_artifact_sha256": list(self.source_artifact_sha256),
            "independence_groups": list(self.independence_groups),
            "uses_evaluator_gt": False,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceProvenance":
        return cls(
            primitive_ref=str(value["primitive_ref"]),
            input_refs=tuple(str(item) for item in value.get("input_refs", ())),
            source_artifact_sha256=tuple(
                str(item) for item in value.get("source_artifact_sha256", ())
            ),
            independence_groups=tuple(
                str(item) for item in value.get("independence_groups", ())
            ),
            uses_evaluator_gt=bool(value.get("uses_evaluator_gt", False)),
        )


@dataclass(frozen=True)
class EvidenceDatum:
    """One named datum in exactly one epistemic domain."""

    name: str
    kind: EvidenceKind
    value: Any
    status: EvidenceStatus
    confidence: float | None
    provenance: EvidenceProvenance
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise EvidenceContractError("evidence name must be non-empty")
        if self.status not in _ALLOWED_STATUS[self.kind]:
            raise EvidenceContractError(
                f"{self.kind.value} evidence cannot have status {self.status.value}"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise EvidenceContractError("confidence must be within [0, 1]")
        if self.status is EvidenceStatus.UNAVAILABLE and self.value is not None:
            raise EvidenceContractError("unavailable evidence must have value=None")
        if self.status is not EvidenceStatus.UNAVAILABLE and self.value is None:
            raise EvidenceContractError("available evidence must carry a value")
        if self.kind is EvidenceKind.VERIFIED and self.status is EvidenceStatus.TRUSTED:
            if not any(ref.startswith("measurement:") for ref in self.derived_from):
                raise EvidenceContractError(
                    "trusted verified evidence requires declared measurement support"
                )
        for ref in self.derived_from:
            if not ref.startswith(
                ("measurement:", "hypothesis:", "verified_evidence:")
            ):
                raise EvidenceContractError(f"invalid evidence reference: {ref}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "value": _jsonable(self.value),
            "status": self.status.value,
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict(),
            "derived_from": list(self.derived_from),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceDatum":
        return cls(
            name=str(value["name"]),
            kind=EvidenceKind(str(value["kind"])),
            value=value.get("value"),
            status=EvidenceStatus(str(value["status"])),
            confidence=(
                None
                if value.get("confidence") is None
                else float(value["confidence"])
            ),
            provenance=EvidenceProvenance.from_dict(value["provenance"]),
            derived_from=tuple(str(item) for item in value.get("derived_from", ())),
        )


@dataclass(frozen=True)
class ObservationEvidencePacket:
    """One frame of observer output, without expected effects or residuals."""

    observer_id: str
    observer_version: str
    frame_id: int | str
    measurements: Mapping[str, EvidenceDatum]
    hypotheses: Mapping[str, EvidenceDatum]
    verified: Mapping[str, EvidenceDatum]
    status: EvidenceStatus

    def __post_init__(self) -> None:
        if not self.observer_id or not self.observer_version:
            raise EvidenceContractError("observer id and version are required")
        if self.status not in {
            EvidenceStatus.TRUSTED,
            EvidenceStatus.AMBIGUOUS,
            EvidenceStatus.UNAVAILABLE,
        }:
            raise EvidenceContractError(
                "packet status must be trusted/ambiguous/unavailable"
            )

        domains = (
            ("measurement", EvidenceKind.MEASUREMENT, dict(self.measurements)),
            ("hypothesis", EvidenceKind.HYPOTHESIS, dict(self.hypotheses)),
            ("verified_evidence", EvidenceKind.VERIFIED, dict(self.verified)),
        )
        seen: set[str] = set()
        for prefix, expected_kind, mapping in domains:
            for key, datum in mapping.items():
                if key != datum.name:
                    raise EvidenceContractError(
                        f"evidence key/name mismatch: {key!r} != {datum.name!r}"
                    )
                if datum.kind is not expected_kind:
                    raise EvidenceContractError(
                        f"{key} belongs to {prefix}, not {datum.kind.value}"
                    )
                qualified = f"{prefix}:{key}"
                if qualified in seen:
                    raise EvidenceContractError(f"duplicate evidence: {qualified}")
                seen.add(qualified)

        measurements = domains[0][2]
        hypotheses = domains[1][2]
        verified = domains[2][2]
        for datum in verified.values():
            for ref in datum.derived_from:
                prefix, name = ref.split(":", 1)
                source = {
                    "measurement": measurements,
                    "hypothesis": hypotheses,
                    "verified_evidence": verified,
                }[prefix].get(name)
                if source is None:
                    raise EvidenceContractError(
                        f"verified evidence references missing datum {ref}"
                    )
                if (
                    datum.status is EvidenceStatus.TRUSTED
                    and prefix == "measurement"
                    and source.status is not EvidenceStatus.AVAILABLE
                ):
                    raise EvidenceContractError(
                        f"trusted evidence depends on unavailable measurement {ref}"
                    )

        object.__setattr__(self, "measurements", MappingProxyType(measurements))
        object.__setattr__(self, "hypotheses", MappingProxyType(hypotheses))
        object.__setattr__(self, "verified", MappingProxyType(verified))

    def to_dict(self) -> dict[str, Any]:
        return {
            "observer_id": self.observer_id,
            "observer_version": self.observer_version,
            "frame_id": self.frame_id,
            "status": self.status.value,
            "measurement": {
                key: value.to_dict() for key, value in self.measurements.items()
            },
            "hypothesis": {
                key: value.to_dict() for key, value in self.hypotheses.items()
            },
            "verified_evidence": {
                key: value.to_dict() for key, value in self.verified.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObservationEvidencePacket":
        def load_domain(name: str) -> dict[str, EvidenceDatum]:
            raw = value.get(name, {})
            if not isinstance(raw, Mapping):
                raise EvidenceContractError(f"{name} must be an object")
            return {
                str(key): EvidenceDatum.from_dict(item)
                for key, item in raw.items()
            }

        return cls(
            observer_id=str(value["observer_id"]),
            observer_version=str(value["observer_version"]),
            frame_id=value["frame_id"],
            measurements=load_domain("measurement"),
            hypotheses=load_domain("hypothesis"),
            verified=load_domain("verified_evidence"),
            status=EvidenceStatus(str(value["status"])),
        )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise EvidenceContractError(
        f"evidence value is not JSON serializable: {type(value).__name__}"
    )
