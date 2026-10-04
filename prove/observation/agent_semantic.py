"""Typed Agent semantic-selection port for current-frame re-grounding.

The Agent is neither an evaluator nor a source of physical ground truth.  It
receives one RGB frame plus detector proposals and returns either one proposal
identifier or an explicit abstention.  The result is a hypothesis until
independently supported by segmentation and RGB-D geometry measurements.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from .catalog import build_foundation_registry
from .observer import ObserverCompiler, ObserverSpec


class AgentSemanticPortError(ValueError):
    """Raised when a selector decision is unbound, incomplete, or GT-tainted."""


_FORBIDDEN_TOKENS = (
    "simulator_gt",
    "target_instance_mask",
    "object_pose_gt",
    "residual_gt",
    "evaluator_gt",
    "oracle",
)


@dataclass(frozen=True)
class SemanticSelection:
    """One runtime Agent decision bound to detector and RGB bytes."""

    case_id: str
    frame_id: int
    status: str
    candidate_id: str | None
    box_xyxy: tuple[float, float, float, float] | None
    detector_score: float | None
    reason: str
    rgb_sha256: str
    source_detections_sha256: str
    source_decisions_sha256: str

    def __post_init__(self) -> None:
        if self.status not in {"selected", "abstained"}:
            raise AgentSemanticPortError("semantic status must be selected/abstained")
        selected = self.status == "selected"
        if selected != (self.candidate_id is not None):
            raise AgentSemanticPortError("selected status and candidate_id disagree")
        if selected != (self.box_xyxy is not None):
            raise AgentSemanticPortError("selected status and proposal box disagree")
        if selected != (self.detector_score is not None):
            raise AgentSemanticPortError("selected status and detector score disagree")
        if not self.case_id or not self.reason:
            raise AgentSemanticPortError("semantic selection metadata is incomplete")
        for digest in (
            self.rgb_sha256,
            self.source_detections_sha256,
            self.source_decisions_sha256,
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise AgentSemanticPortError("semantic provenance requires SHA-256")

    @property
    def identity_hypothesis(self) -> bool | None:
        return True if self.status == "selected" else None


class AgentSemanticSelector(Protocol):
    """Runtime port implemented by an Agent service or a sealed replay."""

    selector_id: str
    selector_version: str

    def select(self, *, case_id: str, frame_id: int) -> SemanticSelection: ...


class RecordedAgentSemanticSelector:
    """Reproducible adapter over sealed Agent decisions from a prior runtime run."""

    selector_id = "prove.agent.current_frame_semantic_selector"
    selector_version = "1.0.0"

    def __init__(self, rows: Mapping[tuple[str, int], SemanticSelection]) -> None:
        if not rows:
            raise AgentSemanticPortError("recorded selector cannot be empty")
        self._rows = MappingProxyType(dict(rows))

    @classmethod
    def from_artifact_pairs(
        cls,
        pairs: Sequence[tuple[Path, Path]],
    ) -> "RecordedAgentSemanticSelector":
        rows: dict[tuple[str, int], SemanticSelection] = {}
        for detections_path, decisions_path in pairs:
            detections_bytes = detections_path.read_bytes()
            decisions_bytes = decisions_path.read_bytes()
            detections_sha = hashlib.sha256(detections_bytes).hexdigest()
            decisions_sha = hashlib.sha256(decisions_bytes).hexdigest()
            detections = json.loads(detections_bytes)
            decisions = json.loads(decisions_bytes)
            if detections.get("format") != "prove.gap-policy-skill-detections.v1":
                raise AgentSemanticPortError("unsupported detector artifact")
            if decisions.get("format") != "prove.agent-semantic-selection.v1":
                raise AgentSemanticPortError("unsupported Agent decision artifact")
            if decisions.get("runtime_gt_access") is not False:
                raise AgentSemanticPortError("Agent selector must deny runtime GT access")
            if decisions.get("source_detections", {}).get("sha256") != detections_sha:
                raise AgentSemanticPortError("Agent decisions do not bind detector bytes")
            serialized = decisions_bytes.lower()
            if any(token.encode() in serialized for token in _FORBIDDEN_TOKENS):
                raise AgentSemanticPortError("Agent decisions reference evaluator-only data")

            detections_by_key = {
                (str(item["case_id"]), int(item["frame_index"])): item
                for item in detections.get("records", ())
            }
            decisions_by_key = {
                (str(item["case_id"]), int(item["frame_index"])): item
                for item in decisions.get("records", ())
            }
            if len(detections_by_key) != len(detections.get("records", ())):
                raise AgentSemanticPortError("duplicate detector frame")
            if len(decisions_by_key) != len(decisions.get("records", ())):
                raise AgentSemanticPortError("duplicate Agent decision frame")
            if set(detections_by_key) != set(decisions_by_key):
                raise AgentSemanticPortError("Agent decisions do not cover detector frames")

            for key, detector_row in detections_by_key.items():
                if key in rows:
                    raise AgentSemanticPortError(f"duplicate semantic replay frame: {key}")
                decision = decisions_by_key[key]
                selection = str(decision["selection"])
                proposals = list(detector_row.get("detections", ()))
                if selection == "none":
                    candidate_id = None
                    box = None
                    score = None
                    status = "abstained"
                else:
                    offset = ord(selection) - ord("A")
                    if offset < 0 or offset >= len(proposals):
                        raise AgentSemanticPortError(
                            f"Agent selected unavailable proposal {selection} for {key}"
                        )
                    proposal = proposals[offset]
                    raw_box = proposal["box"]
                    candidate_id = selection
                    box = (
                        float(raw_box["x1"]),
                        float(raw_box["y1"]),
                        float(raw_box["x2"]),
                        float(raw_box["y2"]),
                    )
                    score = float(proposal["score"])
                    status = "selected"
                rgb_sha = str(detector_row.get("frame_artifact", {}).get("sha256", ""))
                rows[key] = SemanticSelection(
                    case_id=key[0],
                    frame_id=key[1],
                    status=status,
                    candidate_id=candidate_id,
                    box_xyxy=box,
                    detector_score=score,
                    reason=str(decision.get("reason", "")).strip(),
                    rgb_sha256=rgb_sha,
                    source_detections_sha256=detections_sha,
                    source_decisions_sha256=decisions_sha,
                )
        return cls(rows)

    @property
    def frame_keys(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self._rows))

    def select(self, *, case_id: str, frame_id: int) -> SemanticSelection:
        try:
            return self._rows[(case_id, frame_id)]
        except KeyError as exc:
            raise AgentSemanticPortError(
                f"no sealed Agent decision for {case_id}/{frame_id}"
            ) from exc


def build_agent_assisted_regrounding_spec() -> ObserverSpec:
    """Build the formal Observer graph without modifying the Foundation Catalog."""

    return ObserverSpec.from_dict(
        {
            "observer_id": "prove.observer.agent_assisted_current_frame_regrounding",
            "version": "1.0.0",
            "parent_id": "prove.observer.always_on_independent_identity_verification@1.0.0",
            "primary_modification": "agent_semantic_candidate_verification_with_explicit_abstention",
            "inputs": [
                _input("reground_requested", "bool", "measurement", "observer_scheduler"),
                _input("agent_identity_hypothesis", "optional_bool", "hypothesis", "agent_rgb_semantics"),
                _input("target_world_position", "vector3", "measurement", "sam3_rgbd_geometry"),
                _input("target_mask_sha256", "sha256_digest", "measurement", "sam3_segmentation"),
                _input("mask_score", "float", "measurement", "sam3_segmentation"),
                _input("geometry_valid", "bool", "measurement", "rgbd_geometry"),
                _input("eef_world_position", "vector3", "measurement", "robot_state"),
            ],
            "state": [],
            "nodes": [
                {
                    "node_id": "target_eef_relation",
                    "primitive": "prove.relation.relative_vector@1.0.0",
                    "bindings": {
                        "target_position": "$input.target_world_position",
                        "reference_position": "$input.eef_world_position",
                    },
                }
            ],
            "evidence_outputs": [
                _output("reground_requested", "$input.reground_requested", "measurement"),
                _output("agent_identity_hypothesis", "$input.agent_identity_hypothesis", "hypothesis"),
                _output("target_world_position", "$input.target_world_position", "measurement"),
                _output("target_mask_sha256", "$input.target_mask_sha256", "measurement"),
                _output("mask_score", "$input.mask_score", "measurement"),
                _output("geometry_valid", "$input.geometry_valid", "measurement"),
                _output("eef_world_position", "$input.eef_world_position", "measurement"),
                _output("target_eef_relative_vector", "$node.target_eef_relation.relative_vector", "measurement"),
                _output(
                    "target_identity_current_frame",
                    "$input.agent_identity_hypothesis",
                    "verified_evidence",
                    supports=("target_world_position", "target_mask_sha256", "mask_score", "geometry_valid"),
                    value_rule={"op": "eq", "value": True},
                ),
                _output(
                    "target_eef_relation",
                    "$node.target_eef_relation.relative_vector",
                    "verified_evidence",
                    supports=("target_world_position", "eef_world_position"),
                ),
            ],
            "gates": [
                _gate(target, "$input.reground_requested", "truthy")
                for target in ("target_identity_current_frame", "target_eef_relation")
            ]
            + [
                _gate(target, "$input.agent_identity_hypothesis", "truthy")
                for target in ("target_identity_current_frame", "target_eef_relation")
            ]
            + [
                _gate(target, "$input.geometry_valid", "truthy")
                for target in ("target_identity_current_frame", "target_eef_relation")
            ]
            + [
                _gate(target, "$input.mask_score", "gte", 0.5)
                for target in ("target_identity_current_frame", "target_eef_relation")
            ],
            "abstention_rules": [],
            "state_updates": {},
        }
    )


def compile_agent_assisted_regrounding_observer():
    """Compile the formal graph against the frozen Foundation Catalog."""

    return ObserverCompiler(build_foundation_registry()).compile(
        build_agent_assisted_regrounding_spec()
    )


def _input(name: str, type_name: str, kind: str, group: str) -> dict[str, str]:
    return {
        "name": name,
        "type": type_name,
        "kind": kind,
        "independence_group": group,
    }


def _output(
    name: str,
    source: str,
    kind: str,
    *,
    supports: Sequence[str] = (),
    value_rule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "name": name,
        "source": source,
        "kind": kind,
        "supporting_measurements": list(supports),
    }
    if value_rule is not None:
        value["value_rule"] = dict(value_rule)
    return value


def _gate(target: str, source: str, op: str, value: Any = None) -> dict[str, Any]:
    predicate: dict[str, Any] = {"source": source, "op": op}
    if op not in {"truthy", "falsy"}:
        predicate["value"] = value
    return {"target": target, "predicate": predicate, "on_fail": "unavailable"}
