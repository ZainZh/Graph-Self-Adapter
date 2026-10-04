"""Load the manually governed Foundation Primitive Catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .primitive import PrimitiveContractError, PrimitiveMeta, PrimitiveRegistry
from .tool_index import ProveToolIndex
from .primitives import (
    change_point_score,
    cosine_feature_match,
    depth_to_world_points,
    get_eef_pose,
    get_gripper_state,
    mask_to_world_points,
    motion_consistency,
    point_in_upright_envelope,
    project_world_points,
    relative_drift,
    relative_vector,
    robust_centroid,
    transform_points,
    update_multi_hypothesis,
    upright_pointcloud_envelope,
)


CATALOG_PATH = Path(__file__).with_name("catalogs") / "foundation_v1.json"

_BUILTINS: dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {
    "prove.geometry.depth_to_world_points@1.0.0": depth_to_world_points,
    "prove.geometry.mask_to_world_points@1.0.0": mask_to_world_points,
    "prove.geometry.transform_points@1.0.0": transform_points,
    "prove.geometry.robust_centroid@1.0.0": robust_centroid,
    "prove.geometry.project_world_points@1.0.0": project_world_points,
    "prove.relation.relative_vector@1.0.0": relative_vector,
    "prove.relation.relative_drift@1.0.0": relative_drift,
    "prove.temporal.change_point_score@1.0.0": change_point_score,
    "prove.temporal.motion_consistency@1.0.0": motion_consistency,
    "prove.belief.multi_hypothesis_update@1.0.0": update_multi_hypothesis,
    "prove.features.cosine_match@1.0.0": cosine_feature_match,
    "prove.robot.get_eef_pose@1.0.0": get_eef_pose,
    "prove.robot.get_gripper_state@1.0.0": get_gripper_state,
    "prove.geometry.upright_pointcloud_envelope@1.0.0": (
        upright_pointcloud_envelope
    ),
    "prove.relation.point_in_upright_envelope@1.0.0": point_in_upright_envelope,
}


def load_foundation_catalog(
    path: Path = CATALOG_PATH,
) -> Mapping[str, Any]:
    """Parse and validate one immutable catalog snapshot."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format",
        "catalog_id",
        "version",
        "status",
        "governance",
        "primitives",
    }
    if set(raw) != expected:
        raise PrimitiveContractError(
            f"foundation catalog keys must be {sorted(expected)}"
        )
    if raw["format"] != "prove.foundation-primitive-catalog.v1":
        raise PrimitiveContractError("unsupported foundation catalog format")
    if raw["status"] != "frozen":
        raise PrimitiveContractError("foundation catalog must be frozen")
    governance = raw["governance"]
    if governance != {
        "mutation": "manual_append_only_version",
        "primitive_search_enabled": False,
        "observer_graph_search_enabled": True,
        "runtime_gt_access": "forbidden",
    }:
        raise PrimitiveContractError("foundation governance policy drift")
    if not isinstance(raw["primitives"], list) or not raw["primitives"]:
        raise PrimitiveContractError("foundation catalog cannot be empty")
    refs: list[str] = []
    for item in raw["primitives"]:
        meta = PrimitiveMeta.from_dict(item)
        refs.append(meta.ref)
    if len(refs) != len(set(refs)):
        raise PrimitiveContractError("foundation primitive refs must be unique")
    return MappingProxyType(raw)


def build_foundation_registry(
    *,
    adapters: Mapping[
        str, Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] | None = None,
    catalog_path: Path = CATALOG_PATH,
) -> PrimitiveRegistry:
    """Build a frozen stable index; optional backends must be bound up front."""

    catalog = load_foundation_catalog(catalog_path)
    provided = dict(adapters or {})
    known_refs = {
        PrimitiveMeta.from_dict(item).ref for item in catalog["primitives"]
    }
    unknown = sorted(set(provided) - known_refs)
    if unknown:
        raise PrimitiveContractError(f"adapters target unknown refs: {unknown}")
    registry = PrimitiveRegistry(
        registry_id=str(catalog["catalog_id"]),
        version=str(catalog["version"]),
    )
    for item in catalog["primitives"]:
        meta = PrimitiveMeta.from_dict(item)
        implementation = provided.get(meta.ref, _BUILTINS.get(meta.ref))
        registry.register(meta, implementation)
    registry.freeze()
    return registry


def build_foundation_tool_index(
    *,
    adapters: Mapping[
        str, Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] | None = None,
    catalog_path: Path = CATALOG_PATH,
) -> ProveToolIndex:
    """Expose the frozen registry through the user-facing indexed tool API."""

    return ProveToolIndex(
        build_foundation_registry(adapters=adapters, catalog_path=catalog_path)
    )


def catalog_artifact(path: Path = CATALOG_PATH) -> dict[str, Any]:
    """Return a byte-addressed catalog reference for evidence records."""

    payload = path.read_bytes()
    return {
        "uri": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
        "media_type": "application/json",
    }
