"""P69 R2 public grounding and retargeting helpers.

This module is intentionally self-contained and starts no simulator.  Runtime
estimators read only public RGB/mask/camera/proprio frames.  Demo/private object
poses may enter only the offline calibration JSON and are represented as fixed
plane heights and demo-frame bias terms; current episode private poses are never
accepted by the runtime APIs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from prove.task_graph.p57_public_geometry import rays
from prove.task_graph.libero_precision_p51 import MixedController, PRECISE, features, project_progress

ROLES = ("gripper", "book", "caddy")
ROLE_INDEX = {name: i for i, name in enumerate(ROLES)}
SCHEMA = "prove.p69-grounding-reference.v1"
UPDATE_SCHEMA = "prove.p69-r2-public-grounding-update.v1"


@dataclass(frozen=True)
class PlanePose:
    xy: tuple[float, float]
    yaw: float
    count: int
    eigenvalues: tuple[float, float]
    view_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RoleDelta:
    role: str
    demo_xy: tuple[float, float]
    current_xy: tuple[float, float]
    delta_xy: tuple[float, float]
    delta_yaw: float
    current_yaw: float
    valid: bool
    reason: str = "ok"

    def to_dict(self) -> dict:
        return asdict(self)


def _array(value, shape, label):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{label} must be finite shape {shape}")
    return arr


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _load_json_binding(binding: Mapping, label: str) -> dict:
    if not isinstance(binding, Mapping) or "path" not in binding or "sha256" not in binding:
        raise ValueError(f"{label} binding requires path and sha256")
    path = Path(binding["path"])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != binding["sha256"]:
        raise ValueError(f"{label} hash mismatch")
    return json.loads(data.decode("utf-8"))


def _nearest_symmetric_yaw(raw_yaw: float, nominal_yaw: float, period: float) -> float:
    if not np.isfinite(raw_yaw) or not np.isfinite(nominal_yaw) or period <= 0:
        raise ValueError("invalid yaw branch inputs")
    k = round((nominal_yaw - raw_yaw) / period)
    return _wrap_pi(raw_yaw + k * period)


def _mask_pixels(mask: np.ndarray, *, max_points: int = 4096) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(xx) > max_points:
        # Deterministic uniform subsampling by index, not random tuning.
        keep = np.linspace(0, len(xx) - 1, max_points).round().astype(int)
        xx, yy = xx[keep], yy[keep]
    return np.column_stack((xx.astype(np.float64) + 0.5, yy.astype(np.float64) + 0.5))


def mask_plane_points(mask: np.ndarray, camera_K: np.ndarray, camera_pose: np.ndarray,
                      plane_world_z: float, *, max_points: int = 4096) -> np.ndarray:
    """Project visible mask pixels onto a fixed world-z plane using public camera data."""
    if not np.isfinite(plane_world_z):
        raise ValueError("plane_world_z must be finite")
    pixels = _mask_pixels(mask, max_points=max_points)
    if len(pixels) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    origins, directions = rays(pixels, camera_K, camera_pose)
    denom = directions[:, 2]
    valid = np.abs(denom) > 1e-9
    t = np.zeros(len(pixels), dtype=np.float64)
    t[valid] = (float(plane_world_z) - origins[valid, 2]) / denom[valid]
    valid &= t > 0
    points = origins[valid] + directions[valid] * t[valid, None]
    points = points[np.isfinite(points).all(axis=1)]
    return points


def plane_pose_from_points(points: np.ndarray, *, nominal_yaw: float = 0.0,
                           symmetry_yaw_period: float = np.pi,
                           min_points: int = 32) -> PlanePose | None:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError("points must be finite [N,3]")
    if len(pts) < min_points:
        return None
    xy = pts[:, :2]
    center = xy.mean(axis=0)
    residual = xy - center
    cov = residual.T @ residual / len(xy)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    raw = float(np.arctan2(axis[1], axis[0]))
    yaw = _nearest_symmetric_yaw(raw, nominal_yaw, symmetry_yaw_period)
    return PlanePose(tuple(center.tolist()), yaw, int(len(pts)), tuple(eigvals.tolist()), 1)


def estimate_role_pose(public: Mapping[str, np.ndarray], calibration: Mapping, role: str,
                       *, min_points: int = 32) -> PlanePose | None:
    if role not in ("book", "caddy"):
        raise ValueError("R2 grounds only book/caddy roles")
    role_cfg = calibration["role_calibration"][role]
    if role_cfg.get("observer", "view0_planar") != "view0_planar":
        raise ValueError(f"role {role} is not configured for planar grounding")
    masks = np.asarray(public["visible_masks"])
    camera_K = np.asarray(public["camera_K"])
    camera_pose = np.asarray(public["camera_pose"])
    if masks.ndim != 4 or masks.shape[0:2] != (2, 3):
        raise ValueError("visible_masks must be [2,3,H,W]")
    all_points = []
    views = calibration.get("views", [0])
    if views != [0]:
        raise ValueError("P69 R2 grounding freezes the global camera only: views must be [0]")
    for view in views:
        pts = mask_plane_points(masks[view, ROLE_INDEX[role]], camera_K[view], camera_pose[view],
                                role_cfg["plane_world_z"])
        if len(pts):
            all_points.append(pts)
    if not all_points:
        return None
    pose = plane_pose_from_points(np.concatenate(all_points, axis=0),
                                  nominal_yaw=role_cfg["demo_pose_xy_yaw"][2],
                                  symmetry_yaw_period=role_cfg.get("symmetry_yaw_period", np.pi),
                                  min_points=min_points)
    if pose is None:
        return None
    # Calibration bias is demo-mask minus training GT pose; add it to each runtime mask estimate.
    bias = np.asarray(role_cfg.get("mask_bias_xy_yaw", [0.0, 0.0, 0.0]), dtype=np.float64)
    if bias.shape != (3,) or not np.isfinite(bias).all():
        raise ValueError("mask_bias_xy_yaw must be finite length3")
    return PlanePose(tuple((np.asarray(pose.xy) + bias[:2]).tolist()), _wrap_pi(pose.yaw + float(bias[2])),
                     pose.count, pose.eigenvalues, 2)




def _estimate_book_delta_rigid3d(public: Mapping[str, np.ndarray], calibration: Mapping) -> RoleDelta:
    role_cfg = calibration["role_calibration"]["book"]
    try:
        from prove.task_graph.p69_book_silhouette import fit_book_pose
    except ImportError as exc:
        raise RuntimeError("R2 book rigid3d silhouette helper is not available") from exc
    reference = _load_json_binding(role_cfg.get("book_reference"), "book_reference")
    masks = np.asarray(public["visible_masks"])
    if masks.ndim != 4 or masks.shape[0:2] != (2, 3):
        raise ValueError("visible_masks must be [2,3,H,W]")
    views = calibration.get("views", [0])
    if views != [0]:
        raise ValueError("P69 R2 book silhouette freezes global camera views=[0]")
    initial = role_cfg.get("initial_delta_xy_yaw")
    if initial is None and isinstance(role_cfg.get("planar_initializer"), Mapping):
        temp = {"views": calibration.get("views", [0]),
                "role_calibration": {"book": dict(role_cfg["planar_initializer"], observer="view0_planar")}}
        rough = estimate_role_pose(public, temp, "book", min_points=int(role_cfg.get("planar_initializer_min_points", 32)))
        if rough is not None:
            demo0 = np.asarray(temp["role_calibration"]["book"]["demo_pose_xy_yaw"], dtype=np.float64)
            initial = [float(rough.xy[0] - demo0[0]), float(rough.xy[1] - demo0[1]), _wrap_pi(float(rough.yaw - demo0[2]))]
    result = fit_book_pose(masks[0, ROLE_INDEX["book"]], np.asarray(public["camera_K"])[0],
                           np.asarray(public["camera_pose"])[0], reference,
                           initial_delta=(0.0, 0.0, 0.0) if initial is None else tuple(initial))
    if not isinstance(result, Mapping):
        raise ValueError("fit_book_pose must return a mapping")
    source_xy = np.asarray(result.get("source_xy", np.asarray(reference["source_body_pose"])[:2, 3]), dtype=np.float64)
    source_yaw = float(np.arctan2(reference["source_body_pose"][1][0], reference["source_body_pose"][0][0]))
    if source_xy.shape != (2,) or not np.isfinite(source_xy).all() or not np.isfinite(source_yaw):
        raise ValueError("book helper/reference must provide finite source pose")
    valid = bool(result.get("valid", result.get("success", False)))
    success = bool(result.get("success", False))
    if not (valid and success):
        return RoleDelta("book", tuple(source_xy.tolist()), tuple(source_xy.tolist()), (0.0, 0.0), 0.0, source_yaw, False,
                         str(result.get("reason", "rigid3d_fit_failed")))
    current_xy = np.asarray(result.get("body_xy"), dtype=np.float64)
    delta_xy = np.asarray(result.get("delta_xy", current_xy - source_xy), dtype=np.float64)
    current_yaw = float(result.get("body_yaw"))
    delta_yaw = float(result.get("delta_yaw", _wrap_pi(current_yaw - source_yaw)))
    if current_xy.shape != (2,) or delta_xy.shape != (2,) or not np.isfinite(current_xy).all() or not np.isfinite(delta_xy).all() or not np.isfinite(current_yaw) or not np.isfinite(delta_yaw):
        raise ValueError("fit_book_pose success returned malformed pose")
    return RoleDelta("book", tuple(source_xy.tolist()), tuple(current_xy.tolist()),
                     tuple(delta_xy.tolist()), _wrap_pi(delta_yaw), _wrap_pi(current_yaw), True, "ok")

def estimate_role_delta(public: Mapping[str, np.ndarray], calibration: Mapping, role: str) -> RoleDelta:
    role_cfg = calibration["role_calibration"][role]
    if role == "book" and role_cfg.get("observer") == "rigid3d_silhouette":
        return _estimate_book_delta_rigid3d(public, calibration)
    pose = estimate_role_pose(public, calibration, role)
    demo = np.asarray(role_cfg["demo_pose_xy_yaw"], dtype=np.float64)
    if demo.shape != (3,) or not np.isfinite(demo).all():
        raise ValueError("demo_pose_xy_yaw must be finite length3")
    if pose is None:
        return RoleDelta(role, tuple(demo[:2].tolist()), tuple(demo[:2].tolist()), (0.0, 0.0), 0.0, float(demo[2]), False, "missing_or_small_mask")
    current = np.asarray([pose.xy[0], pose.xy[1], pose.yaw], dtype=np.float64)
    return RoleDelta(role, tuple(demo[:2].tolist()), tuple(current[:2].tolist()),
                     tuple((current[:2] - demo[:2]).tolist()), _wrap_pi(float(current[2] - demo[2])),
                     float(current[2]), True, "ok")


def validate_grounding_reference(value: Mapping) -> dict:
    if value.get("schema") != SCHEMA:
        raise ValueError("wrong P69 grounding reference schema")
    if value.get("role_order") != list(ROLES):
        raise ValueError("role_order must be gripper/book/caddy")
    if value.get("views", [0]) != [0]:
        raise ValueError("P69 R2 grounding reference must freeze global camera views=[0]")
    forbidden = value.get("runtime_forbidden", [])
    if "current_episode_private_pose" not in forbidden:
        raise ValueError("grounding reference must explicitly forbid current private pose at runtime")
    roles = value.get("role_calibration", {})
    for role in ("book", "caddy"):
        cfg = roles.get(role)
        if not isinstance(cfg, Mapping):
            raise ValueError(f"missing role_calibration.{role}")
        observer = cfg.get("observer", "view0_planar")
        if role == "book" and observer == "rigid3d_silhouette":
            ref = _load_json_binding(cfg.get("book_reference"), "book_reference")
            if ref.get("schema") != "prove.p69-native-book-silhouette.v1":
                raise ValueError("unsupported book silhouette reference schema")
            _array(ref.get("box_center"), (3,), "book.box_center")
            _array(ref.get("box_halfsize"), (3,), "book.box_halfsize")
            _array(ref.get("source_body_pose"), (4, 4), "book.source_body_pose")
        elif observer == "view0_planar":
            _array(cfg.get("demo_pose_xy_yaw"), (3,), f"{role}.demo_pose_xy_yaw")
            _array(cfg.get("mask_bias_xy_yaw", [0, 0, 0]), (3,), f"{role}.mask_bias_xy_yaw")
            if not np.isfinite(float(cfg.get("plane_world_z"))):
                raise ValueError(f"{role}.plane_world_z must be finite")
            if float(cfg.get("symmetry_yaw_period", np.pi)) <= 0:
                raise ValueError(f"{role}.symmetry_yaw_period must be positive")
        else:
            raise ValueError(f"unsupported {role} observer: {observer}")
    return dict(value)


def _se2_transform_position(position: np.ndarray, delta: RoleDelta) -> np.ndarray:
    p = np.asarray(position, dtype=np.float64).copy()
    c0 = np.asarray(delta.demo_xy, dtype=np.float64)
    c1 = np.asarray(delta.current_xy, dtype=np.float64)
    yaw = float(delta.delta_yaw)
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float64)
    p[:2] = c1 + R @ (p[:2] - c0)
    return p


def _se2_transform_rotvec(rotvec: np.ndarray, delta: RoleDelta) -> np.ndarray:
    zrot = Rotation.from_euler("z", float(delta.delta_yaw)).as_matrix()
    base = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    return Rotation.from_matrix(zrot @ base).as_rotvec()


def _blend_position(position: np.ndarray, book: RoleDelta, caddy: RoleDelta, weight: float) -> np.ndarray:
    return (1.0 - weight) * _se2_transform_position(position, book) + weight * _se2_transform_position(position, caddy)


def retarget_demo_arrays(positions: np.ndarray, rotations: np.ndarray, book: RoleDelta, caddy: RoleDelta) -> tuple[np.ndarray, np.ndarray]:
    """Retarget source EEF trajectory by Root's R2 role assignment.

    Source intervals are the reviewed semantic demo boundaries.  s1/s2/s3/s4
    follow book, s5 blends book->caddy, s6 and later follow caddy.
    """
    pos = np.asarray(positions, dtype=np.float64).copy()
    rot = np.asarray(rotations, dtype=np.float64).copy()
    if pos.ndim != 2 or pos.shape[1] != 3 or rot.shape != pos.shape:
        raise ValueError("positions/rotations must both be [T,3]")
    n = len(pos)
    for i in range(n):
        if i < 85:
            d, w = book, None
        elif i < 130:
            frac = (i - 85) / max(1, 129 - 85)
            d, w = None, float(np.clip(frac, 0.0, 1.0))
        else:
            d, w = caddy, None
        if w is None:
            pos[i] = _se2_transform_position(pos[i], d)
            rot[i] = _se2_transform_rotvec(rot[i], d)
        else:
            pos[i] = _blend_position(pos[i], book, caddy, w)
            yaw = (1.0 - w) * book.delta_yaw + w * caddy.delta_yaw
            rot[i] = _se2_transform_rotvec(rot[i], RoleDelta("blend", book.demo_xy, book.demo_xy, (0.0, 0.0), yaw, 0.0, True))
    return pos, rot


class PublicRetargetVariantController(MixedController):
    """R2 retargeted controller with frozen P51/P69 action head.

    The underlying model and original stage-local paths stay frozen.  The full
    demo EEF trajectory is retargeted for code stages; precise stages transform
    their original stage paths on the fly.  ``self.reference`` is the caddy
    retarget of the nominal reference so ``insert_entry_domain`` remains the
    original P51 calculation with a public reference transform.
    """
    revision = "p69-r2-public-planar-retarget-v1"

    def __init__(self, source, graph, checkpoint, offset, *, book_delta: RoleDelta, caddy_delta: RoleDelta):
        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("grasp offset must be finite length3")
        if np.linalg.norm(offset) > 1e-12:
            raise ValueError("P69 R2 retarget scope requires zero P52 grasp offset")
        super().__init__(source, graph, checkpoint)
        self.book_delta = book_delta
        self.caddy_delta = caddy_delta
        new_positions, new_rotations = retarget_demo_arrays(self.code.positions, self.code.rotations, book_delta, caddy_delta)
        for arr in (new_positions, new_rotations):
            arr.setflags(write=False)
        self.code.positions = new_positions
        self.code.rotations = new_rotations
        self.reference = self._transform_reference(self.nominal_reference, caddy_delta)
        self.reference_source = "public_visual_adapter"

    @staticmethod
    def _transform_reference(reference: np.ndarray, delta: RoleDelta) -> np.ndarray:
        T = np.asarray(reference, dtype=np.float64).copy()
        T[:3, 3] = _se2_transform_position(T[:3, 3], delta)
        T[:3, :3] = Rotation.from_euler("z", delta.delta_yaw).as_matrix() @ T[:3, :3]
        return T

    @staticmethod
    def _transform_path(path: Mapping, delta: RoleDelta) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(path["positions"], dtype=np.float64)
        rotations = np.asarray(path["rotations"], dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 3 or rotations.shape != (len(positions), 3, 3):
            raise ValueError("precise path must contain positions [N,3] and rotations [N,3,3]")
        Rz = Rotation.from_euler("z", delta.delta_yaw).as_matrix()
        out_pos = positions.copy()
        out_rot = rotations.copy()
        for i in range(len(out_pos)):
            out_pos[i] = _se2_transform_position(out_pos[i], delta)
            out_rot[i] = Rz @ out_rot[i]
        return out_pos, out_rot

    def __call__(self, payload):
        stage = payload["stage_id"]
        if stage != self.stage_id:
            raise ValueError("stage entry required")
        if stage not in PRECISE:
            action = self.code(payload)
            self.diagnostics = dict(self.code.diagnostics, controller="r2_retargeted_code_eef_feedback")
            return action
        delta = self.book_delta if stage == "s3" else self.caddy_delta
        reference = self._transform_reference(self.nominal_reference, delta)
        positions, rotations = self._transform_path(self.paths[stage], delta)
        self.progress = project_progress(self.position, self.rotation, positions, rotations, self.progress)
        x = features(self.position, self.rotation, self.gripper, reference, self.progress, stage)
        with torch.no_grad():
            local = self.model(torch.from_numpy(x)).numpy()
        R = reference[:3, :3]
        action = np.r_[R @ local[:3], R @ local[3:], 1.]
        action = np.clip(action, -1., 1.)
        self.diagnostics = dict(controller="r2_public_retargeted_frozen_neural_action",
                                progress=float(self.progress), features=x.tolist(), local_action=local.tolist(),
                                reference_source="public_visual_adapter_book" if stage == "s3" else "public_visual_adapter_caddy",
                                adapter_enabled=True)
        return action


def load_reference(path: str | Path) -> dict:
    return validate_grounding_reference(json.loads(Path(path).read_text()))
