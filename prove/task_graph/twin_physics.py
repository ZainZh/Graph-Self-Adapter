"""Force-driven MuJoCo twin adapter with a strict runtime/evaluator boundary.

The runtime receives robot proprioception, contact evidence and visible RGB-D
role observations.  Continuous object state is retained only for contact,
numerical-safety bookkeeping and the post-episode evaluator.  After
``initialize`` all motion is produced by actuator forces and ``mj_step``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import ConvexHull


ARMS = ("left", "right")
ARM_KP = np.tile(np.array([1500., 1800., 1500., 700., 700., 500.]), 2)
ARM_KD = np.tile(np.array([65., 70., 60., 18., 18., 14.]), 2)


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}, got {result.shape}")
    return result.copy()


def _load_json(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return json.loads(Path(value).read_text())


def _load_npz(value: str | Path | Mapping[str, Any] | None) -> dict[str, np.ndarray] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): np.asarray(v).copy() for k, v in value.items()}
    with np.load(Path(value), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


@dataclass(frozen=True)
class Proprioception:
    tcp_position: np.ndarray
    tcp_rotation: np.ndarray
    joint_pos: np.ndarray
    joint_velocity: np.ndarray
    finger_qpos: np.ndarray


@dataclass(frozen=True)
class RoleObservation:
    role: str
    present: bool
    missing_reason: str | None
    centroid_world_m: np.ndarray | None
    visible_fraction: float
    pixel_count: int
    bbox_xyxy: tuple[int, int, int, int] | None
    mask: np.ndarray


@dataclass(frozen=True)
class VisibleRGBDFrame:
    physical_time_sec: float
    sample_monotonic_sec: float
    rgb: np.ndarray
    depth_m: np.ndarray
    camera_K: np.ndarray
    camera_pose_world: np.ndarray
    roles: dict[str, RoleObservation]
    camera_convention: str = "opencv_x_right_y_down_z_forward"


@dataclass(frozen=True)
class StepResult:
    physical_time_sec: float
    sample_monotonic_sec: float
    physics_steps: int
    proprioception: Proprioception
    contact_evidence: dict[str, Any]
    visible: VisibleRGBDFrame | None
    numerical_safety: dict[str, Any]


@dataclass(frozen=True)
class NominalTrajectory:
    times: np.ndarray
    joints: np.ndarray
    gripper: np.ndarray
    tcp_position: np.ndarray
    tcp_rotation: np.ndarray


class VisibleRGBD:
    """Low-frequency visible-role observer; never reads object body positions."""

    def __init__(self, world: "TwinWorld", *, width: int, height: int, camera: str,
                 min_visible_pixels: int = 12):
        if width <= 0 or height <= 0:
            raise ValueError("renderer dimensions must be positive")
        if min_visible_pixels <= 0:
            raise ValueError("min_visible_pixels must be positive")
        self.world = world
        self.width = int(width)
        self.height = int(height)
        self.camera = camera
        self.min_visible_pixels = int(min_visible_pixels)
        self.renderer = mujoco.Renderer(world.model, height=self.height, width=self.width)

    def close(self) -> None:
        self.renderer.close()

    def _render(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.world.data, camera=self.camera)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.world.data, camera=self.camera)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.world.data, camera=self.camera)
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        return rgb, depth, segmentation

    def _camera(self) -> tuple[np.ndarray, np.ndarray]:
        camera_id = self.world.model.camera(self.camera).id
        resolution = self.world.model.cam_resolution[camera_id].astype(np.float64)
        sensor = self.world.model.cam_sensorsize[camera_id].astype(np.float64)
        intrinsic = self.world.model.cam_intrinsic[camera_id].astype(np.float64)
        if np.all(resolution > 0) and np.all(sensor > 0) and np.all(intrinsic[:2] > 0):
            sx, sy = self.width / resolution[0], self.height / resolution[1]
            fx = intrinsic[0] / sensor[0] * resolution[0] * sx
            fy = intrinsic[1] / sensor[1] * resolution[1] * sy
            cx = (resolution[0] / 2 - intrinsic[2] / sensor[0] * resolution[0]) * sx
            cy = (resolution[1] / 2 + intrinsic[3] / sensor[1] * resolution[1]) * sy
        else:
            fovy = math.radians(float(self.world.model.cam_fovy[camera_id]))
            fy = 0.5 * self.height / math.tan(0.5 * fovy)
            fx, cx, cy = fy, (self.width - 1) / 2, (self.height - 1) / 2
        K = np.array([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]], dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)
        # MuJoCo/OpenGL cameras use +Y up and look along -Z.  Public RGB-D
        # geometry uses OpenCV optical coordinates: +X right, +Y down, +Z out.
        pose[:3, :3] = (self.world.data.cam_xmat[camera_id].reshape(3, 3) @
                        np.diag([1., -1., -1.]))
        pose[:3, 3] = self.world.data.cam_xpos[camera_id]
        return K, pose

    def observe(self, roles: Sequence[str] | None = None) -> VisibleRGBDFrame:
        rgb, depth, segmentation = self._render()
        K, pose = self._camera()
        selected = tuple(roles) if roles is not None else tuple(self.world.role_geom_ids)
        unknown = sorted(set(selected) - set(self.world.role_geom_ids))
        if unknown:
            raise KeyError(f"unknown semantic roles: {unknown}")
        # MuJoCo segmentation pixels are ``(object_id, object_type)``.
        object_id = segmentation[..., 0]
        object_type = segmentation[..., 1]
        rows: dict[str, RoleObservation] = {}
        yy_grid, xx_grid = np.indices(depth.shape)
        for role in selected:
            geom_ids = self.world.role_geom_ids[role]
            mask = ((object_type == int(mujoco.mjtObj.mjOBJ_GEOM)) &
                    np.isin(object_id, np.fromiter(geom_ids, dtype=np.int32)))
            mask &= np.isfinite(depth) & (depth > 0)
            yy, xx = np.nonzero(mask)
            if not len(xx):
                rows[role] = RoleObservation(role, False, "not_visible", None, 0., 0, None, mask.copy())
                continue
            if len(xx) < self.min_visible_pixels:
                rows[role] = RoleObservation(
                    role, False, "insufficient_visible_depth", None,
                    float(len(xx) / (self.width * self.height)), int(len(xx)),
                    (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())),
                    mask.copy(),
                )
                continue
            z = depth[mask]
            optical = np.column_stack(((xx_grid[mask] - K[0, 2]) * z / K[0, 0],
                                       (yy_grid[mask] - K[1, 2]) * z / K[1, 1], z))
            points = optical @ pose[:3, :3].T + pose[:3, 3]
            finite = np.isfinite(points).all(axis=1)
            if not finite.any():
                rows[role] = RoleObservation(role, False, "invalid_depth", None, 0., 0, None, mask.copy())
                continue
            centroid = np.median(points[finite], axis=0)
            rows[role] = RoleObservation(
                role, True, None, centroid,
                float(len(xx) / (self.width * self.height)), int(len(xx)),
                (int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())),
                mask.copy(),
            )
        return VisibleRGBDFrame(
            float(self.world.data.time), time.monotonic(), rgb, depth, K, pose, rows
        )


class TwinWorld:
    """Actuator-only twin runtime for graph and stage-local controllers."""

    def __init__(
        self,
        scene: str | Path | mujoco.MjModel,
        reconstruction: str | Path | Mapping[str, Any],
        reference: str | Path | Mapping[str, Any],
        motor_targets: str | Path | Mapping[str, Any] | None = None,
        *,
        width: int = 640,
        height: int = 360,
        control_hz: float = 10.,
        camera: str = "main",
        initial_translation: Mapping[str, Sequence[float]] | None = None,
        min_visible_pixels: int = 12,
        cup_release_control: str = "legacy_object_distance",
        render: bool = True,
    ):
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.model = scene if isinstance(scene, mujoco.MjModel) else mujoco.MjModel.from_xml_path(str(Path(scene)))
        self.data = mujoco.MjData(self.model)
        self.reconstruction = _load_json(reconstruction)
        self.reference = _load_npz(reference)
        self.motor_targets = _load_npz(motor_targets)
        if self.reference is None:
            raise ValueError("reference trajectory is required")
        self.control_hz = float(control_hz)
        self.camera = camera
        self.width, self.height = int(width), int(height)
        if cup_release_control not in ("legacy_object_distance", "portable_tcp_displacement_v1"):
            raise ValueError(f"unsupported cup release control {cup_release_control!r}")
        self.cup_release_control = cup_release_control
        self._initialized = False
        self._ended = False
        self._state_writes_after_initialization = 0
        self._previous_joint_command: np.ndarray | None = None
        self._previous_gripper_command = np.ones(2, dtype=np.float64)
        self._held_by_arm: dict[str, int] = {}
        self._ever_held: set[tuple[str, str]] = set()
        self._ever_loaded: set[tuple[str, str]] = set()
        self._release_pause: dict[str, tuple[int, float, dict[int, float], np.ndarray]] = {}
        self._numerical_integrity_ok = True
        self._numerical_diagnostics: list[dict[str, Any]] = []
        self._native_warning_messages: list[str] = []
        self._previous_warning_callback = mujoco.get_mju_user_warning()
        def capture_warning(message: str) -> None:
            text = str(message)
            self._native_warning_messages.append(text)
            if any(name in text for name in (
                    "BADQPOS", "BADQVEL", "BADQACC", "value in QPOS", "value in QVEL",
                    "value in QACC")):
                self._numerical_integrity_ok = False
                self._numerical_diagnostics.append({
                    "kind": "native_mujoco_warning", "message": text,
                    "physical_time_sec": float(self.data.time),
                })
        self._warning_callback = capture_warning
        mujoco.set_mju_user_warning(self._warning_callback)
        self._first_visible_centroid: dict[str, np.ndarray] = {}
        self._visible_motion: dict[str, tuple[np.ndarray, float, float]] = {}
        self._private_goal_true_since: dict[tuple[str, str, str, str], float] = {}
        self._private_role_vertices: dict[str, np.ndarray] = {}
        self._trace: list[dict[str, Any]] = []
        self._build_indices()
        self._apply_initial_translation(initial_translation or {})
        mujoco.mj_forward(self.model, self.data)
        self._initial_object_position = self.data.xpos[self.object_body_ids].copy()
        self._initialize_bookkeeping()
        self.observer = (VisibleRGBD(
            self, width=width, height=height, camera=camera,
            min_visible_pixels=min_visible_pixels) if render else None)
        self.nominal_trajectory = self._build_nominal_trajectory()

    @classmethod
    def from_run(cls, run: str | Path, **kwargs: Any) -> "TwinWorld":
        root = Path(run)
        motor = root / "motor_targets.npz"
        return cls(root / "scene.xml", root / "reconstruction.json", root / "reference.npz",
                   motor if motor.is_file() else None, **kwargs)

    def _build_indices(self) -> None:
        qpos, dof, actuator = [], [], []
        finger_rows = []
        for arm_index, arm in enumerate(ARMS):
            for index in range(1, 7):
                joint = self.model.joint(f"{arm}_j{index}")
                qpos.append(joint.qposadr[0]); dof.append(joint.dofadr[0])
                actuator.append(self.model.actuator(f"{arm}_motor{index}").id)
            fingers = []
            for index, sign in ((1, -1), (2, 1)):
                joint = self.model.joint(f"{arm}_gripper{index}")
                aid = self.model.actuator(f"{arm}_gripper_motor{index}").id
                fingers.append((joint, aid, arm_index, sign))
            finger_rows.append(fingers)
        self.arm_qpos_indices = np.asarray(qpos, dtype=np.int32)
        self.arm_dof_indices = np.asarray(dof, dtype=np.int32)
        self.arm_actuator_indices = np.asarray(actuator, dtype=np.int32)
        self.fingers = tuple(item for row in finger_rows for item in row)
        self.finger_actuator_indices = np.asarray([item[1] for item in self.fingers], dtype=np.int32)
        names = [self.model.body(i).name or "" for i in range(self.model.nbody)]
        object_names = [name for name in names if name.startswith("object_")]
        self.object_body_ids = np.asarray([self.model.body(name).id for name in object_names], dtype=np.int32)
        self.object_body_names = tuple(object_names)
        self.role_body_ids = {name.removeprefix("object_"): self.model.body(name).id for name in object_names}
        self.role_geom_ids: dict[str, frozenset[int]] = {}
        for role, body_id in self.role_body_ids.items():
            ids = []
            for geom_id in range(self.model.ngeom):
                current = int(self.model.geom_bodyid[geom_id])
                while current > 0 and current != body_id:
                    current = int(self.model.body_parentid[current])
                if current == body_id:
                    ids.append(geom_id)
            self.role_geom_ids[role] = frozenset(ids)
        self.finger_body_map = {
            self.model.body(f"{arm}_finger{i}").id: (arm, i)
            for arm in ARMS for i in (1, 2)
        }
        self.object_specs = {
            self.model.body(f"object_{row['name']}").id: row
            for row in self.reconstruction.get("objects", [])
            if f"object_{row['name']}" in object_names
        }

    def _apply_initial_translation(self, translations: Mapping[str, Sequence[float]]) -> None:
        # Capture the unmodified kinematic frame once.  Requested values are
        # world translations, whereas ``body_pos`` is expressed by its parent.
        mujoco.mj_forward(self.model, self.data)
        requested: dict[int, np.ndarray] = {}
        role_by_body: dict[int, str] = {}
        for role, value in translations.items():
            canonical_role = str(role).removeprefix("object_")
            delta = _array(value, (3,), f"initial_translation[{role}]")
            body = self.model.body(f"object_{canonical_role}")
            requested[int(body.id)] = delta
            role_by_body[int(body.id)] = canonical_role
        self._requested_initial_translation_world = {
            role_by_body[body_id]: delta.copy() for body_id, delta in requested.items()
        }

        def depth(body_id: int) -> int:
            value = 0
            while body_id:
                value += 1
                body_id = int(self.model.body_parentid[body_id])
            return value

        parent_world_rotation = {
            body_id: self.data.xmat[int(self.model.body_parentid[body_id])].reshape(3, 3).copy()
            for body_id in requested
        }
        for body_id in sorted(requested, key=depth):
            delta = requested[body_id]
            free_joint = None
            joint_start = int(self.model.body_jntadr[body_id])
            joint_count = int(self.model.body_jntnum[body_id])
            for joint_id in range(joint_start, joint_start + joint_count):
                if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    free_joint = joint_id
                    break
            if free_joint is not None:
                address = int(self.model.jnt_qposadr[free_joint])
                self.data.qpos[address:address + 3] += delta
            else:
                ancestor = int(self.model.body_parentid[body_id])
                inherited_world = np.zeros(3, dtype=np.float64)
                while ancestor:
                    if ancestor in requested:
                        inherited_world = requested[ancestor]
                        break
                    ancestor = int(self.model.body_parentid[ancestor])
                residual_world = delta - inherited_world
                residual_parent = residual_world @ parent_world_rotation[body_id]
                self.model.body_pos[body_id] += residual_parent

    def _initialize_bookkeeping(self) -> None:
        self._peak_lift = {name: 0. for name in self.object_body_names}
        self._bilateral_steps = {name: {arm: 0 for arm in ARMS} for name in self.object_body_names}
        self._force_bilateral_steps = {name: {arm: 0 for arm in ARMS} for name in self.object_body_names}
        self._peak_force = {name: {arm: 0. for arm in ARMS} for name in self.object_body_names}
        self._peak_penetration = {key: 0. for key in (
            "robot_object", "object_object", "robot_table", "robot_self",
            "robot_robot", "object_table", "compliant_bristles")}
        self._maximum_workspace_displacement = 0.
        self._maximum_joint_limit_violation = 0.
        self._joint_limit_excursion = {}
        self._limited_joints = np.asarray([
            j for j in range(self.model.njnt)
            if self.model.jnt_limited[j] and self.model.jnt_type[j] in
            (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        ], dtype=np.int32)
        self._limited_qpos = self.model.jnt_qposadr[self._limited_joints]
        self._finger_targets: dict[int, float] = {}
        self._current_held = {
            name: {arm: False for arm in ARMS} for name in self.object_body_names
        }
        self._current_loaded = {
            name: {arm: False for arm in ARMS} for name in self.object_body_names
        }
        self._current_object_contacts: set[frozenset[int]] = set()
        self._current_object_contact_geoms: dict[frozenset[int], frozenset[str]] = {}
        self._current_object_table_contacts: set[int] = set()

    def _set_robot_state(self, joints: np.ndarray, gripper: np.ndarray, qvelocity: np.ndarray | None) -> None:
        self.data.qpos[self.arm_qpos_indices] = joints.reshape(-1)
        self.data.qvel[self.arm_dof_indices] = 0. if qvelocity is None else qvelocity.reshape(-1)
        travel = float(self.model.numeric("gripper_closing_travel").data[0])
        for joint, aid, arm_index, sign in self.fingers:
            target = sign * travel * (1 - np.clip(gripper[arm_index], 0, 1))
            self.data.qpos[joint.qposadr[0]] = target
            self.data.qvel[joint.dofadr[0]] = 0.
            self.data.ctrl[aid] = target
        self.data.ctrl[self.arm_actuator_indices] = joints.reshape(-1)

    def initialize(self, joints: Any, gripper: Any, *, qvelocity: Any | None = None) -> Proprioception:
        if self._initialized:
            raise RuntimeError("TwinWorld may be initialized only once")
        q = _array(joints, (2, 6), "joints")
        g = _array(gripper, (2,), "gripper")
        v = None if qvelocity is None else _array(qvelocity, (2, 6), "qvelocity")
        self._set_robot_state(q, g, v)
        mujoco.mj_forward(self.model, self.data)
        self._finger_targets = {aid: float(self.data.qpos[joint.qposadr[0]]) for joint, aid, _, _ in self.fingers}
        self._previous_joint_command = q.copy()
        self._previous_gripper_command = g.copy()
        self._initialized = True
        self._sample_trace(q)
        return self.proprioception()

    def _build_nominal_trajectory(self) -> NominalTrajectory:
        source = self.motor_targets or self.reference
        times = _array(source["timestamps"], (len(source["timestamps"]),), "timestamps")
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("nominal timestamps must be strictly increasing")
        joints = np.asarray(source["joints"], dtype=np.float64)
        gripper = np.asarray(source["gripper"], dtype=np.float64)
        if joints.shape != (len(times), 2, 6) or gripper.shape != (len(times), 2):
            raise ValueError("nominal trajectory has invalid shape")
        scratch = mujoco.MjData(self.model)
        positions = np.empty((len(times), 2, 3), dtype=np.float64)
        rotations = np.empty((len(times), 2, 3, 3), dtype=np.float64)
        for row in range(len(times)):
            scratch.qpos[self.arm_qpos_indices] = joints[row].reshape(-1)
            travel = float(self.model.numeric("gripper_closing_travel").data[0])
            for joint, _, arm_index, sign in self.fingers:
                scratch.qpos[joint.qposadr[0]] = sign * travel * (1 - np.clip(gripper[row, arm_index], 0, 1))
            mujoco.mj_forward(self.model, scratch)
            for arm_index, arm in enumerate(ARMS):
                site = scratch.site(f"{arm}_tcp")
                positions[row, arm_index] = site.xpos
                rotations[row, arm_index] = site.xmat.reshape(3, 3)
        return NominalTrajectory(times.copy(), joints.copy(), gripper.copy(), positions, rotations)

    def proprioception(self) -> Proprioception:
        self._require_initialized()
        positions, rotations, fingers = [], [], []
        for arm in ARMS:
            site = self.data.site(f"{arm}_tcp")
            positions.append(site.xpos.copy()); rotations.append(site.xmat.reshape(3, 3).copy())
            fingers.append([self.data.qpos[self.model.joint(f"{arm}_gripper{i}").qposadr[0]] for i in (1, 2)])
        return Proprioception(
            np.asarray(positions), np.asarray(rotations),
            self.data.qpos[self.arm_qpos_indices].reshape(2, 6).copy(),
            self.data.qvel[self.arm_dof_indices].reshape(2, 6).copy(),
            np.asarray(fingers),
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("TwinWorld must be initialized first")
        if self._ended:
            raise RuntimeError("TwinWorld episode has ended")

    def _servo(self, q: np.ndarray, g: np.ndarray, v: np.ndarray) -> None:
        self.data.ctrl[self.arm_actuator_indices] = np.clip(
            ARM_KP * (q.reshape(-1) - self.data.qpos[self.arm_qpos_indices]) +
            ARM_KD * (v.reshape(-1) - self.data.qvel[self.arm_dof_indices]) +
            self.data.qfrc_bias[self.arm_dof_indices], -100., 100.)
        finger_force = float(self.reconstruction.get("finger_force_limit_N", 8.))
        damping = float(self.reconstruction.get("finger_velocity_damping_Ns_m", 8.))
        opening_speed = float(self.reconstruction.get("opening_velocity_limit_m_s", 1e6))
        open_buffer = float(self.reconstruction.get("opening_stop_buffer_m", 0.))
        travel = float(self.model.numeric("gripper_closing_travel").data[0])
        for arm_index, arm in enumerate(ARMS):
            if g[arm_index] > .05 and self._previous_gripper_command[arm_index] <= .05 and arm in self._held_by_arm:
                object_id = self._held_by_arm[arm]
                asset = self.object_specs.get(object_id, {}).get("asset", "")
                if asset == "025_mug" or asset.startswith("065-"):
                    clearance = float(self.reconstruction.get("cup_release_clearance_m", .001))
                    aperture = {
                        i: float(self.data.qpos[self.model.joint(f"{arm}_gripper{i}").qposadr[0]]) +
                        (clearance if i == 1 else -clearance) for i in (1, 2)
                    }
                    self._release_pause[arm] = (
                        object_id, float(self.data.time), aperture,
                        self.data.site(f"{arm}_tcp").xpos.copy())
            if g[arm_index] <= .05:
                self._release_pause.pop(arm, None)
            if arm in self._release_pause:
                object_id, started, _, release_tcp = self._release_pause[arm]
                if self.cup_release_control == "portable_tcp_displacement_v1":
                    clearance_reached = np.linalg.norm(
                        self.data.site(f"{arm}_tcp").xpos - release_tcp) > .15
                else:
                    # Historical compatibility only.  v02 must explicitly use
                    # portable_tcp_displacement_v1, which never reads object pose.
                    clearance_reached = np.linalg.norm(
                        self.data.site(f"{arm}_tcp").xpos - self.data.body(object_id).xpos) > .15
                if clearance_reached or self.data.time - started > 1.:
                    self._release_pause.pop(arm, None)
        self._previous_gripper_command = g.copy()
        dt = float(self.model.opt.timestep)
        for joint, aid, arm_index, sign in self.fingers:
            arm = ARMS[arm_index]
            target = sign * (open_buffer + (travel - open_buffer) * (1 - np.clip(g[arm_index], 0, 1)))
            if arm in self._release_pause:
                aperture = self._release_pause[arm][2]
                target = sign * np.clip(sign * aperture[1 if sign == -1 else 2], open_buffer, travel)
            closure = max(sign * target, sign * self._finger_targets[aid] - opening_speed * dt)
            target = sign * closure
            self._finger_targets[aid] = target
            qa, da = joint.qposadr[0], joint.dofadr[0]
            self.data.ctrl[aid] = np.clip(
                1600 * (target - self.data.qpos[qa]) - damping * self.data.qvel[da] + self.data.qfrc_bias[da],
                -finger_force, finger_force)

    def _update_bookkeeping(self) -> None:
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self._numerical_integrity_ok = False
            return
        touching: dict[tuple[int, str], set[int]] = {}
        loaded: dict[tuple[int, str], set[int]] = {}
        force_sums: dict[tuple[int, str], float] = {}
        object_contacts: set[frozenset[int]] = set()
        object_contact_geoms: dict[frozenset[int], set[str]] = {}
        object_table_contacts: set[int] = set()
        object_set = set(map(int, self.object_body_ids))
        soft = {i for i in range(self.model.ngeom) if (self.model.geom(i).name or "").startswith("broom_collision_") and self.model.geom_solref[i, 0] > .02}
        for contact_id, contact in enumerate(self.data.contact):
            if min(contact.geom1, contact.geom2) < 0:
                continue
            b1, b2 = (int(self.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            pair = None
            if b1 in self.finger_body_map and b2 in object_set:
                pair = (b2, *self.finger_body_map[b1])
            elif b2 in self.finger_body_map and b1 in object_set:
                pair = (b1, *self.finger_body_map[b2])
            if pair:
                oid, arm, finger = pair
                touching.setdefault((oid, arm), set()).add(finger)
                force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, contact_id, force)
                if force[0] > .01:
                    loaded.setdefault((oid, arm), set()).add(finger)
                force_sums[(oid, arm)] = force_sums.get((oid, arm), 0.) + float(force[0])
            if b1 in object_set and b2 in object_set:
                force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, contact_id, force)
                if force[0] > .01:
                    object_pair = frozenset((b1, b2))
                    object_contacts.add(object_pair)
                    object_contact_geoms.setdefault(object_pair, set()).update(
                        (self.model.geom(contact.geom1).name or "",
                         self.model.geom(contact.geom2).name or ""))
            robot1 = (self.model.body(b1).name or "").startswith(("left_", "right_"))
            robot2 = (self.model.body(b2).name or "").startswith(("left_", "right_"))
            table = (self.model.geom(contact.geom1).name == "table" or self.model.geom(contact.geom2).name == "table")
            if contact.geom1 in soft or contact.geom2 in soft: kind = "compliant_bristles"
            elif robot1 and robot2: kind = "robot_self" if (self.model.body(b1).name.startswith("left_")) == (self.model.body(b2).name.startswith("left_")) else "robot_robot"
            elif (robot1 and b2 in object_set) or (robot2 and b1 in object_set): kind = "robot_object"
            elif b1 in object_set and b2 in object_set: kind = "object_object"
            elif (robot1 or robot2) and table: kind = "robot_table"
            elif (b1 in object_set or b2 in object_set) and table: kind = "object_table"
            else: continue
            if kind == "object_table":
                force = np.zeros(6); mujoco.mj_contactForce(self.model, self.data, contact_id, force)
                if force[0] > .01:
                    object_table_contacts.add(b1 if b1 in object_set else b2)
            self._peak_penetration[kind] = max(self._peak_penetration[kind], max(0., -float(contact.dist)))
        for name in self.object_body_names:
            for arm in ARMS:
                self._current_held[name][arm] = False
                self._current_loaded[name][arm] = False
        self._current_object_contacts = object_contacts
        self._current_object_contact_geoms = {
            pair: frozenset(names) for pair, names in object_contact_geoms.items()
        }
        self._current_object_table_contacts = object_table_contacts
        for (oid, arm), fingers in touching.items():
            if len(fingers) == 2:
                self._bilateral_steps[self.model.body(oid).name][arm] += 1
        for (oid, arm), fingers in loaded.items():
            name = self.model.body(oid).name
            if fingers:
                self._current_loaded[name][arm] = True
                self._ever_loaded.add((name, arm))
            if len(fingers) == 2:
                self._force_bilateral_steps[name][arm] += 1
                self._held_by_arm[arm] = oid
                self._current_held[name][arm] = True
                self._ever_held.add((name, arm))
            self._peak_force[name][arm] = max(self._peak_force[name][arm], force_sums[(oid, arm)])
        displacement = self.data.xpos[self.object_body_ids] - self._initial_object_position
        self._maximum_workspace_displacement = max(self._maximum_workspace_displacement, float(np.linalg.norm(displacement, axis=1).max(initial=0.)))
        for index, name in enumerate(self.object_body_names):
            self._peak_lift[name] = max(self._peak_lift[name], float(displacement[index, 2]))
        if len(self._limited_joints):
            ranges = self.model.jnt_range[self._limited_joints]
            values = self.data.qpos[self._limited_qpos]
            excursions = np.maximum(ranges[:, 0] - values, values - ranges[:, 1])
            self._maximum_joint_limit_violation = max(self._maximum_joint_limit_violation, float(excursions.max(initial=0.)))
            for joint_id, excursion in zip(self._limited_joints, excursions):
                name = self.model.joint(int(joint_id)).name
                self._joint_limit_excursion[name] = max(
                    self._joint_limit_excursion.get(name, 0.), max(0., float(excursion)))

    def _sample_trace(self, command: np.ndarray) -> None:
        self._trace.append({
            "physical_time_sec": float(self.data.time),
            "sample_monotonic_sec": time.monotonic(),
            "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(), "commanded_joints": command.copy(),
        })

    def _record_numerical_step(self, before_time: float) -> None:
        now = float(self.data.time)
        arrays = {
            "qpos": self.data.qpos,
            "qvel": self.data.qvel,
            "qacc": self.data.qacc,
        }
        nonfinite = [name for name, value in arrays.items() if not np.isfinite(value).all()]
        if nonfinite:
            self._numerical_integrity_ok = False
            self._numerical_diagnostics.append({
                "kind": "nonfinite_state", "arrays": nonfinite,
                "physical_time_sec": now,
            })
        expected = before_time + float(self.model.opt.timestep)
        if now + 1e-12 < expected:
            self._numerical_integrity_ok = False
            self._numerical_diagnostics.append({
                "kind": "physical_time_regression", "before_time_sec": before_time,
                "after_time_sec": now, "expected_min_time_sec": expected,
            })

    def step_command_callback(
        self,
        callback: Callable[[float], tuple[Any, Any, Any | None]],
        duration_sec: float,
        *,
        observe: bool = True,
        roles: Sequence[str] | None = None,
        include_endpoint: bool = False,
    ) -> StepResult:
        self._require_initialized()
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        start, end = float(self.data.time), float(self.data.time + duration_sec)
        steps = 0; last_q = None
        step_limit = int(np.ceil(duration_sec / self.model.opt.timestep))
        if include_endpoint:
            step_limit += 1
        while steps < step_limit:
            q_value, g_value, v_value = callback(min(float(self.data.time), end))
            q = _array(q_value, (2, 6), "joints")
            g = _array(g_value, (2,), "gripper")
            if v_value is None:
                basis = self._previous_joint_command if self._previous_joint_command is not None else q
                v = (q - basis) / max(float(self.model.opt.timestep), end - start)
            else:
                v = _array(v_value, (2, 6), "qvelocity")
            self._servo(q, g, v)
            before_step_time = float(self.data.time)
            mujoco.mj_step(self.model, self.data)
            self._record_numerical_step(before_step_time)
            self._update_bookkeeping()
            self._previous_joint_command = q.copy(); last_q = q; steps += 1
            if self.cup_release_control == "portable_tcp_displacement_v1" and not self._numerical_integrity_ok:
                break
        if last_q is None:
            raise RuntimeError("no physics step executed")
        self._sample_trace(last_q)
        visible = self.observe_visible(roles) if observe else None
        return StepResult(float(self.data.time), time.monotonic(), steps, self.proprioception(),
                          self.contact_evidence(), visible, self.numerical_safety())

    def step_command(
        self,
        joints: Any,
        gripper: Any,
        qvelocity: Any | None = None,
        *,
        duration_sec: float | None = None,
        observe: bool = True,
        roles: Sequence[str] | None = None,
    ) -> StepResult:
        q = _array(joints, (2, 6), "joints")
        g = _array(gripper, (2,), "gripper")
        duration = 1 / self.control_hz if duration_sec is None else float(duration_sec)
        if qvelocity is None:
            basis = self._previous_joint_command if self._previous_joint_command is not None else q
            v = (q - basis) / duration
        else:
            v = _array(qvelocity, (2, 6), "qvelocity")
        return self.step_command_callback(lambda _: (q, g, v), duration, observe=observe, roles=roles)

    def observe_visible(self, roles: Sequence[str] | None = None) -> VisibleRGBDFrame:
        self._require_initialized()
        if self.observer is None:
            raise RuntimeError("visible observer disabled")
        return self.observer.observe(roles)

    def jacobian(self, arm: str, joints: Any | None = None) -> np.ndarray:
        """Return the TCP geometric Jacobian from robot state or scratch FK."""
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        self._require_initialized()
        arm_index = ARMS.index(arm)
        state = self.data
        if joints is not None:
            q = _array(joints, (6,), "joints")
            state = mujoco.MjData(self.model)
            state.qpos[:] = self.model.qpos0
            indices = self.arm_qpos_indices.reshape(2, 6)[arm_index]
            state.qpos[indices] = q
            mujoco.mj_forward(self.model, state)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, state, jacp, jacr, self.model.site(f"{arm}_tcp").id)
        dofs = self.arm_dof_indices.reshape(2, 6)[arm_index]
        return np.vstack((jacp[:, dofs], jacr[:, dofs]))

    def solve_ik_delta(
        self,
        arm: str,
        delta_pose: Any,
        joints: Any | None = None,
        *,
        damping: float = 1e-5,
        max_step: float = .08,
    ) -> np.ndarray:
        """Apply one damped least-squares pose increment without mutating state."""
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        delta = _array(delta_pose, (6,), "delta_pose")
        if damping <= 0 or max_step <= 0:
            raise ValueError("damping and max_step must be positive")
        current = (self.proprioception().joint_pos[ARMS.index(arm)] if joints is None
                   else _array(joints, (6,), "joints"))
        J = self.jacobian(arm, current)
        update = np.linalg.solve(J.T @ J + np.eye(6) * damping, J.T @ delta)
        return current + np.clip(update, -max_step, max_step)

    def visible_stage_facts(
        self,
        role: str,
        frame: VisibleRGBDFrame,
        *,
        arm: str,
        support: str | None = None,
        near_distance_m: float = .08,
        lift_delta_m: float = .04,
        relation_distance_m: float = .10,
        stabilized_sec: float = .20,
        goal_relation: str | None = None,
    ) -> dict[str, bool]:
        """Derive stage booleans only from visible geometry and contact flags."""
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if min(near_distance_m, lift_delta_m, relation_distance_m, stabilized_sec) < 0:
            raise ValueError("stage thresholds must be nonnegative")
        if role not in frame.roles:
            raise KeyError(f"role {role!r} was not requested in this visible frame")
        observation = frame.roles[role]
        support_observation = None if support is None else frame.roles.get(support)
        visible = bool(observation.present and observation.centroid_world_m is not None)
        if visible and role not in self._first_visible_centroid:
            self._first_visible_centroid[role] = observation.centroid_world_m.copy()
        visual_stable = False
        if visible:
            previous = self._visible_motion.get(role)
            if previous is None:
                stable_since = frame.physical_time_sec
            else:
                prior_centroid, _, stable_since = previous
                if np.linalg.norm(observation.centroid_world_m - prior_centroid) > .005:
                    stable_since = frame.physical_time_sec
            self._visible_motion[role] = (
                observation.centroid_world_m.copy(), frame.physical_time_sec, stable_since)
            visual_stable = frame.physical_time_sec - stable_since >= stabilized_sec
        tcp = self.proprioception().tcp_position[ARMS.index(arm)]
        near = bool(visible and np.linalg.norm(observation.centroid_world_m - tcp) <= near_distance_m)
        lifted = bool(
            visible and role in self._first_visible_centroid and
            observation.centroid_world_m[2] - self._first_visible_centroid[role][2] >= lift_delta_m
        )
        relation = bool(
            visible and support_observation is not None and support_observation.present and
            support_observation.centroid_world_m is not None and
            np.linalg.norm(observation.centroid_world_m - support_observation.centroid_world_m)
            <= relation_distance_m
        )
        name = f"object_{role}"
        known = name in self._force_bilateral_steps
        grip_dwell = bool(known and self._force_bilateral_steps[name][arm] * self.model.opt.timestep >= stabilized_sec)
        released = bool(known and (name, arm) in self._ever_held and not self._current_held[name][arm])
        loaded_released = bool(known and (name, arm) in self._ever_loaded and not self._current_loaded[name][arm])
        supported_contact = False
        if support is not None and f"object_{support}" in self.object_body_names and name in self.object_body_names:
            supported_contact = frozenset((self.model.body(name).id, self.model.body(f"object_{support}").id)) in self._current_object_contacts
        table_supported = bool(
            known and self.model.body(name).id in self._current_object_table_contacts)
        stabilized = bool(released and visual_stable)
        facts = {
            "visible": visible,
            "observation_missing": not visible,
            "near": near,
            "lifted": lifted,
            "relation": relation,
            "stabilized": stabilized,
            "grip_dwell": grip_dwell,
            "released": released,
            "loaded_released": loaded_released,
            "loaded_contact": bool(known and self._current_loaded[name][arm]),
            "supported_contact": bool(supported_contact),
            # Stable graph vocabulary aliases.
            "near_object": near,
            "bilateral_contact": bool(known and self._current_held[name][arm]),
            "single_finger_loaded_contact": bool(
                known and self._current_loaded[name][arm]),
            "object_support_contact": bool(supported_contact),
            "object_table_support_contact": table_supported,
            "lift_clear": lifted,
            "near_target": relation,
            "object_stabilized": stabilized,
            "finite_simulation": bool(self.numerical_safety()["simulation_finite"]),
        }
        if goal_relation is not None:
            facts[str(goal_relation)] = relation
        return facts

    def _private_local_point(self, support: str, point_world: np.ndarray) -> np.ndarray:
        """Evaluator-side geometry helper; callers expose only Boolean results."""
        body = self.data.body(f"object_{support}")
        return (np.asarray(point_world) - body.xpos) @ body.xmat.reshape(3, 3)

    def _private_role_mesh_vertices(self, role: str) -> np.ndarray:
        """Cache constant body-local visual mesh vertices for Boolean judges."""
        if role in self._private_role_vertices:
            return self._private_role_vertices[role]
        if role not in self.role_geom_ids:
            raise KeyError(f"unknown semantic role {role!r}")
        body = self.data.body(f"object_{role}")
        visual = [g for g in self.role_geom_ids[role]
                  if int(self.model.geom_group[g]) == 2 and self.model.geom_rgba[g, 3] > 0]
        candidates = visual or list(self.role_geom_ids[role])
        rows: list[np.ndarray] = []
        for geom_id in candidates:
            geom_type = int(self.model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh_id = int(self.model.geom_dataid[geom_id])
                start = int(self.model.mesh_vertadr[mesh_id])
                count = int(self.model.mesh_vertnum[mesh_id])
                local = self.model.mesh_vert[start:start + count]
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                size = self.model.geom_size[geom_id]
                local = np.asarray([
                    (x, y, z) for x in (-size[0], size[0])
                    for y in (-size[1], size[1]) for z in (-size[2], size[2])
                ])
            else:
                continue
            world = (local @ self.data.geom_xmat[geom_id].reshape(3, 3).T +
                     self.data.geom_xpos[geom_id])
            rows.append((world - body.xpos) @ body.xmat.reshape(3, 3))
        if not rows:
            raise ValueError(f"role {role!r} has no usable visual mesh geometry")
        vertices = np.concatenate(rows)
        self._private_role_vertices[role] = vertices
        return vertices

    def _private_supported(self, role: str, support: str,
                           geom_prefix: str | None = None) -> bool:
        pair = frozenset((self.role_body_ids[role], self.role_body_ids[support]))
        if pair not in self._current_object_contacts:
            return False
        if geom_prefix is None:
            return True
        return any(name.startswith(geom_prefix)
                   for name in self._current_object_contact_geoms.get(pair, ()))

    def _private_in_container(self, role: str, support: str) -> bool | None:
        geometry = self.reconstruction.get("container_collision_geometry", {}).get(support)
        if geometry is None:
            return None
        point = self._private_local_point(support, self.data.body(f"object_{role}").xpos)
        axis = np.asarray(geometry["axis_xy_m"], dtype=np.float64)
        radius = float(geometry["outer_radius_m"][-1]) - float(geometry["wall_thickness_m"])
        profile = np.asarray(geometry["profile_z_m"], dtype=np.float64)
        return bool(np.linalg.norm(point[:2] - axis) < radius and
                    profile[0] - .006 < point[2] < profile[-1] + .025)

    def _private_in_box(self, role: str, support: str, *, box_slack: float = -.006) -> bool:
        vertices = self._private_role_mesh_vertices(support)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        point = self._private_local_point(support, self.data.body(f"object_{role}").xpos)
        return bool(np.all(point[:2] > lo[:2] - box_slack) and
                    np.all(point[:2] < hi[:2] + box_slack) and
                    lo[2] < point[2] < hi[2] + .01)

    def _private_released(self, role: str, arm: str) -> bool:
        name = f"object_{role}"
        return bool((name, arm) in self._ever_loaded and not self._current_loaded[name][arm])

    def _private_debounce(self, key: tuple[str, str, str, str], value: bool | None,
                          debounce_sec: float) -> bool | None:
        if value is not True:
            self._private_goal_true_since.pop(key, None)
            return value
        since = self._private_goal_true_since.setdefault(key, float(self.data.time))
        return bool(float(self.data.time) - since + 1e-12 >= debounce_sec)

    def stage_goal_facts(
        self,
        object_role: str,
        target_role: str,
        *,
        arm: str,
        goal_relation: str,
        debounce_sec: float = 0.,
    ) -> dict[str, bool | None]:
        """Return private Boolean supervisors without exposing continuous truth.

        Geometry and body state are read only inside this judge.  Its result is
        suitable for a stage transition, but is neither a whole-task evaluator
        nor a policy/adapter observation.
        """
        self._require_initialized()
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        if not 0. <= debounce_sec <= .15:
            raise ValueError("debounce_sec must be between 0 and .15")
        if object_role not in self.role_body_ids:
            raise KeyError(f"unknown semantic role {object_role!r}")
        relation = str(goal_relation)
        symbolic_target_relations = {"accessible_to_right_arm", "lifted"}
        if (relation not in symbolic_target_relations and
                target_role not in self.role_body_ids):
            raise KeyError(f"unknown semantic role {target_role!r}")

        stable_keys = (
            "goal_inside", "goal_nested", "goal_supported", "goal_hanging",
            "goal_handover", "goal_drawer_open", "goal_lifted",
            "goal_sweep_inside",
        )
        facts: dict[str, bool | None] = {key: None for key in stable_keys}
        selected_key: str | None = None
        raw: bool | None = None
        released = self._private_released(object_role, arm)

        if relation == "inside":
            selected_key = "goal_inside"
            inside = self._private_in_container(object_role, target_role)
            if inside is None:
                inside = self._private_in_box(object_role, target_role)
                inside = bool(inside and self._private_supported(object_role, target_role))
            raw = bool(released and inside)
        elif relation == "nested_with_support":
            selected_key = "goal_nested"
            inside = self._private_in_container(object_role, target_role)
            if inside is not None:
                vertices = self._private_role_mesh_vertices(object_role)
                moving = self.data.body(f"object_{object_role}")
                world_vertices = vertices @ moving.xmat.reshape(3, 3).T + moving.xpos
                local_vertices = np.asarray([
                    self._private_local_point(target_role, point) for point in world_vertices
                ])
                geometry = self.reconstruction["container_collision_geometry"][target_role]
                insertion = float(geometry["profile_z_m"][-1]) - float(local_vertices[:, 2].min())
                raw = bool(released and inside and insertion > .015 and
                           self._private_supported(object_role, target_role))
        elif relation == "supported":
            selected_key = "goal_supported"
            contact = self._private_supported(object_role, target_role)
            asset = str(self.object_specs[self.role_body_ids[target_role]].get("asset", ""))
            if target_role == "plate" or "plate" in asset:
                spatial = self._private_in_container(object_role, target_role)
            elif target_role == "box" or "box" in asset:
                vertices = self._private_role_mesh_vertices(target_role)
                point = self._private_local_point(
                    target_role, self.data.body(f"object_{object_role}").xpos)
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                spatial = bool(np.all(point[:2] > lo[:2] - .01) and
                               np.all(point[:2] < hi[:2] + .01) and
                               point[2] > hi[2])
            else:
                spatial = True
            raw = bool(released and spatial and contact)
        elif relation == "handle_supported_by_peg":
            selected_key = "goal_hanging"
            pair = frozenset((self.role_body_ids[object_role], self.role_body_ids[target_role]))
            names = self._current_object_contact_geoms.get(pair, ())
            target_asset = str(
                self.object_specs[self.role_body_ids[target_role]].get("asset", ""))
            peg_prefixes = {
                "mug_rack_single_v2": ("rack_collision_2", "rack_collision_3"),
            }.get(target_asset)
            raw = (None if peg_prefixes is None else bool(
                released and any(name.startswith(peg_prefixes) for name in names)))
        elif relation == "accessible_to_right_arm":
            selected_key = "goal_handover"
            raw = bool(released and self.role_body_ids[object_role] in
                       self._current_object_table_contacts)
        elif relation == "drawer_extended":
            selected_key = "goal_drawer_open"
            try:
                address = int(self.model.joint("drawer_slide").qposadr[0])
            except KeyError:
                raw = None
            else:
                raw = bool(self.data.qpos[address] > .035)
        elif relation == "lifted":
            selected_key = "goal_lifted"
            index = self.object_body_names.index(f"object_{object_role}")
            raw = bool(self.data.xpos[self.role_body_ids[object_role], 2] -
                       self._initial_object_position[index, 2] > .04)
        elif relation == "receptacle_ready_and_held":
            name = f"object_{object_role}"
            raw = bool(self._current_held[name][arm])
        elif relation == "block_inside_dustpan_with_tools_held":
            selected_key = "goal_sweep_inside"
            vertices = self._private_role_mesh_vertices(target_role)
            # dustpan_v2 compiles its 23 floor sectors first, as in the
            # canonical asset used by the original evaluator.
            if str(self.object_specs[self.role_body_ids[target_role]].get("asset")) != "dustpan_v2":
                raw = None
            else:
                floor_geoms = sorted(
                    (g for g in self.role_geom_ids[target_role]
                     if (self.model.geom(g).name or "").startswith("dustpan_visual_")),
                    key=lambda g: int((self.model.geom(g).name or "").rsplit("_", 1)[-1]),
                )[:23]
                floor_rows = []
                support = self.data.body(f"object_{target_role}")
                for geom_id in floor_geoms:
                    mesh_id = int(self.model.geom_dataid[geom_id])
                    start = int(self.model.mesh_vertadr[mesh_id])
                    count = int(self.model.mesh_vertnum[mesh_id])
                    world = (self.model.mesh_vert[start:start + count] @
                             self.data.geom_xmat[geom_id].reshape(3, 3).T +
                             self.data.geom_xpos[geom_id])
                    floor_rows.append((world - support.xpos) @ support.xmat.reshape(3, 3))
                floor = np.concatenate(floor_rows) if floor_rows else vertices
                hull = ConvexHull(floor[:, :2])
                point = self._private_local_point(
                    target_role, self.data.body(f"object_{object_role}").xpos)
                inside = bool(np.all(hull.equations[:, :2] @ point[:2] +
                                     hull.equations[:, 2] < -.005) and
                              floor[:, 2].min() + .004 < point[2] < floor[:, 2].min() + .045)
                broom = self._current_held.get("object_broom", {}).get("right", False)
                pan = self._current_held.get(f"object_{target_role}", {}).get("left", False)
                broom_dwell = self._force_bilateral_steps.get("object_broom", {}).get("right", 0)
                pan_dwell = self._force_bilateral_steps.get(f"object_{target_role}", {}).get("left", 0)
                dt = float(self.model.opt.timestep)
                raw = bool(inside and broom and pan and broom_dwell * dt > 3. and pan_dwell * dt > 5.)

        key = (object_role, target_role, arm, relation)
        completed = self._private_debounce(key, raw, float(debounce_sec))
        if selected_key is not None:
            facts[selected_key] = completed
        facts[relation] = completed
        facts["stage_goal_complete"] = completed
        return facts

    def contact_evidence(self) -> dict[str, Any]:
        dt = float(self.model.opt.timestep)
        roles = {}
        for name in self.object_body_names:
            role = name.removeprefix("object_")
            held = copy.deepcopy(self._current_held[name])
            loaded = copy.deepcopy(self._current_loaded[name])
            supported_by = sorted(
                other.removeprefix("object_") for other in self.object_body_names
                if other != name and frozenset((self.model.body(name).id, self.model.body(other).id)) in self._current_object_contacts)
            roles[role] = {
                "held_by": held,
                "loaded_by": loaded,
                "supported_by": supported_by,
                "object_table_support_contact": bool(
                    self.model.body(name).id in self._current_object_table_contacts),
                "bilateral_contact_sec": {arm: self._bilateral_steps[name][arm] * dt for arm in ARMS},
                "bilateral_force_contact_sec": {arm: self._force_bilateral_steps[name][arm] * dt for arm in ARMS},
                "peak_summed_normal_contact_force_N": copy.deepcopy(self._peak_force[name]),
            }
        return {"physical_time_sec": float(self.data.time), "roles": roles}

    def numerical_safety(self) -> dict[str, Any]:
        finite = bool(self._numerical_integrity_ok and np.isfinite(self.data.qpos).all() and
                      np.isfinite(self.data.qvel).all() and np.isfinite(self.data.qacc).all())
        joint_limits_ok = {}
        for joint_id in self._limited_joints:
            name = self.model.joint(int(joint_id)).name
            threshold = .0002 if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_SLIDE else .005
            joint_limits_ok[name] = self._joint_limit_excursion.get(name, 0.) < threshold
        limits_ok = all(joint_limits_ok.values())
        collision_ok = max(self._peak_penetration[k] for k in ("robot_self", "robot_robot", "robot_table")) < .001
        bounded = self._maximum_workspace_displacement < 1.5
        return {
            "simulation_finite": finite,
            "numerical_integrity_ok": bool(self._numerical_integrity_ok),
            "numerical_diagnostics": copy.deepcopy(self._numerical_diagnostics),
            "native_warning_messages": list(self._native_warning_messages),
            "cup_release_control": self.cup_release_control,
            "workspace_bounded": bounded,
            "maximum_workspace_displacement_m": self._maximum_workspace_displacement,
            "maximum_joint_limit_violation": self._maximum_joint_limit_violation,
            "joint_limit_excursion": copy.deepcopy(self._joint_limit_excursion),
            "joint_limits_ok": joint_limits_ok,
            "maximum_penetration_mm": {k: v * 1000 for k, v in self._peak_penetration.items()},
            "passed": bool(finite and bounded and limits_ok and collision_ok),
        }

    def evaluator_trace(self) -> tuple[dict[str, Any], ...]:
        """Evaluator-only full state trace; never pass this to an action builder."""
        return tuple(copy.deepcopy(self._trace))

    def finalize_evaluator(self, task_outcome: Callable[..., Any] | None = None) -> dict[str, Any]:
        self._require_initialized()
        evidence = self.contact_evidence()
        objects = {
            f"object_{role}": {
                "peak_lift_mm": self._peak_lift[f"object_{role}"] * 1000,
                "bilateral_contact_sec": row["bilateral_contact_sec"],
                "bilateral_force_contact_sec": row["bilateral_force_contact_sec"],
                "peak_summed_normal_contact_force_N": row["peak_summed_normal_contact_force_N"],
                "final_position_m": self.data.body(f"object_{role}").xpos.tolist(),
            } for role, row in evidence["roles"].items()
        }
        outcome = None if task_outcome is None else task_outcome(
            self.model, self.data, self.reconstruction, objects)
        result = {
            "task_outcome": outcome,
            "numerical_safety": self.numerical_safety(),
            "contact_evidence": evidence,
            "state_writes_after_initialization": self._state_writes_after_initialization,
            "object_external_forces": bool(np.any(self.data.xfrc_applied[self.object_body_ids])),
            "grasp_welds": int(sum(
                self.model.eq_type[e] == mujoco.mjtEq.mjEQ_WELD and
                (int(self.model.eq_obj1id[e]) in self.object_body_ids or int(self.model.eq_obj2id[e]) in self.object_body_ids)
                for e in range(self.model.neq))),
        }
        self._ended = True
        return result

    def close(self) -> None:
        if self.observer is not None:
            self.observer.close(); self.observer = None
        if mujoco.get_mju_user_warning() is self._warning_callback:
            mujoco.set_mju_user_warning(self._previous_warning_callback)


def quick_nominal_replay(
    run: str | Path,
    *,
    max_duration_sec: float | None = None,
    render: bool = False,
    evaluator: Callable[..., Any] | None = None,
    initial_translation: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Engineering replay of existing motor targets; never calls twin.simulate."""
    world = TwinWorld.from_run(run, render=render, initial_translation=initial_translation)
    trajectory = world.nominal_trajectory
    qfun = PchipInterpolator(trajectory.times, trajectory.joints, axis=0, extrapolate=False)
    vfun = qfun.derivative()
    gfun = lambda t: np.array([np.interp(t, trajectory.times, trajectory.gripper[:, arm]) for arm in range(2)])
    # Canonical replay initializes actual velocity to zero; desired velocity
    # enters through the servo on the first and later physics steps.
    world.initialize(trajectory.joints[0], trajectory.gripper[0])
    trajectory_start = float(trajectory.times[0])
    duration = float(trajectory.times[-1] - trajectory_start)
    if max_duration_sec is not None:
        duration = min(duration, float(max_duration_sec))
    try:
        result = world.step_command_callback(
            lambda t: (
                qfun(min(trajectory_start + t, trajectory.times[-1])),
                gfun(min(trajectory_start + t, trajectory.times[-1])),
                vfun(min(trajectory_start + t, trajectory.times[-1])),
            ),
            duration, observe=render, include_endpoint=True,
        )
        final = world.finalize_evaluator(evaluator)
        return {
            "schema": "graph-self-adapter.quick-nominal-replay.v1",
            "engineering_only": True,
            "physical_time_sec": result.physical_time_sec,
            "physics_steps": result.physics_steps,
            **final,
        }
    finally:
        world.close()
