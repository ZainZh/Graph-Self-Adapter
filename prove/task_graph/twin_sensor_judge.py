"""Causal task-generic stage evidence from public sensors only.

No simulator imports, physical object state, collision bounds or success labels.
Visible surface relations are proxies, never certified physical support/contact.
"""
from dataclasses import dataclass, field
from typing import Mapping
import numpy as np

PACKET_SCHEMA = 'graph-self-adapter.sensor-packet.v1'


@dataclass(frozen=True)
class GapCalibration:
    """Measured affine encoder -> jaw aperture; no inferred object dimensions."""
    offset_m: float
    encoder_gains: tuple[float, float]
    open_gap_m: float
    closed_gap_m: float
    provenance: str

    def aperture(self, encoders):
        return float(self.offset_m + np.dot(self.encoder_gains, encoders))


@dataclass(frozen=True)
class ArmSensors:
    q: np.ndarray
    qdot: np.ndarray
    tcp_position_m: np.ndarray
    tcp_rotation: np.ndarray
    finger_encoders: np.ndarray
    actual_aperture_m: float
    commanded_aperture_m: float
    calibration_source: str
    open_gap_m: float
    closed_gap_m: float
    timestamp_sec: float
    measurement_source: str = "unknown"


@dataclass(frozen=True)
class SensorPacket:
    timestamp_sec: float
    received_sec: float
    role_points_m: Mapping[str, np.ndarray]
    arms: Mapping[str, ArmSensors]
    sources: Mapping[str, str]
    schema: str = PACKET_SCHEMA


@dataclass(frozen=True)
class JudgeConfig:
    max_age_sec: float = .25
    debounce_sec: float = .2
    jaw_radius_m: float = .065
    obstruction_m: float = .003
    gap_stability_m: float = .002
    motion_min_m: float = .006
    comotion_error_m: float = .006
    settled_distance_m: float = .005
    relation_tolerance_m: float = .035
    lift_delta_m: float = .04
    min_pixels: int = 8
    motion_window_sec: float = .6
    rotation_min_rad: float = .10
    max_sensor_skew_sec: float = .05

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in self.__dict__.values()):
            raise ValueError('judge thresholds must be finite and positive')
        if not isinstance(self.min_pixels, int):
            raise ValueError('min_pixels must be an integer')


def rgbd_points(depth_m, masks, camera_K, camera_pose_world, *, min_pixels=8):
    """Unproject observed depth pixels using measured camera calibration."""
    depth = np.asarray(depth_m)
    K, pose = np.asarray(camera_K), np.asarray(camera_pose_world)
    if K.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(pose).all():
        raise ValueError('invalid camera calibration')
    points = {}
    for role, mask in masks.items():
        if np.asarray(mask).shape != depth.shape:
            raise ValueError('mask/depth shape mismatch')
        y, x = np.where(np.asarray(mask, bool) & np.isfinite(depth) & (depth > 0))
        if len(x) < min_pixels:
            continue
        rays = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(K).T
        xyz = rays * depth[y, x, None]
        points[role] = xyz @ pose[:3, :3].T + pose[:3, 3]
    return points


class SensorJudge:
    def __init__(self, config=JudgeConfig()):
        self.config = config
        self.packet = None
        self.valid = False
        self.reason = 'no_packet'
        self.history = {}
        self.baseline = {}
        self.since = {}
        self.last_timestamp = None
        self.diagnostics = {}
        self.anchors = {}
        self.settle_anchors = {}
        self.acquired = set()
        self.basic_cache = {}
        self.now_sec = None

    def ingest(self, packet: SensorPacket, now_sec: float):
        """Exactly once per packet. Duplicate reads never advance debounce."""
        if packet is self.packet:
            return self.expire(now_sec)
        self.packet = packet
        self.now_sec = now_sec
        timestamp = packet.timestamp_sec
        self.valid = (packet.schema == PACKET_SCHEMA and np.isfinite(timestamp)
                      and np.isfinite(now_sec) and np.isfinite(packet.received_sec)
                      and timestamp <= packet.received_sec <= now_sec
                      and 0 <= now_sec - timestamp <= self.config.max_age_sec
                      and (self.last_timestamp is None or timestamp > self.last_timestamp))
        if not self.valid:
            self.reason = 'stale_out_of_order_or_invalid_packet'
            self.since.clear(); self.history.clear(); self.anchors.clear(); self.settle_anchors.clear()
            return False
        self.reason = None
        self.last_timestamp = timestamp
        self.previous_history = dict(self.history)
        self.history = {}
        for key in list(self.anchors):
            if key[0] not in packet.role_points_m:
                self._reset_role(key[0])
        for role, points in packet.role_points_m.items():
            points = np.asarray(points)
            if points.ndim == 2 and points.shape[1] == 3 and len(points) >= self.config.min_pixels and np.isfinite(points).all():
                center = np.median(points, axis=0)
                self.baseline.setdefault(role, center.copy())
                for arm, robot in packet.arms.items():
                    self.history[(role, arm)] = (timestamp, center.copy(),
                        np.asarray(robot.tcp_position_m).copy(), robot.actual_aperture_m)
        return True

    def expire(self, now_sec):
        """Mandatory decision-clock tick: a stopped stream cannot retain True."""
        self.now_sec = now_sec
        if (self.packet is None or not np.isfinite(now_sec)
                or not 0 <= now_sec - self.packet.timestamp_sec <= self.config.max_age_sec):
            self.valid = False
            self.reason = 'expired_or_invalid_decision_clock'
            for store in (self.history, self.anchors, self.settle_anchors, self.since, self.basic_cache):
                store.clear()
        return self.valid

    def _reset_role(self, role):
        for store in (self.history, self.anchors, self.settle_anchors, self.since, self.basic_cache):
            for key in list(store):
                if key[0] == role or isinstance(key[0], tuple) and key[0][0] == role:
                    store.pop(key, None)

    def _unknown(self, keys, reason):
        self.diagnostics = {key: {'value': None, 'reason': reason,
                                  'sources': dict(self.packet.sources) if self.packet else {}}
                            for key in keys}
        return dict.fromkeys(keys)

    def _debounce(self, key, value):
        if value is not True:
            self.since.pop(key, None)
            return value
        start = self.since.setdefault(key, self.packet.timestamp_sec)
        return bool(self.packet.timestamp_sec - start + 1e-9 >= self.config.debounce_sec)

    def _center(self, role):
        points = np.asarray(self.packet.role_points_m.get(role, []))
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < self.config.min_pixels or not np.isfinite(points).all():
            return None
        return np.median(points, axis=0)

    def facts(self, role, arm, *, target=None, relation=None, reference_delta=None):
        keys = ('visible', 'observation_missing', 'near', 'near_object', 'holding',
                'bilateral_contact', 'loaded_contact', 'single_finger_loaded_contact',
                'grip_dwell', 'released', 'loaded_released', 'lifted', 'lift_clear',
                'relation', 'near_target', 'stabilized', 'object_stabilized',
                'supported_contact', 'object_support_contact', 'object_table_support_contact')
        facts = dict.fromkeys(keys)
        if not self.valid:
            return self._unknown(keys, self.reason)
        robot = self.packet.arms.get(arm)
        center = self._center(role)
        if robot is None or center is None:
            self._reset_role(role)
            return self._unknown(keys, 'missing_arm_or_visible_geometry')
        cached = self.basic_cache.get((role, arm))
        if cached is not None and cached[0] == self.packet.timestamp_sec:
            self.diagnostics = {name: dict(row) for name, row in cached[2].items()}
            return self._relations(role, arm, dict(cached[1]), target, relation, reference_delta)
        arrays = ((robot.q, (6,)), (robot.qdot, (6,)), (robot.tcp_position_m, (3,)),
                  (robot.tcp_rotation, (3, 3)))
        scalars = (robot.commanded_aperture_m, robot.open_gap_m, robot.closed_gap_m, robot.timestamp_sec)
        if (any(np.asarray(value).shape != shape or not np.isfinite(value).all() for value, shape in arrays)
                or not np.isfinite(scalars).all() or robot.open_gap_m <= robot.closed_gap_m
                or not robot.calibration_source
                or not np.allclose(robot.tcp_rotation.T @ robot.tcp_rotation, np.eye(3), atol=1e-4)
                or np.linalg.det(robot.tcp_rotation) < .99):
            self._reset_role(role)
            return self._unknown(keys, 'malformed_robot_or_calibration')
        skew = self.packet.timestamp_sec - robot.timestamp_sec
        if skew < 0 or skew > self.config.max_sensor_skew_sec or self.now_sec - robot.timestamp_sec > self.config.max_age_sec:
            self._reset_role(role)
            return self._unknown(keys, 'stale_or_unsynchronized_robot_sample')
        cfg, t = self.config, self.packet.timestamp_sec
        key = (role, arm)
        previous = self.previous_history.get(key)
        if previous is not None and t - previous[0] > cfg.max_age_sec:
            self._reset_role(role)
            previous = None
        local_points = (self.packet.role_points_m[role] - robot.tcp_position_m) @ robot.tcp_rotation
        near = bool(np.any(np.linalg.norm(local_points, axis=1) <= cfg.jaw_radius_m))
        measured = (robot.measurement_source in ('measured_encoder', 'simulator_actual_joint_encoder')
                    and np.asarray(robot.finger_encoders).shape == (2,)
                    and np.isfinite(robot.finger_encoders).all()
                    and isinstance(robot.actual_aperture_m, (float, int, np.floating))
                    and np.isfinite(robot.actual_aperture_m))
        if not measured:
            self._reset_role(role)
            facts = self._unknown(keys, 'actual_gripper_feedback_missing_or_command_placeholder')
            facts.update(visible=True, observation_missing=False, near=near, near_object=near)
            for name in ('visible', 'observation_missing', 'near', 'near_object'):
                self.diagnostics[name]['value'] = facts[name]
                self.diagnostics[name]['reason'] = None
            return facts
        obstructed = bool(robot.actual_aperture_m - robot.commanded_aperture_m >= cfg.obstruction_m)
        if not near or not obstructed:
            self.anchors.pop(key, None)
        anchor = self.anchors.setdefault(key, (t, center.copy(), robot.tcp_position_m.copy(),
                    robot.tcp_rotation.copy(), robot.actual_aperture_m))
        stable_gap = abs(robot.actual_aperture_m - anchor[4]) <= cfg.gap_stability_m
        if not stable_gap:
            self.anchors[key] = (t, center.copy(), robot.tcp_position_m.copy(), robot.tcp_rotation.copy(), robot.actual_aperture_m)
        comotion = None
        if previous is not None:
            object_delta = center - anchor[1]
            tcp_delta = robot.tcp_position_m - anchor[2]
            angle = float(np.arccos(np.clip((np.trace(anchor[3].T @ robot.tcp_rotation) - 1) / 2, -1, 1)))
            excited = np.linalg.norm(tcp_delta) >= cfg.motion_min_m or angle >= cfg.rotation_min_rad
            if excited:
                local_now = (center - robot.tcp_position_m) @ robot.tcp_rotation
                local_then = (anchor[1] - anchor[2]) @ anchor[3]
                comotion = bool(np.linalg.norm(local_now - local_then) <= cfg.comotion_error_m
                                and np.linalg.norm(object_delta) >= cfg.motion_min_m)
        if t - anchor[0] >= cfg.motion_window_sec:
            self.anchors[key] = (t, center.copy(), robot.tcp_position_m.copy(), robot.tcp_rotation.copy(), robot.actual_aperture_m)
        settle = self.settle_anchors.setdefault(key, (t, center.copy()))
        if np.linalg.norm(center - settle[1]) > cfg.settled_distance_m:
            settle = (t, center.copy())
            self.settle_anchors[key] = settle
        settled = bool(previous is not None and t - settle[0] + 1e-9 >= cfg.debounce_sec)
        holding_raw = bool(near and obstructed and stable_gap and comotion is not False)
        holding = self._debounce((key, 'holding'), holding_raw)
        if holding is True:
            self.acquired.add(key)
        # Opening relative to command is direct encoder evidence, not contact release.
        opened = bool(robot.commanded_aperture_m > robot.actual_aperture_m - cfg.gap_stability_m
                      and not obstructed and robot.actual_aperture_m >= robot.closed_gap_m + .8 * (robot.open_gap_m - robot.closed_gap_m))
        released = self._debounce((key, 'released'), bool(opened and key in self.acquired))
        lift = bool(center[2] - self.baseline[role][2] >= cfg.lift_delta_m)
        facts.update(visible=True, observation_missing=False, near=near, near_object=near,
                     holding=holding, bilateral_contact=holding, loaded_contact=holding,
                     single_finger_loaded_contact=holding, grip_dwell=holding,
                     released=released, loaded_released=released, lifted=lift, lift_clear=lift,
                     relation=None, near_target=None,
                     stabilized=self._debounce((key, 'settled'), bool(opened and settled is True)),
                     supported_contact=None, object_support_contact=None)
        facts['object_stabilized'] = facts['stabilized']
        strength = 'motion_verified' if comotion is True else 'gap_obstruction_visible_vicinity_only'
        self.diagnostics = {name: {'value': value, 'sources': dict(self.packet.sources),
                                  'evidence_strength': strength if name in ('holding', 'bilateral_contact', 'loaded_contact', 'grip_dwell') else 'visible_geometry_encoder_proxy',
                                  'physical_contact_verified': False,
                                  'legacy_contact_name_is_sensor_alias_only': name in ('bilateral_contact', 'loaded_contact', 'single_finger_loaded_contact', 'supported_contact', 'object_support_contact'),
                                  'robot_camera_skew_sec': skew,
                                  'robot_sample_timestamp_sec': robot.timestamp_sec,
                                  'timestamp_sec': t, 'age_sec': self.now_sec - t,
                                  'reason': 'unsupported_or_unobservable_relation' if value is None else None}
                            for name, value in facts.items()}
        self.basic_cache[key] = (t, dict(facts), {name: dict(row) for name, row in self.diagnostics.items()})
        return self._relations(role, arm, facts, target, relation, reference_delta)

    def _relations(self, role, arm, facts, target, relation, reference_delta):
        center, target_center = self._center(role), self._center(target) if target else None
        relation_proxy = None
        if center is not None and target_center is not None and reference_delta is not None:
            reference = np.asarray(reference_delta, dtype=float)
            if reference.shape == (3,) and np.isfinite(reference).all():
                relation_proxy = bool(np.linalg.norm(center - target_center - reference) <= self.config.relation_tolerance_m)
        placement = None
        if relation == 'supported' and relation_proxy is not None:
            placement = self._debounce(((role, arm), 'placement', target), bool(
                facts['released'] is True and relation_proxy and facts['stabilized'] is True))
        facts.update(relation=relation_proxy, near_target=relation_proxy,
                     supported_contact=placement, object_support_contact=placement)
        for name in ('relation', 'near_target', 'supported_contact', 'object_support_contact'):
            self.diagnostics[name]['value'] = facts[name]
            self.diagnostics[name]['reason'] = 'unsupported_or_unobservable_relation' if facts[name] is None else None
        return facts

    def handover(self, role, arm, fixed_visible_anchor):
        """Released/settled visible object in a declared offline handover region."""
        values = self.facts(role, arm)
        if not self.valid or fixed_visible_anchor is None or self._center(role) is None:
            return None
        anchor = np.asarray(fixed_visible_anchor, dtype=float)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            return None
        near = np.linalg.norm(self._center(role) - anchor) <= self.config.relation_tolerance_m
        value = None if values['released'] is None or values['stabilized'] is None else bool(
            near and values['released'] and values['stabilized'])
        value = self._debounce(((role, arm), 'handover'), value)
        self.diagnostics['goal_handover'] = {'value': value,
            'source': 'fixed_offline_visible_handover_region_plus_current_RGBD_encoder',
            'prior_is_fresh_measurement': False, 'physical_contact_verified': False}
        return value
