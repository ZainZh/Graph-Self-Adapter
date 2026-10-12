"""Additive simulator sensor proxy. Independent safety/evaluator remain privileged."""
import hashlib
import json
from pathlib import Path
import numpy as np
from prove.task_graph.twin_sensor_judge import (ArmSensors, GapCalibration, SensorPacket,
                                              SensorJudge, rgbd_points)


def sensor_holding_contract(stage, values):
    contract = stage.get('holding_contract', 'none')
    if contract == 'none':
        return True
    if contract == 'bilateral_contact':
        return values.get('bilateral_contact')
    if contract == 'loaded_contact':
        return values.get('single_finger_loaded_contact')
    if contract == 'held_or_supported':
        held = values.get('bilateral_contact')
        supported = (values.get('object_table_support_contact')
            if stage.get('target') == 'table_handover_region' and stage.get('goal_relation') == 'accessible_to_right_arm'
            else values.get('object_support_contact'))
        if held is True or supported is True:
            return True
        return False if held is False and supported is False else None
    return None


def carrying_gate_required(stage):
    return (stage.get('phase') in ('Lift', 'Move', 'Align')
            and stage.get('holding_contract', 'none') != 'none'
            and (any(row.get('predicate') == 'holding' for row in stage.get('invariant_predicates', []))
                 or any(name in ('holding', 'bilateral_contact', 'loaded_contact') for name in stage.get('invariants', []))))


def raw_proprio_servo(world, physics, q, g, v):
    """Original robot gains/limits, no object/contact-dependent release decisions."""
    world.data.ctrl[world.arm_actuator_indices] = np.clip(
        physics.ARM_KP * (q.reshape(-1) - world.data.qpos[world.arm_qpos_indices]) +
        physics.ARM_KD * (v.reshape(-1) - world.data.qvel[world.arm_dof_indices]) +
        world.data.qfrc_bias[world.arm_dof_indices], -100., 100.)
    travel = float(world.model.numeric('gripper_closing_travel').data[0])
    force = float(world.reconstruction.get('finger_force_limit_N', 8.))
    damping = float(world.reconstruction.get('finger_velocity_damping_Ns_m', 8.))
    opening_speed = float(world.reconstruction.get('opening_velocity_limit_m_s', 1e6))
    buffer = float(world.reconstruction.get('opening_stop_buffer_m', 0.))
    for joint, aid, arm_index, sign in world.fingers:
        target = buffer + (travel - buffer) * (1 - np.clip(g[arm_index], 0, 1))
        target = sign * max(target, sign * world._finger_targets[aid] - opening_speed * world.model.opt.timestep)
        world._finger_targets[aid] = target
        qa, da = joint.qposadr[0], joint.dofadr[0]
        world.data.ctrl[aid] = np.clip(1600 * (target - world.data.qpos[qa])
            - damping * world.data.qvel[da] + world.data.qfrc_bias[da], -force, force)
    world._previous_gripper_command = np.asarray(g).copy()


def packet_from_visible(frame, proprio, commands, calibrations):
    if frame.camera_convention != 'opencv_x_right_y_down_z_forward':
        raise ValueError('sensor packet requires explicit OpenCV camera convention')
    points = rgbd_points(frame.depth_m, {role: obs.mask for role, obs in frame.roles.items()},
                         frame.camera_K, frame.camera_pose_world)
    arms = {}
    for index, arm in enumerate(('left', 'right')):
        cal = calibrations[arm]
        arms[arm] = ArmSensors(np.asarray(proprio.joint_pos[index]).copy(),
            np.asarray(proprio.joint_velocity[index]).copy(),
            np.asarray(proprio.tcp_position[index]).copy(), np.asarray(proprio.tcp_rotation[index]).copy(),
            np.asarray(proprio.finger_qpos[index]).copy(), cal.aperture(proprio.finger_qpos[index]),
            cal.closed_gap_m + float(commands[index]) * (cal.open_gap_m - cal.closed_gap_m),
            cal.provenance, cal.open_gap_m, cal.closed_gap_m, float(frame.physical_time_sec), "simulator_actual_joint_encoder")
    timestamp = float(frame.physical_time_sec)
    return SensorPacket(timestamp, timestamp, points, arms, {
        'mask': 'simulator_GT_semantic_mask_proxy_not_external_segmentation',
        'geometry': 'rendered_RGBD_pixels_plus_camera_calibration_only',
        'robot': 'actual_q_qdot_finger_encoders_robot_FK',
        'clock': 'simulator_sensor_timestamp_not_wall_time',
        'aperture': 'encoder_equivalent_sim_proxy_requires_real_measured_calibration'})


def sensor_world_class(physics, *, config, references, judge_mode, seal_path, timeout_sec=1., provenance=None):
    if judge_mode not in ('sensor_only', 'privileged_gt_control'):
        raise ValueError('explicit judge mode required')

    class SensorWorld(physics.TwinWorld):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sensor_judge = SensorJudge(config)
            travel = float(self.model.numeric('gripper_closing_travel').data[0])
            calibration = GapCalibration(2 * travel, (1., -1.), 2 * travel, 0.,
                'sim_encoder_equivalent_opening_not_measured_physical_inner_jaw_gap')
            self.sensor_calibrations = dict(left=calibration, right=calibration)
            self.sensor_stage = None
            self.sensor_runtime = None
            self.sensor_records = []
            self.sensor_packets = {}
            self.sensor_gate_records = []
            self.sensor_last_frame = None
            self.sensor_packet_timestamp = None
            self.sensor_unknown_since = None
            self.sensor_motion_blocked = False
            self.sensor_evaluator_context = False

        def _servo(self, q, g, v):
            raw_proprio_servo(self, physics, q, g, v)

        def contact_evidence(self):
            # Independent evaluator receives its original evidence only after sealing.
            if self.sensor_evaluator_context:
                return super().contact_evidence()
            # StepResult must never expose legacy contact flags to the controller.
            return {'source': 'disabled_sensor_proxy', 'roles': {}, 'judge_mode': judge_mode}

        def observe_visible(self, *args, **kwargs):
            frame = super().observe_visible(*args, **kwargs)
            self.sensor_last_frame = frame
            if self.sensor_packet_timestamp != float(frame.physical_time_sec):
                packet = packet_from_visible(frame, self.proprioception(),
                    self._previous_gripper_command, self.sensor_calibrations)
                self.sensor_judge.ingest(packet, float(self.data.time))
                self.sensor_packet_timestamp = packet.timestamp_sec
                index = len([key for key in self.sensor_packets if key.endswith('_timestamp')])
                prefix = f'packet{index:05d}_'
                self.sensor_packets[prefix + 'timestamp'] = np.array(packet.timestamp_sec)
                for role, points in packet.role_points_m.items():
                    self.sensor_packets[prefix + 'role_' + role] = points.copy()
                for arm, values in packet.arms.items():
                    for key in ('q', 'qdot', 'tcp_position_m', 'tcp_rotation', 'finger_encoders',
                                'actual_aperture_m', 'commanded_aperture_m', 'timestamp_sec'):
                        self.sensor_packets[prefix + arm + '_' + key] = np.asarray(getattr(values, key)).copy()
            return frame

        def _sensor_facts(self, role, arm, support=None, relation=None):
            self.sensor_judge.expire(float(self.data.time))
            stage = self.sensor_stage or {}
            reference = references.get('stage_relations', {}).get(stage.get('stage_id'), {})
            values = self.sensor_judge.facts(role, arm, target=support, relation=relation,
                                           reference_delta=reference.get('object_target_end'))
            if relation == 'accessible_to_right_arm':
                prior = references.get('handover_anchors', {}).get(stage.get('cycle_id'), {})
                anchor = prior.get('centroid_world_m', reference.get('target_end'))
                values['object_table_support_contact'] = self.sensor_judge.handover(role, arm, anchor)
                self.sensor_judge.diagnostics['object_table_support_contact'] = {
                    'value': values['object_table_support_contact'],
                    'source': 'fixed_visible_handover_region_plus_sensor_released_settled',
                    'legacy_contact_name_is_sensor_alias_only': True,
                    'physical_contact_verified': False, 'prior_is_fresh_measurement': False}
            self.sensor_records.append({'timestamp_sec': float(self.data.time),
                'stage_id': stage.get('stage_id'), 'role': role, 'arm': arm,
                'facts': values, 'diagnostics': self.sensor_judge.diagnostics})
            return values

        def visible_stage_facts(self, role, frame, *, arm, support=None, goal_relation=None, **kwargs):
            if judge_mode == 'privileged_gt_control':
                self._sensor_facts(role, arm, support, goal_relation)
                return super().visible_stage_facts(role, frame, arm=arm, support=support,
                                                  goal_relation=goal_relation, **kwargs)
            # Never call legacy visible_stage_facts in sensor mode.
            values = self._sensor_facts(role, arm, support, goal_relation)
            values['finite_simulation'] = bool(self.numerical_safety()['simulation_finite'])
            return values

        def stage_goal_facts(self, object_role, target_role, *, arm, goal_relation, **kwargs):
            values = self._sensor_facts(object_role, arm, target_role, goal_relation)
            out = dict.fromkeys(('goal_inside', 'goal_nested', 'goal_supported', 'goal_hanging',
                                 'goal_handover', 'goal_drawer_open', 'goal_lifted', 'goal_sweep_inside'))
            if goal_relation == 'supported':
                out['goal_supported'] = values['supported_contact']
            elif goal_relation == 'lifted':
                out['goal_lifted'] = values['lifted']
            elif goal_relation == 'accessible_to_right_arm':
                stage = self.sensor_stage or {}
                prior = references.get('handover_anchors', {}).get(stage.get('cycle_id'), {})
                anchor = prior.get('centroid_world_m')
                if anchor is None:
                    anchor = references.get('stage_relations', {}).get(stage.get('stage_id'), {}).get('target_end')
                out['goal_handover'] = self.sensor_judge.handover(object_role, arm, anchor)
            selected = {'supported': 'goal_supported', 'lifted': 'goal_lifted',
                        'accessible_to_right_arm': 'goal_handover'}.get(goal_relation)
            value = out.get(selected) if selected else None
            out[goal_relation] = out['stage_goal_complete'] = value
            self.sensor_records.append({'timestamp_sec': float(self.data.time),
                'stage_goal_requested_relation': goal_relation, 'values': out,
                'diagnostics': self.sensor_judge.diagnostics})
            if judge_mode == 'privileged_gt_control':
                return super().stage_goal_facts(object_role, target_role, arm=arm, goal_relation=goal_relation, **kwargs)
            return out

        def step_command_callback(self, callback, duration_sec, *args, **kwargs):
            stage = self.sensor_stage or {}
            self.sensor_motion_blocked = False
            if judge_mode == 'sensor_only' and carrying_gate_required(stage):
                values = self._sensor_facts(stage['object'], stage['arm'], stage.get('target'), stage.get('goal_relation'))
                holding = sensor_holding_contract(stage, values)
                if holding is not True:
                    self.sensor_motion_blocked = True
                    now = float(self.data.time)
                    if self.sensor_unknown_since is None:
                        self.sensor_unknown_since = now
                    reason = 'unknown_holding_pose_hold' if holding is None else 'holding_false_pose_hold'
                    if holding is False or now - self.sensor_unknown_since >= timeout_sec:
                        reason = 'unknown_stop' if holding is None else 'holding_false_stop'
                        if self.sensor_runtime is not None:
                            self.sensor_runtime.state.failure = reason
                    self.sensor_gate_records.append({'timestamp_sec': now, 'stage_id': stage.get('stage_id'), 'reason': reason})
                    prop = self.proprioception()
                    q, g = np.asarray(prop.joint_pos).copy(), self._previous_gripper_command.copy()
                    callback = lambda _: (q, g, np.zeros_like(q))
                else:
                    self.sensor_unknown_since = None
            return super().step_command_callback(callback, duration_sec, *args, **kwargs)

        def finalize_evaluator(self, task_outcome):
            # Seal sensor decisions and controller trace BEFORE independent GT evaluation.
            path = Path(seal_path)
            if path.exists():
                raise FileExistsError(path)
            archive = path.with_name('SENSOR_PACKETS.npz')
            if archive.exists():
                raise FileExistsError(archive)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(archive, **self.sensor_packets)
            payload = {'schema': 'graph-self-adapter.sensor-prediction-seal.v1',
                'judge_mode': judge_mode, 'servo': 'raw_proprio_only_v1',
                'sealed_before_finalize_evaluator': True,
                'execution_variant': 'sensor_only_judge' if judge_mode == 'sensor_only' else 'explicit_gt_judge_control',
                'actual_runtime_provenance': provenance or {},
                'public_sensor_packet_archive': {'path': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()},
                'sensor_records': self.sensor_records, 'motion_gates': self.sensor_gate_records,
                'controller_trace': [] if self.sensor_runtime is None else self.sensor_runtime.trace}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
            self.sensor_prediction_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.sensor_evaluator_context = True
            try:
                return super().finalize_evaluator(task_outcome)
            finally:
                self.sensor_evaluator_context = False

    return SensorWorld
