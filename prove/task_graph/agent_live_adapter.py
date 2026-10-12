"""Additive v06 semantic gates over frozen v04 sensor and A2 motion code."""
from collections import deque
import numpy as np
from prove.task_graph.agent_live_loop import HandoffError, PauseBroker, VARIANT
from prove.task_graph.twin_sensor_transition_adapter import transition_world_class

PERMISSION = 'semantic.continue_placing_permission'
ALIGN_READY = 'semantic.align_ready'
PLACEMENT = 'semantic.placement_complete'


def install_agent_world(physics, *, broker_options, bindings, **sensor_kwargs):
    Base = transition_world_class(physics, **sensor_kwargs)
    class AgentWorld(Base):
        def __init__(self, *args, **kwargs):
            self.live_initializations = 0; self.live_robot_state_writes = 0
            self.live_external_force_writes = 0; self.live_holding = False
            self.live_hold_by_stage = {}
            self.live_frames = deque(maxlen=3); self.live_permission = None
            super().__init__(*args, **kwargs)
            import mujoco
            self.live_overview_renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            self.live_broker = PauseBroker(**broker_options)
            self.live_seal_hash = None

        def initialize(self, *args, **kwargs):
            self.live_initializations += 1
            if self.live_initializations != 1:
                raise HandoffError('multiple initialization forbidden')
            return super().initialize(*args, **kwargs)

        def _set_robot_state(self, *args, **kwargs):
            self.live_robot_state_writes += 1
            if self.live_initializations != 1 or self.live_robot_state_writes != 1:
                raise HandoffError('robot restore outside single original initialization')
            return super()._set_robot_state(*args, **kwargs)

        def live_counters(self):
            return {'initialization_count': self.live_initializations,
                    'robot_initial_state_write_count': self.live_robot_state_writes,
                    'restore_after_initialization_count': int(self._state_writes_after_initialization),
                    'external_force_write_count': self.live_external_force_writes}

        def live_hash(self):
            # Opaque integrity proof only: values never enter requests, judgment or actions.
            import hashlib
            import mujoco
            signature = mujoco.mjtState.mjSTATE_INTEGRATION
            state = np.empty(mujoco.mj_stateSize(self.model, signature), dtype=np.float64)
            mujoco.mj_getState(self.model, self.data, state, signature)
            return hashlib.sha256(state.tobytes()).hexdigest()

        def _servo(self, *args, **kwargs):
            if np.any(self.data.xfrc_applied) or np.any(self.data.qfrc_applied):
                self.live_external_force_writes += 1
                raise HandoffError('external-force injection forbidden')
            return super()._servo(*args, **kwargs)

        def observe_visible(self, *args, **kwargs):
            frame = super().observe_visible(*args, **kwargs)
            timestamp = float(frame.physical_time_sec)
            if not self.live_frames or self.live_frames[-1][0] != timestamp:
                renderer = self.live_overview_renderer
                renderer.disable_depth_rendering(); renderer.disable_segmentation_rendering()
                renderer.update_scene(self.data, camera='overview')
                self.live_frames.append((timestamp, np.asarray(frame.rgb).copy(), renderer.render().copy()))
            return frame

        def live_capture(self):
            self.observe_visible()
            p = self.proprioception(); t = float(self.data.time); proprio = {}
            for index, arm in enumerate(('left', 'right')):
                cal = self.sensor_calibrations[arm]
                proprio[arm] = {'robot_timestamp_sec': t, 'q': p.joint_pos[index].tolist(),
                    'qdot': p.joint_velocity[index].tolist(), 'tcp_position_m': p.tcp_position[index].tolist(),
                    'tcp_rotation': p.tcp_rotation[index].tolist(), 'finger_encoders': p.finger_qpos[index].tolist(),
                    'actual_aperture_m': float(cal.aperture(p.finger_qpos[index])),
                    'commanded_aperture_m': float(cal.closed_gap_m + self._previous_gripper_command[index] * (cal.open_gap_m - cal.closed_gap_m)),
                    'measurement_source': 'simulator_actual_joint_encoder', 'calibration_source': cal.provenance}
            return [(t, rgb, view) for t, main, overview in self.live_frames
                    for rgb, view in ((main, self.camera), (overview, 'overview'))], proprio

        def live_hold(self, duration):
            p = self.proprioception(); q = np.asarray(p.joint_pos).copy()
            g = self._previous_gripper_command.copy()
            self.live_holding = True
            start = float(self.data.time)
            sid = (self.sensor_stage or {}).get('stage_id')
            try:
                # Preserve frozen servo; bypass only the repeating semantic preflight.
                physics.TwinWorld.step_command_callback(self, lambda _: (q, g, np.zeros_like(q)), duration,
                                                       observe=False, include_endpoint=False)
                self.observe_visible()
                if not self.numerical_safety()['simulation_finite']:
                    raise HandoffError('nonfinite bounded hold')
            finally:
                self.live_hold_by_stage[sid] = self.live_hold_by_stage.get(sid, 0.) + float(self.data.time) - start
                self.live_holding = False

        def live_ask(self, stage, fact_ids):
            try:
                answer = self.live_broker.ask(stage, fact_ids, capture=self.live_capture,
                    state_hash=self.live_hash, physical_time=lambda: float(self.data.time),
                    hold=self.live_hold, counters=self.live_counters)
                if answer['decision'] in ('local_failure', 'structural_failure'):
                    raise HandoffError('Root ' + answer['decision'] + ': ' + answer['reason'])
                return answer
            except HandoffError as exc:
                if self.sensor_runtime is not None:
                    self.sensor_runtime.state.failure = 'agent_fail_closed: ' + str(exc)
                raise

        def _sensor_facts(self, role, arm, support=None, relation=None):
            facts = super()._sensor_facts(role, arm, support, relation)
            stage = self.sensor_stage or {}
            if (stage.get('phase') == 'Align' and stage.get('holding_contract') == 'held_or_supported'
                and not self.live_holding and self.live_permission is not None):
                sid, timestamp, value = self.live_permission
                if sid == stage['stage_id'] and 0 <= float(self.data.time) - timestamp <= .25:
                    facts['support_transition_candidate'] = value
            return facts

        def step_command_callback(self, callback, duration_sec, *args, **kwargs):
            stage = self.sensor_stage or {}
            if stage.get('phase') == 'Align' and stage.get('holding_contract') == 'held_or_supported':
                from prove.task_graph.twin_sensor_transition_adapter import routed_holding_contract
                values = self._sensor_facts(stage['object'], stage['arm'], stage.get('target'), stage.get('goal_relation'))
                if routed_holding_contract(stage, values) is not True:
                    answer = self.live_ask(stage, [PERMISSION])
                    value = answer['judged_facts'].get(PERMISSION)
                    self.live_permission = (stage['stage_id'], float(self.data.time), value)
                    if value is not True:
                        raise HandoffError('Root did not authorize local placement continuation')
            return super().step_command_callback(callback, duration_sec, *args, **kwargs)

        def close(self):
            self.live_overview_renderer.close()
            return super().close()

        def finalize_evaluator(self, outcome):
            from pathlib import Path
            path = Path(sensor_kwargs['seal_path']).with_name('AGENT_PREDICTIONS_SEALED.json')
            self.live_seal_hash = self.live_broker.seal(path, bindings=bindings, counters=self.live_counters())
            return super().finalize_evaluator(outcome)
    return AgentWorld


def semantic_contract(stage, facts, diagnostics, *, ask):
    """Keep robot progress, actual release and safety; semantic facts only add gates."""
    facts = dict(facts); diagnostics = list(diagnostics)
    phase = stage.get('phase')
    if phase not in ('Align', 'Release'):
        return facts, diagnostics
    rows = stage.get('completion_predicates', [])
    ready = all(facts.get(row['id']) is True for row in rows
                if row['predicate'] in ('reference_progress', 'actor_command_reached'))
    if not ready:
        facts['semantic.goal'] = None
        return facts, diagnostics
    if phase == 'Align':
        ids = [PERMISSION, ALIGN_READY]
    else:
        released_rows = [row for row in rows if row['predicate'] == 'released']
        if not released_rows or not all(facts.get(row['id']) is True for row in released_rows):
            facts['semantic.goal'] = None
            return facts, diagnostics
        ids = [PLACEMENT]
    reply = ask(stage, ids)
    judged = reply['judged_facts']
    permitted_decision = reply['decision'] == 'complete'
    fact = ALIGN_READY if phase == 'Align' else PLACEMENT
    value = judged.get(fact) if permitted_decision else False
    facts['semantic.goal'] = value
    if phase == 'Release':
        for row in rows:
            if row['predicate'] == 'stage_goal':
                facts[row['id']] = value
    diagnostics.append({'id': 'semantic.goal', 'predicate': 'actual_root_visual_semantics', 'value': value,
                        'decision': reply['decision'], 'facts': judged, 'actions': None})
    return facts, diagnostics
