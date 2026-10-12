"""One bounded sensor repair; original judge implementation remains immutable."""
from dataclasses import dataclass
import numpy as np
from prove.task_graph.twin_sensor_judge import JudgeConfig, SensorJudge


@dataclass(frozen=True)
class RepairConfig(JudgeConfig):
    release_opening_margin_m: float = .003


class RepairJudge(SensorJudge):
    """Provisional obstruction/vicinity holding, not measured physical contact."""
    def __init__(self, config=RepairConfig()):
        super().__init__(config)
        self.repair_acquired = set()
        self.blocked_aperture = {}
        self.repair_cache = {}
        self.previous_aperture = {}
        self.in_base_facts = False

    def facts(self, role, arm, *, target=None, relation=None, reference_delta=None):
        robot = self.packet.arms.get(arm) if self.packet is not None else None
        if self.valid and robot is not None:
            scalars = (robot.actual_aperture_m, robot.commanded_aperture_m, robot.closed_gap_m, robot.open_gap_m)
            finite = all(isinstance(value, (int, float, np.floating)) and np.isfinite(value) for value in scalars)
            tolerance = self.config.gap_stability_m
            if not finite or not (robot.closed_gap_m < robot.open_gap_m
                and robot.closed_gap_m - tolerance <= robot.actual_aperture_m <= robot.open_gap_m + tolerance
                and robot.closed_gap_m - tolerance <= robot.commanded_aperture_m <= robot.open_gap_m + tolerance):
                self._reset_role(role)
                for name in ('repair_holding','repair_released','repair_settled'):
                    self.since.pop(((role,arm),name),None)
                keys = ('visible','observation_missing','near','near_object','holding','bilateral_contact',
                    'loaded_contact','single_finger_loaded_contact','grip_dwell','released','loaded_released',
                    'lifted','lift_clear','relation','near_target','stabilized','object_stabilized',
                    'supported_contact','object_support_contact','object_table_support_contact')
                return self._unknown(keys, 'calibrated_aperture_range_or_missing_nonfinite')
        self.in_base_facts = True
        try:
            base = super().facts(role, arm, target=target, relation=relation, reference_delta=reference_delta)
        finally:
            self.in_base_facts = False
        # Parent validates freshness, actual-feedback provenance and packet shapes.
        if base['holding'] is None:
            self.since.pop(((role, arm), 'repair_holding'), None)
            self.since.pop(((role, arm), 'repair_released'), None)
            self.since.pop(((role, arm), 'repair_settled'), None)
            return base
        key, t = (role, arm), self.packet.timestamp_sec
        cached = self.repair_cache.get(key)
        if cached is not None and cached[0] == t:
            self.diagnostics = {name: dict(row) for name, row in cached[2].items()}
            return self._relations(role, arm, dict(cached[1]), target, relation, reference_delta)
        robot, cfg = self.packet.arms[arm], self.config
        actual, command = robot.actual_aperture_m, robot.commanded_aperture_m
        near = base['near']
        obstruction = actual - command >= cfg.obstruction_m
        previous_blocked = self.blocked_aperture.get(key)
        # Offset-independent actual opening from the last sensor-confirmed block.
        opening_command = previous_blocked is not None and command >= previous_blocked + cfg.release_opening_margin_m
        measured_opening = bool(previous_blocked is not None and opening_command
            and actual >= previous_blocked + cfg.release_opening_margin_m and not obstruction)
        holding = self._debounce((key, 'repair_holding'), bool(near and obstruction and not measured_opening))
        if holding is True:
            self.repair_acquired.add(key)
            self.acquired.add(key)  # inherited handover helper uses the same sensor acquisition
        if near and obstruction and not opening_command and key in self.repair_acquired:
            self.blocked_aperture[key] = actual
        released = self._debounce((key, 'repair_released'), bool(
            key in self.repair_acquired and measured_opening))
        settle = self.settle_anchors.get(key)
        settled = bool(settle is not None and t - settle[0] + 1e-9 >= cfg.debounce_sec)
        stabilized = self._debounce((key, 'repair_settled'), bool(measured_opening and settled))
        facts = dict(base)
        facts.update(holding=holding, bilateral_contact=holding, loaded_contact=holding,
            single_finger_loaded_contact=holding, grip_dwell=holding,
            released=released, loaded_released=released,
            stabilized=stabilized, object_stabilized=stabilized,
            supported_contact=None, object_support_contact=None)
        prior = self.previous_aperture.get(key, actual)
        aperture_change = actual - prior
        self.previous_aperture[key] = actual
        motion_strength = self.diagnostics['holding'].get('evidence_strength')
        strength = ('motion_correlated_provisional_obstruction_vicinity' if motion_strength == 'motion_verified'
                    else 'provisional_obstruction_vicinity_motion_unconfirmed')
        for name in ('holding', 'bilateral_contact', 'loaded_contact', 'single_finger_loaded_contact', 'grip_dwell',
                     'released', 'loaded_released', 'stabilized', 'object_stabilized'):
            self.diagnostics[name] = {**self.diagnostics.get(name, {}), 'value':facts[name],
                'repair_variant':'persistent_obstruction_compliance_release_relative_v1',
                'evidence_strength':strength if name in ('holding','bilateral_contact','loaded_contact','grip_dwell') else 'measured_relative_opening_visible_settling_proxy',
                'aperture_change_m':aperture_change,
                'aperture_compliance_uncertainty':abs(aperture_change) > cfg.gap_stability_m,
                'aperture_change_is_hard_detachment_veto':False,
                'centroid_motion_is_hard_detachment_veto':False,
                'previous_blocked_aperture_m':previous_blocked,
                'actual_aperture_m':actual, 'commanded_aperture_m':command,
                'opening_margin_m':cfg.release_opening_margin_m,
                'physical_contact_verified':False, 'rigid_attachment_verified':False}
        self.repair_cache[key] = (t, dict(facts), {name:dict(row) for name,row in self.diagnostics.items()})
        return self._relations(role, arm, facts, target, relation, reference_delta)

    def _relations(self, role, arm, facts, target, relation, reference_delta):
        result = super()._relations(role, arm, facts, target, relation, reference_delta)
        if self.in_base_facts:
            return result
        value = None
        if relation == 'supported' and result['relation'] is not None:
            value = self._debounce(((role, arm), 'repair_placement', target), bool(
                result['released'] is True and result['stabilized'] is True and result['relation']))
        result['supported_contact'] = result['object_support_contact'] = value
        for name in ('supported_contact','object_support_contact'):
            self.diagnostics[name]['value'] = value
            self.diagnostics[name]['physical_contact_verified'] = False
        return result
