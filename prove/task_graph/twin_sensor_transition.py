"""Align-only public-sensor transition permission, never placement completion."""
import numpy as np


def support_transition_candidate(judge, stage, facts, reference, *, observed_role, observed_target, handover_anchor=None):
    diagnostic={'permission_only':True,'measured_support_contact':False,'placement_completed':False,
                'source':'prior_sensor_acquisition_current_RGBD_visible_relation_cumulative_settling',
                'phase':stage.get('phase'),'holding_contract':stage.get('holding_contract')}
    if stage.get('phase')!='Align' or stage.get('holding_contract')!='held_or_supported':
        return None,{**diagnostic,'reason':'unavailable_outside_declared_Align_transition'}
    if (observed_target is None and stage.get('target')=='table_handover_region'
        and stage.get('goal_relation')=='accessible_to_right_arm'):
        observed_target='table_handover_region'
    if observed_target is None and observed_role==stage.get('object'):
        return None,{**diagnostic,'reason':'missing_declared_target_observation'}
    if observed_role!=stage.get('object') or observed_target!=stage.get('target'):
        return False,{**diagnostic,'reason':'wrong_declared_role_or_target'}
    if not judge.valid or judge.packet is None or facts.get('holding') is None:
        return None,{**diagnostic,'reason':'stale_missing_or_invalid_actual_sensors'}
    key=(stage['object'],stage['arm'])
    if key not in judge.repair_acquired:
        return False,{**diagnostic,'reason':'no_prior_sensor_acquisition'}
    center=judge._center(stage['object'])
    if center is None:
        return None,{**diagnostic,'reason':'missing_fresh_object_geometry'}
    if reference.get('authority')!='offline_visible_rgbd_reference_only':
        return None,{**diagnostic,'reason':'missing_permitted_fixed_visible_reference'}
    target=judge._center(stage['target'])
    if target is not None:
        delta=reference.get('object_target_end')
        if delta is None:return None,{**diagnostic,'reason':'missing_visible_relative_reference'}
        delta=np.asarray(delta,dtype=float)
        if delta.shape!=(3,) or not np.isfinite(delta).all():
            return None,{**diagnostic,'reason':'invalid_visible_relative_reference'}
        distance=float(np.linalg.norm(center-target-delta))
        diagnostic['target_source']='fresh_current_RGBD_role_geometry'
    elif stage.get('target')=='table_handover_region' and stage.get('goal_relation')=='accessible_to_right_arm':
        if handover_anchor is None:return None,{**diagnostic,'reason':'missing_declared_fixed_handover_region'}
        anchor=np.asarray(handover_anchor,dtype=float)
        if anchor.shape!=(3,) or not np.isfinite(anchor).all():
            return None,{**diagnostic,'reason':'invalid_fixed_visible_handover_region'}
        distance=float(np.linalg.norm(center-anchor))
        diagnostic.update(target_source='declared_fixed_offline_visible_handover_region',prior_is_fresh_measurement=False)
    else:
        return None,{**diagnostic,'reason':'missing_fresh_declared_target_geometry'}
    diagnostic['visible_relation_error_m']=distance
    if distance>judge.config.relation_tolerance_m:
        return False,{**diagnostic,'reason':'visible_geometry_far_from_declared_target_relation'}
    settle=judge.settle_anchors.get(key)
    elapsed=0. if settle is None else judge.packet.timestamp_sec-settle[0]
    diagnostic['cumulative_settling_sec']=elapsed
    diagnostic['settling_window_sec']=judge.config.debounce_sec
    if elapsed+1e-9<judge.config.debounce_sec:
        return None,{**diagnostic,'reason':'in_region_waiting_cumulative_settling'}
    return True,{**diagnostic,'reason':'local_Align_permission_only'}


def transition_holding_contract(stage, facts, strict_contract):
    strict=strict_contract(stage,facts)
    if stage.get('phase')!='Align' or stage.get('holding_contract')!='held_or_supported':
        return strict
    candidate=facts.get('support_transition_candidate')
    if strict is True or candidate is True:return True
    return False if strict is False and candidate is False else None
