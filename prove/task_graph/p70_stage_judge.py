"""P70 privileged Boolean stage judges; never supplies pose to the controller."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from prove.task_graph.libero_supervisor import LiberoSupervisor, angle, corners


def _bound_json(binding):
    p=Path(binding['path']); raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=binding['sha256']:
        raise ValueError('Stage judge bound file changed: '+str(p))
    return json.loads(raw)


def independent_facts(report, private, calibration, entry_relation, near, minimum_bottom_world_z):
    """Pure private-to-Boolean conversion; unknown attachment is not success."""
    old=report.facts
    B=np.asarray(private['world_book']); C=np.asarray(private['world_caddy'])
    local=np.linalg.inv(C)@B
    desired=np.asarray(entry_relation)
    bottom=float((corners(calibration['book_collision_center'],np.asarray(calibration['book_collision_halfsize']))@B[:3,:3].T+B[:3,3])[:,2].min())
    back_top=float((corners(np.zeros(3),np.asarray(calibration['back_halfsize']))@C[:3,:3].T+C[:3,3])[:,2].max())
    rot_error=angle(local[:3,:3],desired[:3,:3])
    xy_error=float(np.linalg.norm(local[:2,3]-desired[:2,3]))
    minimum_bottom_world_z=float(minimum_bottom_world_z)
    if not np.isfinite(minimum_bottom_world_z):raise ValueError('Invalid frozen safety height')
    safe=bottom>=minimum_bottom_world_z
    attached=old.get('attached') is True
    fingers=private['finger_contacts']
    facts=dict(grasp_ready=old.get('above_grasp_approach') is True,
        at_grasp_pose=old.get('grasp_ready') is True,
        attached=attached,
        bilateral_contact=bool(fingers is not None and all(fingers)),
        lift_clear=bool(attached and safe and old.get('clear_of_table') is True),
        orient_clear=bool(attached and safe and old.get('clear_of_table') is True and rot_error<=calibration['tolerances']['angle_rad']),
        above_destination_vicinity=bool(attached and safe and xy_error<=calibration['tolerances']['vicinity_position'] and bottom>=back_top and rot_error<=calibration['tolerances']['angle_rad']),
        near_entry=bool(attached and near['near_ready']),
        wrong_target=private['wrong_target'] is True,
        object_lost=old.get('object_lost') is True,
        official_success=bool(private['official_success']))
    diagnostics=dict(authority='supervisor_private_only',original_facts=dict(old),
        original_diagnostics=dict(report.diagnostics),near_gate=dict(near),
        entry_rotation_error_rad=rot_error,entry_xy_error_m=xy_error,
        book_bottom_world_z=bottom,back_top_world_z=back_top,
        frozen_minimum_bottom_world_z=minimum_bottom_world_z,safe_clearance=safe,
        attachment_known=old.get('attached') is not None)
    assert all(type(v) is bool for v in facts.values())
    return facts,diagnostics


class StageJudge:
    def __init__(self,env,names,child,config):
        self.env,self.names,self.child=env,names,child
        calibration=json.loads(Path(child['prefix']['calibration']).read_text())
        calibration=copy.deepcopy(calibration)
        demo=_bound_json(config['grasp_policy']['demo_relations'])
        for name,role in [('approach','pregrasp'),('grasp','grasp')]:
            T=np.asarray(demo['relations'][role]['T_book_eef'])
            calibration['references'][name]=dict(position=T[:3,3].tolist(),rotation=T[:3,:3].tolist())
        relations=_bound_json(config['object_relation_targets'])
        self.minimum_bottom_world_z=float(relations['safety_heights']['book_bottom_min_world_z'])
        if not np.isfinite(self.minimum_bottom_world_z):raise ValueError('Invalid safety height')
        self.entry=np.asarray(relations['T_back_site_book_entry'])
        if self.entry.shape!=(4,4) or not np.isfinite(self.entry).all():raise ValueError('Invalid frozen entry relation')
        self.calibration=calibration
        self.supervisor=LiberoSupervisor(env,calibration,instance_id='p70_stage_qualification')
        self.last_t=None

    def observe(self,obs,stage,t):
        from tools.task_graph.collect_libero_p52_v02 import private_measurement
        from tools.task_graph.p54_full_task import near_gate,OLD_TOL,OLD_GAP_MAX
        if stage not in ('approach_book','short_grasp','lift_clear','orient_clear','transport_above_caddy','near_entry','p68_insert'):
            raise ValueError('Unknown P70 stage')
        if self.last_t is not None and t<self.last_t:raise ValueError('Noncausal judge timestamp')
        self.last_t=t
        # Original supervisor advances its 3-frame history only on new sim time.
        report=self.supervisor.observe(obs,'s3')
        private=private_measurement(self.env,obs,self.names,self.child['relation'])
        near=near_gate(self.child['relation'],private,dict(near_position_tolerance=OLD_TOL,near_gap_max=OLD_GAP_MAX))
        facts,diag=independent_facts(report,private,self.calibration,self.entry,near,self.minimum_bottom_world_z)
        diag.update(stage=stage,t=int(t),sim_time=float(self.env.sim.data.time))
        return facts,diag
