"""P52 teacher v02: footprint-centered mouth alignment and gated descent.

Private training GT only. Historical teacher, force settings and native goal
remain unchanged; this is a new frozen geometry/controller candidate.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .p52_training_teacher import TeacherConfig,TrainingOnlyBookTeacher as BaseTeacher,box_vertices,held_compensated_eef,norm_limit


def centered_relation(relation):
    result=copy.deepcopy(relation)
    original=np.asarray(relation['final_caddy_book'],float)
    points=box_vertices(relation['book_collision_center'],relation['book_collision_halfsize'])@original[:3,:3].T+original[:3,3]
    center=(points.min(0)+points.max(0))/2
    projected_half=(points.max(0)-points.min(0))/2
    half=np.asarray(relation['back_halfsize'])
    slack=half[:2]-projected_half[:2]
    if np.any(slack<=0):raise ValueError('demonstrated orientation does not fit the mouth footprint')
    target=original.copy();target[:2,3]-=center[:2]
    result.update(original_final_caddy_book=original.tolist(),final_caddy_book=target.tolist(),
                  footprint_center_xy_correction=(-center[:2]).tolist(),
                  nominal_projected_halfwidth=projected_half[:2].tolist(),
                  mouth_clearance_margin=float(.2*np.min(slack)),margin_rule='20 percent of minimum nominal footprint-to-mouth halfwidth slack')
    return result


class TrainingOnlyBookTeacher(BaseTeacher):
    authority='private_training_teacher_gt_v02'
    def __init__(self,relation,config=TeacherConfig()):
        if 'original_final_caddy_book' not in relation:relation=centered_relation(relation)
        super().__init__(relation,config)
        self.recoveries=0

    def footprint(self,C_B):
        points=box_vertices(self.center,self.half)@C_B[:3,:3].T+C_B[:3,3]
        lower=points[:,:2].min(0);upper=points[:,:2].max(0)
        half=np.asarray(self.relation['back_halfsize'])[:2]
        margin=self.relation['mouth_clearance_margin']
        return bool(np.all(lower>=-half+margin) and np.all(upper<=half-margin)),lower,upper

    def command(self,private):
        W_C=np.asarray(private['world_caddy']);W_B=np.asarray(private['world_book']);W_E=np.asarray(private['world_eef'])
        C_B=np.linalg.inv(W_C)@W_B
        floor=max(float(private['rim_height']),float(self.relation['demonstrated_clearance_bottom']))+self.margin
        if self.initial_clear is None:self.initial_clear=self.raised(C_B,floor)
        goal=bool(private['official_success']);self.goal_streak=self.goal_streak+1 if goal else 0
        bilateral=private['finger_contacts'];self.lost_streak=self.lost_streak+1 if bilateral is not None and not any(bilateral) and not goal else 0
        qualified,lower,upper=self.footprint(C_B)
        correction_event=None
        if self.phase=='insert' and not qualified and not goal:
            self.initial_clear=self.raised(C_B,floor);self.phase='lift_clear';self.arrivals=0;self.recoveries+=1
            correction_event='descent_stopped_footprint_outside_mouth'
        final=np.asarray(self.relation['final_caddy_book'])
        targets={'lift_clear':self.initial_clear,'align_above':self.raised(final,floor),'insert':final}
        target=targets[self.phase]
        dp=target[:3,3]-C_B[:3,3];dr=Rotation.from_matrix(target[:3,:3]@C_B[:3,:3].T).as_rotvec()
        reached=np.linalg.norm(dp)<=self.position_tolerance and np.linalg.norm(dr)<=self.rotation_tolerance
        if self.phase=='align_above':reached=bool(reached and qualified)
        self.arrivals=self.arrivals+1 if reached else 0
        if self.arrivals>=self.config.reached_consecutive and self.phase!='insert':
            self.phase='align_above' if self.phase=='lift_clear' else 'insert';self.arrivals=0;target=targets[self.phase]
        desired=held_compensated_eef(W_C,target,W_E,W_B)
        translation=norm_limit(desired[:3,3]-W_E[:3,3],self.config.translation_step_limit)
        rotation=norm_limit(Rotation.from_matrix(desired[:3,:3]@W_E[:3,:3].T).as_rotvec(),self.config.rotation_step_limit)
        action=np.r_[translation/self.config.position_scale,rotation/self.config.rotation_scale,1.]
        if goal:action[:6]=0.
        if not np.isfinite(action).all():raise ValueError('nonfinite training teacher action')
        reason='qualified_goal_hold' if self.goal_streak>=self.config.goal_hold_frames else ('teacher_object_lost' if self.lost_streak>=self.config.loss_contact_frames else None)
        label=dict(authority=self.authority,teacher_phase=self.phase,expert_action=action.tolist(),desired_world_eef=desired.tolist(),
                   desired_caddy_book=target.tolist(),actual_caddy_book=C_B.tolist(),actual_eef_book=(np.linalg.inv(W_E)@W_B).tolist(),
                   clearance_floor=floor,position_tolerance=self.position_tolerance,rotation_tolerance=self.rotation_tolerance,
                   official_success=goal,goal_streak=self.goal_streak,stop_reason=reason,
                   footprint_qualified=qualified,footprint_xy_min=lower.tolist(),footprint_xy_max=upper.tolist(),
                   footprint_margin=self.relation['mouth_clearance_margin'],recovery_count=self.recoveries,recovery_event=correction_event)
        return action,label
