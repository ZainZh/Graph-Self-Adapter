"""P53 v03 near-contact local alignment plus insertion, privileged labels only."""
import numpy as np
from scipy.spatial.transform import Rotation
from .p53_micro_insert import PublicPosePath,geometry
from .p52_training_teacher import box_vertices,held_compensated_eef,norm_limit


def near_supervisor(relation,private):
    g=geometry(relation,private);f=private['finger_contacts'];tol=min(relation['book_collision_halfsize'])/4
    return dict(near_ready=bool(f is not None and all(f) and np.all(np.asarray(g['footprint_slack'])>=-tol) and 0<=g['bottom_gap']<=2*tol and not private['official_success'] and not private['wrong_target']),
                code_completed_goal=bool(private['official_success']),structural_stop=bool(private['wrong_target'] or f is None or not any(f)))


class MicroTeacher:
    """No distant lift/recovery; all local-align/insert actions share H24."""
    def __init__(self,relation):
        self.relation=relation;self.goal_hold=0;self.phase='local_align';self.entry_gap=None
    def command(self,private):
        g=geometry(self.relation,private);f=private['finger_contacts'];official=bool(private['official_success']);attached=f is not None and any(f)
        wrong=bool(private['wrong_target']);tol=min(self.relation['book_collision_halfsize'])/4
        if self.entry_gap is None:self.entry_gap=max(0.,g['bottom_gap'])
        self.goal_hold=self.goal_hold+1 if official and attached and not wrong else 0
        stop=('wrong_target' if wrong else 'micro_contact_unavailable' if f is None else 'micro_contact_lost' if not attached else
              'micro_clearance_exit' if g['bottom_gap']>3*tol else
              'micro_lateral_domain_exit' if np.any(np.asarray(g['footprint_slack']) < -tol) else None)
        actual=np.linalg.inv(private['world_caddy'])@private['world_book'];final=np.asarray(self.relation['final_caddy_book']);event=None
        if self.phase=='insert' and not g['footprint_qualified'] and not official:
            if g['bottom_gap']<0:stop=stop or 'micro_unsafe_realign_below_rim'
            else:self.phase='local_align';event='local_realign_same_entry_clearance'
        if self.phase=='local_align':
            if g['bottom_gap']<0 and not official:stop=stop or 'micro_alignment_below_rim'
            angle=float(Rotation.from_matrix(final[:3,:3]@actual[:3,:3].T).magnitude())
            if g['footprint_qualified'] and angle<=np.arctan(tol/max(self.relation['book_collision_halfsize'])):
                self.phase='insert'
        target=final.copy()
        if self.phase=='local_align':
            points=box_vertices(self.relation['book_collision_center'],self.relation['book_collision_halfsize'])@target[:3,:3].T+target[:3,3]
            bottom=float(np.min(-points[:,2]));required=private['rim_height']+self.entry_gap
            target[2,3]-=max(0.,required-bottom)
        W_E=private['world_eef'];desired=held_compensated_eef(private['world_caddy'],target,W_E,private['world_book'])
        dp=desired[:3,3]-W_E[:3,3];dr=Rotation.from_matrix(desired[:3,:3]@W_E[:3,:3].T).as_rotvec()
        action=np.r_[norm_limit(dp,.01)/.05,norm_limit(dr,.05)/.5,1.]
        if official:action[:6]=0.
        if self.goal_hold>=3:stop=stop or 'qualified_goal_contact_hold'
        label=dict(authority='private_training_micro_teacher_gt_v03',teacher_phase=self.phase,expert_action=action.tolist(),desired_world_eef=desired.tolist(),desired_caddy_book=target.tolist(),actual_eef_book=(np.linalg.inv(W_E)@private['world_book']).tolist(),actual_caddy_book=actual.tolist(),official_success=official,goal_streak=self.goal_hold,stop_reason=stop,entry_gap=self.entry_gap,local_realign_event=event,**g)
        return action,label
