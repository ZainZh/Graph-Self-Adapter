"""P53 public-proprio Code path and separate privileged micro training teacher."""
import numpy as np
from scipy.spatial.transform import Rotation
from .p52_training_teacher import box_vertices,held_compensated_eef,norm_limit


def geometry(relation,private):
    C_B=np.linalg.inv(private['world_caddy'])@private['world_book']
    points=box_vertices(relation['book_collision_center'],relation['book_collision_halfsize'])@C_B[:3,:3].T+C_B[:3,3]
    half=np.asarray(relation['back_halfsize'])[:2]
    slack=np.minimum(points[:,:2].min(0)+half,half-points[:,:2].max(0))
    return dict(bottom_gap=float(np.min(-points[:,2])-private['rim_height']),
                footprint_qualified=bool(np.all(slack>=relation['mouth_clearance_margin'])),footprint_slack=slack.tolist())


def near_supervisor(relation,private):
    """Privileged evaluator: only Boolean gate/stop may route the public Code."""
    g=geometry(relation,private);f=private['finger_contacts'];tol=min(relation['book_collision_halfsize'])/4
    return dict(near_ready=bool(f is not None and all(f) and g['footprint_qualified'] and 0<=g['bottom_gap']<=tol and not private['official_success'] and not private['wrong_target']),
                code_completed_goal=bool(private['official_success']),
                structural_stop=bool(private['wrong_target'] or f is None or not any(f)))


class PublicPosePath:
    """No environment handle or object/target state accepted by the controller."""
    def __init__(self,references,position_tolerance,rotation_tolerance):
        self.p=np.asarray([x['position'] for x in references]);self.R=Rotation.from_quat([x['quaternion_xyzw'] for x in references]).as_matrix()
        self.ptol=position_tolerance;self.rtol=rotation_tolerance;self.index=0
    def command(self,position,quaternion,start,end):
        p=np.asarray(position);R=Rotation.from_quat(quaternion).as_matrix();self.index=max(self.index,start)
        while self.index<end:
            dp=self.p[self.index]-p;dr=Rotation.from_matrix(self.R[self.index]@R.T).as_rotvec()
            if np.linalg.norm(dp)>self.ptol or np.linalg.norm(dr)>self.rtol:break
            self.index+=1
        dp=self.p[self.index]-p;dr=Rotation.from_matrix(self.R[self.index]@R.T).as_rotvec()
        reached=bool(self.index==end and np.linalg.norm(dp)<=self.ptol and np.linalg.norm(dr)<=self.rtol)
        return np.r_[norm_limit(dp,.01)/.05,norm_limit(dr,.05)/.5,1.],reached,dict(reference_index=self.index,position_error=dp.tolist(),rotation_error=dr.tolist())


class MicroTeacher:
    """Training labels only: direct final relation, no lift/align recovery phases."""
    def __init__(self,relation):self.relation=relation;self.goal_hold=0
    def command(self,private):
        g=geometry(self.relation,private);f=private['finger_contacts'];official=bool(private['official_success'])
        attached=f is not None and any(f);wrong=bool(private['wrong_target'])
        self.goal_hold=self.goal_hold+1 if official and attached and not wrong else 0
        tol=min(self.relation['book_collision_halfsize'])/4
        stop=('wrong_target' if wrong else 'micro_contact_unavailable' if f is None else 'micro_contact_lost' if not attached else
              'micro_footprint_exit' if not g['footprint_qualified'] else 'micro_clearance_exit' if g['bottom_gap']>tol+.01 else
              'qualified_goal_contact_hold' if self.goal_hold>=3 else None)
        target=np.asarray(self.relation['final_caddy_book']);W_E=private['world_eef'];desired=held_compensated_eef(private['world_caddy'],target,W_E,private['world_book'])
        dp=desired[:3,3]-W_E[:3,3];dr=Rotation.from_matrix(desired[:3,:3]@W_E[:3,:3].T).as_rotvec()
        action=np.r_[norm_limit(dp,.01)/.05,norm_limit(dr,.05)/.5,1.]
        if official:action[:6]=0.
        label=dict(authority='private_training_micro_teacher_gt',teacher_phase='micro_insert',expert_action=action.tolist(),
            desired_world_eef=desired.tolist(),actual_eef_book=(np.linalg.inv(W_E)@private['world_book']).tolist(),
            actual_caddy_book=(np.linalg.inv(private['world_caddy'])@private['world_book']).tolist(),
            official_success=official,goal_streak=self.goal_hold,stop_reason=stop,**g)
        return action,label
