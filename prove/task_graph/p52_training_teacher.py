"""P52 privileged training-only book-relative teacher, never a runtime policy.

GT held transforms compensate actual grasp offsets. No force, gains, mass,
friction or success predicate is changed. Scene updates require real physics.
"""
from dataclasses import dataclass, asdict
import itertools
import numpy as np
from scipy.spatial.transform import Rotation


def transform(position, rotation):
    value=np.eye(4);value[:3,3]=position;value[:3,:3]=rotation
    return value


def box_vertices(center, halfsize):
    return np.asarray(center)+np.asarray(list(itertools.product((-1,1),repeat=3)))*np.asarray(halfsize)


def held_compensated_eef(world_caddy, caddy_book_goal, world_eef, world_book):
    eef_book=np.linalg.inv(world_eef)@world_book
    return world_caddy@caddy_book_goal@np.linalg.inv(eef_book)


@dataclass(frozen=True)
class TeacherConfig:
    max_steps:int=180
    translation_step_limit:float=.01
    rotation_step_limit:float=.05
    position_scale:float=.05
    rotation_scale:float=.5
    reached_consecutive:int=2
    goal_hold_frames:int=3
    loss_contact_frames:int=5


def norm_limit(vector, bound):
    vector=np.asarray(vector,float)
    return vector*min(1.,bound/max(float(np.linalg.norm(vector)),1e-12))


class TrainingOnlyBookTeacher:
    authority='private_training_teacher_gt'
    def __init__(self, relation, config=TeacherConfig()):
        self.relation=relation;self.config=config
        self.center=np.asarray(relation['book_collision_center'])
        self.half=np.asarray(relation['book_collision_halfsize'])
        # Geometry-derived margins and tolerances, fixed before any outcome.
        self.margin=float(np.min(self.half))
        self.position_tolerance=self.margin/4
        self.rotation_tolerance=float(np.arctan(self.position_tolerance/np.max(self.half)))
        self.phase='lift_clear';self.arrivals=0;self.initial_clear=None;self.goal_streak=0;self.lost_streak=0

    def bottom(self, caddy_book):
        points=box_vertices(self.center,self.half)@caddy_book[:3,:3].T+caddy_book[:3,3]
        return float(np.min(points@np.array([0.,0.,-1.])))

    def raised(self, pose, floor):
        out=pose.copy();out[:3,3]+=np.array([0.,0.,-1.])*max(0.,floor-self.bottom(out))
        return out

    def command(self, private):
        """Inputs are explicitly privileged labels, disallowed in runtime models."""
        W_C=np.asarray(private['world_caddy']);W_B=np.asarray(private['world_book']);W_E=np.asarray(private['world_eef'])
        C_B=np.linalg.inv(W_C)@W_B
        floor=max(float(private['rim_height']),float(self.relation['demonstrated_clearance_bottom']))+self.margin
        if self.initial_clear is None:self.initial_clear=self.raised(C_B,floor)
        final=np.asarray(self.relation['final_caddy_book'])
        above=self.raised(final,floor)
        targets={'lift_clear':self.initial_clear,'align_above':above,'insert':final}
        goal=bool(private['official_success'])
        self.goal_streak=self.goal_streak+1 if goal else 0
        bilateral=private['finger_contacts']
        self.lost_streak=self.lost_streak+1 if bilateral is not None and not any(bilateral) and not goal else 0
        target=targets[self.phase]
        dp=target[:3,3]-C_B[:3,3]
        dr=Rotation.from_matrix(target[:3,:3]@C_B[:3,:3].T).as_rotvec()
        reached=np.linalg.norm(dp)<=self.position_tolerance and np.linalg.norm(dr)<=self.rotation_tolerance
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
        label=dict(authority=self.authority,teacher_phase=self.phase,expert_action=action.tolist(),
                   desired_world_eef=desired.tolist(),desired_caddy_book=target.tolist(),actual_caddy_book=C_B.tolist(),
                   actual_eef_book=(np.linalg.inv(W_E)@W_B).tolist(),clearance_floor=floor,
                   position_tolerance=self.position_tolerance,rotation_tolerance=self.rotation_tolerance,
                   official_success=goal,goal_streak=self.goal_streak,stop_reason=reason)
        return action,label
