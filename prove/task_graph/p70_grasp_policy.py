"""Frozen object-relative short grasp action MLP; no demo path index or runtime GT."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

SCHEMA='prove.p70-relative-grasp-policy.v1'
ACTION_SCALES=np.array([.2,.2,.2,.1,.1,.1],np.float32)


def rigid(value):
    T=np.asarray(value,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(T[:3,:3])<.999:
        raise ValueError('Finite proper rigid transform required')
    return T


def relative_features(current_book_eef,goal_book_eef,gripper_qpos):
    current,goal=rigid(current_book_eef),rigid(goal_book_eef)
    grip=np.asarray(gripper_qpos,float)
    if grip.shape!=(2,) or not np.isfinite(grip).all():raise ValueError('Finite two-finger proprioception required')
    dp=goal[:3,3]-current[:3,3]
    dr=Rotation.from_matrix(goal[:3,:3]@current[:3,:3].T).as_rotvec()
    return np.r_[dp/.05,dr/.5,grip/.04].astype(np.float32)


class GraspPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(8,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,6),torch.nn.Tanh())
        self.register_buffer('action_scales',torch.tensor(ACTION_SCALES))

    def forward(self,x):return self.net(x)*self.action_scales

    @torch.no_grad()
    def command(self,T_world_book,T_world_eef,gripper_qpos,goal_book_eef,close=False):
        book,eef=rigid(T_world_book),rigid(T_world_eef)
        current=np.linalg.inv(book)@eef
        x=relative_features(current,goal_book_eef,gripper_qpos)
        local=self(torch.as_tensor(x,device=self.action_scales.device)).cpu().numpy()
        # Norm caps match the teacher's explicit OSC per-command bounds.
        for start,cap in ((0,.2),(3,.1)):
            part=local[start:start+3];local[start:start+3]=part*min(1.,cap/max(float(np.linalg.norm(part)),1e-12))
        action=np.r_[book[:3,:3]@local[:3],book[:3,:3]@local[3:],1. if close else -1.]
        return action,dict(controller='p70_frozen_object_relative_grasp_mlp',features=x.tolist(),
            current_book_eef=current.tolist(),goal_book_eef=np.asarray(goal_book_eef).tolist(),
            local_action=local.tolist(),actual_action=action.tolist(),close=bool(close),network_weights_frozen=not any(p.requires_grad for p in self.parameters()),
            source='public_book_estimate_and_robot_proprioception',waypoint_index_used=False)

    @classmethod
    def load(cls,path,device='cpu'):
        payload=torch.load(path,map_location=device,weights_only=True)
        if payload.get('schema')!=SCHEMA:raise ValueError('Wrong grasp checkpoint')
        model=cls().to(device);model.load_state_dict(payload['state_dict']);model.eval().requires_grad_(False)
        return model
