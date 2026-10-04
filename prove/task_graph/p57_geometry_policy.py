"""P57 B stage-local geometry policy around the frozen A2 origin readout.

Only public UV rays and proprioception update z. Expert actions are accepted by
the offline trainer, never by this module or the runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from prove.task_graph.p57_public_geometry import rays
from prove.task_graph.p57_ray_geometry import _frame
from prove.task_graph.p57_translation_geometry import fit_origin_translations
from tools.task_graph.p57_translation_readout import (predict_pixels, predict_relative,
                                                       public_matrix)


PUBLIC_KEYS = frozenset({'raw','boxes','visible','proprio','mask_tokens','mask_valid',
                         'camera_K','camera_pose','eef_transform'})
ARMS = ('geometry_prior','geometry_fixed','geometry_online','geometry_direct')


@dataclass(frozen=True)
class PolicyConfig:
    hidden: int = 64
    latent: int = 6
    history: int = 8
    inner_steps: int = 3
    inner_lr: float = .65
    damping: float = 1e-4
    geometry_scale: float = .01
    translation_scale: float = .05
    rotation_scale: float = .5
    translation_cap: float = .2
    rotation_cap: float = .1


def prior_frames(readout):
    prior = readout['baseline_pose_prior']
    book = prior['book_in_eef_anchors'].numpy()
    goal = prior['desired_world_anchors'].numpy()
    held_frame = _frame(book)[0]
    goal_frame = _frame(goal)[0]
    if held_frame is None or goal_frame is None:
        raise ValueError('Degenerate frozen A2 train-only pose prior')
    return (held_frame[:3,3],goal_frame[:3,3],
            held_frame[:3,:3],goal_frame[:3,:3])


def strict_public(public):
    if set(public) != PUBLIC_KEYS:
        raise ValueError('P57 B public input keys differ from allowlist')
    value = {}
    for key, item in public.items():
        value[key] = torch.as_tensor(item, dtype=torch.bool if key == 'mask_valid' else torch.float32)
    expected = dict(raw=(2,3,3072), boxes=(2,3,4), visible=(2,3), proprio=(18,),
                    mask_tokens=(2,3,55), mask_valid=(2,3), camera_K=(2,3,3),
                    camera_pose=(2,4,4), eef_transform=(4,4))
    for key,shape in expected.items():
        if tuple(value[key].shape) != shape or not torch.isfinite(value[key]).all():
            raise ValueError('Invalid public '+key)
    if not torch.equal(value['visible'].bool(),value['mask_valid']):
        raise ValueError('Cached and live mask visibility differ')
    return value


def predict_public_uv(readout, public):
    public = strict_public(public)
    input_episode = dict(input=dict(features={key:public[key][None] for key in ('raw','boxes','visible','proprio')},
                                    **{key:public[key][None] for key in PUBLIC_KEYS-{'raw','boxes','visible','proprio'}}))
    x, obs = public_matrix(input_episode, readout['mode'])
    uv = predict_pixels(predict_relative(readout['model'],readout['coef'],x),obs)[0]
    return uv.detach().numpy(), public['mask_valid'][:,1:3].numpy(), public


def public_rays(uv, role_valid, K, camera, eef):
    """A,C,valid [role,view] in book EEF and goal world coordinates."""
    uv=np.asarray(uv,np.float64);valid=np.asarray(role_valid,bool)
    K=np.asarray(K,np.float64);camera=np.asarray(camera,np.float64);eef=np.asarray(eef,np.float64)
    A=np.zeros((2,2,3,3),np.float32);C=np.zeros((2,2,3),np.float32)
    for view in range(2):
        for role in range(2):
            if not valid[view,role]:continue
            origin,direction=rays(uv[view,role],K[view],camera[view])
            if role==0:
                origin=eef[:3,:3].T@(origin-eef[:3,3])
                direction=eef[:3,:3].T@direction
            A[role,view]=np.eye(3)-np.outer(direction,direction)
            C[role,view]=origin
    return torch.from_numpy(A),torch.from_numpy(C),torch.from_numpy(valid.T.copy())


def instant_geometry(uv, valid, K, camera, eef):
    fit=fit_origin_translations(uv[None],valid[None],K[None],camera[None],eef[None],history=1)[0]['instant']
    if not fit['valid']['book'] or not fit['valid']['goal']:
        return None,fit
    p_book=eef[:3,:3].T@(fit['p_book_world']-eef[:3,3])
    g=np.r_[p_book,fit['p_goal_world']].astype(np.float32)
    return g,fit


def action_context(public, R_held, R_goal):
    eef=public['eef_transform'].numpy();R_desired=R_goal@R_held.T
    dr=Rotation.from_matrix(R_desired@eef[:3,:3].T).as_rotvec().astype(np.float32)
    box=public['boxes'][:,1,:]
    wh=(box[:,2:]-box[:,:2]).reshape(4)
    grip=public['proprio'][9:11]
    return (torch.from_numpy(eef[:3,3].copy()).float(),torch.from_numpy(R_desired.copy()).float(),
            torch.from_numpy(dr),wh.float(),grip.float())


class GeometryPolicy(nn.Module):
    def __init__(self,config=PolicyConfig()):
        super().__init__()
        if config.latent!=6 or config.hidden!=64 or config.history!=8:
            raise ValueError('P57 B architecture mismatch')
        self.config=config
        self.B=nn.Parameter(torch.eye(6))
        self.z0=nn.Parameter(torch.zeros(6))
        self.policy=nn.Sequential(nn.Linear(12,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,6))

    def decoded_geometry(self,g,z):
        return g+self.config.geometry_scale*torch.matmul(self.B,z)

    def action(self,g,z,eef_pos,R_desired,dr,bboxwh,gripper):
        q=self.decoded_geometry(g,z)
        p_book,p_goal=q[:3],q[3:]
        dp=p_goal-R_desired@p_book-eef_pos
        feature=torch.cat((dp/self.config.translation_scale,dr/self.config.rotation_scale,bboxwh,gripper))
        a=torch.tanh(self.policy(feature))
        translation=a[:3]*torch.clamp(self.config.translation_cap/a[:3].norm().clamp_min(1e-8),max=1.)
        rotation=a[3:]*torch.clamp(self.config.rotation_cap/a[3:].norm().clamp_min(1e-8),max=1.)
        return torch.cat((translation,rotation))

    def command(self,*arguments):
        a=self.action(*arguments)
        return torch.cat((a,a.new_ones(1)))

    def _geometry_system(self,g,support):
        A,C,valid=support
        mask=valid[...,None,None]
        # Match the global valid-ray average used by ray_loss exactly.
        h=(A*mask).sum((0,2))/valid.sum().clamp_min(1)
        H=torch.block_diag(h[0],h[1])
        metric=(self.B.T@H@self.B+self.config.damping*torch.eye(6,device=g.device,dtype=g.dtype)).detach()
        return H,metric

    def ray_loss(self,g,z,support):
        A,C,valid=support
        q=self.decoded_geometry(g,z).reshape(2,3)
        d=(q[None,:,None,:]-C)/self.config.geometry_scale
        error=torch.einsum('savi,savij,savj->sav',d,A,d)
        return .5*(error*valid).sum()/valid.sum().clamp_min(1)

    def adapt(self,g,z,support,*,create_graph=False):
        _,metric=self._geometry_system(g,support)
        losses=[]
        for _ in range(self.config.inner_steps):
            if not z.requires_grad:z=z.detach().requires_grad_(True)
            loss=self.ray_loss(g,z,support)
            gradient,=torch.autograd.grad(loss,z,create_graph=create_graph)
            direction=torch.linalg.solve(metric,gradient)
            direction=direction*torch.clamp(1./direction.norm().clamp_min(1e-8),max=1.)
            z=z-self.config.inner_lr*direction
            losses.append(loss)
            if not create_graph:z=z.detach().requires_grad_(True)
        return z,torch.stack(losses)

    def direct(self,g,z0,support):
        A,C,valid=support
        _,metric=self._geometry_system(g,support)
        d=(g.reshape(2,3)[None,:,None,:]-C)/self.config.geometry_scale
        projected=torch.einsum('savij,savj->savi',A,d)*valid[...,None]
        count=valid.sum().clamp_min(1)
        rhs=-self.B.T@projected.sum((0,2)).reshape(6)/count+self.config.damping*z0
        return torch.linalg.solve(metric,rhs)


class GeometryPolicyRuntime:
    def __init__(self,model,readout,arm='geometry_fixed',device='cpu'):
        if arm not in ARMS:raise ValueError('Unknown P57 B arm')
        self.model=model.to(device).eval().requires_grad_(False)
        self.readout=readout;self.arm=arm;self.device=device
        self.book_prior,self.goal_prior,self.R_held,self.R_goal=prior_frames(readout)
        self.reset()

    def reset(self):
        self.z=self.model.z0.detach().clone().to(self.device)
        self.history=[];self.records=[];self.t=0

    def step(self,public):
        uv,role_valid,p=predict_public_uv(self.readout,public)
        k=p['camera_K'].numpy();camera=p['camera_pose'].numpy();eef=p['eef_transform'].numpy()
        A,C,valid=public_rays(uv,role_valid,k,camera,eef)
        self.history.append((A.to(self.device),C.to(self.device),valid.to(self.device)))
        self.history=self.history[-self.model.config.history:]
        if self.arm=='geometry_prior':
            g=np.r_[self.book_prior,self.goal_prior].astype(np.float32);fit=None
        else:
            g,fit=instant_geometry(uv,role_valid,k,camera,eef)
            if g is None:raise ValueError('Current public origin rays are degenerate')
        support=tuple(torch.stack([entry[i] for entry in self.history]) for i in range(3))
        gt=torch.tensor(g,device=self.device)
        z_before=self.z.detach().clone()
        with torch.no_grad():loss_before=float(self.model.ray_loss(gt,z_before,support))
        if self.t>=1 and self.arm=='geometry_online':
            with torch.enable_grad():self.z,_=self.model.adapt(gt,self.z,support)
            self.z=self.z.detach()
        elif self.t>=1 and self.arm=='geometry_direct':
            with torch.no_grad():self.z=self.model.direct(gt,self.model.z0.detach(),support).detach()
        eef_pos,R_desired,dr,bboxwh,gripper=action_context(p,self.R_held,self.R_goal)
        args=(gt,self.z,*[v.to(self.device) for v in (eef_pos,R_desired,dr,bboxwh,gripper)])
        with torch.no_grad():
            action=self.model.command(*args).detach().cpu().numpy()
            fixed_args=(gt,self.model.z0.detach(),*[v.to(self.device) for v in (eef_pos,R_desired,dr,bboxwh,gripper)])
            fixed_action=self.model.command(*fixed_args).detach().cpu().numpy()
            fixed_g=self.model.decoded_geometry(gt,self.model.z0.detach()).detach().cpu().numpy()
            decoded=self.model.decoded_geometry(gt,self.z).detach().cpu().numpy()
            loss_after=float(self.model.ray_loss(gt,self.z,support))
        telemetry=dict(t=self.t,arm=self.arm,g=g.tolist(),z=self.z.detach().cpu().tolist(),
                       z_before=z_before.cpu().tolist(),z_delta=float((self.z-z_before).norm().detach().cpu()),
                       decoded_g=decoded.tolist(),fixed_z_decoded_g=fixed_g.tolist(),
                       geometry_delta_m=float(np.linalg.norm(decoded-fixed_g)),
                       action=action.tolist(),fixed_z_same_obs_action=fixed_action.tolist(),
                       action_delta_norm=float(np.linalg.norm(action[:6]-fixed_action[:6])),
                       ray_loss_before=loss_before,ray_loss_after=loss_after,
                       inner_steps=(self.model.config.inner_steps if self.t>=1 and self.arm=='geometry_online' else 0),
                       valid_rays=int(valid.sum()),ray_A=A.numpy().tolist(),ray_C=C.numpy().tolist(),
                       ray_valid=valid.numpy().tolist(),pred_uv=uv.tolist(),
                       fit_valid=(fit is None or all(fit['valid'].values())))
        self.records.append(telemetry);self.t+=1
        return action,telemetry


def load_geometry_policy(policy_head_path,a2_head_path,device='cpu'):
    import hashlib
    a2path=Path(a2_head_path)
    digest=hashlib.sha256(a2path.read_bytes()).hexdigest()
    payload=torch.load(policy_head_path,map_location='cpu',weights_only=True)
    if payload.get('schema')!='p57.geometry-policy.v1' or payload.get('a2_head_sha256')!=digest:
        raise ValueError('P57 B checkpoint/A2 readout binding mismatch')
    readout=torch.load(a2path,map_location='cpu',weights_only=True)
    if readout.get('schema')!='p57.a2-translation-readout.v1' or readout.get('mode')!='mask_geometry':
        raise ValueError('P57 B requires frozen A2 mask readout')
    model=GeometryPolicy(PolicyConfig(**payload['config']))
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.to(device).eval(),readout,payload
