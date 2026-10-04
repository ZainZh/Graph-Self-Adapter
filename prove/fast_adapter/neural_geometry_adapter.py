"""P43 learned visual geometry, persistent neural latent, fixed P41 controller.

The encoder and rank-eight latent decoder are trained. PnP initialization,
calibrated ray loss, Gauss-Newton metric, SE3 extraction and action decoding are
explicit analytic priors. No evaluator or simulator object state is accepted.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from prove.fast_adapter.geometry3d_adapter import initial, pose, project, rays

GEOMETRY_SCALE = .05
FEATURE_DIM = 95  # LiLa48 + corners8 + camera12 + proprio12 + role bbox/visible15

def observation_features(obs, visual_mode='normal'):
    visual=np.asarray(obs['local'],np.float32).reshape(48).copy()
    if visual_mode=='zero': visual[:]=0
    corners=np.asarray(obs['corners'],np.float32).reshape(8)/224.-1.
    return np.concatenate([visual,corners,np.asarray(obs['camera_position'])/.2,
        np.asarray(obs['camera_rotation']).reshape(9),np.asarray(obs['position'])/.2,
        np.asarray(obs['rotation']).reshape(9),np.asarray(obs.get('role_bbox',np.zeros((3,4)))).reshape(12)/448.,
        np.asarray(obs.get('role_visible',np.zeros(3))).reshape(3)]).astype(np.float32)


def align_observation(obs, reference):
    """Resolve only the unmarked rectangle's cyclic correspondence."""
    uv=project(reference,obs)
    choices=[np.roll(np.asarray(obs['corners']),-k,axis=0) for k in range(4)]
    return dict(obs,corners=min(choices,key=lambda x:np.square(x-uv).sum()))


def ray_matrices(obs):
    v=rays(obs);a=np.eye(3)[None]-v[:,:,None]*v[:,None,:]
    return a.astype(np.float32),np.broadcast_to(obs['camera_position'],(4,3)).astype(np.float32).copy()


class NeuralGeometry(nn.Module):
    def __init__(self, task, feature_dim=FEATURE_DIM, width=96, latent_dim=8):
        super().__init__();self.task=task;self.feature_dim=feature_dim;self.width=width;self.latent_dim=latent_dim
        self.visual_geometry_token=nn.Sequential(nn.Linear(feature_dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.initial_latent=nn.Linear(width,latent_dim)
        self.current_residual=nn.Sequential(nn.Linear(width*2+12,width),nn.SiLU(),nn.Linear(width,12))
        # A learned low-rank decoder, initialized using task nominal frame only.
        a,b=(.019,.016) if task=='insert' else (.018,.019)
        local=np.array([[-a,b,0],[a,b,0],[a,-b,0],[-a,-b,0]])
        rotation=np.eye(3) if task=='insert' else np.array([[1,0,0],[0,0,-1],[0,1,0]])
        corners=local@rotation.T
        columns=[]
        for j in range(3): columns.append(np.tile(np.eye(3)[j],(4,1)))
        for j in range(3): columns.append(np.cross(np.eye(3)[j],corners)/.04)
        columns.extend([np.tile(rotation[:,0],(4,1))*np.array([-1,1,1,-1])[:,None],
                        np.tile(rotation[:,1],(4,1))*np.array([1,1,-1,-1])[:,None]])
        basis=np.stack([v.reshape(12) for v in columns],axis=1)
        basis=basis/np.maximum(np.linalg.norm(basis,axis=0,keepdims=True),1e-8)
        self.latent_basis=nn.Parameter(torch.tensor(basis,dtype=torch.float32))
        nn.init.normal_(self.initial_latent.weight,std=.005);nn.init.zeros_(self.initial_latent.bias)
        nn.init.normal_(self.current_residual[-1].weight,std=.005);nn.init.zeros_(self.current_residual[-1].bias)

    def encode(self, first, current, reference):
        token0=self.visual_geometry_token(first);token=self.visual_geometry_token(current)
        offset=self.current_residual(torch.cat([token0,token,reference.flatten(-2)/.2],dim=-1))
        return self.initial_latent(token0),offset

    def decode(self, reference, offset, z):
        return reference+GEOMETRY_SCALE*(offset+z@self.latent_basis.T).reshape(*z.shape[:-1],4,3)

    def ray_loss(self, points, A, C):
        # A [B,S,4,3,3], C [B,S,4,3]; visible-only support.
        difference=(points[:,None]-C)/GEOMETRY_SCALE
        return .5*torch.einsum('bski,bskij,bskj->bsk',difference,A,difference).sum(-1).mean(-1)

    def adapt(self, reference, offset, z, A, C, steps=3, create_graph=False):
        # Learned decoder Jacobian supplies the latent metric. Analytic rays
        # provide the observation metric; no target dimensions are consulted.
        b=self.latent_basis.reshape(4,3,self.latent_dim)
        h=A.mean(1)
        metric=torch.einsum('kid,bkij,kje->bde',b,h,b)
        metric=metric+torch.eye(self.latent_dim,device=z.device)[None]*1e-4
        metric=metric.detach()  # stop-gradient metric; gradient itself remains meta-differentiable
        records=[]
        for _ in range(steps):
            if not z.requires_grad: z=z.detach().requires_grad_(True)
            loss=self.ray_loss(self.decode(reference,offset,z),A,C)
            gradient,=torch.autograd.grad(loss.sum(),z,create_graph=create_graph)
            change=torch.linalg.solve(metric,gradient.unsqueeze(-1)).squeeze(-1)
            change=change*torch.clamp(1./change.norm(dim=-1,keepdim=True).clamp_min(1e-8),max=1.)
            next_z=z-.65*change
            records.append(dict(loss=float(loss.mean().detach()),gradient=float(gradient.norm(dim=-1).mean().detach()),delta=float((next_z-z).norm(dim=-1).mean().detach())))
            z=next_z if create_graph else next_z.detach().requires_grad_(True)
        return z,records

    def config(self):
        return dict(task=self.task,feature_dim=self.feature_dim,width=self.width,latent_dim=self.latent_dim)


class NeuralGeometryLatent:
    """Runtime state: neural weights frozen; z alone can receive gradients."""
    def __init__(self,model,arm='continuous',visual_mode='normal',device='cpu'):
        self.model=model.to(device).eval();self.model.requires_grad_(False)
        self.arm=arm;self.visual_mode=visual_mode;self.device=device
        self.reference=None;self.first=None;self.z=None;self.initial_z=None;self.points=None;self.history=[];self.records=[]

    def tensor(self,x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)

    def update(self,obs,t):
        if obs['corners'] is None:return
        if self.reference is None:
            self.reference=initial(obs,self.model.task)
            aligned=align_observation(obs,self.reference)
            self.first=observation_features(aligned,self.visual_mode)
        aligned=align_observation(obs,self.points if self.points is not None else self.reference)
        current=observation_features(aligned,self.visual_mode)
        reference=self.tensor(self.reference)[None]
        with torch.no_grad():
            z0,offset=self.model.encode(self.tensor(self.first)[None],self.tensor(current)[None],reference)
        if self.z is None:self.z=z0.detach();self.initial_z=self.z.clone()
        before=self.z.clone();fixed_points=self.model.decode(reference,offset,self.z).detach()
        self.history.append(ray_matrices(aligned));support=self.history if len(self.history)<=32 else self.history[:12]+self.history[-20:]
        A=self.tensor(np.stack([x[0] for x in support]))[None];C=self.tensor(np.stack([x[1] for x in support]))[None]
        loss_before=float(self.model.ray_loss(fixed_points,A,C)[0])
        update_allowed=self.arm=='continuous' or (self.arm=='prefix' and t<12)
        inner=[]
        if update_allowed and len(self.history)>=3:
            self.z,inner=self.model.adapt(reference,offset,self.z,A,C)
            self.z=self.z.detach()
        self.points=self.model.decode(reference,offset,self.z).detach().cpu().numpy()[0]
        loss_after=float(self.model.ray_loss(self.tensor(self.points)[None],A,C)[0])
        self.records.append(dict(t=int(t),S=len(support),U=len(inner),z=self.z.cpu().numpy()[0].tolist(),
            latent_change=float((self.z-before).norm()),z_from_initial=float((self.z-self.initial_z).norm()),
            initial_z_points=self.model.decode(reference,offset,self.initial_z).detach().cpu().numpy()[0].tolist(),point_change=float(np.linalg.norm(self.points-fixed_points.cpu().numpy()[0])),
            support_loss_before=loss_before,support_loss_after=loss_after,inner=inner,
            pre_update_points=fixed_points.cpu().numpy()[0].tolist()))

    def estimate(self):return None if self.points is None else pose(self.points)


def load_model(path,device='cpu'):
    package=torch.load(path,map_location=device,weights_only=False)
    model=NeuralGeometry(**package['model_config']);model.load_state_dict(package['model']);model.eval()
    return model,package
