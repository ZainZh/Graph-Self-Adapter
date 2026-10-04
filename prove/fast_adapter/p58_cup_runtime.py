"""Frozen P58 cup functional-frame runtime; public observations only."""
from __future__ import annotations

import numpy as np
import torch

from prove.fast_adapter.p58_cup_functional import (AdapterConfig, FunctionalFrameAdapter,
    camera_pose, camera_to_world, points_to_pose, public_features, ray_system, support_window)
from prove.task_graph.p57_public_geometry import mask_token


ARMS = ('fixed','online','direct')


def _numpy_readout(package):
    return {k:(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v)
            for k,v in package['readout'].items()}


def load_runtime(path, arm='online', device='cpu'):
    package=torch.load(path,map_location='cpu',weights_only=False)
    if package.get('schema')!='prove.p58-cup-functional.v1': raise ValueError('Wrong P58 cup checkpoint')
    config=package['config'];representation=package['representation']
    adapter=FunctionalFrameAdapter(torch.zeros(config['point_count']*3,config['latent_dim']),
        AdapterConfig(representation=representation,latent_dim=config['latent_dim']))
    adapter.load_state_dict(package['model']);adapter.to(device).eval().requires_grad_(False)
    return CupFunctionalRuntime(adapter,_numpy_readout(package),representation,
        arm,package.get('mean_rotation'),device)


class CupFunctionalRuntime:
    def __init__(self,model,readout,representation,arm,mean_rotation=None,device='cpu'):
        if arm not in ARMS: raise ValueError('Unknown P58 cup arm')
        self.model,self.readout,self.representation,self.arm,self.device=model,readout,representation,arm,device
        self.mean_rotation=None if mean_rotation is None or not len(mean_rotation) else np.asarray(mean_rotation)
        self.reset()

    def reset(self):
        self.z=self.model.z0.detach().clone()[None]
        self.history=[];self.records=[];self.points=None;self.update_events=0

    def _predict(self,obs,masks):
        tokens=np.zeros((3,55),np.float32);valid=np.zeros(3,bool)
        for role in range(3):tokens[role],valid[role]=mask_token(np.asarray(masks[role],np.uint8))
        cam=camera_pose(obs['camera_position'],obs['camera_rotation'])
        eef=camera_pose(obs['position'],obs['rotation'])
        feature=public_features(tokens,valid,obs['K'],cam,eef)
        r=self.readout;pc=((feature-r['mean'])/r['scale'])@r['components'].T
        pred=np.r_[1.,pc]@r['coef'];n=self.model.config.point_count
        bbox=tokens[2,:4];center=(bbox[:2]+bbox[2:])*.5;extent=np.maximum(bbox[2:]-bbox[:2],1/448.)
        uv=(center[None]+extent[None]*pred[:2*n].reshape(n,2))*448.
        xyz=pred[2*n:].reshape(n,3)*.01
        base=camera_to_world(xyz,cam)
        A,C=ray_system(uv,obs['K'],cam)
        return base,uv,A,C,bool(valid[2])

    def update(self,obs,masks,t):
        base,uv,A,C,valid=self._predict(obs,masks)
        before=self.z.detach().clone();loss_before=loss_after=None
        if valid:
            self.history.append((A,C,np.ones(self.model.config.point_count,bool)))
            self.history=support_window(self.history)
        if self.history:
            At=torch.tensor(np.stack([x[0] for x in self.history]),dtype=torch.float32,device=self.device)[None]
            Ct=torch.tensor(np.stack([x[1] for x in self.history]),dtype=torch.float32,device=self.device)[None]
            Vt=torch.tensor(np.stack([x[2] for x in self.history]),dtype=torch.bool,device=self.device)[None]
            g=torch.tensor(base,dtype=torch.float32,device=self.device)[None]
            loss_before=float(self.model.ray_loss(g,self.z,At,Ct,Vt)[0])
            if len(self.history)>=3:
                if self.arm=='online': self.z,_=self.model.adapt(g,self.z,At,Ct,Vt)
                elif self.arm=='direct': self.z=self.model.direct(g,self.z,At,Ct,Vt).detach()
                if self.arm!='fixed': self.update_events+=1
            loss_after=float(self.model.ray_loss(g,self.z,At,Ct,Vt)[0])
            self.points=self.model.decode(g,self.z).detach().cpu().numpy()[0]
        self.records.append(dict(t=int(t),S=len(self.history),valid=valid,uv=uv.tolist(),
            z=self.z.detach().cpu().numpy()[0].tolist(),latent_change=float((self.z-before).norm()),
            loss_before=loss_before,loss_after=loss_after,update_events=self.update_events,
            points=None if self.points is None else self.points.tolist()))
        return self.estimate()

    def estimate(self):
        if self.points is None:return None
        grasp=self.points[0]
        if self.representation=='functional_frame':R=points_to_pose(self.points)[:3,:3]
        else:R=self.mean_rotation
        # Algebraic bridge: make the unchanged P45 controller's grip expression
        # equal the predicted semantic grasp point, without a truth size input.
        size=np.array([.003,0.])
        p=grasp+R[:,2]*.004
        return p,R,size
