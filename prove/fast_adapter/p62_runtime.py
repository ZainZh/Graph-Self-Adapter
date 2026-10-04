"""P62 frozen RGB policy frontend, with public-contour-only persistent z updates.

No simulator, target configuration, PnP or detected corner is accepted here.
The first observed own-EEF pose fixes the episode coordinate frame.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from prove.fast_adapter.p61_encoder import P61RelationEncoder,first_eef_anchor_transforms
from prove.fast_adapter.p61_adaptation import corrected_components,contour_residual
from prove.fast_adapter.p62_adaptation import adapt_z,candidate_configuration
from prove.fast_adapter.p60_contour import ContourMeasurement,causal_support,outer_hull_boundary
from prove.fast_adapter.visual_alignment import FrozenDinoROI

PUBLIC_POSE_KEYS=('K','camera_position','camera_rotation','position','rotation')

def pose_matrix(position,rotation):
    matrix=np.eye(4,dtype=np.float32)
    matrix[:3,:3]=np.asarray(rotation);matrix[:3,3]=np.asarray(position)
    return matrix

class P62Runtime:
    def __init__(self,checkpoint,model_dir,arm,device='cuda'):
        if arm not in ('fixed','translation_size5'):raise ValueError('unknown P62 arm')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['schema']!='prove.p61-relation-checkpoint.v1':raise ValueError('checkpoint schema')
        self.encoder=P61RelationEncoder(**payload['constructor'])
        self.encoder.load_state_dict(payload['state_dict'],strict=True)
        self.encoder.to(device).eval().requires_grad_(False)
        self.vision=FrozenDinoROI(model_dir,device=device,microbatch=1)
        self.device=device;self.arm=arm
        self.anchor=None;self.z=torch.zeros(8,device=device)
        self.history=[];self.records=[];self.geometry=None;self._estimate=None;self.last_t=-1

    def update(self,public,rgb,masks,t):
        if set(public)!=set(PUBLIC_POSE_KEYS):raise ValueError('only public camera/own-pose fields accepted')
        if t<=self.last_t:raise ValueError('observations must strictly increase')
        self.last_t=t
        rgb=np.asarray(rgb);masks=np.asarray(masks,dtype=bool)
        if rgb.shape!=(448,448,3) or masks.shape!=(3,448,448):raise ValueError('native448 role image required')
        if not all(np.isfinite(public[k]).all() for k in PUBLIC_POSE_KEYS):raise ValueError('nonfinite public pose')
        world_eef=torch.as_tensor(pose_matrix(public['position'],public['rotation']),device=self.device)
        world_camera=torch.as_tensor(pose_matrix(public['camera_position'],public['camera_rotation']),device=self.device)
        if self.anchor is None:self.anchor=world_eef.clone()
        anchor_camera,anchor_eef=first_eef_anchor_transforms(self.anchor,world_camera,world_eef)
        valid=masks.reshape(3,-1).any(-1)
        with torch.no_grad():
            pixels=torch.from_numpy(rgb.copy()).to(self.device).permute(2,0,1)[None].float()/255
            pixels=((pixels-self.vision.mean)/self.vision.std).to(torch.bfloat16)
            hidden=self.vision.model(pixel_values=pixels,output_hidden_states=True,return_dict=True).hidden_states
            # Preserve the materialized train-cache precision contract.
            layers=torch.stack([hidden[i][:,self.vision.prefix:] for i in (-12,-8,-4)],1).to(torch.float16).float()
            patch=F.avg_pool2d(torch.from_numpy(masks.copy()).to(self.device).float(),16,16)
            self.geometry=self.encoder(dino_layers=layers[:,None],role_patch_masks=patch[None,None],
                role_valid=torch.as_tensor(valid,device=self.device)[None,None],
                K=torch.as_tensor(public['K'],device=self.device,dtype=torch.float32)[None,None],
                T_anchor_camera=anchor_camera[None,None],T_anchor_eef=anchor_eef[None],image_hw=(448,448)).geometry8[0].detach()
        if not torch.isfinite(self.geometry).all():raise ValueError('nonfinite encoder geometry')
        boundary,info=outer_hull_boundary(masks[2]);admitted=boundary is not None
        if admitted:
            self.history.append(ContourMeasurement(boundary,np.asarray(info['hull_vertices'],dtype=np.float32),
                np.asarray(public['K']),np.asarray(public['camera_position']),np.asarray(public['camera_rotation']),(448,448),int(t)))
        support=causal_support(self.history);before=self.z.clone();inner=[]
        pre_loss=post_loss=None
        if support and valid[2]:
            with torch.no_grad():pre_loss=float(.5*contour_residual(self.geometry,self.z,self.anchor,support).square().mean())
            if admitted and len(support)>=3 and self.arm=='translation_size5':
                with torch.enable_grad():self.z,inner=adapt_z(self.geometry,self.z,self.anchor,support,candidate=self.arm,create_graph=False)
            with torch.no_grad():post_loss=float(.5*contour_residual(self.geometry,self.z,self.anchor,support).square().mean())
        with torch.no_grad():
            center,rotation,size=corrected_components(self.geometry,self.z)
            center=self.anchor[:3,:3]@center+self.anchor[:3,3];rotation=self.anchor[:3,:3]@rotation
            self._estimate=tuple(x.detach().cpu().numpy() for x in (center,rotation,size)) if valid[2] else None
        if not torch.equal(self.z[3:6],torch.zeros_like(self.z[3:6])):raise RuntimeError('inactive rotation latent changed')
        self.records.append(dict(t=int(t),candidate=self.arm,admitted=admitted,role_valid=valid.tolist(),
            geometry8=self.geometry.cpu().numpy(),z_before=before.cpu().numpy(),z_after=self.z.cpu().numpy(),
            U=len(inner),inner=inner,S=len(support),support_times=[m.t for m in support],
            pre_feedback_loss=pre_loss,post_feedback_loss=post_loss,T_world_anchor=self.anchor.cpu().numpy()))
        return admitted

    def estimate(self):
        return self._estimate
