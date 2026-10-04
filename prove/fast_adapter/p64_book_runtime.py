"""P64 book runtime restricted to current pixels/masks, camera and robot state."""
import numpy as np
import torch
from prove.fast_adapter.p64_book_bridge import (
    BookPriors,BookMeasurement,visible_boundary,adapt_book_z,frozen_book_action)
from prove.fast_adapter.p61_encoder import first_eef_anchor_transforms
from prove.task_graph.p57_public_geometry import eef_transform
from prove.fast_adapter.neural_geometry_visual import mask_geometry
from torch.nn import functional as F

PUBLIC_KEYS=frozenset(('rgb','visible_masks','camera_K','camera_pose','eef_pos',
                      'eef_quat_xyzw','gripper_qpos','joint_pos','joint_vel'))
ENCODER_KEYS=('dino_layers','role_patch_masks','role_valid','K','T_anchor_camera','T_anchor_eef')


def encoder_inputs(public,device):
    return {k:public[k].to(device).float().unsqueeze(0) if k!='role_valid'
            else public[k].to(device).bool().unsqueeze(0) for k in ENCODER_KEYS}


def decode_measurements(rows):
    return [BookMeasurement(**{k:(v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v)
                              for k,v in row.items()}) for row in rows]


class BookRuntime:
    def __init__(self,encoder,policy,priors,arm,vision,device='cuda'):
        if arm not in ('fixed','continuous'):raise ValueError('unknown arm')
        self.encoder=encoder.to(device).eval().requires_grad_(False)
        self.policy=policy.to(device).eval().requires_grad_(False)
        self.priors=priors if isinstance(priors,BookPriors) else BookPriors(**priors)
        self.arm=arm;self.vision=vision;self.device=device
        self.reset()

    def reset(self):
        self.anchor=None;self.z=torch.zeros(9,device=self.device);self.history=[];self.t=0

    def exit(self):
        self.reset()

    def step(self,frame):
        if set(frame)!=PUBLIC_KEYS:raise ValueError('book runtime accepts only allowlisted public frame')
        rgb=np.asarray(frame['rgb']);masks=np.asarray(frame['visible_masks'],bool)
        if rgb.shape!=(2,448,448,3) or masks.shape!=(2,3,448,448):raise ValueError('native448 dual-view required')
        eef=torch.as_tensor(eef_transform(frame['eef_pos'],frame['eef_quat_xyzw']),dtype=torch.float32,device=self.device)
        if self.anchor is None:self.anchor=eef.detach().clone()
        camera=torch.as_tensor(frame['camera_pose'],dtype=torch.float32,device=self.device)
        ac,ae=first_eef_anchor_transforms(self.anchor,camera,eef)
        with torch.no_grad():
            pixels=torch.from_numpy(rgb.copy()).to(self.device).permute(0,3,1,2).float()/255
            pixels=((pixels-self.vision.mean)/self.vision.std).to(torch.bfloat16)
            hidden=self.vision.model(pixel_values=pixels,output_hidden_states=True,return_dict=True).hidden_states
            layers=torch.stack([hidden[i][:,self.vision.prefix:] for i in (-12,-8,-4)],1).to(torch.float16).float()
            patch=F.avg_pool2d(torch.tensor(masks,device=self.device,dtype=torch.float32).reshape(6,1,448,448),16,16).reshape(2,3,28,28)
            public=dict(dino_layers=layers,role_patch_masks=patch,role_valid=patch.flatten(-2).any(-1),
                K=torch.as_tensor(frame['camera_K'],device=self.device,dtype=torch.float32),T_anchor_camera=ac,T_anchor_eef=ae)
            geometry=self.encoder(**encoder_inputs(public,self.device),image_hw=(448,448)).geometry[0].detach()
        added=0
        for view in range(2):
            for role in (1,2):
                boundary=visible_boundary(masks[view,role],np.any(np.delete(masks[view],role,axis=0),axis=0))
                if boundary is None:continue
                self.history.append(BookMeasurement(role,boundary,np.asarray(frame['camera_K'][view]),
                    np.asarray(frame['camera_pose'][view]),eef.cpu().numpy(),self.t,self.anchor.cpu().numpy()));added+=1
        self.history=[m for m in self.history if m.t>=self.t-7]
        before=self.z.clone();updates=[]
        if self.arm=='continuous' and self.t>=1 and added and self.history:
            with torch.enable_grad():self.z,updates=adapt_book_z(geometry,self.z,self.priors,self.history)
            self.z=self.z.detach()
        boxes=np.stack([mask_geometry(m)[0] for m in masks])/448.
        wh=(boxes[:,1,2:]-boxes[:,1,:2]).reshape(4)
        args=(self.priors,eef,wh,frame['gripper_qpos'],self.anchor)
        with torch.no_grad():
            action=frozen_book_action(self.policy,geometry,self.z,*args)
            fixed=frozen_book_action(self.policy,geometry,torch.zeros_like(self.z),*args)
        record=dict(t=self.t,arm=self.arm,geometry9=geometry.cpu().tolist(),z_before=before.cpu().tolist(),
            z_after=self.z.cpu().tolist(),admitted_measurements=added,support_measurements=len(self.history),
            support_times=[m.t for m in self.history],updates=updates,
            action=action.cpu().tolist(),same_observation_fixed_action=fixed.cpu().tolist(),
            action_delta_norm=float((action-fixed).norm()),orientation_update=False,
            scale_direct_action_channels=False)
        self.t+=1
        return action.cpu().numpy(),record
