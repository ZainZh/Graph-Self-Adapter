"""Current frozen LiLa/DINO cup encoder with persistent public contour z."""
import numpy as np
import torch
from torch.nn import functional as F
from prove.fast_adapter.p64_common import load_encoder
from prove.fast_adapter.p64_cup import public_pose, adapt
from prove.fast_adapter.p61_adaptation import corrected_components
from prove.fast_adapter.p61_encoder import first_eef_anchor_transforms
from prove.fast_adapter.p60_contour import outer_hull_boundary, ContourMeasurement, causal_support
from prove.fast_adapter.visual_alignment import FrozenDinoROI
from tools.fast_adapter.p64_cup_train import matrix, geometry8


class CupRuntime:
    def __init__(self, checkpoint, model_dir, arm, device='cuda'):
        if arm not in ('fixed','translation_size5'):raise ValueError('Unknown cup arm')
        self.encoder=load_encoder(checkpoint,device=device)
        if self.encoder.task!='cup':raise ValueError('Cup checkpoint required')
        self.vision=FrozenDinoROI(model_dir,device=device,microbatch=1)
        self.arm,self.device=arm,device;self.z=torch.zeros(8,device=device)
        self.anchor=None;self.history=[];self.records=[];self.geometry=None;self.last_t=-1
        self._estimate=None

    def update(self, public, rgb, masks, t):
        if set(public)!=set(('K','camera_position','camera_rotation','position','rotation')):
            raise ValueError('Public allowlist only')
        public=public_pose(public)
        if t<=self.last_t:raise ValueError('Observations must be causal')
        self.last_t=t
        rgb=np.asarray(rgb);masks=np.asarray(masks,dtype=bool)
        if rgb.shape!=(448,448,3) or masks.shape!=(3,448,448):raise ValueError('Native448 roles required')
        eef=matrix(public['position'],public['rotation']).to(self.device)
        camera=matrix(public['camera_position'],public['camera_rotation']).to(self.device)
        if self.anchor is None:self.anchor=eef.clone()
        ac,ae=first_eef_anchor_transforms(self.anchor,camera,eef)
        valid=masks.reshape(3,-1).any(-1)
        with torch.no_grad():
            pixels=torch.from_numpy(rgb.copy()).to(self.device).permute(2,0,1)[None].float()/255
            hidden=self.vision.model(pixel_values=((pixels-self.vision.mean)/self.vision.std).to(torch.bfloat16),output_hidden_states=True,return_dict=True).hidden_states
            layers=torch.stack([hidden[i][:,self.vision.prefix:] for i in (-12,-8,-4)],1).to(torch.float16).float()
            patch=F.avg_pool2d(torch.tensor(masks,device=self.device,dtype=torch.float32),16,16)
            value=self.encoder(dino_layers=layers[:,None],role_patch_masks=patch[None,None],
                role_valid=torch.tensor(valid,device=self.device)[None,None],K=torch.tensor(public['K'],device=self.device,dtype=torch.float32)[None,None],
                T_anchor_camera=ac[None,None],T_anchor_eef=ae[None],image_hw=(448,448)).geometry[0]
            self.geometry=geometry8(value,self.anchor).detach()
        boundary,info=outer_hull_boundary(masks[2]);admitted=boundary is not None
        if admitted:self.history.append(ContourMeasurement(boundary,np.asarray(info['hull_vertices'],dtype=np.float32),public['K'],public['camera_position'],public['camera_rotation'],(448,448),int(t)))
        support=causal_support(self.history);before=self.z.clone();inner=[]
        if admitted and len(support)>=3:
            with torch.enable_grad():self.z,inner=adapt(self.geometry,self.z,self.anchor,support,arm=self.arm)
        with torch.no_grad():
            center,rotation,size=corrected_components(self.geometry,self.z)
            estimate=(self.anchor[:3,:3]@center+self.anchor[:3,3],self.anchor[:3,:3]@rotation,size)
            self._estimate=tuple(x.cpu().numpy() for x in estimate) if valid[2] else None
        if bool(self.z[3:6].abs().any()):raise RuntimeError('Orientation update forbidden')
        self.records.append(dict(t=int(t),admitted=admitted,S=len(support),support_times=[m.t for m in support],
            geometry8=self.geometry.cpu().numpy(),z_before=before.cpu().numpy(),z_after=self.z.cpu().numpy(),
            inner=inner,U=len(inner),T_world_anchor=self.anchor.cpu().numpy(),role_valid=valid.tolist()))
        return admitted

    def estimate(self):return self._estimate
