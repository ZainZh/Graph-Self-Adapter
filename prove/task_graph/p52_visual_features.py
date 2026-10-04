"""Frozen pretrained DINOv3 + official LiLa spatial fusion for P52.

Only fusion is consumed, not LiLa query adapter. Role identities are allowed
visible GT masks; the goal role denotes the visible caddy, not hidden mouth/site.
"""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
from prove.fast_adapter.visual_alignment import FrozenDinoROI, DEFAULT_MODEL
from prove.fast_adapter.neural_geometry_visual import mask_geometry


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


class FrozenStageVision(FrozenDinoROI):
    def __init__(self, model_dir=DEFAULT_MODEL,
                 lila_root='data/sources/lila-wam-reference-20260923', device='cpu'):
        super().__init__(model_dir=model_dir, device=device, microbatch=1)
        root = Path(lila_root)
        source, weights = root/'models/vla_model_fm.py', root/'VISUAL_WEIGHTS.pt'
        spec = importlib.util.spec_from_file_location('p52_official_lila', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.fusion = module.MultiLayerConcatFusion(1024, 3, 1024,
                        proj_type='linear', pre_norm=True).to(self.device).eval()
        self.fusion.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True)['fusion'], strict=True)
        self.fusion.requires_grad_(False)
        self.binding = {'dino': str(Path(model_dir).resolve()), 'lila': str(root.resolve()),
                        'consumed': 'DINO hidden[-12,-8,-4] + LiLa fusion; no query adapter',
                        'lila_sha256': sha256(weights),
                        'dino_sha256': sha256(Path(model_dir)/'model.safetensors'),
                        'lila_source_sha256': sha256(source),
                        'frontend_source_sha256': sha256(__file__)}

    def encode(self, rgb, masks):
        rgb, masks = np.asarray(rgb), np.asarray(masks)
        if rgb.ndim != 4 or rgb.shape[1:] != (448,448,3):
            raise ValueError('Requires native448 RGB [views,448,448,3]')
        if masks.shape != (len(rgb),3,448,448):
            raise ValueError('Requires visible masks [views,3,448,448]')
        rows=[]
        for image, mask in zip(rgb, masks):
            with torch.inference_mode():
                pixels=torch.tensor(image.copy(),device=self.device).permute(2,0,1)[None].float()/255
                hidden=self.model(pixel_values=((pixels-self.mean)/self.std).to(torch.bfloat16),
                                  output_hidden_states=True).hidden_states
                fused=self.fusion([hidden[i].float() for i in (-12,-8,-4)])
                patch=fused[:,self.prefix:].reshape(1,784,1024)
                weight=F.avg_pool2d(torch.tensor(mask.copy(),device=self.device).float()[None],16,16).flatten(2)
                weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-6)
                yy,xx=torch.meshgrid(torch.linspace(-1,1,28,device=self.device),torch.linspace(-1,1,28,device=self.device),indexing='ij')
                basis=torch.stack((torch.ones_like(xx),xx,yy)).flatten(1)
                raw=torch.einsum('brp,kp,bpc->brkc',weight,basis,patch).flatten(2)
            rows.append(raw[0].detach().cpu().numpy())
        result=np.stack(rows)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite pretrained visual output')
        return result


def public_features(frame, vision):
    """Explicit allowlist. No object pose, site, contact, success or teacher input."""
    rgb, masks=frame['rgb'],frame['visible_masks']
    boxes,visible=zip(*(mask_geometry(m) for m in masks))
    rot=Rotation.from_quat(frame['eef_quat_xyzw']).as_matrix()[:,:2].T.reshape(-1)
    proprio=np.concatenate((frame['eef_pos'],rot,frame['gripper_qpos'],frame['joint_pos'])).astype(np.float32)
    return dict(raw=vision.encode(rgb,masks), boxes=np.asarray(boxes,dtype=np.float32)/448,
                visible=np.asarray(visible,dtype=np.float32),proprio=proprio)
