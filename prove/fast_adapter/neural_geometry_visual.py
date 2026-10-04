"""P43 public RGB/visible-mask frontend, separate from privileged collection.

Actor/Object/Goal are simulator-visible part identities (not learned masks).
No target pose, dimensions, contacts, success or configuration enter this module.
The frozen DINOv3 ViT-L/16 and official frozen LiLa three-layer fusion match P40.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROLES = ('actor', 'object', 'goal')
RAW_ROLE_DIM = 3072
ROLE_DIM = 16


def mask_geometry(masks):
    """Visible pixel xyxy boxes; absent regions remain exactly zero."""
    masks = np.asarray(masks, dtype=bool)
    boxes = np.zeros((3, 4), dtype=np.float32)
    visible = masks.reshape(3, -1).any(1)
    for i in np.flatnonzero(visible):
        yy, xx = np.nonzero(masks[i])
        boxes[i] = [xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]
    return boxes, visible


def observe_roles(env):
    """Return public P41 observation, wrist RGB, and 3 visible role masks.

    The parent environment's visible-corner observer is used without changing
    its frozen code. Segmentation exposes only rendered geometry identities.
    """
    obs, rgb, goal_mask = env.observe()
    env.renderer.enable_segmentation_rendering()
    seg = env.renderer.render().copy()
    env.renderer.disable_segmentation_rendering()
    names = [(env.model.geom(i).name or '') for i in range(env.model.ngeom)]
    actor = [i for i, name in enumerate(names) if name == 'mount' or name.startswith('pad')]
    obj = [i for i, name in enumerate(names) if name == 'peg' or name.startswith('cup_')]
    rendered = seg[:, :, 1] == int(env.mj.mjtObj.mjOBJ_GEOM)
    masks = np.stack([np.isin(seg[:, :, 0], actor) & rendered,
                      np.isin(seg[:, :, 0], obj) & rendered,
                      goal_mask.astype(bool)])
    boxes, visible = mask_geometry(masks)
    obs = dict(obs, role_bbox=boxes, role_visible=visible)
    return obs, rgb, masks.astype(np.uint8)


class FrozenRoleVision:
    """RGB -> role appearance/spatial moments -> train-only PCA16 per role.

    encode returns [N,48] with a fitted PCA, otherwise [N,3,3072]. Missing
    masks always yield zero, including after PCA centering. No test-stat fitting
    or implicit updates are supported. Pixel preprocessing preserves 448px.
    """
    def __init__(self, pca_path=None, model_dir='assets/dinov3', device='cuda', microbatch=8):
        from prove.fast_adapter.lila_frontend import ThreeFrontends
        self.net = ThreeFrontends(model_dir, device=device, microbatch=microbatch, selected='lila')
        self.device = torch.device(device)
        self.microbatch = microbatch
        self.pca = None
        if pca_path is not None:
            self.set_pca(torch.load(Path(pca_path), map_location='cpu', weights_only=True))

    def set_pca(self, pca):
        if pca.get('fit_split') != 'train':
            raise ValueError('Only a recorded training-only PCA can be consumed')
        self.pca = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in pca.items()}

    def project(self, raw, visible):
        if self.pca is None:
            return raw
        local = torch.einsum('brd,rdk->brk', raw - self.pca['mean'], self.pca['basis']) / self.pca['scale']
        return (local * visible[..., None]).flatten(1)

    def encode(self, images, masks):
        images = np.asarray(images)
        masks = np.asarray(masks)
        if images.ndim != 4 or images.shape[1:] != (448, 448, 3):
            raise ValueError('P43 requires native 448x448 RGB')
        if masks.shape != (len(images), 3, 448, 448):
            raise ValueError('Expected visible Actor/Object/Goal masks')
        rows = []
        for start in range(0, len(images), self.microbatch):
            with torch.inference_mode():
                rgb = torch.as_tensor(np.array(images[start:start+self.microbatch], copy=True), device=self.device).permute(0, 3, 1, 2).float()/255
                pix = ((rgb-self.net.mean)/self.net.std).to(torch.bfloat16)
                hidden = self.net.model(pixel_values=pix, output_hidden_states=True, return_dict=True).hidden_states
                if len(hidden) != 25:
                    raise RuntimeError('Unexpected DINOv3 layer count')
                fused = self.net.fusion([hidden[i].float() for i in (-12, -8, -4)])
                patch = fused[:, self.net.prefix:].reshape(-1, 28*28, 1024)
                mask = torch.as_tensor(masks[start:start+self.microbatch], device=self.device).float()
                weight = F.avg_pool2d(mask, 16, 16).flatten(2)
                visible = weight.sum(-1) > 0
                weight = weight/weight.sum(-1, keepdim=True).clamp_min(1e-6)
                yy, xx = torch.meshgrid(torch.linspace(-1, 1, 28, device=self.device), torch.linspace(-1, 1, 28, device=self.device), indexing='ij')
                spatial = torch.stack((torch.ones_like(xx), xx, yy)).flatten(1)
                raw = torch.einsum('brp,kp,bpc->brkc', weight, spatial, patch).flatten(2)
                result = self.project(raw, visible)
                if not torch.isfinite(result).all():
                    raise RuntimeError('Nonfinite visual feature')
            rows.append(result.detach().clone())
        return torch.cat(rows)


def fit_role_pca(train_raw, train_visible, seed=430, device='cuda'):
    """Explicit training-only offline operation, never called by runtime."""
    x = torch.as_tensor(train_raw, device=device, dtype=torch.float32).reshape(-1, 3, RAW_ROLE_DIM)
    visible = torch.as_tensor(train_visible, device=device, dtype=torch.bool).reshape(-1, 3)
    means, bases, scales, counts = [], [], [], []
    torch.manual_seed(seed)
    for role in range(3):
        rows = x[:, role][visible[:, role]]
        if len(rows) < ROLE_DIM + 1:
            raise ValueError(f'Insufficient visible training frames for {ROLES[role]}: {len(rows)}')
        mean = rows.mean(0)
        centered = rows - mean
        _, _, basis = torch.pca_lowrank(centered, q=ROLE_DIM+8, center=False, niter=5)
        basis = basis[:, :ROLE_DIM]
        scale = (centered @ basis).std(0).clamp_min(.01)
        means.append(mean.cpu()); bases.append(basis.cpu()); scales.append(scale.cpu()); counts.append(len(rows))
    return dict(mean=torch.stack(means), basis=torch.stack(bases), scale=torch.stack(scales),
                fit_split='train', roles=list(ROLES), visible_training_frames=counts,
                raw_role_dim=RAW_ROLE_DIM, role_dim=ROLE_DIM, seed=seed)


PUBLIC_ARRAY_KEYS = ('corners', 'corners_valid', 'camera_position', 'camera_rotation', 'K',
                     'position', 'rotation', 'role_bbox', 'role_visible', 'local', 't',
                     'previous_action', 'seeds')


def array_observation(arrays, episode, frame):
    """Convert a label-free runtime NPZ frame to the online observation dict."""
    obs = {k: arrays[k][episode, frame].copy() for k in
           ('camera_position', 'camera_rotation', 'K', 'position', 'rotation', 'role_bbox', 'role_visible', 'local')}
    obs['corners'] = arrays['corners'][episode, frame].copy() if arrays['corners_valid'][episode, frame] else None
    return obs
