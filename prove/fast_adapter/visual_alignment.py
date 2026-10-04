"""P18 RGB-only observation path; privileged state stays in the simulator.

Ground-truth boxes select DINO regions. Box coordinates are NOT returned as
features. The nominal controller separately uses GT centers, as declared.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image, ImageDraw

DINO_SHA256 = 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179'
DEFAULT_MODEL = '/home/zheng/github/LiLa-WAM/dinov3_pretrain/dinov3-vitl16-pretrain-lvd1689m'
WORLD_HALF = 1.75
IMAGE_SIZE = 224


def appearance(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(size=(n, 8)).astype(np.float32)


def render_batch(state, target, style):
    """Pure renderer: no dynamics/domain identifier input, deterministic RGB."""
    state = np.asarray(state, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    centers = (np.stack([state, target], axis=1) / (2 * WORLD_HALF) + .5) * IMAGE_SIZE
    y, x = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE]
    imgs, boxes = [], []
    for pts, s in zip(centers, style):
        bg = np.array([35, 40, 45]) + 22 * s[:3]
        texture = 4 * np.sin(x * (.07 + .03*s[6]) + 6*s[7]) + 3*np.cos(y*.085+3*s[4])
        rgb = np.clip(bg[None,None,:] + texture[:,:,None], 0, 255).astype(np.uint8)
        im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(im)
        ox, oy = map(float, pts[0]); tx, ty = map(float, pts[1])
        ob = [ox-8, oy-8, ox+8, oy+8]; tb = [tx-13, ty-13, tx+13, ty+13]
        draw.rectangle(tb, outline=(int(55+35*s[3]), int(185+45*s[4]), int(180+55*s[5])), width=3)
        draw.rectangle(ob, fill=(int(205+45*s[3]), int(90+45*s[4]), int(40+30*s[5])))
        draw.line((ox-5,oy,ox+5,oy), fill=(255,220,170), width=2)
        imgs.append(np.asarray(im)); boxes.append([ob, tb])
    return np.stack(imgs), np.asarray(boxes, dtype=np.float32)


def visible(state):
    return (np.abs(np.asarray(state)) <= WORLD_HALF - 13 * 2 * WORLD_HALF / IMAGE_SIZE).all(axis=-1)


class FrozenDinoROI:
    """Actual frozen ViT-L/16, same official preprocessing at native 224x224.

    Float32 ROI features are ordinary detached tensors (not inference tensors),
    allowing gradients through the small projection downstream.
    """
    def __init__(self, model_dir=DEFAULT_MODEL, device='cuda', microbatch=8):
        from transformers import AutoImageProcessor, AutoModel
        self.device = torch.device(device)
        self.microbatch = microbatch
        self.processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16).to(self.device).eval()
        self.model.requires_grad_(False)
        self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1,3,1,1)
        self.std = torch.tensor(self.processor.image_std, device=self.device).view(1,3,1,1)
        assert self.model.config.hidden_size == 1024 and self.model.config.patch_size == 16
        self.prefix = 1 + self.model.config.num_register_tokens

    def encode(self, rgb, boxes):
        results = []
        for start in range(0,len(rgb),self.microbatch):
            with torch.inference_mode():
                pixels = torch.as_tensor(np.array(rgb[start:start+self.microbatch],copy=True),device=self.device).permute(0,3,1,2).float()/255
                pixels = ((pixels-self.mean)/self.std).to(torch.bfloat16)
                feats = self.model(pixel_values=pixels).last_hidden_state[:,self.prefix:].float().reshape(-1,14,14,1024)
                b = torch.as_tensor(boxes[start:start+self.microbatch],device=self.device)
                lo = torch.arange(14,device=self.device)*16
                wx = (torch.minimum(b[:,:,2,None],lo+16)-torch.maximum(b[:,:,0,None],lo)).clamp(min=0)
                wy = (torch.minimum(b[:,:,3,None],lo+16)-torch.maximum(b[:,:,1,None],lo)).clamp(min=0)
                weights = wy[:,:,:,None]*wx[:,:,None,:]
                pooled = torch.einsum('bryx,byxc->brc',weights,feats)/weights.sum((-1,-2)).clamp(min=1)[...,None]
            results.append(pooled.detach().clone())
        return torch.cat(results)

    def observe(self,state,target,style):
        rgb, boxes = render_batch(state,target,style)
        return self.encode(rgb,boxes)
