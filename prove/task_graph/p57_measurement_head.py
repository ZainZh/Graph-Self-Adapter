"""P57 Stage A: public-observation virtual-anchor measurement head.

The three anchors describe a book frame and a desired book frame. They are
training/evaluation labels, not visible corners or runtime target coordinates.
No private label is accepted by ``forward``.
"""
from dataclasses import dataclass

import torch
from torch import nn


MODES = ("mask_geometry", "dino_lila", "fusion")
INPUT_KEYS = frozenset({"raw", "boxes", "visible", "proprio", "mask_tokens",
                        "mask_valid", "camera_K", "camera_pose", "eef_transform"})


@dataclass(frozen=True)
class MeasurementConfig:
    mode: str = "fusion"
    raw_dim: int = 3072
    mask_dim: int = 55
    width: int = 96
    image_size: int = 448


def _mlp(in_dim, width):
    return nn.Sequential(nn.Linear(in_dim, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU())


class GeometryMeasurementHead(nn.Module):
    """Predict per-view UV for three virtual anchors of book and desired frame.

    Output is in native image pixels, shape ``[...,2,2,3,2]`` with roles book,
    desired-book-frame. For each role its three points are predicted relative to
    that role's *visible object* bounding box (book or whole caddy respectively).
    The caddy box does not itself identify the mouth or a visible anchor.
    """

    def __init__(self, config=MeasurementConfig()):
        super().__init__()
        if config.mode not in MODES or config.width <= 0 or config.image_size <= 0:
            raise ValueError("Invalid P57 measurement configuration")
        self.config = config
        self.mask_encoder = _mlp(config.mask_dim, config.width) if config.mode in ("mask_geometry", "fusion") else None
        self.raw_encoder = _mlp(config.raw_dim, config.width) if config.mode in ("dino_lila", "fusion") else None
        self.fuse = nn.Linear(config.width * 2, config.width) if config.mode == "fusion" else None
        # Geometry, robot state and role identity accompany the selected visual input.
        self.aux_dim = 47
        self.context = nn.Sequential(nn.Linear(config.width * 4, config.width), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(config.width + self.aux_dim, config.width), nn.SiLU(),
                                  nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, 6))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("raw_mean", torch.zeros(2, 3, config.raw_dim))
        self.register_buffer("raw_scale", torch.ones(2, 3, config.raw_dim))
        self.register_buffer("mask_mean", torch.zeros(2, 3, config.mask_dim))
        self.register_buffer("mask_scale", torch.ones(2, 3, config.mask_dim))
        self.register_buffer("aux_mean", torch.zeros(2, 2, self.aux_dim))
        self.register_buffer("aux_scale", torch.ones(2, 2, self.aux_dim))

    @staticmethod
    def _validate(obs, config):
        if set(obs) != INPUT_KEYS:
            raise ValueError(f"Public input keys must be exactly {sorted(INPUT_KEYS)}")
        expected = {"raw": (2, 3, config.raw_dim), "boxes": (2, 3, 4),
                    "visible": (2, 3), "proprio": (18,),
                    "mask_tokens": (2, 3, config.mask_dim), "mask_valid": (2, 3),
                    "camera_K": (2, 3, 3), "camera_pose": (2, 4, 4),
                    "eef_transform": (4, 4)}
        leading = obs["proprio"].shape[:-1]
        for key, suffix in expected.items():
            value = obs[key]
            if not isinstance(value, torch.Tensor) or value.shape != (*leading, *suffix):
                raise ValueError(f"Invalid public {key} shape")
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite public {key}")
        if obs["mask_valid"].dtype != torch.bool:
            raise ValueError("mask_valid must be boolean")
        if not torch.equal(obs["visible"].bool(), obs["mask_valid"]):
            raise ValueError("Cached visibility and public mask validity differ")
        return leading

    def _auxiliary(self, obs):
        # Object roles 1=book and 2=caddy. Camera and EEF transforms are public.
        box = obs["boxes"][..., 1:3, :]
        vis = obs["visible"][..., 1:3, None]
        k = obs["camera_K"]
        calib = torch.stack((k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2]), -1) / self.config.image_size
        cam = obs["camera_pose"]
        cam_pos = cam[..., :3, 3]
        cam_rot = cam[..., :3, :2].reshape(*cam.shape[:-2], 6)
        eef = obs["eef_transform"]
        eef_pos = eef[..., :3, 3]
        eef_rot = eef[..., :3, :2].reshape(*eef.shape[:-2], 6)
        # EEF position in the camera frame gives moving wrist and static camera
        # observations a common geometric reference without object truth.
        rel_eef = torch.matmul(cam[..., :3, :3].transpose(-1, -2),
                               (eef_pos[..., None, :] - cam_pos).unsqueeze(-1)).squeeze(-1)
        two = lambda x: x.unsqueeze(-2).expand(*x.shape[:-1], 2, x.shape[-1])
        proprio = obs["proprio"].unsqueeze(-2).unsqueeze(-2).expand(*box.shape[:-1], 18)
        role = torch.eye(2, device=box.device, dtype=box.dtype).expand(*box.shape[:-2], 2, 2)
        eef_rot_views = eef_rot.unsqueeze(-2).expand(*cam.shape[:-2], 6)
        aux = torch.cat((box, vis, two(calib), two(cam_pos), two(cam_rot), two(rel_eef),
                         two(eef_rot_views), proprio, role), -1)
        if aux.shape[-1] != self.aux_dim:
            raise RuntimeError("P57 auxiliary feature dimension mismatch")
        return aux

    def forward(self, obs):
        self._validate(obs, self.config)
        all_mask = obs["mask_valid"]
        mask = all_mask[..., 1:3]
        tokens = []
        if self.mask_encoder is not None:
            m = (obs["mask_tokens"] - self.mask_mean) / self.mask_scale
            tokens.append(self.mask_encoder(m))
        if self.raw_encoder is not None:
            r = (obs["raw"] - self.raw_mean) / self.raw_scale
            tokens.append(self.raw_encoder(r))
        visual = self.fuse(torch.cat(tokens, -1)) if self.fuse is not None else tokens[0]
        visual = visual * all_mask.unsqueeze(-1)
        book, caddy, actor = visual[..., 1, :], visual[..., 2, :], visual[..., 0, :]
        common = torch.cat((book, caddy, actor), -1).unsqueeze(-2).expand(*mask.shape, self.config.width * 3)
        role = visual[..., 1:3, :]
        visual = self.context(torch.cat((role, common), -1)) * mask.unsqueeze(-1)
        aux = (self._auxiliary(obs) - self.aux_mean) / self.aux_scale
        delta = self.head(torch.cat((visual, aux), -1)).reshape(*mask.shape, 3, 2)
        box = obs["boxes"][..., 1:3, :] * self.config.image_size
        center = (box[..., :2] + box[..., 2:]) * .5
        size = (box[..., 2:] - box[..., :2]).clamp_min(1.)
        uv = center[..., None, :] + size[..., None, :] * delta
        return uv, mask

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
