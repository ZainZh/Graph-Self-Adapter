"""Task-trained LiLa ROI relation encoder for bounded P61 insertion.

The module consumes *frozen* dense DINOv3 hidden states.  It preserves the
official LiLa fusion and 64-query adapter architectures and loads their weights
strictly.  Public masks and calibration condition a shared per-role encoder;
no detected corner, PnP result, simulator pose, or evaluator signal is an
input.

All geometry is expressed in an episode-fixed anchor whose origin and axes are
the first observed EEF frame. ``T_anchor_camera`` maps camera coordinates into
that anchor, and ``T_anchor_eef`` maps the current EEF frame into it.  The
geometry head predicts the slot *opening* as
``[center_xyz_m, rotvec_xyz_rad, log_half_size_xy_m]``.
The persistent z correction and its online optimizer live in the P61
adaptation module, but use this same fixed coordinate frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ROLE_NAMES = ("actor_robot", "held_object", "goal_slot")
HIDDEN_STATE_INDICES = (-12, -8, -4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def first_eef_anchor_transforms(
    T_world_first_eef: Tensor, T_world_camera: Tensor, T_world_eef: Tensor,
) -> tuple[Tensor, Tensor]:
    """Convert public poses to the fixed first-EEF episode anchor.

    All transforms use the column-vector convention ``p_parent = T_parent_child
    @ p_child``. Inputs are camera/EEF-to-world poses, so the returned values
    are camera-to-anchor and current-EEF-to-anchor respectively.
    """
    if T_world_first_eef.shape[-2:] != (4, 4):
        raise ValueError("T_world_first_eef must end in [4,4]")
    if T_world_camera.shape[-2:] != (4, 4) or T_world_eef.shape[-2:] != (4, 4):
        raise ValueError("Camera and EEF transforms must end in [4,4]")
    T_anchor_world = torch.linalg.inv(T_world_first_eef)
    if T_world_camera.ndim == T_anchor_world.ndim + 1:
        T_anchor_camera = T_anchor_world.unsqueeze(-3) @ T_world_camera
    else:
        T_anchor_camera = T_anchor_world @ T_world_camera
    return T_anchor_camera, T_anchor_world @ T_world_eef


def calibrated_anchor_rays(K: Tensor, T_anchor_camera: Tensor, pixels_uv1: Tensor) -> Tensor:
    """Map pixel homogeneous coordinates to unit rays in the fixed anchor.

    ``K`` uses ordinary OpenCV image coordinates (+x right, +y down, +z
    forward), while the P45/P60 camera transform is OpenGL (+x right, +y up,
    camera looks toward -z).  The explicit ``diag(1,-1,-1)`` conversion makes
    this agree with ``u=cx+f*x/-z, v=cy-f*y/-z``.
    """
    rays_cv = torch.einsum("...ij,...pj->...pi", torch.linalg.inv(K.float()).to(K.dtype),
                           pixels_uv1)
    cv_to_gl = torch.tensor((1.0, -1.0, -1.0), dtype=K.dtype, device=K.device)
    rays_gl = rays_cv * cv_to_gl
    rays_anchor = torch.einsum("...ij,...pj->...pi", T_anchor_camera[..., :3, :3], rays_gl)
    return F.normalize(rays_anchor, dim=-1)


def _official_state(checkpoint: Path) -> tuple[dict[str, Tensor], dict[str, Tensor], str]:
    package = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if "fusion" in package and "adapter" in package:
        return dict(package["fusion"]), dict(package["adapter"]), "visual_weights"
    state = package.get("model_state_dict", package)
    if not isinstance(state, Mapping):
        raise ValueError("LiLa checkpoint has no tensor state dictionary")

    def subset(prefix: str) -> dict[str, Tensor]:
        return {key[len(prefix):]: value for key, value in state.items()
                if key.startswith(prefix)}

    fusion = subset("concat_fusion.")
    adapter = subset("dino_adapters.0.")
    if not fusion or not adapter:
        raise ValueError("LiLa checkpoint lacks concat_fusion/dino_adapters.0 weights")
    return fusion, adapter, "official_checkpoint"


def load_official_lila(source: str | Path, checkpoint: str | Path) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Construct exact official modules and strictly load their pretrained state."""
    source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
    if not source.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Missing LiLa source/checkpoint: {source}, {checkpoint}")
    spec = importlib.util.spec_from_file_location("p61_official_lila", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official LiLa source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fusion = module.MultiLayerConcatFusion(
        1024, 3, 1024, proj_type="linear", pre_norm=True)
    adapter = module.VisualFeatureAdapter(
        1024, 768, num_queries=64, num_heads=8, num_layers=4, dropout=0.0)
    fusion_state, adapter_state, layout = _official_state(checkpoint)
    fusion.load_state_dict(fusion_state, strict=True)
    adapter.load_state_dict(adapter_state, strict=True)
    binding = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_layout": layout,
        "fusion_class": type(fusion).__name__,
        "adapter_class": type(adapter).__name__,
        "fusion_shape": "MultiLayerConcatFusion(1024,3,1024,linear,pre_norm)",
        "adapter_shape": "VisualFeatureAdapter(1024,768,64queries,8heads,4layers,dropout0)",
        "strict_load": True,
    }
    return fusion, adapter, binding


@dataclass(frozen=True)
class P61EncoderOutput:
    u: Tensor
    geometry8: Tensor
    role_tokens: Tensor


class _FourQueryReducer(nn.Module):
    """Shared 64x768 -> 4x256 role aggregation."""

    def __init__(self) -> None:
        super().__init__()
        self.memory = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256))
        self.queries = nn.Parameter(torch.empty(1, 4, 256))
        layer = nn.TransformerDecoderLayer(
            256, 8, 1024, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(256)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, values: Tensor) -> Tensor:
        memory = self.memory(values)
        queries = self.queries.expand(values.shape[0], -1, -1)
        return self.norm(self.decoder(queries, memory))


class P61RelationEncoder(nn.Module):
    """Shared Actor/Object/Goal ROI encoder and task functional geometry head.

    ``dino_layers`` may contain only dense patches or may retain DINO
    CLS/register prefix tokens. Patch count is defined by
    ``role_patch_masks.shape[-2:]`` and the prefix count is inferred as
    ``N - Hp*Wp``. DINO values are always detached here, including in the
    task-finetuned LiLa arm.
    """

    def __init__(
        self,
        lila_source: str | Path,
        lila_checkpoint: str | Path,
        *,
        finetune_lila: bool,
        geometry_mean8: Tensor | Sequence[float],
        max_views: int = 4,
        adapter_microbatch: int = 8,
    ) -> None:
        super().__init__()
        if max_views < 1 or adapter_microbatch < 1:
            raise ValueError("max_views and adapter_microbatch must be positive")
        self.fusion, self.visual_adapter, self.lila_binding = load_official_lila(
            lila_source, lila_checkpoint)
        self.max_views = int(max_views)
        self.adapter_microbatch = int(adapter_microbatch)
        # mask, image xy, ROI-local xy, anchor ray, camera origin, visibility
        self.spatial_encoder = nn.Sequential(
            nn.Linear(12, 256), nn.GELU(), nn.Linear(256, 1024))
        nn.init.normal_(self.spatial_encoder[-1].weight, std=1e-3)
        nn.init.zeros_(self.spatial_encoder[-1].bias)
        self.role_embedding = nn.Embedding(len(ROLE_NAMES), 1024)
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        self.query_reducer = _FourQueryReducer()
        self.relation_role_embedding = nn.Embedding(len(ROLE_NAMES), 256)
        self.view_embedding = nn.Embedding(self.max_views, 256)
        self.eef_encoder = nn.Sequential(nn.Linear(9, 256), nn.GELU(), nn.Linear(256, 256))
        self.relation_cls = nn.Parameter(torch.empty(1, 1, 256))
        relation_layer = nn.TransformerEncoderLayer(
            256, 8, 1024, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True)
        self.relation_transformer = nn.TransformerEncoder(relation_layer, num_layers=2)
        self.relation_norm = nn.LayerNorm(256)
        self.geometry_head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 8))
        nn.init.normal_(self.geometry_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.geometry_head[-1].bias)
        geometry_mean = torch.as_tensor(geometry_mean8, dtype=torch.float32)
        if geometry_mean.shape != (8,) or not torch.isfinite(geometry_mean).all():
            raise ValueError("geometry_mean8 must be a finite train-only [8] statistic")
        self.register_buffer("geometry_mean8", geometry_mean.clone())
        with torch.no_grad():
            nn.init.trunc_normal_(self.relation_cls, std=0.02)
        self.set_lila_trainable(finetune_lila)

    @property
    def finetune_lila(self) -> bool:
        return any(parameter.requires_grad for parameter in self.visual_adapter.parameters())

    def set_lila_trainable(self, enabled: bool) -> None:
        """Toggle only official LiLa weights; DINO remains outside and detached."""
        for module in (self.fusion, self.visual_adapter):
            module.requires_grad_(bool(enabled))
            module.train(self.training and bool(enabled))

    def train(self, mode: bool = True) -> "P61RelationEncoder":
        super().train(mode)
        # Frozen official modules remain deterministic eval modules. Finetuned
        # modules also have dropout=0, but train mode is retained explicitly.
        if not self.finetune_lila:
            self.fusion.eval()
            self.visual_adapter.eval()
        return self

    @staticmethod
    def _validate(
        dino_layers: Tensor, role_patch_masks: Tensor, role_valid: Tensor,
        K: Tensor, T_anchor_camera: Tensor, T_anchor_eef: Tensor,
        image_hw: Sequence[int], max_views: int,
    ) -> tuple[int, int, int, int, int]:
        if dino_layers.ndim != 5 or dino_layers.shape[2] != 3 or dino_layers.shape[-1] != 1024:
            raise ValueError("dino_layers must be [B,V,3,N,1024]")
        batch, views, _, tokens, _ = dino_layers.shape
        if role_patch_masks.ndim != 5 or role_patch_masks.shape[:3] != (batch, views, 3):
            raise ValueError("role_patch_masks must be [B,V,3,Hp,Wp]")
        hp, wp = role_patch_masks.shape[-2:]
        if tokens < hp * wp:
            raise ValueError("DINO token count is smaller than the declared patch grid")
        expected = {
            "role_valid": (batch, views, 3), "K": (batch, views, 3, 3),
            "T_anchor_camera": (batch, views, 4, 4), "T_anchor_eef": (batch, 4, 4),
        }
        actual = {"role_valid": tuple(role_valid.shape), "K": tuple(K.shape),
                  "T_anchor_camera": tuple(T_anchor_camera.shape),
                  "T_anchor_eef": tuple(T_anchor_eef.shape)}
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(f"{name} must have shape {shape}, got {actual[name]}")
        if views > max_views or len(image_hw) != 2 or min(map(int, image_hw)) <= 0:
            raise ValueError("Invalid view count or image_hw")
        return batch, views, tokens, hp, wp

    @staticmethod
    def _rotation_6d(transform: Tensor) -> Tensor:
        # First two columns, matching the repository's continuous rotation convention.
        return transform[..., :3, :2].transpose(-1, -2).reshape(*transform.shape[:-2], 6)

    def _spatial_features(
        self, masks: Tensor, valid: Tensor, K: Tensor,
        T_anchor_camera: Tensor, image_hw: Sequence[int], prefix: int,
    ) -> tuple[Tensor, Tensor]:
        """Return role token conditioning and ROI gates as [B,V,R,N,*]."""
        batch, views, roles, hp, wp = masks.shape
        dtype, device = K.dtype, K.device
        height, width = map(int, image_hw)
        ys = (torch.arange(hp, device=device, dtype=dtype) + 0.5) * (height / hp)
        xs = (torch.arange(wp, device=device, dtype=dtype) + 0.5) * (width / wp)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        image_x = 2.0 * xx / width - 1.0
        image_y = 2.0 * yy / height - 1.0
        ones = torch.ones_like(xx)
        pixels = torch.stack((xx, yy, ones), dim=-1).reshape(1, 1, hp * wp, 3)
        rays_anchor = calibrated_anchor_rays(
            K, T_anchor_camera, pixels.expand(batch, views, -1, -1))
        origins = T_anchor_camera[..., :3, 3]

        mask = masks.to(dtype).clamp(0.0, 1.0)
        support = mask > 0
        gx = torch.arange(wp, device=device).view(1, 1, 1, 1, wp)
        gy = torch.arange(hp, device=device).view(1, 1, 1, hp, 1)
        xmin = torch.where(support, gx, wp).amin(dim=(-2, -1)).to(dtype)
        xmax = torch.where(support, gx, -1).amax(dim=(-2, -1)).to(dtype)
        ymin = torch.where(support, gy, hp).amin(dim=(-2, -1)).to(dtype)
        ymax = torch.where(support, gy, -1).amax(dim=(-2, -1)).to(dtype)
        cx, cy = (xmin + xmax + 1.0) / 2.0, (ymin + ymax + 1.0) / 2.0
        sx, sy = (xmax - xmin + 1.0).clamp_min(1.0), (ymax - ymin + 1.0).clamp_min(1.0)
        local_x = ((torch.arange(wp, device=device, dtype=dtype).view(1, 1, 1, 1, wp)
                    + 0.5 - cx[..., None, None]) / (0.5 * sx[..., None, None]))
        local_y = ((torch.arange(hp, device=device, dtype=dtype).view(1, 1, 1, hp, 1)
                    + 0.5 - cy[..., None, None]) / (0.5 * sy[..., None, None]))
        local_x = local_x.expand(batch, views, roles, hp, wp)
        local_y = local_y.expand(batch, views, roles, hp, wp)
        visible = valid.to(dtype)[..., None, None].expand(batch, views, roles, hp, wp)
        shared = lambda value: value[:, :, None].expand(batch, views, roles, *value.shape[2:])
        aux = torch.cat((
            mask[..., None],
            shared(image_x.reshape(1, 1, hp, wp, 1)).expand(batch, views, roles, hp, wp, 1),
            shared(image_y.reshape(1, 1, hp, wp, 1)).expand(batch, views, roles, hp, wp, 1),
            local_x[..., None], local_y[..., None],
            shared(rays_anchor.reshape(batch, views, hp, wp, 3)),
            shared(origins[:, :, None, None, :].expand(batch, views, hp, wp, 3)),
            visible[..., None]), dim=-1).reshape(batch, views, roles, hp * wp, 12)
        patch_condition = self.spatial_encoder(aux)
        zeros = torch.zeros(batch, views, roles, prefix, 1024,
                            dtype=patch_condition.dtype, device=device)
        condition = torch.cat((zeros, patch_condition), dim=3)
        prefix_gate = valid.to(dtype)[..., None].expand(batch, views, roles, prefix)
        gate = torch.cat((prefix_gate, mask.flatten(-2)), dim=-1)
        return condition, gate

    def _adapter_chunks(self, memory: Tensor, padding: Tensor) -> Tensor:
        """Run the exact official adapter weights with a genuine ROI mask."""
        outputs = []
        for start in range(0, memory.shape[0], self.adapter_microbatch):
            stop = start + self.adapter_microbatch
            values = memory[start:stop]
            blocked = padding[start:stop]
            normalized = self.visual_adapter.input_norm(values)
            projected = self.visual_adapter.feature_proj(normalized)
            queries = self.visual_adapter.query_embed.expand(values.shape[0], -1, -1)
            outputs.append(self.visual_adapter.transformer_decoder(
                queries, projected, memory_key_padding_mask=blocked))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        *,
        dino_layers: Tensor,
        role_patch_masks: Tensor,
        role_valid: Tensor,
        K: Tensor,
        T_anchor_camera: Tensor,
        T_anchor_eef: Tensor,
        image_hw: Sequence[int],
    ) -> P61EncoderOutput:
        batch, views, tokens, hp, wp = self._validate(
            dino_layers, role_patch_masks, role_valid, K, T_anchor_camera,
            T_anchor_eef, image_hw, self.max_views)
        device = next(self.parameters()).device
        tensors = (dino_layers, role_patch_masks, role_valid, K,
                   T_anchor_camera, T_anchor_eef)
        if any(value.device != device for value in tensors):
            raise ValueError(f"All inputs must already be on encoder device {device}")
        model_dtype = next(self.fusion.parameters()).dtype
        dense = dino_layers.detach().to(model_dtype)
        flat_layers = dense.reshape(batch * views, 3, tokens, 1024)
        selected = [flat_layers[:, index] for index in range(3)]
        if self.finetune_lila:
            fused = self.fusion(selected)
        else:
            with torch.no_grad():
                fused = self.fusion(selected)
        fused = fused.reshape(batch, views, tokens, 1024)
        prefix = tokens - hp * wp
        condition, gate = self._spatial_features(
            role_patch_masks, role_valid, K.to(model_dtype),
            T_anchor_camera.to(model_dtype), image_hw, prefix)
        role_ids = torch.arange(3, device=device)
        role_bias = self.role_embedding(role_ids).view(1, 1, 3, 1, 1024)
        memory = (fused[:, :, None] * gate[..., None] + condition + role_bias)
        memory = memory.reshape(batch * views * 3, tokens, 1024)
        # Prefix/global tokens are deliberately excluded: this is a role ROI,
        # not a global-image adapter presented under an ROI name.
        prefix_blocked = torch.ones(batch, views, 3, prefix,
                                    dtype=torch.bool, device=device)
        patch_blocked = role_patch_masks <= 0
        padding = torch.cat((prefix_blocked, patch_blocked.flatten(-2)), dim=-1)
        padding = padding.reshape(batch * views * 3, tokens)
        missing = padding.all(dim=-1)
        if missing.any():
            # Transformer attention over an entirely masked memory is NaN.
            # Missing roles receive one zero dummy KV and are masked from the
            # relation transformer below, so they carry no observation data.
            padding = padding.clone()
            padding[missing, 0] = False
            memory = memory.clone()
            memory[missing, 0] = 0
        queries = self._adapter_chunks(memory, padding)
        role_tokens = self.query_reducer(queries).reshape(batch, views, 3, 4, 256)
        role_tokens = role_tokens * role_valid[..., None, None].to(role_tokens.dtype)
        relation_role = self.relation_role_embedding(role_ids).view(1, 1, 3, 1, 256)
        view_ids = torch.arange(views, device=device)
        relation_view = self.view_embedding(view_ids).view(1, views, 1, 1, 256)
        relation_roles = role_tokens + relation_role + relation_view
        relation_roles = relation_roles.reshape(batch, views * 3 * 4, 256)
        position = T_anchor_eef[..., :3, 3]
        rotation6 = self._rotation_6d(T_anchor_eef)
        eef_token = self.eef_encoder(torch.cat((position, rotation6), dim=-1)).unsqueeze(1)
        cls = self.relation_cls.expand(batch, -1, -1)
        sequence = torch.cat((cls, eef_token, relation_roles), dim=1)
        invalid = (~role_valid.bool())[..., None].expand(batch, views, 3, 4)
        padding = torch.cat((torch.zeros(batch, 2, dtype=torch.bool, device=device),
                             invalid.reshape(batch, -1)), dim=1)
        encoded = self.relation_transformer(sequence, src_key_padding_mask=padding)
        u = self.relation_norm(encoded[:, 0])
        geometry8 = self.geometry_mean8.to(u.dtype) + self.geometry_head(u)
        if not torch.isfinite(u).all() or not torch.isfinite(geometry8).all():
            raise FloatingPointError("P61 encoder produced non-finite output")
        return P61EncoderOutput(u=u, geometry8=geometry8, role_tokens=role_tokens)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        lila = list(self.fusion.parameters()) + list(self.visual_adapter.parameters())
        lila_ids = {id(parameter) for parameter in lila}
        task = [parameter for parameter in self.parameters() if id(parameter) not in lila_ids]
        return {"lila": lila, "task": task}

    def gradient_audit(self) -> dict[str, Any]:
        """Report trainability and finite/nonzero gradients without mutating state."""
        result: dict[str, Any] = {"finetune_lila": self.finetune_lila, "groups": {}}
        groups = self.parameter_groups()
        detailed = {
            "fusion": list(self.fusion.parameters()),
            "visual_adapter": list(self.visual_adapter.parameters()),
            "lila": groups["lila"],
            "task": groups["task"],
        }
        for name, parameters in detailed.items():
            trainable = [parameter for parameter in parameters if parameter.requires_grad]
            gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
            result["groups"][name] = {
                "parameters": sum(parameter.numel() for parameter in parameters),
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "grad_tensors": len(gradients),
                "all_gradients_finite": all(torch.isfinite(gradient).all().item()
                                            for gradient in gradients),
                "nonzero_gradient_tensors": sum(torch.count_nonzero(gradient).item() > 0
                                                for gradient in gradients),
            }
        return result
