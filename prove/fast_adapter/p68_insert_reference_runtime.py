"""P68 insert reference-state runtime built on frozen P61/P62 components.

The runtime accepts the same public RGB/mask/camera/own-pose fields as P62.  It
never accepts condition seeds, simulator configuration, target truth, corners or
PnP output.  Reference arms freeze a public visual geometry estimate and then
reuse the original P62 translation_size5 latent update math against subsequent
public contour feedback.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from prove.fast_adapter.p61_encoder import P61RelationEncoder, first_eef_anchor_transforms
from prove.fast_adapter.p61_adaptation import corrected_components, contour_residual
from prove.fast_adapter.p62_adaptation import TRANSLATION_SIZE5, adapt_z, candidate_configuration
from prove.fast_adapter.p60_contour import ContourMeasurement, causal_support, outer_hull_boundary
from prove.fast_adapter.visual_alignment import FrozenDinoROI
from prove.fast_adapter.p62_runtime import PUBLIC_POSE_KEYS, pose_matrix

FRAME_FIXED = "frame_fixed"
REFERENCE_FIXED = "reference_fixed"
ENTRY_ONCE = "entry_once"
CONTINUOUS = "continuous"
P68_ARMS = (FRAME_FIXED, REFERENCE_FIXED, ENTRY_ONCE, CONTINUOUS)
PREFIX_STEPS = 12
REFERENCE_UPDATE_CANDIDATE = TRANSLATION_SIZE5


def p68_candidate_configuration() -> dict[str, object]:
    """Return the unchanged P62 translation_size5 numerical update contract."""
    return candidate_configuration(REFERENCE_UPDATE_CANDIDATE)


class P68InsertReferenceRuntime:
    """Frozen insert runtime for current-frame and fixed-reference arms.

    Before ``PREFIX_STEPS`` every arm emits the current-frame fixed estimate so
    the physical observation prefix is common.  Reference arms may lock a public
    finite geometry before then, but that reference becomes action-relevant only
    at and after the prefix.  ``entry_once`` performs exactly one update; the
    continuous arm updates on every post-prefix admitted feedback frame with at
    least three causal support records.  Rotation latent coordinates are
    structurally inactive and checked after each observation.
    """

    def __init__(self, checkpoint, model_dir, arm, device="cuda"):
        if arm not in P68_ARMS:
            raise ValueError(f"unknown P68 arm {arm!r}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["schema"] != "prove.p61-relation-checkpoint.v1":
            raise ValueError("checkpoint schema")
        self.encoder = P61RelationEncoder(**payload["constructor"])
        self.encoder.load_state_dict(payload["state_dict"], strict=True)
        self.encoder.to(device).eval().requires_grad_(False)
        self.vision = FrozenDinoROI(model_dir, device=device, microbatch=1)
        self.device = device
        self.arm = arm
        self.anchor = None
        self.z = torch.zeros(8, device=device)
        self.history: list[ContourMeasurement] = []
        self.records: list[dict[str, object]] = []
        self.geometry = None
        self.reference_geometry = None
        self.reference_t: int | None = None
        self.entry_update_done = False
        self.update_events = 0
        self._estimate = None
        self.last_t = -1

    def _encode_geometry(self, public, rgb, masks, valid):
        world_eef = torch.as_tensor(pose_matrix(public["position"], public["rotation"]), device=self.device)
        world_camera = torch.as_tensor(pose_matrix(public["camera_position"], public["camera_rotation"]), device=self.device)
        if self.anchor is None:
            self.anchor = world_eef.clone()
        anchor_camera, anchor_eef = first_eef_anchor_transforms(self.anchor, world_camera, world_eef)
        with torch.no_grad():
            pixels = torch.from_numpy(np.asarray(rgb).copy()).to(self.device).permute(2, 0, 1)[None].float() / 255
            pixels = ((pixels - self.vision.mean) / self.vision.std).to(torch.bfloat16)
            hidden = self.vision.model(pixel_values=pixels, output_hidden_states=True, return_dict=True).hidden_states
            layers = torch.stack([hidden[i][:, self.vision.prefix:] for i in (-12, -8, -4)], 1).to(torch.float16).float()
            patch = F.avg_pool2d(torch.from_numpy(np.asarray(masks).copy()).to(self.device).float(), 16, 16)
            geometry = self.encoder(
                dino_layers=layers[:, None],
                role_patch_masks=patch[None, None],
                role_valid=torch.as_tensor(valid, device=self.device)[None, None],
                K=torch.as_tensor(public["K"], device=self.device, dtype=torch.float32)[None, None],
                T_anchor_camera=anchor_camera[None, None],
                T_anchor_eef=anchor_eef[None],
                image_hw=(448, 448),
            ).geometry8[0].detach()
        if not torch.isfinite(geometry).all():
            raise ValueError("nonfinite encoder geometry")
        return geometry

    def _maybe_lock_reference(self, t: int, valid, admitted: bool) -> bool:
        if self.arm == FRAME_FIXED or self.reference_geometry is not None:
            return False
        if bool(admitted) and bool(valid[2]) and self.geometry is not None and torch.isfinite(self.geometry).all():
            self.reference_geometry = self.geometry.detach().clone()
            self.reference_t = int(t)
            return True
        return False

    def _estimation_geometry(self, t: int):
        if self.arm == FRAME_FIXED:
            return self.geometry, torch.zeros_like(self.z), "current_frame"
        if t < PREFIX_STEPS:
            return self.geometry, torch.zeros_like(self.z), "prefix_current_frame"
        if self.reference_geometry is None:
            return None, torch.zeros_like(self.z), "waiting_reference"
        return self.reference_geometry, self.z, "reference"

    def _adapt_geometry(self):
        if self.arm == FRAME_FIXED:
            return None
        if self.reference_geometry is None:
            return None
        return self.reference_geometry

    def update(self, public, rgb, masks, t):
        if set(public) != set(PUBLIC_POSE_KEYS):
            raise ValueError("only public camera/own-pose fields accepted")
        if t <= self.last_t:
            raise ValueError("observations must strictly increase")
        self.last_t = int(t)
        rgb = np.asarray(rgb)
        masks = np.asarray(masks, dtype=bool)
        if rgb.shape != (448, 448, 3) or masks.shape != (3, 448, 448):
            raise ValueError("native448 role image required")
        if not all(np.isfinite(public[k]).all() for k in PUBLIC_POSE_KEYS):
            raise ValueError("nonfinite public pose")
        valid = masks.reshape(3, -1).any(-1)
        self.geometry = self._encode_geometry(public, rgb, masks, valid)

        boundary, info = outer_hull_boundary(masks[2])
        admitted = boundary is not None
        reference_locked_now = self._maybe_lock_reference(int(t), valid, admitted)
        if admitted:
            self.history.append(ContourMeasurement(
                boundary,
                np.asarray(info["hull_vertices"], dtype=np.float32),
                np.asarray(public["K"]),
                np.asarray(public["camera_position"]),
                np.asarray(public["camera_rotation"]),
                (448, 448),
                int(t),
            ))
        support = causal_support(self.history)
        before = self.z.clone()
        inner = []
        pre_loss = post_loss = None
        adapt_geometry = self._adapt_geometry()
        can_update = (
            adapt_geometry is not None
            and admitted
            and int(t) >= PREFIX_STEPS
            and len(support) >= 3
            and bool(valid[2])
            and self.arm in (ENTRY_ONCE, CONTINUOUS)
            and (self.arm != ENTRY_ONCE or not self.entry_update_done)
        )
        if support and bool(valid[2]) and adapt_geometry is not None:
            with torch.no_grad():
                pre_loss = float(0.5 * contour_residual(adapt_geometry, self.z, self.anchor, support).square().mean())
            if can_update:
                with torch.enable_grad():
                    self.z, inner = adapt_z(
                        adapt_geometry,
                        self.z,
                        self.anchor,
                        support,
                        candidate=REFERENCE_UPDATE_CANDIDATE,
                        create_graph=False,
                    )
                self.update_events += 1
                update_phase = "entry" if not self.entry_update_done else "continuous"
                self.entry_update_done = True
            with torch.no_grad():
                post_loss = float(0.5 * contour_residual(adapt_geometry, self.z, self.anchor, support).square().mean())

        if not can_update:
            update_phase = None
        estimate_geometry, estimate_z, estimate_source = self._estimation_geometry(int(t))
        with torch.no_grad():
            if (self.arm != FRAME_FIXED or bool(valid[2])) and estimate_geometry is not None:
                center, rotation, size = corrected_components(estimate_geometry, estimate_z)
                center = self.anchor[:3, :3] @ center + self.anchor[:3, 3]
                rotation = self.anchor[:3, :3] @ rotation
                self._estimate = tuple(x.detach().cpu().numpy() for x in (center, rotation, size))
            else:
                self._estimate = None
        if not torch.equal(self.z[3:6], torch.zeros_like(self.z[3:6])):
            raise RuntimeError("inactive rotation latent changed")
        self.records.append(dict(
            t=int(t),
            arm=self.arm,
            candidate=REFERENCE_UPDATE_CANDIDATE if self.arm in (ENTRY_ONCE, CONTINUOUS) else "fixed",
            admitted=bool(admitted),
            role_valid=valid.tolist(),
            geometry8=self.geometry.cpu().numpy(),
            reference_geometry8=None if self.reference_geometry is None else self.reference_geometry.cpu().numpy(),
            reference_t=self.reference_t,
            reference_locked_now=bool(reference_locked_now),
            reference_action_active=bool(self.arm != FRAME_FIXED and int(t) >= PREFIX_STEPS and self.reference_geometry is not None),
            estimate_source=estimate_source,
            z_before=before.cpu().numpy(),
            z_after=self.z.cpu().numpy(),
            U=len(inner),
            update_events=int(self.update_events),
            entry_update_done=bool(self.entry_update_done),
            update_phase=update_phase,
            inner=inner,
            S=len(support),
            support_times=[m.t for m in support],
            pre_feedback_loss=pre_loss,
            post_feedback_loss=post_loss,
            T_world_anchor=self.anchor.cpu().numpy(),
            prefix_steps=PREFIX_STEPS,
        ))
        return admitted

    def estimate(self):
        return self._estimate

    def reset_skill_state(self):
        """Explicit episode-end cleanup for queued runners; frozen weights remain loaded."""
        self.z.zero_()
        self.history.clear()
        self.records.clear()
        self.geometry = None
        self.reference_geometry = None
        self.reference_t = None
        self.entry_update_done = False
        self.update_events = 0
        self._estimate = None
        self.anchor = None
        self.last_t = -1
