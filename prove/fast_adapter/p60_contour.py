"""P60 public-contour adaptation for the frozen P45 neural geometry model.

This candidate deliberately changes only the online measurement objective.
P45's corner/PnP initialization, current observation encoder, learned rank-eight
decoder and pose extraction remain intact.  Runtime gradients update ``z``
only; model parameters are frozen.

The insert measurement is the *outer* convex hull of a visibly closed goal
frame.  This matches the functional outer-frame convention used to train P45.
The hole is only a public visibility/admission check and is never fitted as the
target contour.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

from prove.fast_adapter.geometry3d_adapter import initial, pose
from prove.fast_adapter.neural_geometry_adapter import align_observation
from prove.fast_adapter.p45_neural import features, load_model


def model_state_sha256(model) -> str:
    """Stable content hash used to prove runtime updates leave weights intact."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ContourMeasurement:
    """One causal public image measurement; all image coordinates are pixels."""

    boundary: np.ndarray
    hull_vertices: np.ndarray
    K: np.ndarray
    camera_position: np.ndarray
    camera_rotation: np.ndarray
    image_size: tuple[int, int]
    t: int


def _uniform_closed_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3 or count < 4:
        raise ValueError("A closed boundary needs at least three vertices and four samples")
    end = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(end - points, axis=1)
    keep = lengths > 1e-9
    points, end, lengths = points[keep], end[keep], lengths[keep]
    if len(points) < 3 or not np.isfinite(lengths).all() or lengths.sum() <= 0:
        raise ValueError("Degenerate boundary")
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    distance = np.arange(count, dtype=np.float64) * cumulative[-1] / count
    edge = np.searchsorted(cumulative[1:], distance, side="right")
    fraction = (distance - cumulative[edge]) / lengths[edge]
    return (points[edge] + fraction[:, None] * (end[edge] - points[edge])).astype(np.float32)


def outer_hull_boundary(goal_mask: np.ndarray, samples: int = 32):
    """Extract the public outer-frame boundary or fail closed.

    All visible mask contours are concatenated before taking their convex hull,
    exactly matching the original P45 insert observation convention.  A child
    contour larger than 25 pixels is required as evidence of a closed opening.
    Returns ``(boundary_or_none, diagnostics)``.
    """
    mask = np.asarray(goal_mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError("goal_mask must be two-dimensional")
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    info = {
        "opening_visible": False,
        "boundary_kind": "outer_convex_hull",
        "sample_count": 0,
        "reason": "closed visible opening unavailable",
    }
    if hierarchy is None or not contours:
        return None, info
    opening = any(h[3] >= 0 and cv2.contourArea(c) > 25
                  for c, h in zip(contours, hierarchy[0]))
    if not opening:
        return None, info
    hull = cv2.convexHull(np.concatenate(contours)).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull.astype(np.float32)) <= 25:
        return None, dict(info, opening_visible=True, reason="degenerate outer hull")
    boundary = _uniform_closed_polyline(hull, samples)
    return boundary, dict(info, opening_visible=True, sample_count=int(samples),
                          hull_vertices=hull.astype(float).tolist(),
                          reason="valid closed outer hull")


def causal_support(history: Sequence[ContourMeasurement]):
    """Keep P45's causal support contract: prefix 12 plus recent 20."""
    return list(history) if len(history) <= 32 else list(history[:12]) + list(history[-20:])


def dense_perimeter(corners: torch.Tensor, samples_per_edge: int = 8) -> torch.Tensor:
    """Uniformly sample each segment of an ordered four-point 3-D perimeter."""
    if corners.shape[-2:] != (4, 3):
        raise ValueError("corners must end in [4,3]")
    if samples_per_edge < 1:
        raise ValueError("samples_per_edge must be positive")
    fraction = torch.arange(samples_per_edge, device=corners.device,
                            dtype=corners.dtype) / samples_per_edge
    start = corners.unsqueeze(-2)
    end = torch.roll(corners, -1, dims=-2).unsqueeze(-2)
    sampled = start + fraction.reshape(*([1] * (corners.ndim - 2)), 1, -1, 1) * (end - start)
    return sampled.reshape(*corners.shape[:-2], 4 * samples_per_edge, 3)


def project_world_points(points: torch.Tensor, K: torch.Tensor,
                         camera_position: torch.Tensor,
                         camera_rotation: torch.Tensor) -> torch.Tensor:
    """P45 world-to-image projection, including its OpenGL negative-z sign."""
    cam = (points - camera_position) @ camera_rotation
    depth = -cam[..., 2]
    if bool((depth.detach() <= 1e-6).any()):
        raise ValueError("Decoded contour is behind or on the camera plane")
    u = K[0, 2] + K[0, 0] * cam[..., 0] / depth
    v = K[1, 2] - K[1, 1] * cam[..., 1] / depth
    return torch.stack((u, v), dim=-1)


def _point_to_polyline_residual(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    """Point-to-nearest cyclic segment residual with a local ICP assignment."""
    starts = polyline
    ends = torch.roll(polyline, -1, dims=-2)
    segment = ends - starts
    delta = points[..., :, None, :] - starts[..., None, :, :]
    fraction = (delta * segment[..., None, :, :]).sum(-1)
    fraction = fraction / (segment.square().sum(-1)[..., None, :].clamp_min(1e-12))
    fraction = fraction.clamp(0.0, 1.0)
    closest = starts[..., None, :, :] + fraction[..., None] * segment[..., None, :, :]
    residual = points[..., :, None, :] - closest
    # Nearest-segment assignment is the fixed local ICP correspondence.  The
    # selected point-on-segment remains differentiable within that assignment.
    selected = residual.detach().square().sum(-1).argmin(-1)
    gather = selected[..., None, None].expand(*selected.shape, 1, 2)
    return residual.gather(-2, gather).squeeze(-2)


def symmetric_contour_residual(predicted: torch.Tensor, observed: torch.Tensor,
                               image_size: tuple[int, int], *,
                               predicted_polyline: torch.Tensor | None = None,
                               observed_polyline: torch.Tensor | None = None) -> torch.Tensor:
    """Bidirectional point-to-segment Chamfer residual in normalized pixels."""
    scale = predicted.new_tensor([float(image_size[1]), float(image_size[0])])
    predicted, observed = predicted / scale, observed / scale
    predicted_polyline = (predicted if predicted_polyline is None
                          else predicted_polyline / scale)
    observed_polyline = (observed if observed_polyline is None
                         else observed_polyline / scale)
    return torch.cat((_point_to_polyline_residual(predicted, observed_polyline),
                      _point_to_polyline_residual(observed, predicted_polyline)), dim=-2)


def _measurement_residual(points: torch.Tensor, measurements: Iterable[ContourMeasurement],
                          samples_per_edge: int = 8) -> torch.Tensor:
    perimeter = dense_perimeter(points, samples_per_edge)
    residuals = []
    for measurement in measurements:
        tensor = lambda x: torch.as_tensor(x, dtype=points.dtype, device=points.device)
        projected = project_world_points(perimeter, tensor(measurement.K),
                                         tensor(measurement.camera_position),
                                         tensor(measurement.camera_rotation))
        projected_vertices = project_world_points(points, tensor(measurement.K),
                                                  tensor(measurement.camera_position),
                                                  tensor(measurement.camera_rotation))
        observed = tensor(measurement.boundary)
        observed_vertices = tensor(measurement.hull_vertices)
        residuals.append(symmetric_contour_residual(projected, observed,
                                                     measurement.image_size,
                                                     predicted_polyline=projected_vertices,
                                                     observed_polyline=observed_vertices).reshape(-1))
    if not residuals:
        return points.new_zeros(0)
    return torch.cat(residuals)


class P60ContourLatent:
    """Frozen P45 model with a persistent contour-updated eight-D latent.

    Arms:
      ``fixed`` keeps the first latent but still re-encodes the current frame;
      ``continuous`` persists three GN steps after each admitted observation;
      ``current_only`` independently reinitializes and encodes every frame;
      ``batch_fit`` restarts from the initial latent and takes 15 iterations of
      the same nonlinear contour objective (an iterative reference, not Direct).
    """

    ARMS = ("fixed", "continuous", "current_only", "batch_fit")

    def __init__(self, model, arm="continuous", device="cpu",
                 boundary_samples=32, samples_per_edge=8, damping=1e-4,
                 step_scale=.65, step_clip=1.0, inner_steps=3, batch_fit_steps=15):
        if arm not in self.ARMS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {self.ARMS}")
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.arm, self.device = arm, device
        self.model_state_sha256 = model_state_sha256(self.model)
        self.boundary_samples, self.samples_per_edge = boundary_samples, samples_per_edge
        self.damping, self.step_scale, self.step_clip = damping, step_scale, step_clip
        self.inner_steps, self.batch_fit_steps = inner_steps, batch_fit_steps
        self.reference = self.first = self.z = self.initial_z = None
        self.current_offset = self.points = None
        self.last_t = None
        self.history: list[ContourMeasurement] = []
        self.records = []

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _decode(self, reference, offset, z):
        return self.model.decode(reference, offset, z)

    def _residual(self, reference, offset, z, support):
        points = self._decode(reference, offset, z)[0]
        return _measurement_residual(points, support, self.samples_per_edge)

    def _adapt(self, reference, offset, start_z, support, steps):
        z = start_z.detach()
        records = []
        latent_dim = int(z.shape[-1])
        identity = torch.eye(latent_dim, dtype=z.dtype, device=z.device)
        for _ in range(steps):
            z = z.detach().requires_grad_(True)
            residual = self._residual(reference, offset, z, support)
            loss = .5 * residual.square().mean()
            gradient, = torch.autograd.grad(loss, z, retain_graph=False)

            def residual_at(active):
                return self._residual(reference, offset, active, support)

            # Vectorized forward AD evaluates all eight latent tangents in one
            # transform; it does not run a separate view loop per dimension.
            jacobian = torch.func.jacfwd(residual_at)(z).squeeze(-2)
            metric = jacobian.T @ jacobian / max(1, residual.numel())
            metric = metric + identity * self.damping
            change = torch.linalg.solve(metric, gradient.reshape(-1, 1)).reshape_as(z)
            norm = change.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            change = change * torch.clamp(self.step_clip / norm, max=1.0)
            next_z = (z - self.step_scale * change).detach()
            records.append({
                "loss": float(loss.detach()),
                "gradient": float(gradient.norm().detach()),
                "delta": float((next_z - z.detach()).norm()),
                "local_correspondence": "nearest cyclic segment",
            })
            z = next_z
        return z, records

    def _measurement(self, obs, t, goal_mask):
        boundary, info = outer_hull_boundary(goal_mask, self.boundary_samples)
        if boundary is None:
            return None, info
        height, width = np.asarray(goal_mask).shape
        arrays = {name: np.asarray(obs[name]) for name in
                  ("K", "camera_position", "camera_rotation")}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            return None, dict(info, reason="non-finite public camera")
        measurement = ContourMeasurement(
            boundary=boundary,
            hull_vertices=np.asarray(info["hull_vertices"], dtype=np.float32),
            K=arrays["K"].astype(np.float32),
            camera_position=arrays["camera_position"].astype(np.float32),
            camera_rotation=arrays["camera_rotation"].astype(np.float32),
            image_size=(int(height), int(width)), t=int(t))
        return measurement, info

    def _encode_current(self, obs, reset=False):
        if reset or self.reference is None:
            reference_np = initial(obs, self.model.geometry_task)
            aligned = align_observation(obs, reference_np)
            first_np = features(aligned, self.model.representation)
        else:
            reference_np, first_np = self.reference, self.first
        target = reference_np if reset or self.points is None else self.points
        aligned = align_observation(obs, target)
        current_np = features(aligned, self.model.representation)
        reference = self.tensor(reference_np)[None]
        with torch.no_grad():
            z0, offset = self.model.encode(self.tensor(first_np)[None],
                                           self.tensor(current_np)[None], reference)
        return reference_np, first_np, reference, z0.detach(), offset.detach(), aligned

    def update(self, obs, t, goal_mask):
        """Consume one public frame; return True iff the frame was admitted."""
        if self.last_t is not None and int(t) <= self.last_t:
            raise ValueError("P60 observations must have strictly increasing t")
        self.last_t = int(t)
        if obs.get("corners") is None:
            if self.arm == "current_only":
                self.points = None
            return False
        measurement, boundary_info = self._measurement(obs, t, goal_mask)
        if measurement is None:
            if self.arm == "current_only":
                self.points = None
            return False

        reset = self.arm == "current_only"
        reference_np, first_np, reference, z0, offset, _ = self._encode_current(obs, reset)
        if reset:
            self.reference, self.first = reference_np, first_np
            before = z0.clone()
            self.z = self.initial_z = z0.clone()
            support = [measurement]
        else:
            if self.reference is None:
                self.reference, self.first = reference_np, first_np
            if self.z is None:
                self.z, self.initial_z = z0.clone(), z0.clone()
            before = self.z.clone()
            self.history.append(measurement)
            support = causal_support(self.history)

        fixed_points = self._decode(reference, offset, self.z).detach()
        residual_before = _measurement_residual(fixed_points[0], support, self.samples_per_edge)
        loss_before = float(.5 * residual_before.square().mean())
        inner = []
        if self.arm == "continuous" and len(self.history) >= 3:
            self.z, inner = self._adapt(reference, offset, self.z, support, self.inner_steps)
        elif self.arm == "batch_fit" and len(self.history) >= 3:
            self.z, inner = self._adapt(reference, offset, self.initial_z,
                                        support, self.batch_fit_steps)

        self.current_offset = offset.detach()
        decoded = self._decode(reference, offset, self.z).detach()
        self.points = decoded.cpu().numpy()[0]
        residual_after = _measurement_residual(decoded[0], support, self.samples_per_edge)
        loss_after = float(.5 * residual_after.square().mean())
        pre_estimate = pose(fixed_points.cpu().numpy()[0])
        post_estimate = pose(self.points)
        serialize_estimate = lambda estimate: {
            "position": estimate[0].tolist(), "rotation": estimate[1].tolist(),
            "half_size": estimate[2].tolist(),
        }
        self.records.append({
            "t": int(t), "arm": self.arm, "S": len(support), "U": len(inner),
            "z": self.z.cpu().numpy()[0].tolist(),
            "initial_z": self.initial_z.cpu().numpy()[0].tolist(),
            "latent_change": float((self.z - before).norm()),
            "z_from_initial": float((self.z - self.initial_z).norm()),
            "current_offset": offset.cpu().numpy()[0].tolist(),
            "support_loss_before": loss_before, "support_loss_after": loss_after,
            "pre_update_points": fixed_points.cpu().numpy()[0].tolist(),
            "points": self.points.tolist(), "inner": inner,
            "measurement": boundary_info,
            "support_timestamps": [entry.t for entry in support],
            "model_state_sha256": self.model_state_sha256,
            "control_shadow_pre_update": serialize_estimate(pre_estimate),
            "control_shadow_post_update": serialize_estimate(post_estimate),
        })
        return True

    def estimate(self):
        return None if self.points is None else pose(self.points)


def load_p60(path, arm="continuous", device="cpu", **kwargs):
    """Load the frozen P45 checkpoint and wrap it without altering its bytes."""
    model, package = load_model(path, device=device)
    return P60ContourLatent(model, arm=arm, device=device, **kwargs), package
