"""P64 actual LIBERO book bridge; no simulator or evaluator inputs.

This is a new task-specific use of P61's LiLa relation representation, not a
claim that its synthetic slot checkpoint transfers unchanged. Geometry9 is
book-body origin in EEF, caddy back-site origin in first-EEF anchor, log book axis scales.
Rotations and actual visual-mesh vertices are fixed, train-derived priors.
"""
from dataclasses import dataclass
import cv2
import numpy as np
import torch
from torch import nn
from scipy.spatial import ConvexHull
from prove.fast_adapter.p60_contour import (
    _uniform_closed_polyline, _point_to_polyline_residual, project_world_points)


@dataclass(frozen=True)
class BookPriors:
    book_vertices: np.ndarray
    caddy_vertices: np.ndarray
    book_scale_center: np.ndarray
    R_book_eef: np.ndarray
    R_caddy_world: np.ndarray
    final_caddy_book: np.ndarray

    def __post_init__(self):
        for name in ('book_vertices', 'caddy_vertices'):
            array = np.asarray(getattr(self, name))
            if array.ndim != 2 or array.shape[1] != 3 or len(array) < 4 or not np.isfinite(array).all():
                raise ValueError('actual train visual mesh vertices required: ' + name)
        for name in ('R_book_eef', 'R_caddy_world'):
            rotation = np.asarray(getattr(self, name))
            if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1):
                raise ValueError('fixed train rotation required: ' + name)
        final = np.asarray(self.final_caddy_book)
        center = np.asarray(self.book_scale_center)
        if final.shape != (4, 4) or not np.isfinite(final).all() or not np.allclose(final[3], [0, 0, 0, 1]):
            raise ValueError('train caddy-to-book goal transform required')
        if not np.allclose(final[:3,:3].T @ final[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(final[:3,:3]),1):
            raise ValueError('goal rotation must be proper orthonormal')
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('train book geometry scale center required')


@dataclass(frozen=True)
class BookMeasurement:
    role: int  # 1 book, 2 caddy; never an evaluator target
    boundary: np.ndarray
    K: np.ndarray
    camera_pose: np.ndarray
    eef_transform: np.ndarray
    t: int
    T_world_anchor: np.ndarray


def visible_boundary(mask, other_roles, samples=32):
    """Observed outer-hull samples supported by visible mask, away from occluders.

    One-sided residual below does not force an invisible predicted edge toward
    an occlusion boundary. This is a conservative partial-outline measurement;
    it does not establish full observability or recover hidden edges.
    """
    mask = np.asarray(mask, np.uint8)
    occluders = np.asarray(other_roles, np.uint8)
    if mask.ndim != 2 or occluders.shape != mask.shape:
        raise ValueError('same-size public masks required')
    if not mask.any():
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(np.concatenate(contours)).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull.astype(np.float32)) <= 25:
        return None
    boundary = _uniform_closed_polyline(hull, samples)
    edge = np.zeros_like(mask)
    cv2.drawContours(edge, contours, -1, 1, 1)
    distance = cv2.distanceTransform(1-edge, cv2.DIST_L2, 3)
    blocked = cv2.dilate(occluders, np.ones((7, 7), np.uint8))
    xy = np.rint(boundary).astype(int)
    x, y = xy.T
    h, w = mask.shape
    keep = (x > 1) & (x < w-2) & (y > 1) & (y < h-2)
    keep &= (distance[y, x] <= 1.5) & (blocked[y, x] == 0)
    return boundary[keep] if keep.sum() >= 4 else None


def corrected_geometry(geometry9, z):
    if geometry9.shape != (9,) or z.shape != (9,):
        raise ValueError('single geometry9 and z9 required')
    return geometry9 + z * geometry9.new_tensor([.02]*6 + [.25]*3)


def world_vertices(geometry9, z, priors, measurement):
    q = corrected_geometry(geometry9, z)
    tensor = lambda a: torch.as_tensor(a, device=q.device, dtype=q.dtype)
    if measurement.role == 1:
        center = tensor(priors.book_scale_center)
        vertices = center + (tensor(priors.book_vertices)-center)*q[6:9].exp()
        vertices = vertices @ tensor(priors.R_book_eef).T + q[:3]
        eef = tensor(measurement.eef_transform)
        return vertices @ eef[:3, :3].T + eef[:3, 3]
    if measurement.role == 2:
        anchor = tensor(measurement.T_world_anchor)
        center = anchor[:3, :3] @ q[3:6] + anchor[:3, 3]
        return tensor(priors.caddy_vertices) @ tensor(priors.R_caddy_world).T + center
    raise ValueError('book/caddy roles only')


def silhouette_residual(geometry9, z, priors, measurements, hull_indices=None):
    """Public visible boundary -> projected actual-mesh convex envelope.

    Freeze hull membership at each GN linearization, as nearest-edge ICP does.
    Sizes affect the book mesh; caddy nominal shape is an explicit fixed prior.
    """
    residuals = []
    for i, measurement in enumerate(measurements):
        tensor = lambda a: torch.as_tensor(a, device=z.device, dtype=z.dtype)
        camera = tensor(measurement.camera_pose)
        pixels = project_world_points(world_vertices(geometry9, z, priors, measurement),
            tensor(measurement.K), camera[:3, 3], camera[:3, :3])
        indices = (ConvexHull(pixels.detach().cpu().numpy()).vertices.copy()
                   if hull_indices is None else hull_indices[i])
        hull = pixels[torch.as_tensor(indices, device=z.device)]
        residuals.append(_point_to_polyline_residual(tensor(measurement.boundary), hull).flatten())
    if not residuals:
        raise ValueError('at least one public silhouette measurement required')
    return torch.cat(residuals)


def adapt_book_z(geometry9, z, priors, measurements, *, create_graph=False):
    """Shared three-step GN; missing visible support preserves persistent z."""
    if not measurements:
        return z, []
    from prove.fast_adapter.p64_common import adapt_z
    def factory(active):
        indices=[]
        for m in measurements:
            camera=torch.as_tensor(m.camera_pose,dtype=z.dtype,device=z.device)
            pixels=project_world_points(world_vertices(geometry9,active,priors,m),
                torch.as_tensor(m.K,dtype=z.dtype,device=z.device),camera[:3,3],camera[:3,:3])
            indices.append(ConvexHull(pixels.detach().cpu().numpy()).vertices.copy())
        return lambda value:silhouette_residual(geometry9,value,priors,measurements,indices)
    return adapt_z(lambda value:silhouette_residual(geometry9,value,priors,measurements),
                   z,create_graph=create_graph,residual_factory=factory)


def frozen_book_action(policy, geometry9, z, priors, eef_transform, bboxwh, gripper, T_world_anchor):
    """Connect new task geometry to the exact frozen P57 action network.

    Book size affects silhouette estimation and observed bbox input; the old
    action head has no explicit scale channels. Report this limitation.
    """
    q = corrected_geometry(geometry9, z)
    tensor = lambda a: torch.as_tensor(a, device=q.device, dtype=q.dtype)
    eef = tensor(eef_transform)
    rotation = tensor(priors.R_caddy_world)
    anchor = tensor(T_world_anchor)
    caddy_world = anchor[:3, :3] @ q[3:6] + anchor[:3, 3]
    goal = caddy_world + rotation @ tensor(priors.final_caddy_book)[:3, 3]
    desired = rotation @ tensor(priors.final_caddy_book)[:3, :3] @ tensor(priors.R_book_eef).T
    from scipy.spatial.transform import Rotation
    dr = tensor(Rotation.from_matrix((desired @ eef[:3, :3].T).detach().cpu().numpy()).as_rotvec())
    geometry6 = torch.cat((q[:3], goal))
    # B/z0 correction belongs to the historical estimator, not the action MLP.
    # New decoded geometry reaches the exact same policy.action via zero latent.
    return policy.command(geometry6, torch.zeros_like(policy.z0), eef[:3, 3], desired,
                          dr, tensor(bboxwh), tensor(gripper))
