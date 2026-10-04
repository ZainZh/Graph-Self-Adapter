"""P45 controlled representation comparison around a frozen neural geometry head.

The three arms use the same width, latent dimension and parameter count. Geometry
zeros 48 appearance inputs; fusion uses the P43 absolute moments and boxes;
relative_fusion changes both visual moment coordinates and box parameterization.
All retain visible corners, camera calibration, tool proprioception, PnP and the
analytic ray objective. This is not a pure RGB or cross-attention experiment.
"""
from __future__ import annotations

import numpy as np
import torch

from prove.fast_adapter.geometry3d_adapter import initial, pose
from prove.fast_adapter.neural_geometry_adapter import (
    FEATURE_DIM, NeuralGeometry, align_observation, observation_features, ray_matrices,
)

REPRESENTATIONS = ('geometry', 'absolute_fusion', 'relative_fusion')
ARMS = ('fixed', 'continuous', 'prefix3', 'mismatched')
TASK_GEOMETRY = {'cup': 'cup', 'insert': 'insert', 'slider': 'cup'}


def relation_boxes(boxes, visible):
    """Same 12 numbers: goal xyxy plus actor/object center-to-goal and size.

    Together with visibility, this is invertible for every visible role. It
    adds no target truth or information beyond the original visible boxes.
    """
    bbox = np.asarray(boxes, dtype=np.float32).reshape(3, 4) / 448.
    seen = np.asarray(visible, dtype=bool).reshape(3)
    result = bbox.copy()
    anchor = (bbox[2, :2] + bbox[2, 2:]) * .5 if seen[2] else np.zeros(2, np.float32)
    for role in (0, 1):
        if seen[role]:
            result[role] = np.r_[(bbox[role, :2] + bbox[role, 2:]) * .5 - anchor,
                                bbox[role, 2:] - bbox[role, :2]]
        else:
            result[role] = 0.
    if not seen[2]:
        result[2] = 0.
    return result.flatten()


def features(obs, representation):
    if representation not in REPRESENTATIONS:
        raise ValueError(f'Unknown P45 representation: {representation}')
    public = dict(obs)
    if representation == 'geometry':
        # Do not read appearance, including a possible nonfinite visual cache.
        public['local'] = np.zeros(48, np.float32)
    result = observation_features(public)
    if representation == 'relative_fusion':
        result[80:92] = relation_boxes(obs['role_bbox'], obs['role_visible'])
    if result.shape != (FEATURE_DIM,) or not np.isfinite(result).all():
        raise ValueError('P45 features must be finite and 95-dimensional')
    return result


class P45NeuralGeometry(NeuralGeometry):
    def __init__(self, task, representation, feature_dim=FEATURE_DIM, width=96, latent_dim=8, geometry_task=None):
        if task not in TASK_GEOMETRY or representation not in REPRESENTATIONS:
            raise ValueError('Unknown P45 task or representation')
        geometry_task = geometry_task or TASK_GEOMETRY[task]
        if geometry_task != TASK_GEOMETRY[task] or feature_dim != FEATURE_DIM:
            raise ValueError('P45 nominal geometry frame or input budget mismatch')
        super().__init__(geometry_task, feature_dim=feature_dim, width=width, latent_dim=latent_dim)
        self.task = task
        self.geometry_task = geometry_task
        self.representation = representation

    def config(self):
        return dict(super().config(), geometry_task=self.geometry_task, representation=self.representation)


class P45Latent:
    """Weights stay frozen; all arms re-encode the current public observation.

    prefix3 permits exactly three update EVENTS, each with three gradient steps.
    It does not mean t<12 or three support images. Mismatched transfers the
    donor's z-z0 at the latest time <= current time, preserving receiver z0.
    Donor provenance and independent condition assignment belong to the runner.
    """
    def __init__(self, model, arm='continuous', device='cpu', donor_deltas=None):
        if arm not in ARMS:
            raise ValueError(f'Unknown P45 update arm: {arm}')
        if arm == 'mismatched' and not donor_deltas:
            raise ValueError('Mismatched adaptation requires a sealed donor delta-z schedule')
        if arm != 'mismatched' and donor_deltas is not None:
            raise ValueError('Donor input is only allowed for the mismatched arm')
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.arm, self.device = arm, device
        self.reference = self.first = self.z = self.initial_z = self.points = None
        self.history, self.records = [], []
        self.update_events = 0
        self.donor_deltas = {}
        for t, delta in (donor_deltas or {}).items():
            value = np.asarray(delta, np.float32)
            if value.shape != (model.latent_dim,) or not np.isfinite(value).all() or int(t) < 0:
                raise ValueError('Invalid causal donor delta-z schedule')
            self.donor_deltas[int(t)] = value

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def update(self, obs, t):
        if obs['corners'] is None:
            return
        if self.reference is None:
            self.reference = initial(obs, self.model.geometry_task)
            aligned = align_observation(obs, self.reference)
            self.first = features(aligned, self.model.representation)
        aligned = align_observation(obs, self.points if self.points is not None else self.reference)
        current = features(aligned, self.model.representation)
        reference = self.tensor(self.reference)[None]
        with torch.no_grad():
            z0, offset = self.model.encode(self.tensor(self.first)[None], self.tensor(current)[None], reference)
        if self.z is None:
            self.z = z0.detach()
            self.initial_z = self.z.clone()
        before = self.z.clone()
        fixed_points = self.model.decode(reference, offset, self.z).detach()
        self.history.append(ray_matrices(aligned))
        support = self.history if len(self.history) <= 32 else self.history[:12] + self.history[-20:]
        A = self.tensor(np.stack([entry[0] for entry in support]))[None]
        C = self.tensor(np.stack([entry[1] for entry in support]))[None]
        loss_before = float(self.model.ray_loss(fixed_points, A, C)[0])
        inner, donor_t = [], None
        allowed = self.arm == 'continuous' or (self.arm == 'prefix3' and self.update_events < 3)
        if allowed and len(self.history) >= 3:
            self.z, inner = self.model.adapt(reference, offset, self.z, A, C, steps=3)
            self.z = self.z.detach()
            self.update_events += 1
        elif self.arm == 'mismatched':
            available = [time for time in self.donor_deltas if time <= int(t)]
            if available:
                donor_t = max(available)
                self.z = self.initial_z + self.tensor(self.donor_deltas[donor_t])[None]
        self.points = self.model.decode(reference, offset, self.z).detach().cpu().numpy()[0]
        loss_after = float(self.model.ray_loss(self.tensor(self.points)[None], A, C)[0])
        self.records.append(dict(
            t=int(t), S=len(support), U=len(inner), update_events=self.update_events,
            z=self.z.cpu().numpy()[0].tolist(), initial_z=self.initial_z.cpu().numpy()[0].tolist(),
            delta_z=(self.z - self.initial_z).cpu().numpy()[0].tolist(), donor_t=donor_t,
            latent_change=float((self.z - before).norm()), z_from_initial=float((self.z - self.initial_z).norm()),
            initial_z_points=self.model.decode(reference, offset, self.initial_z).detach().cpu().numpy()[0].tolist(),
            point_change=float(np.linalg.norm(self.points - fixed_points.cpu().numpy()[0])),
            support_loss_before=loss_before, support_loss_after=loss_after, inner=inner,
            pre_update_points=fixed_points.cpu().numpy()[0].tolist(), representation=self.model.representation,
        ))

    def estimate(self):
        return None if self.points is None else pose(self.points)


def load_model(path, device='cpu'):
    package = torch.load(path, map_location=device, weights_only=False)
    if package.get('schema') != 'prove.p45-neural-checkpoint.v1':
        raise ValueError('P45 runtime requires a P45 checkpoint')
    model = P45NeuralGeometry(**package['model_config'])
    model.load_state_dict(package['model'])
    return model.eval(), package
