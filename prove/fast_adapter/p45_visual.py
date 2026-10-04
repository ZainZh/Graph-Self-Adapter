"""P45 train-only projections of frozen, visible-role LiLa moments.

Both visual arms start from the SAME frozen P43 appearance/x/y moments. The
relative arm changes the moment coordinates to each visible bounding box;
this is not a second visual backbone or a newly introduced local ROI branch.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from prove.fast_adapter.neural_geometry_visual import FrozenRoleVision, fit_role_pca

VISUAL_REPRESENTATIONS = ('absolute_fusion', 'relative_fusion')
PCA_SCHEMA = 'prove.p45-role-projection.v1'


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_training_binding(binding, verify_file=True):
    """Require a separately sealed list of training inputs, never test stats."""
    if binding.get('split') != 'train':
        raise ValueError('Projection fitting requires an explicit train split')
    seeds = binding.get('seeds', [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Projection fitting requires unique training seeds')
    if not isinstance(binding.get('sha256'), str) or len(binding['sha256']) != 64:
        raise ValueError('Projection training-input digest is missing')
    if verify_file and _sha(binding['path']) != binding['sha256']:
        raise ValueError('Projection training inputs changed after sealing')


def relative_moments(raw, boxes, visible):
    """Invertible box-centered coordinates BEFORE rank-16 PCA truncation.

    P43's 28 patch coordinates run from -1 to +1, with centers 8..440px.
    Boxes are pixel-edge xyxy. A patch-width floor protects tiny visible parts;
    absent masks remain zero. No target dimensions or pose are consulted.
    """
    x = torch.as_tensor(raw)
    if not x.is_floating_point():
        x = x.float()
    if x.shape[-2:] != (3, 3072):
        raise ValueError('Expected three appearance/x/y role moments')
    bbox = torch.as_tensor(boxes, device=x.device, dtype=x.dtype)
    seen = torch.as_tensor(visible, device=x.device, dtype=torch.bool)
    if bbox.shape != x.shape[:-1] + (4,) or seen.shape != x.shape[:-1]:
        raise ValueError('Role boxes/visibility must match the raw moment batch')
    parts = x.reshape(*x.shape[:-1], 3, 1024)
    centers = (bbox[..., :2] + bbox[..., 2:]) * .5
    centers = (centers - 8.) / 216. - 1.
    extents = ((bbox[..., 2:] - bbox[..., :2]) / 432.).clamp_min(1. / 27.)
    moments = (parts[..., 1:, :] - centers[..., None] * parts[..., :1, :]) / extents[..., None]
    return torch.cat((parts[..., :1, :], moments), dim=-2).flatten(-2) * seen[..., None]


def validate_projection(package, representation=None, verify_training_file=False):
    if package.get('schema') != PCA_SCHEMA or package.get('fit_split') != 'train':
        raise ValueError('A P45 training-only projection is required')
    kind = package.get('representation')
    if kind not in VISUAL_REPRESENTATIONS or (representation is not None and kind != representation):
        raise ValueError('Projection/representation mismatch')
    validate_training_binding(package['training_binding'], verify_training_file)
    if tuple(package['mean'].shape) != (3, 3072) or tuple(package['basis'].shape) != (3, 3072, 16):
        raise ValueError('Unexpected P45 projection dimensions')
    if tuple(package['scale'].shape) != (3, 16) or not torch.all(package['scale'] > 0):
        raise ValueError('Projection whitening scales must be positive')
    if not all(torch.isfinite(package[key]).all() for key in ('mean', 'basis', 'scale')):
        raise ValueError('Projection contains nonfinite parameters')


def fit_projections(train_raw, train_visible, train_boxes, training_binding, seed=450, device='cuda'):
    validate_training_binding(training_binding)
    raw = torch.as_tensor(train_raw, dtype=torch.float32, device=device)
    visible = torch.as_tensor(train_visible, dtype=torch.bool, device=device)
    if raw.ndim != 4 or raw.shape[0] != len(training_binding['seeds']):
        raise ValueError('Raw projection input must preserve the declared training episode axis')
    normalized = relative_moments(raw, train_boxes, visible)
    packages = {}
    for kind, values in zip(VISUAL_REPRESENTATIONS, (raw, normalized)):
        package = fit_role_pca(values, visible, seed=seed, device=device)
        package.update(schema=PCA_SCHEMA, representation=kind, training_binding=dict(training_binding),
                       raw_input_shape=list(raw.shape),
                       moment_frame='image' if kind == 'absolute_fusion' else 'visible-role-box')
        validate_projection(package, kind)
        packages[kind] = package
    return packages


def project_features(raw, visible, boxes, package, device='cpu'):
    validate_projection(package)
    x = torch.as_tensor(raw, dtype=torch.float32, device=device)
    seen = torch.as_tensor(visible, dtype=torch.bool, device=device)
    if x.shape[-2:] != (3, 3072) or seen.shape != x.shape[:-1]:
        raise ValueError('Raw role features and visibility have incompatible shapes')
    if package['representation'] == 'relative_fusion':
        x = relative_moments(x, boxes, seen)
    mean, basis, scale = (package[key].to(device) for key in ('mean', 'basis', 'scale'))
    projected = torch.einsum('...rd,rdk->...rk', x - mean, basis) / scale
    projected = (projected * seen[..., None]).flatten(-2)
    if not torch.isfinite(projected).all():
        raise ValueError('Nonfinite projected visual features')
    return projected


class FrozenP45Vision:
    """Reuse one frozen P43 encoder, apply a separately frozen P45 projection."""
    def __init__(self, pca_path, representation, model_dir='assets/dinov3', device='cuda', microbatch=1, raw_vision=None):
        self.package = torch.load(pca_path, map_location='cpu', weights_only=True)
        validate_projection(self.package, representation)
        self.raw_vision = raw_vision if raw_vision is not None else FrozenRoleVision(
            model_dir=model_dir, device=device, microbatch=microbatch)
        self.device = device

    def encode(self, images, masks, boxes=None):
        from prove.fast_adapter.neural_geometry_visual import mask_geometry
        if boxes is None:
            boxes = np.stack([mask_geometry(mask)[0] for mask in masks])
        visible = np.asarray(masks).reshape(len(masks), 3, -1).any(-1)
        raw = self.raw_vision.encode(images, masks)
        return project_features(raw, visible, boxes, self.package, device=self.device)
