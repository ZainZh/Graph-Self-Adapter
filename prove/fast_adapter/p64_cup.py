"""P64 cup-specific geometry bridge; no simulator truth enters runtime math.

The current LiLa relation encoder predicts the front handle surface center and
OUTER half sizes. Unlike the insertion task there is no extra rim expansion.
Orientation is the declared, fixed yaw-zero task prior in both evaluation arms.
The rendered roles mean robot, cup body with support context, and handle front.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from prove.fast_adapter.geometry3d_env import configuration as historical_configuration
from prove.fast_adapter.p60_contour import _measurement_residual
from prove.fast_adapter.p61_adaptation import corrected_components
from prove.fast_adapter.p62_adaptation import _adapt_residual, P62InnerConfig

ROLE_NAMES = ('actor_robot', 'cup_body_and_support_context', 'functional_handle_front')
PUBLIC_KEYS = ('K', 'camera_position', 'camera_rotation', 'position', 'rotation')
NOMINAL_ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], dtype=np.float32)


def configuration(task, seed, split='id'):
    """Evaluator-only configuration; size and position strata do not mix.

    Existing historical cup size ranges are retained, but historical OOD yaw
    is explicitly removed. Position OOD changes one XY axis by 40--55 mm from
    the training center; the other remains within its training range.
    """
    if task != 'cup' or split not in ('id', 'size', 'position'):
        raise ValueError('P64 cup supports only id/size/position')
    cfg = historical_configuration('cup', seed, 'geometry' if split == 'size' else 'id')
    cfg.update(yaw=0., split=split)
    if split == 'position':
        rng = np.random.default_rng(int(seed) + 640000)
        axis = int(rng.integers(2))
        cfg['center'][axis] = (-.045 if axis == 0 else 0.) + float(rng.choice([-1, 1]) * rng.uniform(.040, .055))
    return cfg


def public_pose(observation):
    """Explicit allowlist strips legacy corners and evaluator metadata."""
    result = {key: np.asarray(observation[key]).copy() for key in PUBLIC_KEYS}
    expected = dict(K=(3, 3), camera_position=(3,), camera_rotation=(3, 3), position=(3,), rotation=(3, 3))
    if any(result[k].shape != expected[k] or not np.isfinite(result[k]).all() for k in PUBLIC_KEYS):
        raise ValueError('Malformed public camera/own pose')
    return result


def fixed_orientation(geometry8, anchor):
    """Replace predicted orientation using the shared task prior, never truth."""
    from scipy.spatial.transform import Rotation
    value = geometry8.clone()
    local = anchor[:3, :3].detach().cpu().numpy().T @ NOMINAL_ROTATION
    value[..., 3:6] = value.new_tensor(Rotation.from_matrix(local).as_rotvec())
    return value


def world_boundary(geometry8, z, anchor):
    center, rotation, half_size = corrected_components(geometry8, z)
    signs = geometry8.new_tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    xy = signs * half_size[..., None, :]
    local = torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)
    anchored = local @ rotation.transpose(-1, -2) + center[..., None, :]
    return anchored @ anchor[:3, :3].T + anchor[:3, 3]


def adapt(geometry8, z, anchor, measurements, *, arm, create_graph=False):
    if arm not in ('fixed', 'translation_size5'):
        raise ValueError('Unknown cup arm')
    if arm == 'fixed':
        return torch.zeros_like(z), []
    if not measurements:
        if bool(z[3:6].detach().abs().any()):
            raise ValueError('Orientation latent must remain zero')
        return z, []
    residual = lambda active: _measurement_residual(world_boundary(geometry8, active, anchor), measurements, 8)
    return _adapt_residual(residual, z, candidate=arm, config=P62InnerConfig(), create_graph=create_graph)


class CupGraspHead(nn.Module):
    """Small learned task action map from outer half-size to local grip offset.

    Train only on training geometry, using the existing controller as teacher.
    No analytical fallback exists in forward; checkpoint weights are required.
    Code phases continue to handle approach, closing and lift.
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    def forward(self, half_size):
        return self.linear(half_size / .01) * .01


def fit_grasp_head(training_half_sizes):
    """Training-only least-squares distillation; never called by runtime."""
    sizes = torch.as_tensor(training_half_sizes, dtype=torch.float64)
    if sizes.ndim != 2 or sizes.shape[1] != 2 or len(sizes) < 3 or not torch.isfinite(sizes).all():
        raise ValueError('Expected finite train-only Nx2 geometry')
    design = torch.cat((sizes / .01, torch.ones_like(sizes[:, :1])), -1)
    if torch.linalg.matrix_rank(design) != 3:
        raise ValueError('Training geometry must span both size axes')
    targets = torch.stack((sizes[:, 0] - .003, torch.zeros_like(sizes[:, 0]), torch.full_like(sizes[:, 0], -.004)), -1) / .01
    solution = torch.linalg.lstsq(design, targets).solution
    head = CupGraspHead()
    with torch.no_grad():
        head.linear.weight.copy_(solution[:2].T)
        head.linear.bias.copy_(solution[2])
    return head


def controller_estimate(geometry8, z, anchor, head):
    """Bridge learned grip location into the unchanged phase controller."""
    center, rotation, size = corrected_components(geometry8, z)
    rotation = anchor[:3, :3] @ rotation
    center = anchor[:3, :3] @ center + anchor[:3, 3]
    grip = center + rotation @ head(size)
    # P45 command uses p + R.x*(size.x-.003) - R.z*.004.
    # This bridge exactly places that command at the neural head's grip.
    return grip + rotation[:, 2] * .004, rotation, size.new_tensor([.003, 0.])
