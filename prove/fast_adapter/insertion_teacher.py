"""P26 privileged TRAINING/REFERENCE teacher; never a learned runtime input.

Old geometry_insert.py is intentionally unchanged. Phase is inferred from the
current geometric relation, so labels can be regenerated on recorded states.
"""
import numpy as np
from prove.fast_adapter.insert3d import STEP, HALF, INNER, SLOT_TOP


def phase_nominal(position, xy, scale):
    p = np.asarray(position, dtype=np.float64)
    xy, s = np.asarray(xy, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    error = np.abs(p[..., :2] - xy)
    centered = np.all(error <= .65*(INNER-HALF[:2])*s[..., None], axis=-1)
    inside = np.all(error <= (INNER-HALF[:2]+.0002)*s[..., None], axis=-1)
    bottom = p[..., 2] - HALF[2]*s
    completed_geometry = inside & (bottom >= .0065*s) & (bottom <= .013*s)
    in_channel = inside & (bottom <= SLOT_TOP*s)
    goal = np.concatenate((np.broadcast_to(xy, p[..., :2].shape),
        (np.where(centered | in_channel, .0385, .110)*s)[..., None]), axis=-1)
    goal = np.where(completed_geometry[..., None], p, goal)
    return np.clip(.65*(goal-p)/STEP, -1., 1.)


def action_labels(positions, configs, matrices, biases):
    """positions [domain,time,3]; truth supplied only by training/evaluator."""
    xy = np.asarray([c['xy'] for c in configs])[:, None, :]
    scale = np.asarray([c['scale'] for c in configs])[:, None]
    nominal = phase_nominal(positions, xy, scale)
    rhs = nominal - np.asarray(biases)[:, None, :]/STEP
    return np.clip(np.linalg.solve(np.asarray(matrices)[:, None], rhs[..., None])[..., 0], -1, 1).astype(np.float32)


def reference_action(env):
    nominal = phase_nominal(env.state(), env.xy, env.scale)
    return np.clip(np.linalg.solve(env.M, nominal-env.bias/STEP), -1, 1).astype(np.float32)
