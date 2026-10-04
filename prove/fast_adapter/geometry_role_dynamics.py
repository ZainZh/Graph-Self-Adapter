"""P36 explicit RGB-geometry interaction token ablations.

Runtime inputs are frozen visual tokens/boxes, proprioception and executed
actions.  No simulator pose, contact, success or future state is read.
"""
import math
import torch
from torch import nn

from prove.fast_adapter.role_dynamics import EFFECT_DIM, observed_effect

GEOMETRY_MODES = ("no_explicit_bbox_geometry", "p34_existing_bbox_geometry", "p36_explicit_relation_geometry")
TASKS = ("mug-handle-lift", "hammer-block")


def _relative(boxes):
    return (boxes[..., 1, :2] - boxes[..., 0, :2]) / boxes[..., 1, 2:].clamp_min(0.03)


def _iou(boxes):
    center = (boxes[..., :2] + 1) * 0.5
    half = boxes[..., 2:] * 0.5
    lo = center - half
    hi = center + half
    inter_lo = torch.maximum(lo[..., 0, :], lo[..., 1, :])
    inter_hi = torch.minimum(hi[..., 0, :], hi[..., 1, :])
    inter_wh = (inter_hi - inter_lo).clamp_min(0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area = (boxes[..., 2].clamp_min(0) * boxes[..., 3].clamp_min(0))
    return inter / (area[..., 0] + area[..., 1] - inter).clamp_min(1e-8)


def geometry_features(boxes, box_velocity, valid, mode):
    """Return a fixed 32D RGB-derived geometry slot.

    `valid` is role visibility from RGB tokens.  Entity geometry is zeroed when
    that entity is not visible; pair geometry is zeroed unless both roles are
    visible, so a missing mask cannot create a synthetic relation.
    """
    if mode not in GEOMETRY_MODES:
        raise ValueError(mode)
    shape = boxes.shape[:-2]
    out = boxes.new_zeros(*shape, 32)
    if mode == "no_explicit_bbox_geometry":
        return out

    role_valid = valid.float()[..., None]
    pair = valid.all(-1).float()[..., None]
    safe_boxes = boxes * role_valid
    safe_vel = box_velocity * role_valid
    rel = _relative(boxes) * pair
    obj_wh = boxes[..., 1, 2:] * valid[..., 1:2].float()
    flags = valid.float()
    existing = torch.cat(
        (
            safe_boxes.flatten(-2),
            safe_vel.flatten(-2) * 20,
            rel,
            obj_wh,
            flags,
        ),
        -1,
    )
    out[..., : existing.shape[-1]] = existing
    if mode == "p34_existing_bbox_geometry":
        return out

    delta = (boxes[..., 0, :2] - boxes[..., 1, :2]) * 0.5 * pair
    norm_delta = delta / boxes[..., 1, 2:].clamp_min(0.03)
    sizes = (boxes[..., :, 2:] * role_valid).flatten(-2)
    dist = delta.square().sum(-1, keepdim=True).sqrt() * pair
    overlap = _iou(boxes)[..., None] * pair
    explicit = torch.cat((delta, norm_delta, sizes, dist, overlap), -1) * pair
    out[..., 22:32] = explicit
    return out


class GeometryRoleDynamics(nn.Module):
    """Same-capacity token model; only the 32D geometry slot changes by mode."""

    def __init__(self, mode="no_explicit_bbox_geometry", task="hammer-block"):
        super().__init__()
        if mode not in GEOMETRY_MODES:
            raise ValueError(mode)
        if task not in TASKS:
            raise ValueError(task)
        self.mode = mode
        self.task = task
        self.task_index = TASKS.index(task)
        self.role_visual = nn.Linear(48 + 1, 256)
        self.proprio_project = nn.Linear(8, 256)
        self.geometry_project = nn.Linear(32, 256)
        self.goal_semantic = nn.Parameter(torch.randn(2, 256) * 0.01)
        self.skill_embedding = nn.Parameter(torch.randn(2, 32) * 0.01)
        self.skill_project = nn.Linear(32, 256)
        self.roles = nn.Parameter(torch.randn(3, 256) * 0.01)
        self.readout = nn.Sequential(nn.Linear(5 * 256, 256), nn.SiLU(), nn.LayerNorm(256))
        self.core = nn.Sequential(nn.Linear(272, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU())
        self.mean = nn.Linear(128, EFFECT_DIM)
        self.gate = nn.Linear(128, 32)
        self.basis = nn.Parameter(torch.randn(EFFECT_DIM, 32) / math.sqrt(32))
        self.z0 = nn.Parameter(torch.zeros(32))
        self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(0.1))))
        self.register_buffer("token_mean", torch.zeros(50))
        self.register_buffer("token_sd", torch.ones(50))
        self.register_buffer("effect_sd", torch.ones(EFFECT_DIM))
        self.register_buffer("direction", torch.tensor([1.0, 0.0]))
        self.register_buffer("progress_sd", torch.tensor(1.0))

    def set_stats(self, stats):
        for key in ("token_mean", "token_sd", "effect_sd", "direction", "progress_sd"):
            getattr(self, key).copy_(stats[key])

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.raw_alpha)

    def encode(self, tokens, boxes, proprio, previous, box_velocity):
        shape = tokens.shape[:-3]
        n = math.prod(shape)
        mask = tokens[..., -1:]
        valid = mask.sum(-2).squeeze(-1) > 0
        appearance = (tokens[..., :48] - self.token_mean[:48]) / self.token_sd[:48]
        pooled = (appearance * mask).sum(-2) / mask.sum(-2).clamp_min(1)
        role_in = torch.cat((pooled, valid[..., None].float()), -1)
        roles = self.role_visual(role_in).reshape(n, 2, 256)
        p = (proprio - proprio.new_tensor([0, 0.65, 0.16, 0.5])) / proprio.new_tensor([0.2, 0.2, 0.2, 0.5])
        own = torch.cat((p, previous), -1).reshape(n, 8)
        actor = roles[:, 0] + self.proprio_project(own) + self.roles[0]
        obj = roles[:, 1] + self.roles[1]
        goal = self.goal_semantic[self.task_index][None].expand(n, -1) + self.roles[2]
        geo = self.geometry_project(geometry_features(boxes, box_velocity, valid, self.mode).reshape(n, 32))
        skill = self.skill_project(self.skill_embedding[self.task_index][None].expand(n, -1))
        h = self.readout(torch.cat((actor, obj, goal, geo, skill), -1))
        return h.reshape(*shape, 256)

    def coefficients(self, u, plans):
        h = self.core(torch.cat((u, plans.flatten(-2)), -1))
        return self.mean(h), 2 * torch.sigmoid(self.gate(h))

    def predict_coeff(self, mean, gate, z):
        return mean + (gate * z) @ self.basis.T

    def loss(self, pred, y, w):
        return ((pred - y).square() * w).sum(-1) / w.sum(-1).clamp_min(1e-8)

    def update(self, z, mean, gate, y, w):
        pred = self.predict_coeff(mean, gate, z[:, None])
        denom = w.sum((-1, -2)).clamp_min(1e-8)
        grad = 32 * ((((pred - y) * w) @ self.basis) * gate).sum(1) / denom[:, None]
        grad = grad + 0.002 * (z - self.z0)
        norm = grad.norm(dim=-1, keepdim=True)
        step = grad * (5 / norm.clamp_min(5))
        nz = z - self.alpha * step
        delta = nz - self.z0
        radius = delta.norm(dim=-1, keepdim=True)
        nz = self.z0 + delta * (5 / radius.clamp_min(5))
        return nz, {
            "loss": (((pred - y).square() * w).sum((-1, -2)) / denom).detach(),
            "grad_norm": norm.detach().squeeze(-1),
            "delta_norm": (nz - z).detach().norm(dim=-1),
        }


def action_candidates(base, task):
    offsets = base.new_zeros(9, 4)
    for k in range(3):
        offsets[1 + 2 * k, k] = 0.25
        offsets[2 + 2 * k, k] = -0.25
    if task == "mug-handle-lift":
        offsets[7, 3] = 0.35
        offsets[8, 3] = -0.35
    else:
        offsets[7, :2] = 0.25
        offsets[8, :2] = -0.25
    actions = (base[:, None] + offsets).clamp(-1, 1)
    if task == "hammer-block":
        actions[..., 3] = 0.9
    return actions


def select_action(net, u, base, z):
    actions = action_candidates(base, net.task)
    action_plans = actions[:, :, None].expand(-1, -1, 4, -1)
    mean, gate = net.coefficients(u[:, None].expand(-1, 9, -1), action_plans)
    pred = net.predict_coeff(mean, gate, z[:, None]) * net.effect_sd
    progress = (pred[..., 128:130] * net.direction).sum(-1) / net.progress_sd
    scores = progress - 0.2 * (actions - base[:, None]).square().sum(-1)
    choice = scores.argmax(-1)
    return actions[torch.arange(len(base), device=base.device), choice], choice, scores
