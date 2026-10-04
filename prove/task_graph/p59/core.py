"""P59 differentiable P50 action path and continual latent adaptation.

P53/P56 already establish unrolled inner updates and future-action outer BC.
P59 changes the adapted representation and feedback: a canonical static
functional-geometry correction, low-dimensional relation prediction, and the
frozen P50 13-to-6 action head.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from prove.task_graph.precision_action_policy import PrecisionActionMLP
from .contracts import P59Contract, PublicTransition, validate_history


P50_CHECKPOINT_SHA256 = "96fdcf6a8ed49e53615bdbff8d3f9cf69ffed85353c18f0584b31e024b4b4fc7"
P50_CHECKPOINT_SCHEMA = "prove.p50-precision-action-checkpoint.v1"


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.square().sum(-1, keepdim=True).clamp_min(eps).sqrt()


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 6:
        raise ValueError("rotation representation must be 6D")
    x = _normalize(value[..., :3])
    y_seed = value[..., 3:]
    y = _normalize(y_seed - (x * y_seed).sum(-1, keepdim=True) * x)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in 3x3")
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def so3_exp_map(rotvec: torch.Tensor) -> torch.Tensor:
    if rotvec.shape[-1] != 3:
        raise ValueError("rotvec must end in 3")
    theta2 = rotvec.square().sum(-1, keepdim=True)
    # torch.where evaluates both branches. Safe denominators keep the inactive
    # exact-form branch finite at zero, including second-order meta-gradients.
    safe_theta2 = theta2.clamp_min(1e-12)
    safe_theta = safe_theta2.sqrt()
    a_exact = torch.sin(safe_theta) / safe_theta
    b_exact = (1 - torch.cos(safe_theta)) / safe_theta2
    a = torch.where(theta2 > 1e-8, a_exact, 1 - theta2 / 6 + theta2.square() / 120)
    b = torch.where(theta2 > 1e-8, b_exact, .5 - theta2 / 24 + theta2.square() / 720)
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*rotvec.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand_as(skew)
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


def so3_log_map(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must end in 3x3")
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1) * .5).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cosine)
    vee = torch.stack((matrix[..., 2, 1] - matrix[..., 1, 2],
                       matrix[..., 0, 2] - matrix[..., 2, 0],
                       matrix[..., 1, 0] - matrix[..., 0, 1]), -1)
    scale = torch.where(theta.abs() > 1e-4, theta / (2 * torch.sin(theta)),
                        .5 + theta.square() / 12)
    return scale[..., None] * vee


def split_geometry(geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if geometry.shape[-1] != 12:
        raise ValueError("control geometry must be [position3, rotation6, half-size2, held-length1]")
    position = geometry[..., :3]
    rotation = rotation_6d_to_matrix(geometry[..., 3:9])
    half_size = geometry[..., 9:11]
    held_length = geometry[..., 11:12]
    if torch.any(half_size <= 0) or torch.any(held_length <= 0):
        raise ValueError("geometry sizes and held length must be positive")
    return position, rotation, half_size, held_length


def pack_geometry(position: torch.Tensor, rotation: torch.Tensor,
                  half_size: torch.Tensor, held_length: torch.Tensor) -> torch.Tensor:
    return torch.cat((position, matrix_to_rotation_6d(rotation), half_size, held_length), -1)


def geometry_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dimensionless control-scaled geometry error, averaged per example."""
    pp, pr, ps, pl = split_geometry(prediction)
    tp, tr, ts, tl = split_geometry(target)
    position = ((pp - tp) / .02).square().mean(-1)
    rotation = (so3_log_map(tr.transpose(-1, -2) @ pr) / .25).square().mean(-1)
    size = ((torch.log(ps) - torch.log(ts)) / .25).square().mean(-1)
    length = ((torch.log(pl) - torch.log(tl)) / .25).square().mean(-1)
    return (position + rotation + size + length).mean()


class CanonicalGeometryCorrection(nn.Module):
    """Apply a static, canonical nine-DOF correction to a fresh public readout.

    z[0:9] has fixed semantics: world position (3), right-composed rotation
    tangent (3), log half-size (2), and log held-length (1).  z[9:16] is
    reserved and forced to zero by the initializer/runtime.  This makes the
    identifiable subspace explicit instead of pretending all 16 coordinates
    can be recovered from a nine-DOF control geometry.
    """

    def __init__(self, *, position_scale: float = .02, rotation_scale: float = .25,
                 log_size_scale: float = .25, log_length_scale: float = .25):
        super().__init__()
        self.register_buffer("scales", torch.tensor(
            [position_scale] * 3 + [rotation_scale] * 3 + [log_size_scale] * 2 + [log_length_scale]))

    def forward(self, public_geometry: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != 16:
            raise ValueError("P59 latent container must be 16D")
        p, rotation, half_size, held_length = split_geometry(public_geometry)
        correction = z[..., :9] * self.scales
        p = p + correction[..., :3]
        rotation = rotation @ so3_exp_map(correction[..., 3:6])
        half_size = half_size * torch.exp(correction[..., 6:8])
        held_length = held_length * torch.exp(correction[..., 8:9])
        return pack_geometry(p, rotation, half_size, held_length)


class FrozenP50Action(nn.Module):
    """Read-only P50 MLP with a differentiable geometry-to-action input path."""

    def __init__(self, checkpoint: str | Path, *, require_canonical_hash: bool = True):
        super().__init__()
        checkpoint = Path(checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if require_canonical_hash and digest != P50_CHECKPOINT_SHA256:
            raise ValueError(f"P50 checkpoint hash mismatch: {digest}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("schema") != P50_CHECKPOINT_SCHEMA:
            raise ValueError("unknown P50 checkpoint schema")
        self.head = PrecisionActionMLP()
        self.head.load_state_dict(payload["state_dict"], strict=True)
        self.head.eval().requires_grad_(False)
        self.checkpoint_sha256 = digest

    @staticmethod
    def features(geometry: torch.Tensor, q: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        p, rotation, half_size, held_length = split_geometry(geometry)
        if q.shape[-1] != 9:
            raise ValueError("q must be actual EEF [position3, rotation6]")
        q_position = q[..., :3]
        q_rotation = rotation_6d_to_matrix(q[..., 3:])
        relative_position = (rotation.transpose(-1, -2) @ (q_position - p)[..., None])[..., 0] / .2
        relative_rotation = so3_log_map(rotation.transpose(-1, -2) @ q_rotation)
        phase = phase.to(device=q.device, dtype=torch.long)
        one_hot = F.one_hot(phase, num_classes=4).to(dtype=q.dtype)
        return torch.cat((relative_position, relative_rotation, half_size / .04,
                          held_length / .1, one_hot), -1)

    def forward(self, geometry: torch.Tensor, q: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(geometry, q, phase))

    @staticmethod
    def decode_world_pose(normalized_action: torch.Tensor, geometry: torch.Tensor,
                          q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact differentiable counterpart of P50 ``decode_action``."""
        _, slot_rotation, _, _ = split_geometry(geometry)
        q_position = q[..., :3]
        q_rotation = rotation_6d_to_matrix(q[..., 3:])
        position = q_position + (slot_rotation @ (normalized_action[..., :3] * .2)[..., None])[..., 0]
        local_delta = so3_exp_map(normalized_action[..., 3:] * .5)
        rotation = slot_rotation @ local_delta @ slot_rotation.transpose(-1, -2) @ q_rotation
        return position, rotation


def world_pose_loss(predicted: tuple[torch.Tensor, torch.Tensor],
                    target: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    pp, pr = predicted
    tp, tr = target
    position = ((pp - tp) / .02).square().mean(-1)
    rotation = (so3_log_map(tr.transpose(-1, -2) @ pr) / .25).square().mean(-1)
    return (position + rotation).mean()


def _mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                         nn.SiLU(), nn.Linear(hidden, output_dim))


class P59AdaptationCore(nn.Module):
    """Public observation -> static z -> functional geometry -> frozen P50 action."""

    def __init__(self, contract: P59Contract, p50_checkpoint: str | Path, *, hidden: int = 128,
                 require_canonical_p50_hash: bool = True):
        super().__init__()
        self.contract = contract
        self.register_buffer("u_mean", torch.zeros(contract.u_dim))
        self.register_buffer("u_scale", torch.ones(contract.u_dim))
        self.register_buffer("normalization_fitted", torch.tensor(False))
        self.public_readout = _mlp(contract.u_dim, hidden, contract.geometry_dim)
        with torch.no_grad():
            self.public_readout[-1].bias[3:9].copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        self.initializer = _mlp(contract.u_dim, hidden, contract.active_latent_dim)
        self.correction = CanonicalGeometryCorrection()
        # The feedback head predicts the next six control relations only. Its
        # target comes from a frozen public readout plus actual next q.
        self.feedback_head = _mlp(9 + contract.action_dim, hidden, 6)
        self.p50 = FrozenP50Action(p50_checkpoint, require_canonical_hash=require_canonical_p50_hash)
        self._readout_frozen = False
        self._runtime_frozen = False

    def _positive_public_geometry(self, raw: torch.Tensor) -> torch.Tensor:
        # The rotation columns are normalized by split_geometry. Positive
        # scalar outputs prevent invalid P50 inputs during early training.
        return torch.cat((raw[..., :9], F.softplus(raw[..., 9:]) + 1e-4), -1)

    def public_geometry(self, u: torch.Tensor) -> torch.Tensor:
        return self._positive_public_geometry(self.public_readout(self._normalize_u(u)))

    def initial(self, u: torch.Tensor) -> torch.Tensor:
        active = self.initializer(self._normalize_u(u))
        reserved = torch.zeros(*active.shape[:-1], self.contract.latent_dim - self.contract.active_latent_dim,
                               dtype=active.dtype, device=active.device)
        return torch.cat((active, reserved), -1)

    def _normalize_u(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.u_mean) / self.u_scale

    def set_public_normalization(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if self._readout_frozen:
            raise ValueError("cannot change public normalization after readout freeze")
        if mean.shape != (self.contract.u_dim,) or scale.shape != (self.contract.u_dim,):
            raise ValueError("public normalization shape mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError("public normalization must be finite with positive scale")
        self.u_mean.copy_(mean)
        self.u_scale.copy_(scale)
        self.normalization_fitted.fill_(True)

    def geometry(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Recompute from this frame's public readout every time. No previous
        # decoded geometry is an input, so a correction cannot accumulate.
        return self.geometry_from_public(self.public_geometry(u), z)

    def geometry_from_public(self, public_geometry: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.correction(public_geometry, z)

    def action(self, u: torch.Tensor, q: torch.Tensor, phase: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.p50(self.geometry(u, z), q, phase)

    def current_only_action(self, u: torch.Tensor, q: torch.Tensor,
                            phase: torch.Tensor) -> torch.Tensor:
        """Per-frame public readout baseline with no persistent correction."""
        z = torch.zeros(*u.shape[:-1], self.contract.latent_dim, dtype=u.dtype, device=u.device)
        return self.action(u, q, phase, z)

    def freeze_runtime(self) -> "P59AdaptationCore":
        if not self._readout_frozen:
            raise ValueError("freeze the pretrained public readout before runtime")
        self.eval().requires_grad_(False)
        self._runtime_frozen = True
        return self

    def freeze_public_readout(self) -> "P59AdaptationCore":
        if not bool(self.normalization_fitted):
            raise ValueError("fit and freeze train-only public normalization first")
        self.public_readout.eval().requires_grad_(False)
        self._readout_frozen = True
        return self

    def feedback_loss(self, transition: PublicTransition, z: torch.Tensor,
                      z0: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
        transition.validate(self.contract)
        geometry = self.geometry(transition.u, z)
        current_relation = self.p50.features(geometry, transition.q, transition.phase)[..., :9]
        prediction = self.feedback_head(torch.cat((current_relation, transition.executed_action), -1))
        # No GT label and no learnable target path: freeze/detach both the
        # public measurement and actual q before constructing the target.
        with torch.no_grad():
            target_geometry = self.public_geometry(transition.next_u).detach()
            target = self.p50.features(target_geometry, transition.next_q.detach(),
                                       transition.next_phase).detach()[..., :6]
        per_example = (prediction - target).square().mean(-1)
        anchor = self.contract.anchor * (z[..., :9] - z0[..., :9]).square().mean(-1)
        value = per_example + anchor
        if reduction == "none":
            return value
        if reduction != "mean":
            raise ValueError("reduction must be mean or none")
        return value.mean()

    def _prepare_feedback(self, transition: PublicTransition) -> tuple:
        """Cache frozen readout outputs without caching any z-dependent value."""
        transition.validate(self.contract)
        with torch.no_grad():
            current_public = self.public_geometry(transition.u).detach()
            next_public = self.public_geometry(transition.next_u).detach()
            target = self.p50.features(next_public, transition.next_q.detach(),
                                       transition.next_phase).detach()[..., :6]
        return current_public, transition.q, transition.phase, transition.executed_action, target

    def _prepared_feedback_loss(self, prepared: tuple, z: torch.Tensor,
                                z0: torch.Tensor) -> torch.Tensor:
        current_public, q, phase, executed_action, target = prepared
        geometry = self.geometry_from_public(current_public, z)
        current_relation = self.p50.features(geometry, q, phase)[..., :9]
        prediction = self.feedback_head(torch.cat((current_relation, executed_action), -1))
        per_example = (prediction - target).square().mean(-1)
        return per_example + self.contract.anchor * (z[..., :9] - z0[..., :9]).square().mean(-1)

    def _adapt_prepared_history(self, prepared_history: list[tuple], z: torch.Tensor,
                                z0: torch.Tensor, *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        window = prepared_history[-self.contract.history:]
        losses = []
        for _ in range(self.contract.inner_steps):
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            loss = torch.stack([self._prepared_feedback_loss(item, z, z0) for item in window]).mean(0).sum()
            gradient, = torch.autograd.grad(loss, z, create_graph=create_graph)
            gradient = torch.cat((gradient[..., :9], torch.zeros_like(gradient[..., 9:])), -1)
            z = z - self.contract.inner_lr * gradient
            z = torch.cat((z[..., :9], torch.zeros_like(z[..., 9:])), -1)
            losses.append(loss)
        return z, torch.stack(losses)

    def adapt_history(self, history: list[PublicTransition], z: torch.Tensor, z0: torch.Tensor,
                      *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        validate_history(history, self.contract)
        window = history[-self.contract.history:]
        losses = []
        for _ in range(self.contract.inner_steps):
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            # Average time, sum independent batch elements: each latent gets
            # deployment's learning rate regardless of outer batch size.
            loss = torch.stack([self.feedback_loss(item, z, z0, reduction="none") for item in window]).mean(0).sum()
            gradient, = torch.autograd.grad(loss, z, create_graph=create_graph)
            # Reserved coordinates are structurally inactive and exactly zero.
            gradient = torch.cat((gradient[..., :9], torch.zeros_like(gradient[..., 9:])), -1)
            z = z - self.contract.inner_lr * gradient
            z = torch.cat((z[..., :9], torch.zeros_like(z[..., 9:])), -1)
            losses.append(loss)
        return z, torch.stack(losses)

    def adapt_sequence(self, support: list[PublicTransition], z0: torch.Tensor,
                       *, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay deployment timing: three updates after every arrived transition."""
        validate_history(support, self.contract)
        z = z0
        # public_readout is frozen before meta training, so these cached
        # tensors preserve the z->correction->relation gradient route.
        prepared_support = [self._prepare_feedback(item) for item in support]
        history: list[tuple] = []
        losses = []
        for prepared in prepared_support:
            history.append(prepared)
            history = history[-self.contract.history:]
            z, transition_losses = self._adapt_prepared_history(history, z, z0, create_graph=create_graph)
            losses.append(transition_losses)
        return z, torch.stack(losses)

    def outer_loss(self, support: list[PublicTransition], query_u: torch.Tensor,
                   query_q: torch.Tensor, query_phase: torch.Tensor,
                   oracle_action: torch.Tensor, oracle_geometry: torch.Tensor,
                   *, query_step_index: int, query_episode_id: str,
                   fixed_weight: float = .5, geometry_weight: float = .25) -> tuple[torch.Tensor, dict]:
        """P53/P56-style unrolled BC on a future query, specialized to P59.

        Oracle action/geometry are offline-only arguments. They are never used
        by ``feedback_loss`` or ``P59Runtime``.
        """
        validate_history(support, self.contract)
        if not self._readout_frozen:
            raise ValueError("public readout must be pretrained and frozen before meta outer training")
        if query_episode_id != support[0].episode_id:
            raise ValueError("query crosses episode boundary")
        if query_step_index < support[-1].next_step_index:
            raise ValueError("outer query must be a future frame after all support transitions")
        if query_u.shape[-1] != self.contract.u_dim or query_q.shape[-1] != 9:
            raise ValueError("query public input shape mismatch")
        z0 = self.initial(support[0].u)
        z, inner_losses = self.adapt_sequence(support, z0, create_graph=True)
        with torch.no_grad():
            query_public_geometry = self.public_geometry(query_u).detach()
        online_geometry = self.geometry_from_public(query_public_geometry, z)
        fixed_geometry = self.geometry_from_public(query_public_geometry, z0)
        online_action = self.p50(online_geometry, query_q, query_phase)
        fixed_action = self.p50(fixed_geometry, query_q, query_phase)
        zero = torch.zeros_like(z)
        current_only_action = self.p50(self.geometry_from_public(query_public_geometry, zero), query_q, query_phase)
        # P50 outputs are expressed in each predicted slot frame. Decode each
        # through its own geometry before comparing commands in world space.
        oracle_pose = self.p50.decode_world_pose(oracle_action, oracle_geometry, query_q)
        online_bc = world_pose_loss(self.p50.decode_world_pose(online_action, online_geometry, query_q), oracle_pose)
        fixed_bc = world_pose_loss(self.p50.decode_world_pose(fixed_action, fixed_geometry, query_q), oracle_pose)
        geometry_supervision = geometry_loss(online_geometry, oracle_geometry)
        control = (1 - fixed_weight) * online_bc + fixed_weight * fixed_bc
        loss = control + geometry_weight * geometry_supervision
        return loss, dict(online_bc=online_bc, fixed_bc=fixed_bc,
                          geometry=geometry_supervision, inner=inner_losses,
                          z0=z0, adapted_z=z, online_action=online_action,
                          fixed_action=fixed_action, current_only_action=current_only_action)


class P59Runtime:
    """A stage-local runtime that updates z only after a real transition arrives."""

    def __init__(self, model: P59AdaptationCore, *, online: bool):
        if not model._runtime_frozen:
            raise ValueError("call freeze_runtime() before constructing runtime")
        self.model = model
        self.online = online
        self.z0: torch.Tensor | None = None
        self.z: torch.Tensor | None = None
        self.history: list[PublicTransition] = []
        self.updates = 0

    def enter(self, first_u: torch.Tensor) -> None:
        with torch.no_grad():
            self.z0 = self.model.initial(first_u).detach()
        self.z = self.z0.clone()
        self.history = []
        self.updates = 0

    def command(self, u: torch.Tensor, q: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if self.z is None:
            raise RuntimeError("stage not entered")
        with torch.no_grad():
            return self.model.action(u, q, phase, self.z)

    def observe_transition(self, transition: PublicTransition) -> torch.Tensor | None:
        if self.z is None or self.z0 is None:
            raise RuntimeError("stage not entered")
        transition.validate(self.model.contract)
        if self.history and transition.step_index != self.history[-1].next_step_index:
            raise ValueError("new runtime transition is not causally consecutive")
        if self.history and transition.episode_id != self.history[-1].episode_id:
            raise ValueError("runtime transition crosses episode boundary")
        self.history.append(transition)
        self.history = self.history[-self.model.contract.history:]
        if not self.online:
            return None
        with torch.enable_grad():
            self.z, losses = self.model.adapt_history(self.history, self.z, self.z0, create_graph=False)
        self.z = self.z.detach()
        self.updates += 1
        return losses.detach()

    def exit(self) -> None:
        self.z0 = self.z = None
        self.history = []
        self.updates = 0
