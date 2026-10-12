"""Small frozen NumPy skill and the contracted stage-local geometric latent.

The checkpoint is trained offline by ``tools/task_graph/twin_train_skills.py``.
Runtime code intentionally has no torch dependency and cannot silently replace a
missing/bad model with an analytic teacher.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _finite(value: Any, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


@dataclass(frozen=True)
class SkillStep:
    action: np.ndarray
    z: np.ndarray
    record: dict[str, Any]


class LocalPoseLatent:
    """Three-dimensional reference-pose correction local to one short phase."""

    def __init__(self, *, enabled: bool, learning_rate: float = 0.25,
                 damping: float = 0.15, huber_delta_m: float = 0.012,
                 limit_m: float = 0.02) -> None:
        self.enabled = bool(enabled)
        self.learning_rate = float(learning_rate)
        self.damping = float(damping)
        self.huber_delta_m = float(huber_delta_m)
        self.limit_m = float(limit_m)
        if min(self.learning_rate, self.damping, self.huber_delta_m, self.limit_m) < 0:
            raise ValueError("latent hyperparameters must be nonnegative")
        self.history: deque[np.ndarray] = deque(maxlen=3)
        self.z = np.zeros(3, dtype=np.float64)
        self.updates = 0

    def reset(self) -> None:
        self.history.clear()
        self.z[:] = 0.0
        self.updates = 0

    def update(self, current_surface_xyz: Any | None,
               reference_surface_xyz: Any | None) -> dict[str, Any]:
        before = self.z.copy()
        known = current_surface_xyz is not None and reference_surface_xyz is not None
        if known:
            residual = _finite(current_surface_xyz, name="current_surface_xyz").reshape(3) - _finite(
                reference_surface_xyz, name="reference_surface_xyz").reshape(3)
            self.history.append(residual)
        else:
            residual = None
        history = np.stack(tuple(self.history)) if self.history else np.empty((0, 3))
        before_error = before - history
        before_loss = self._huber_loss(before_error)
        # current-reference is the desired object-anchored translation correction;
        # training uses the same sign in its final three feature coordinates.
        if self.enabled and known:
            mean = np.mean(np.stack(tuple(self.history)), axis=0)
            grad = np.clip(mean - self.z, -self.huber_delta_m, self.huber_delta_m)
            self.z = self.z + (self.learning_rate / (1.0 + self.damping)) * grad
            self.z = np.clip(self.z, -self.limit_m, self.limit_m)
            self.updates += 1
        after_error = self.z - history
        return {
            "measurement_known": known,
            "surface_residual_m": None if residual is None else residual.tolist(),
            "history_length": len(self.history),
            "z_before_m": before.tolist(),
            "z_after_m": self.z.tolist(),
            "update_index": self.updates,
            "loss_before": before_loss,
            "loss_after": self._huber_loss(after_error),
            "damping": self.damping,
            "update_rule": "huber_gradient_step_divided_by_one_plus_damping",
        }

    def _huber_loss(self, error: np.ndarray) -> float | None:
        if error.size == 0:
            return None
        absolute = np.abs(error)
        loss = np.where(absolute <= self.huber_delta_m, 0.5 * np.square(error),
                        self.huber_delta_m * (absolute - 0.5 * self.huber_delta_m))
        return float(np.mean(loss))


class AttachmentPoseLatent:
    """R2 held-rigid attachment discrepancy in the active TCP frame."""

    def __init__(self, *, enabled: bool, learning_rate: float = .25,
                 damping: float = .15, huber_delta_m: float = .004,
                 limit_m: float = .006, window: int = 3) -> None:
        self.enabled = bool(enabled)
        self.learning_rate = float(learning_rate)
        self.damping = float(damping)
        self.huber_delta_m = float(huber_delta_m)
        self.limit_m = float(limit_m)
        self.history: deque[np.ndarray] = deque(maxlen=int(window))
        if (len(self.history) != 0 or window != 3 or self.learning_rate != .25 or
                self.damping != .15 or self.huber_delta_m != .004 or self.limit_m != .006):
            raise ValueError("R2 attachment feedback hyperparameters must match the frozen contract")
        self.z = np.zeros(3, dtype=np.float64)
        self.updates = 0

    def reset(self) -> None:
        self.history.clear()
        self.z[:] = 0.0
        self.updates = 0

    @staticmethod
    def discrepancy(*, current_object_xyz: Any, current_tcp_xyz: Any,
                    current_tcp_rotation: Any, reference_object_xyz: Any,
                    reference_tcp_xyz: Any, reference_tcp_rotation: Any) -> np.ndarray:
        p = _finite(current_object_xyz, name="current_object_xyz").reshape(3)
        tcp = _finite(current_tcp_xyz, name="current_tcp_xyz").reshape(3)
        rotation = _finite(current_tcp_rotation, name="current_tcp_rotation").reshape(3, 3)
        p_ref = _finite(reference_object_xyz, name="reference_object_xyz").reshape(3)
        tcp_ref = _finite(reference_tcp_xyz, name="reference_tcp_xyz").reshape(3)
        rotation_ref = _finite(reference_tcp_rotation, name="reference_tcp_rotation").reshape(3, 3)
        return rotation.T @ (p - tcp) - rotation_ref.T @ (p_ref - tcp_ref)

    def update(self, discrepancy_local_m: Any | None, *, admitted: bool,
               admission_reason: str, matched_reference_time_sec: float | None) -> dict[str, Any]:
        before = self.z.copy()
        measurement = (None if discrepancy_local_m is None else
                       _finite(discrepancy_local_m, name="attachment_discrepancy_local_m").reshape(3))
        if admitted:
            if measurement is None:
                raise ValueError("admitted R2 feedback requires a finite discrepancy")
            self.history.append(measurement)
        history = np.stack(tuple(self.history)) if self.history else np.empty((0, 3))
        loss_before = self._loss(before, history)
        gradient = None
        if self.enabled and admitted:
            # Exact gradient of mean_i sum_j Huber(z_j - b_ij).
            gradient = np.mean(np.clip(before - history, -self.huber_delta_m,
                                       self.huber_delta_m), axis=0)
            self.z = before - (self.learning_rate / (1.0 + self.damping)) * gradient
            self.z = np.clip(self.z, -self.limit_m, self.limit_m)
            self.updates += 1
        return {
            "feedback_type": "held_rigid_attachment_tcp",
            "admitted": bool(admitted),
            "admission_reason": admission_reason,
            "measurement_known": measurement is not None,
            "matched_reference_time_sec": matched_reference_time_sec,
            "attachment_discrepancy_local_m": None if measurement is None else measurement.tolist(),
            "history_length": len(self.history),
            "z_local_before_m": before.tolist(),
            "z_local_after_m": self.z.tolist(),
            "gradient_local": None if gradient is None else gradient.tolist(),
            "update_index": self.updates,
            "loss_before": loss_before,
            "loss_after": self._loss(self.z, history),
            "loss_definition": "mean_samples_sum_axes_huber",
            "gradient_definition": "mean_samples_clip_individual_z_minus_b",
            "huber_delta_m": self.huber_delta_m,
            "learning_rate": self.learning_rate,
            "damping": self.damping,
            "limit_m": self.limit_m,
        }

    def _loss(self, z: np.ndarray, history: np.ndarray) -> float | None:
        if history.size == 0:
            return None
        error = z.reshape(1, 3) - history
        absolute = np.abs(error)
        terms = np.where(absolute <= self.huber_delta_m, .5 * np.square(error),
                         self.huber_delta_m * (absolute - .5 * self.huber_delta_m))
        return float(np.mean(np.sum(terms, axis=1)))


class FrozenNumpyMLP:
    """Exact NumPy execution of a 64x128 ReLU MLP exported from torch."""

    schema = "prove.twin-local-skill.numpy-mlp.v1"

    def __init__(self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
        self.metadata = dict(metadata)
        if self.metadata.get("schema") != self.schema:
            raise ValueError("unsupported local skill checkpoint schema")
        self.input_mean = _finite(arrays["input_mean"], name="input_mean")
        self.input_scale = _finite(arrays["input_scale"], name="input_scale")
        self.output_mean = _finite(arrays["output_mean"], name="output_mean")
        self.output_scale = _finite(arrays["output_scale"], name="output_scale")
        self.weights = [_finite(arrays[f"w{i}"], name=f"w{i}") for i in range(3)]
        self.biases = [_finite(arrays[f"b{i}"], name=f"b{i}") for i in range(3)]
        dims = (self.input_mean.size, 64, 128, self.output_mean.size)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if w.shape != (dims[i + 1], dims[i]) or b.shape != (dims[i + 1],):
                raise ValueError(f"layer {i} shape mismatch: {w.shape}, {b.shape}")
        if self.input_scale.shape != self.input_mean.shape or self.output_scale.shape != self.output_mean.shape:
            raise ValueError("normalization shape mismatch")
        if np.any(self.input_scale <= 0) or np.any(self.output_scale <= 0):
            raise ValueError("normalization scales must be positive")
        self.action_limit = float(self.metadata.get("action_limit_rad", 0.04))
        self.z_slice = tuple(self.metadata.get("z_feature_slice", [self.input_mean.size - 3, self.input_mean.size]))
        self.correction_slices = tuple(tuple(x) for x in self.metadata.get(
            "correction_feature_slices", [[12, 18], list(self.z_slice)]))
        if self.z_slice[1] - self.z_slice[0] != 3:
            raise ValueError("checkpoint must expose exactly three z features")

    @classmethod
    def load(cls, path: str | Path) -> "FrozenNumpyMLP":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"local skill checkpoint missing: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"metadata_json", "input_mean", "input_scale", "output_mean", "output_scale",
                        "w0", "b0", "w1", "b1", "w2", "b2"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"local skill checkpoint missing arrays: {sorted(missing)}")
            metadata = json.loads(str(data["metadata_json"].item()))
            arrays = {key: data[key].copy() for key in required - {"metadata_json"}}
        return cls(arrays, metadata)

    @property
    def input_dim(self) -> int:
        return int(self.input_mean.size)

    @property
    def output_dim(self) -> int:
        return int(self.output_mean.size)

    def predict(self, features: Any, z: Any) -> np.ndarray:
        x = _finite(features, name="features").reshape(self.input_dim).copy()
        z = _finite(z, name="z").reshape(3)
        x[slice(*self.z_slice)] = z
        x = (x - self.input_mean) / self.input_scale
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.maximum(w @ x + b, 0.0)
        out = self.weights[-1] @ x + self.biases[-1]
        out = out * self.output_scale + self.output_mean
        return np.clip(out, -self.action_limit, self.action_limit)

    def predict_residual(self, features: Any, z: Any) -> np.ndarray:
        """Learned adapter residual, structurally zero at zero geometry/z."""
        features = _finite(features, name="features").reshape(self.input_dim)
        active = self.predict(features, z)
        baseline = features.copy()
        for start, end in self.correction_slices:
            baseline[slice(start, end)] = 0.0
        zero = self.predict(baseline, np.zeros(3))
        return np.clip(active - zero, -self.action_limit, self.action_limit)


class TwinLocalSkill:
    """Frozen policy plus either fixed-z or past/current-feedback online-z."""

    def __init__(self, checkpoint: str | Path, *, online_z: bool,
                 feedback_contract: Mapping[str, Any] | None = None) -> None:
        self.model = FrozenNumpyMLP.load(checkpoint)
        self.feedback_contract = None if feedback_contract is None else dict(feedback_contract)
        if self.feedback_contract is None:
            self.feedback_mode = "legacy_surface_translation"
            self.latent = LocalPoseLatent(enabled=online_z)
            self.eligible_phases: frozenset[str] = frozenset()
        else:
            contract = self.feedback_contract
            if (contract.get("schema") != "graph-self-adapter.twin-feedback-contract.v2" or
                    contract.get("feedback_mode") != "held_visible_attachment"):
                raise ValueError("unsupported R2 local feedback contract")
            exact = {
                "window": 3, "learning_rate": .25, "damping": .15,
                "huber_delta_m": .004, "limit_m": .006,
                "minimum_completed_local_actions": 1,
                "fresh_visible_required": True, "current_holding_required": True,
            }
            mismatched = {key: (contract.get(key), value) for key, value in exact.items()
                          if contract.get(key) != value}
            phases = frozenset(contract.get("eligible_phases", ()))
            if mismatched or phases != frozenset(("Align", "Insert", "Push")):
                raise ValueError(f"R2 local feedback contract mismatch: {mismatched}, phases={sorted(phases)}")
            self.feedback_mode = "held_visible_attachment"
            self.eligible_phases = phases
            self.latent = AttachmentPoseLatent(
                enabled=online_z, learning_rate=contract["learning_rate"],
                damping=contract["damping"], huber_delta_m=contract["huber_delta_m"],
                limit_m=contract["limit_m"], window=contract["window"])
        self.online_z = bool(online_z)
        self.command_index = 0
        self.phase_archive: list[dict[str, Any]] = []

    def reset_phase(self, completed_stage_id: str | None = None) -> None:
        if self.command_index:
            self.phase_archive.append({
                "stage_id": completed_stage_id,
                "feedback_mode": self.feedback_mode,
                "commands": self.command_index,
                "updates": self.latent.updates,
                "z_m": self.latent.z.tolist(),
                "history_length": len(self.latent.history),
            })
        self.latent.reset()
        self.command_index = 0

    def command(self, features: Any, *, current_surface_xyz: Any | None,
                reference_surface_xyz: Any | None, phase: str | None = None,
                holding: bool | None = None,
                matched_reference_time_sec: float | None = None,
                current_tcp_xyz: Any | None = None,
                current_tcp_rotation: Any | None = None,
                reference_tcp_xyz: Any | None = None,
                reference_tcp_rotation: Any | None = None) -> SkillStep:
        if self.feedback_mode == "legacy_surface_translation":
            update = self.latent.update(current_surface_xyz, reference_surface_xyz)
            z_action = self.latent.z.copy()
            z_local = self.latent.z.copy()
        else:
            if phase not in self.eligible_phases:
                reason = "phase_not_eligible"
            elif self.command_index < 1:
                reason = "first_local_action_common_prefix"
            elif holding is not True:
                reason = "current_holding_not_true"
            elif current_surface_xyz is None:
                reason = "fresh_visible_object_missing"
            elif any(value is None for value in (reference_surface_xyz, current_tcp_xyz,
                                                  current_tcp_rotation, reference_tcp_xyz,
                                                  reference_tcp_rotation,
                                                  matched_reference_time_sec)):
                reason = "matched_reference_pose_missing"
            else:
                reason = "admitted"
            admitted = reason == "admitted"
            discrepancy = None if not admitted else AttachmentPoseLatent.discrepancy(
                current_object_xyz=current_surface_xyz, current_tcp_xyz=current_tcp_xyz,
                current_tcp_rotation=current_tcp_rotation,
                reference_object_xyz=reference_surface_xyz,
                reference_tcp_xyz=reference_tcp_xyz,
                reference_tcp_rotation=reference_tcp_rotation)
            update = self.latent.update(
                discrepancy, admitted=admitted, admission_reason=reason,
                matched_reference_time_sec=matched_reference_time_sec)
            rotation = (np.eye(3) if current_tcp_rotation is None else
                        _finite(current_tcp_rotation, name="current_tcp_rotation").reshape(3, 3))
            z_local = self.latent.z.copy()
            z_action = -rotation @ z_local
            z_before_world = -rotation @ np.asarray(update["z_local_before_m"], dtype=np.float64)
            # Compatibility fields are explicitly in the model/action world
            # coordinates for renderer and generic trace consumers.
            update["z_before_m"] = z_before_world.tolist()
            update["z_after_m"] = z_action.tolist()
            update["z_world_m"] = z_action.tolist()
        action = self.model.predict_residual(features, z_action)
        fixed_action = self.model.predict_residual(features, np.zeros(3))
        if self.feedback_mode == "held_visible_attachment":
            rotation = (np.eye(3) if current_tcp_rotation is None else
                        _finite(current_tcp_rotation, name="current_tcp_rotation").reshape(3, 3))
            probe_z = -rotation @ (z_local + np.array([1e-3, 0.0, 0.0]))
        else:
            probe_z = z_action + np.array([1e-3, 0.0, 0.0])
        changed_probe = self.model.predict_residual(features, probe_z)
        record = {
            "controller": "frozen_numpy_mlp",
            "policy_form": "nominal_reference_plus_bias_cancelled_learned_residual",
            "online_z": self.online_z,
            "command_index": self.command_index,
            "normalization": {
                "input_l2": float(np.linalg.norm((np.asarray(features) - self.model.input_mean) / self.model.input_scale)),
                "output_l2_rad": float(np.linalg.norm(action)),
            },
            "latent": update,
            "feedback_mode": self.feedback_mode,
            "action_online_minus_fixed_l2_rad": float(np.linalg.norm(action - fixed_action)),
            "z_action_sensitivity_l2_per_mm": float(np.linalg.norm(changed_probe - action)),
        }
        self.command_index += 1
        return SkillStep(action=action, z=z_action.copy(), record=record)
