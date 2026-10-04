"""P50 external frozen action head, distilled from a programmatic P45 teacher.

The old Fast Adapter and its weights are unmodified. Runtime geometry is an
estimate supplied by the frozen geometry head with fixed or online z. Every
precision pose after the shared observation prefix is produced by this MLP;
analytic phase targets are used only for progress judging. There is no oracle
input, analytic action fallback, or raw untrained z concatenation.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

PHASES = ('align', 'approach', 'insert', 'hold')
CLEARANCES = np.array([.04, .004, -.04, -.04], dtype=np.float64)
INPUT_DIM = 13
OUTPUT_DIM = 6
TRANSLATION_SCALE = .2
ROTATION_SCALE = .5


def training_contract():
    return dict(
        schema='prove.p50-offline-action-distillation.v1',
        task='P45 Panda preheld insert; not LIBERO transfer',
        training_source='synthetic nominal poses with unchanged P45 programmatic controller labels',
        architecture=dict(input_dim=13, hidden=[128, 128], activation='SiLU', output_dim=6, output_activation='linear'),
        input_order=['slot_frame_eef_translation/0.2:3', 'rotvec(slot_R.T@eef_R):3',
                     'estimated_half_sizes/0.04:2', 'visually_calibrated_held_length/0.1:1', 'phase_onehot:4'],
        output_order=['slot_frame_delta_position/0.2:3', 'slot_frame_left_delta_rotvec/0.5:3'],
        domain=dict(translation_lower=[-.12, -.12, -.10], translation_upper=[.12, .12, .22],
                    rotation_axis='normalized isotropic Gaussian', rotation_angle=[0., .6],
                    half_size_lower=[.013, .010], half_size_upper=[.026, .023],
                    held_length=[.035, .09], phases=list(PHASES), phase_sampling='balanced within each batch then shuffled'),
        teacher=dict(module='prove.fast_adapter.robot_geometry_env_p45', class_name='GeometryTaskController',
                     phase_forced=True, t=12, clearances=CLEARANCES.tolist(), gripper_closed=True,
                     rule='p_slot + R_slot[:,2]*(length+clearance), R_slot'),
        training=dict(seed=11, updates=3000, batch_size=256, optimizer='Adam', learning_rate=.001,
                      torch_threads=1, device='cpu', loss='mean squared normalized six-dimensional action',
                      checkpoint_selection='final update only; no grid or query monitoring'),
        teacher_precheck=dict(seed=50150, cases=32, absolute_tolerance=1e-10),
        offline_query=dict(seed=50151, cases=4096, evaluation='once after training seal',
                           gate_max_position_error_m=.002, gate_max_rotation_geodesic_rad=.02),
        deployment=dict(adapter='unchanged P47Latent with P45 insert geometry seed11',
                        adapter_sha256='4c195b345a6a147bdeaa4bb5ec8614600f599196c6919a451616e99d964e3d25',
                        learned_head_frozen=True, adaptation='z only through decoded estimated geometry',
                        physical_rollout_authorized=False),
        claims_excluded=['robot task success', 'geometry OOD success', 'LIBERO transfer', 'human demonstrations', 'reinforcement learning'])


def sample_examples(rng, count, contract=None):
    """Vectorized synthetic ID examples; no environment or adapter is loaded."""
    cfg = (contract or training_contract())['domain']
    p = rng.uniform(cfg['translation_lower'], cfg['translation_upper'], size=(count, 3))
    axis = rng.normal(size=(count, 3))
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-15)
    r = axis * rng.uniform(*cfg['rotation_angle'], size=(count, 1))
    size = rng.uniform(cfg['half_size_lower'], cfg['half_size_upper'], size=(count, 2))
    length = rng.uniform(*cfg['held_length'], size=count)
    phase = np.arange(count) % len(PHASES)
    rng.shuffle(phase)
    features = np.c_[p / TRANSLATION_SCALE, r, size / .04, length / .1, np.eye(4)[phase]]
    delta = -p.copy()
    delta[:, 2] += length + CLEARANCES[phase]
    target = np.c_[delta / TRANSLATION_SCALE, -r / ROTATION_SCALE]
    return features.astype(np.float32), target.astype(np.float32), dict(
        relative_position=p, relative_rotvec=r, size=size, length=length, phase=phase)


def _array(value, shape, label):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'{label} must be finite with shape {shape}')
    return value


def _rotation(value, label):
    value = _array(value, (3, 3), label)
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(value), 1., atol=1e-5):
        raise ValueError(f'{label} must be a proper rotation')
    return value


def encode_state(estimate, proprio, held_length, phase):
    """Only estimated slot frame/size and measured own pose enter the head."""
    if phase not in PHASES:
        raise ValueError('Unknown precision phase')
    p, R, size = estimate
    q, Q = proprio
    p, q = _array(p, (3,), 'estimated position'), _array(q, (3,), 'EEF position')
    R, Q = _rotation(R, 'estimated rotation'), _rotation(Q, 'EEF rotation')
    size = _array(size, (2,), 'estimated half size')
    if np.any(size <= 0) or not np.isfinite(held_length) or held_length <= 0:
        raise ValueError('Positive estimated size and calibrated held length required')
    relative_position = R.T @ (q - p)
    relative_rotation = Rotation.from_matrix(R.T @ Q).as_rotvec()
    return np.r_[relative_position / TRANSLATION_SCALE, relative_rotation, size / .04,
                 held_length / .1, np.eye(4)[PHASES.index(phase)]].astype(np.float32)


def decode_action(normalized_action, estimate_rotation, proprio):
    """Left-compose the predicted local SE(3) increment with current EEF pose."""
    action = _array(normalized_action, (6,), 'normalized neural action')
    R = _rotation(estimate_rotation, 'estimated rotation')
    p, Q = proprio
    p, Q = _array(p, (3,), 'EEF position'), _rotation(Q, 'EEF rotation')
    delta_R = Rotation.from_rotvec(action[3:] * ROTATION_SCALE).as_matrix()
    return p + R @ (action[:3] * TRANSLATION_SCALE), R @ delta_R @ R.T @ Q


class PrecisionActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(INPUT_DIM, 128), nn.SiLU(),
                                    nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, OUTPUT_DIM))

    def forward(self, features):
        return self.layers(features)


def load_frozen_head(path, device='cpu'):
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get('schema') != 'prove.p50-precision-action-checkpoint.v1':
        raise ValueError('Unknown precision action checkpoint')
    if payload['contract'] != training_contract():
        raise ValueError('Checkpoint input/action/training contract mismatch')
    model = PrecisionActionMLP().to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return model


class PrecisionActionController:
    """P45 command API, same shared prefix, neural precision, code phase judge.

    Missing geometry holds the measured EEF pose without constructing a target.
    Malformed geometry/nonfinite model output raises; it never falls back to the
    teacher action. The robot backend retains its existing safety checks.
    """
    def __init__(self, model, start_pos, start_R, *, length, device='cpu'):
        self.model = model.to(device).eval().requires_grad_(False)
        self.device = device
        self.start_pos = _array(start_pos, (3,), 'start position').copy()
        self.start_R = _rotation(start_R, 'start rotation').copy()
        if not np.isfinite(length) or length <= 0:
            raise ValueError('Positive visually calibrated held length required')
        self.length = float(length)
        self.phase = 'observe'
        self.stable_steps = 0
        self.phase_steps = 0
        self.adaptation_frozen = False
        self.records = []
        self._last_t = -1

    def command(self, t, estimate, proprio):
        if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t <= self._last_t:
            raise ValueError('Command indices must increase monotonically')
        self._last_t = int(t)
        q, Q = _array(proprio[0], (3,), 'EEF position'), _rotation(proprio[1], 'EEF rotation')
        if t < 12:
            waypoints = np.array([[.018, 0, 0], [0, .018, 0], [-.018, 0, 0], [0, -.018, .01]])
            return self.start_pos + self.start_R @ waypoints[t // 3], self.start_R.copy(), True, 'shared observation motion'
        if estimate is None:
            self.stable_steps = 0
            return q.copy(), Q.copy(), True, 'valid estimated geometry unavailable: hold measured pose'
        if self.phase == 'observe':
            self.phase = 'align'
        phase = self.phase
        features = encode_state(estimate, (q, Q), self.length, phase)
        with torch.no_grad():
            action = self.model(torch.as_tensor(features, device=self.device)[None])[0].cpu().numpy()
        predicted_position, predicted_rotation = decode_action(action, estimate[1], (q, Q))
        # The analytic goal judges phase progress only. It is never the issued pose.
        p, R, _ = estimate
        judge_position = np.asarray(p) + np.asarray(R)[:, 2] * (self.length + CLEARANCES[PHASES.index(phase)])
        distance = float(np.linalg.norm(q - judge_position))
        angle = float(np.linalg.norm(Rotation.from_matrix(np.asarray(R) @ Q.T).as_rotvec()))
        self.stable_steps = self.stable_steps + 1 if distance < .003 and angle < .045 else 0
        if self.stable_steps >= 3 and phase != 'hold':
            self.phase = PHASES[PHASES.index(phase) + 1]
            self.stable_steps = 0
            self.phase_steps = 0
        self.phase_steps += 1
        self.records.append(dict(t=int(t), phase=phase, next_phase=self.phase,
                                 features=features.tolist(), normalized_action=action.tolist(),
                                 position=predicted_position.tolist(), rotation=predicted_rotation.tolist(),
                                 judge_distance=distance, judge_angle=angle,
                                 command_source='frozen_precision_action_mlp'))
        return predicted_position, predicted_rotation, True, f'neural {phase}'

    def state_dict(self):
        return deepcopy({key: value for key, value in self.__dict__.items() if key != 'model'})
