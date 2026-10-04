#!/usr/bin/env python3
"""One fixed P50 offline distillation run; freeze before train, no simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
import scipy
import torch

from prove.task_graph.precision_action_policy import (
    PHASES, PrecisionActionMLP, sample_examples, training_contract,
)

SOURCES = ['prove/task_graph/precision_action_policy.py',
           'tools/task_graph/train_precision_action_policy.py',
           'tools/task_graph/evaluate_precision_action_policy.py',
           'tests/test_precision_action_policy.py',
           'prove/fast_adapter/robot_geometry_env_p45.py',
           'prove/fast_adapter/geometry3d_env.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def packages():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__))


def teacher_precheck(contract):
    """Check synthetic label math against the existing unmodified controller."""
    from prove.fast_adapter.robot_geometry_env_p45 import GeometryTaskController
    cfg = contract['teacher_precheck']
    rng = np.random.default_rng(cfg['seed'])
    _, _, rows = sample_examples(rng, cfg['cases'], contract)
    position_errors, rotation_errors = [], []
    # Random global transforms verify the local-frame target/composition, not
    # merely a special identity-orientation algebraic case.
    for i in range(cfg['cases']):
        slot_position = rng.uniform(-.3, .3, 3)
        slot_rotation = Rotation.random(random_state=rng).as_matrix()
        relative_rotation = Rotation.from_rotvec(rows['relative_rotvec'][i]).as_matrix()
        eef_position = slot_position + slot_rotation @ rows['relative_position'][i]
        eef_rotation = slot_rotation @ relative_rotation
        phase = PHASES[rows['phase'][i]]
        teacher = GeometryTaskController('insert', eef_position, eef_rotation, length=rows['length'][i])
        teacher.phase = phase
        teacher_p, teacher_R, closed, _ = teacher.command(12, (slot_position, slot_rotation, rows['size'][i]),
                                                        (eef_position, eef_rotation))
        local_delta = -rows['relative_position'][i].copy()
        local_delta[2] += rows['length'][i] + contract['teacher']['clearances'][rows['phase'][i]]
        predicted_p = eef_position + slot_rotation @ local_delta
        predicted_R = slot_rotation @ Rotation.from_rotvec(-rows['relative_rotvec'][i]).as_matrix() @ slot_rotation.T @ eef_rotation
        position_errors.append(float(np.linalg.norm(predicted_p - teacher_p)))
        rotation_errors.append(float(np.linalg.norm(Rotation.from_matrix(predicted_R @ teacher_R.T).as_rotvec())))
        if not closed:
            raise ValueError('Insert teacher unexpectedly released preheld tool')
    result = dict(cases=cfg['cases'], seed=cfg['seed'], max_position_error_m=max(position_errors),
                  max_rotation_geodesic_rad=max(rotation_errors), scope='pure controller math, no robot rollout')
    result['passed'] = max(position_errors + rotation_errors) < cfg['absolute_tolerance']
    if not result['passed']:
        raise ValueError(f'Vectorized teacher mismatch: {result}')
    return result


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('OFFLINE_FREEZE.json', 'TRAINING_START.json', 'TRAINING_SEAL.json', 'OFFLINE_QUERY.json', 'precision_action.pt')):
        raise FileExistsError('Offline run already frozen or attempted; no overwrite/retry')
    contract = training_contract()
    precheck = teacher_precheck(contract)
    freeze = dict(schema='prove.p50-offline-freeze.v1', contract=contract,
                  source_sha256={name: sha(ROOT / name) for name in SOURCES},
                  packages=packages(), teacher_precheck=precheck)
    dump_new(output / 'OFFLINE_FREEZE.json', freeze)
    return freeze


def verify_freeze(output):
    freeze = json.loads((Path(output) / 'OFFLINE_FREEZE.json').read_text())
    if freeze['schema'] != 'prove.p50-offline-freeze.v1' or freeze['contract'] != training_contract():
        raise ValueError('Frozen experiment contract mismatch')
    if freeze['packages'] != packages():
        raise ValueError('Frozen runtime packages changed')
    for name, expected in freeze['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Frozen implementation changed: {name}')
    if not freeze['teacher_precheck']['passed']:
        raise ValueError('Teacher label precheck did not pass')
    return freeze


def train(output):
    output = Path(output)
    freeze = verify_freeze(output)
    cfg = freeze['contract']['training']
    # This exclusive receipt precedes RNG initialization and any weight update.
    dump_new(output / 'TRAINING_START.json', dict(schema='prove.p50-training-start.v1',
                                                offline_freeze_sha256=sha(output / 'OFFLINE_FREEZE.json'),
                                                maximum_updates=cfg['updates'], automatic_retries=0))
    started = time.perf_counter()
    torch.set_num_threads(cfg['torch_threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    model = PrecisionActionMLP().to(cfg['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    model.train()
    dataset_digest = hashlib.sha256()
    for update in range(cfg['updates']):
        features, targets, _ = sample_examples(rng, cfg['batch_size'], freeze['contract'])
        dataset_digest.update(features.tobytes())
        dataset_digest.update(targets.tobytes())
        x, y = torch.from_numpy(features), torch.from_numpy(targets)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.square(model(x) - y).mean()
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite training loss at update {update}')
        loss.backward()
        optimizer.step()
    model.eval().requires_grad_(False)
    payload = dict(schema='prove.p50-precision-action-checkpoint.v1',
                   state_dict=model.state_dict(), contract=freeze['contract'],
                   offline_freeze_sha256=sha(output / 'OFFLINE_FREEZE.json'))
    with (output / 'precision_action.pt').open('xb') as stream:
        torch.save(payload, stream)
    seal = dict(schema='prove.p50-offline-training-seal.v1',
                offline_freeze_sha256=sha(output / 'OFFLINE_FREEZE.json'),
                training_start_sha256=sha(output / 'TRAINING_START.json'),
                checkpoint_sha256=sha(output / 'precision_action.pt'),
                updates_completed=cfg['updates'], training_examples=cfg['updates'] * cfg['batch_size'],
                training_stream_sha256=dataset_digest.hexdigest(),
                final_minibatch_normalized_mse=float(loss.detach()),
                elapsed_seconds=time.perf_counter() - started,
                query_opened=False, robot_rollouts=0, adapter_weights_loaded=False)
    dump_new(output / 'TRAINING_SEAL.json', seal)
    return seal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'train'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps((prepare if args.mode == 'prepare' else train)(args.output), indent=2, allow_nan=False))
