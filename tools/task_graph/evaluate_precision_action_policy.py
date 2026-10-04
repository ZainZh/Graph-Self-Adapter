#!/usr/bin/env python3
"""One post-seal offline action-approximation query; no task success claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from prove.task_graph.precision_action_policy import load_frozen_head, sample_examples
from tools.task_graph.train_precision_action_policy import dump_new, sha, verify_freeze


def error_summary(values):
    return dict(mean=float(np.mean(values)), rms=float(np.sqrt(np.mean(values ** 2))),
                p95=float(np.quantile(values, .95)), maximum=float(np.max(values)))


def evaluate(output):
    output = Path(output)
    freeze = verify_freeze(output)
    seal = json.loads((output / 'TRAINING_SEAL.json').read_text())
    if seal['checkpoint_sha256'] != sha(output / 'precision_action.pt') or seal['offline_freeze_sha256'] != sha(output / 'OFFLINE_FREEZE.json'):
        raise ValueError('Training seal/checkpoint binding changed')
    cfg = freeze['contract']['offline_query']
    dump_new(output / 'OFFLINE_QUERY_START.json', dict(training_seal_sha256=sha(output / 'TRAINING_SEAL.json'),
                                                    query_seed=cfg['seed'], query_cases=cfg['cases']))
    torch.set_num_threads(1)
    model = load_frozen_head(output / 'precision_action.pt')
    features, targets, _ = sample_examples(np.random.default_rng(cfg['seed']), cfg['cases'], freeze['contract'])
    with torch.no_grad():
        prediction = model(torch.from_numpy(features)).cpu().numpy()
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite offline predicted actions')
    position_error = np.linalg.norm((prediction[:, :3] - targets[:, :3]) * .2, axis=1)
    rotations = Rotation.from_rotvec(prediction[:, 3:] * .5) * Rotation.from_rotvec(targets[:, 3:] * .5).inv()
    rotation_error = np.linalg.norm(rotations.as_rotvec(), axis=1)
    # Seal numeric traces before their derived qualification verdict.
    with (output / 'OFFLINE_QUERY_TRACES.npz').open('xb') as stream:
        np.savez_compressed(stream, features=features, targets=targets, predictions=prediction,
                            position_error_m=position_error, rotation_geodesic_rad=rotation_error)
    report = dict(schema='prove.p50-offline-action-query.v1',
                  scope='synthetic ID programmatic-teacher action approximation only',
                  training_seal_sha256=sha(output / 'TRAINING_SEAL.json'),
                  checkpoint_sha256=seal['checkpoint_sha256'],
                  traces_sha256=sha(output / 'OFFLINE_QUERY_TRACES.npz'),
                  query_seed=cfg['seed'], query_cases=cfg['cases'],
                  query_input_sha256=hashlib.sha256(features.tobytes() + targets.tobytes()).hexdigest(),
                  normalized_action_mse=float(np.mean((prediction - targets) ** 2)),
                  position_error_m=error_summary(position_error),
                  rotation_geodesic_rad=error_summary(rotation_error),
                  gates=dict(max_position_below_2mm=bool(position_error.max() < cfg['gate_max_position_error_m']),
                             max_rotation_below_002rad=bool(rotation_error.max() < cfg['gate_max_rotation_geodesic_rad'])),
                  robot_rollouts=0, physical_pilot_authorized=False,
                  exclusions=['task success', 'geometry OOD success', 'LIBERO transfer', 'online adaptation benefit'])
    report['offline_action_approximation_qualified'] = all(report['gates'].values())
    dump_new(output / 'OFFLINE_QUERY.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output), indent=2, allow_nan=False))
