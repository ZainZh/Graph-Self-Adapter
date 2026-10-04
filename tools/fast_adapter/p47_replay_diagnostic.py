"""Opened-P45 local optimizer comparison; no simulator, DINO, or new rollout.

The archived fixed-latent replay defines one common correspondence/history and
current offset. At 3/6/12/all available pre-lock views, fixed, GN3, and direct
start from the same z0. This isolates the local solve, not closed-loop success
or the trajectory of a persistent continuous arm. The full-history endpoint
still uses P45's first12+recent20 support cap. Optional evaluator corner labels
are opened only after these predictions are written and sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from prove.fast_adapter.neural_geometry_adapter import align_observation
from prove.fast_adapter.p45_neural import P45Latent, features
from prove.fast_adapter.p47_neural import (
    REPRESENTATIONS, SOLVER_CONTRACT, direct_quadratic_step, load_model,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    return dict(path=str(path), sha256=sha(path))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def compare_history(model, observations):
    """Return local comparisons on common frozen-z public observation histories."""
    eligible = [obs for obs in observations
                if obs['corners'] is not None and not obs.get('adaptation_frozen', False)]
    if len(eligible) < 3:
        raise ValueError('At least three visible pre-lock observations are needed')
    checkpoints = sorted({count for count in (3, 6, 12, len(eligible)) if count <= len(eligible)})
    fixed = P45Latent(model, 'fixed', 'cpu')
    results = []
    for count, obs in enumerate(eligible, 1):
        anchor = fixed.points.copy() if fixed.points is not None else None
        fixed.update(obs, int(obs['t']))
        if count not in checkpoints:
            continue
        aligned = align_observation(obs, anchor if anchor is not None else fixed.reference)
        reference = fixed.tensor(fixed.reference)[None]
        with torch.no_grad():
            _, offset = model.encode(fixed.tensor(fixed.first)[None],
                                     fixed.tensor(features(aligned, model.representation))[None], reference)
        support = fixed.history if count <= 32 else fixed.history[:12] + fixed.history[-20:]
        A = fixed.tensor(np.stack([entry[0] for entry in support]))[None]
        C = fixed.tensor(np.stack([entry[1] for entry in support]))[None]
        z = fixed.initial_z.clone()
        started = time.perf_counter()
        gn, inner = model.adapt(reference, offset, z.clone(), A, C, steps=3)
        gradient_seconds = time.perf_counter() - started
        direct, numerical = direct_quadratic_step(model, reference, offset, z.clone(), A, C)
        arms = {}
        for arm, value, seconds in (
                ('fixed', z, 0.), ('continuous', gn.detach(), gradient_seconds),
                ('direct', direct, numerical['solve_seconds'])):
            points = model.decode(reference, offset, value).detach()
            arms[arm] = dict(z=value.detach().numpy()[0].tolist(), points=points.numpy()[0].tolist(),
                             loss=float(model.ray_loss(points, A, C)[0]),
                             delta_z_norm=float((value-z).norm()), solve_seconds=seconds)
        arms['continuous']['inner'] = inner
        arms['direct']['solver'] = numerical
        results.append(dict(t=int(obs['t']), available_views=count, support_views=len(support),
                            full_history_endpoint=count == len(eligible), arms=arms))
    return results


def run(cell, out, checkpoint=None, labels=None):
    cell, out = Path(cell), Path(out)
    if out.exists():
        raise FileExistsError(f'Preserve existing diagnostic output: {out}')
    pre = json.loads((cell / 'PRE_RUN.json').read_text())
    seal = json.loads((cell / 'SEAL.json').read_text())
    if (pre.get('phase') != 'development' or not (cell / 'RESULT.json').is_file()
            or pre['cell']['rep'] not in REPRESENTATIONS):
        raise ValueError('Only completed, opened P45 development G/R archives are admitted')
    for name in ('OBS.npy', 'PRE_RUN.json'):
        if seal.get('files', {}).get(name) != sha(cell / name):
            raise ValueError(f'Changed or unsealed diagnostic input: {name}')
    expected = pre['candidate']['checkpoint']
    checkpoint = Path(checkpoint or expected['path'])
    if sha(checkpoint) != expected['sha256']:
        raise ValueError('Checkpoint does not match the archived P45 binding')
    model, package = load_model(checkpoint, 'cpu')
    if (model.task != pre['cell']['task'] or model.representation != pre['cell']['rep']
            or package.get('updates') != 2000 or package.get('seed') != pre['cell']['model_seed']):
        raise ValueError('Unexpected task/representation/model/training budget')
    observations = np.load(cell / 'OBS.npy', allow_pickle=True)
    records = compare_history(model, observations)
    predictions = dict(
        schema='prove.p47-opened-replay-diagnostic.v1',
        scope='opened development; local same-state optimizer comparison; no physical success claims',
        common_history='P45 fixed-z correspondences and support, current offset, incoming z0',
        continuous_meaning='one original three-step GN event from the same incoming z0',
        solver_contract=SOLVER_CONTRACT, cell=pre['cell'], checkpoint=binding(checkpoint),
        observations=binding(cell / 'OBS.npy'), source_seal=binding(cell / 'SEAL.json'),
        records=records,
    )
    out.mkdir(parents=True)
    write(out / 'PREDICTIONS.json', predictions)
    write(out / 'PREDICTION_SEAL.json', dict(
        schema='prove.p47-diagnostic-prediction-seal.v1',
        files={'PREDICTIONS.json': sha(out / 'PREDICTIONS.json')},
        evaluator_labels_opened=False,
    ))
    # Explicitly separate post-prediction evaluation. Labels are optional JSON
    # {"corners": {"<t>": [[x,y,z], ... four corners ...]}} from an opened
    # evaluator archive. They are never passed to the adapter or solve.
    evaluation = dict(labels=None, geometry_available=False, records=[])
    if labels is not None:
        truth = json.loads(Path(labels).read_text())['corners']
        for record in records:
            if str(record['t']) not in truth:
                continue
            corners = np.asarray(truth[str(record['t'])], dtype=float)
            if corners.shape != (4, 3) or not np.isfinite(corners).all():
                raise ValueError('Evaluator corners must be finite 4x3 arrays')
            errors = {}
            for arm, prediction in record['arms'].items():
                points = np.asarray(prediction['points'])
                errors[arm] = dict(
                    center_error_m=float(np.linalg.norm(points.mean(0) - corners.mean(0))),
                    corner_rmse_best_cyclic_m=float(min(np.sqrt(np.mean(np.square(
                        points - np.roll(corners, shift, axis=0)))) for shift in range(4))),
                )
            evaluation['records'].append(dict(t=record['t'], arms=errors))
        evaluation.update(labels=binding(labels), geometry_available=bool(evaluation['records']))
    write(out / 'EVALUATION.json', evaluation)
    return predictions, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', required=True, help='Completed opened P45 development cell directory')
    parser.add_argument('--checkpoint', help='Optional relocated checkpoint with identical sealed SHA256')
    parser.add_argument('--out', required=True, help='New output directory')
    parser.add_argument('--labels', help='Optional separate evaluator corner-label JSON, read after prediction seal')
    args = parser.parse_args()
    torch.set_num_threads(1)
    predictions, evaluation = run(args.cell, args.out, args.checkpoint, args.labels)
    print(json.dumps(dict(output=args.out, endpoints=len(predictions['records']),
                          geometry_available=evaluation['geometry_available'])))


if __name__ == '__main__':
    main()
