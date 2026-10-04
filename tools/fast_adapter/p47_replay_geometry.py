"""Evaluate already sealed opened-development predictions; never take actions.

The optional geometry labels deliberately stay out of the P47 solver replay.
This separate process restores archived simulator states only after checking
the saved prediction seal. No physics step, controller, encoder or adapter is
called. Unordered corner RMSE allows the permutation ambiguity of mask corners.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from tools.fast_adapter.p45_common import make_env, privileged_training_label, sha
from tools.fast_adapter.p47_evaluate import restore


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    return dict(path=str(path), sha256=sha(path))


def geometry_errors(points, truth):
    points, truth = np.asarray(points, dtype=float), np.asarray(truth, dtype=float)
    if points.shape != (4, 3) or truth.shape != (4, 3) or not np.isfinite([points, truth]).all():
        raise ValueError('Require finite predicted and evaluator-only four-corner geometry')
    return dict(center_error_m=float(np.linalg.norm(points.mean(0) - truth.mean(0))),
                unordered_corner_rmse_m=min(float(np.sqrt(np.mean(np.sum(
                    (points - truth[list(order)]) ** 2, axis=1))))
                    for order in itertools.permutations(range(4))))


def evaluate(prediction_path):
    prediction_path = Path(prediction_path)
    predictions = read(prediction_path)
    prediction_seal = prediction_path.parent / 'PREDICTION_SEAL.json'
    seal = read(prediction_seal)
    if (predictions.get('schema') != 'prove.p47-opened-replay-diagnostic.v1'
            or seal.get('evaluator_labels_opened') is not False
            or seal.get('files', {}).get(prediction_path.name) != sha(prediction_path)):
        raise ValueError('Geometry evaluation requires sealed, label-free development predictions')
    source = Path(predictions['observations']['path']).parent
    for key in ('observations', 'source_seal'):
        item = predictions[key]
        if sha(item['path']) != item['sha256']:
            raise ValueError('Changed source binding: ' + key)
    source_seal = read(source / 'SEAL.json')
    for name in ('PRE_RUN.json', 'OBS.npy', 'TRACE.npy'):
        if sha(source / name) != source_seal['files'].get(name):
            raise ValueError('Changed source archive: ' + name)
    pre = read(source / 'PRE_RUN.json')
    if pre.get('phase') != 'development' or pre['cell'] != predictions['cell']:
        raise ValueError('Only the original opened development cell is allowed')
    traces = {int(row['t']): row for row in np.load(source / 'TRACE.npy', allow_pickle=True)}
    cell = pre['cell']
    env = make_env(cell['task'], cell['condition_seed'], cell['split'],
                   render=False, backend=pre['environment'])
    records = []
    try:
        for record in predictions['records']:
            row = traces[record['t']]
            if row['controller_before'].get('adaptation_frozen', False):
                raise ValueError('Geometry replay scope must stay before target motion')
            restore(env, row['state'])
            truth = privileged_training_label(env)['corners']
            records.append(dict(t=record['t'], available_views=record['available_views'],
                                support_views=record['support_views'],
                                full_history_endpoint=record['full_history_endpoint'],
                                evaluator_corners=truth.tolist(),
                                arms={arm: geometry_errors(value['points'], truth)
                                      for arm, value in record['arms'].items()}))
    finally:
        env.close()
    return dict(cell=cell, prediction=binding(prediction_path),
                prediction_seal=binding(prediction_seal),
                source_pre_run=binding(source / 'PRE_RUN.json'),
                source_trace=binding(source / 'TRACE.npy'), records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', action='append', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Preserve existing geometry evaluation: ' + str(out))
    result = dict(schema='prove.p47-opened-replay-geometry.v1',
                  scope='Opened development only; sealed predictions evaluated afterward; not independent success evidence',
                  physics_steps=0, runtime_models_updated=False,
                  geometry_definition='Four visible-surface corners from the original training-label convention; permutation-invariant corner error',
                  implementation=binding(__file__),
                  label_implementation=binding('tools/fast_adapter/p45_common.py'),
                  comparisons=[evaluate(path) for path in args.prediction])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(out), comparisons=len(result['comparisons']),
                          points=sum(len(row['records']) for row in result['comparisons']))))


if __name__ == '__main__':
    main()
