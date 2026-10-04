"""P45 same-data, same-budget representation training; no simulator execution.

Supervised outer labels are separate from the public runtime archive. The inner
loop consumes only calibrated visible corner rays. Checkpoints are always the
final prescribed update, never selected by development performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from prove.fast_adapter.geometry3d_adapter import initial
from prove.fast_adapter.neural_geometry_adapter import GEOMETRY_SCALE, align_observation, ray_matrices
from prove.fast_adapter.neural_geometry_visual import PUBLIC_ARRAY_KEYS
from prove.fast_adapter.p45_neural import P45NeuralGeometry, REPRESENTATIONS, TASK_GEOMETRY, features
from prove.fast_adapter.p45_visual import VISUAL_REPRESENTATIONS, validate_projection
from tools.fast_adapter.p43_train import batch_items, meta_prediction

SOURCES = [
    'prove/fast_adapter/geometry3d_adapter.py',
    'prove/fast_adapter/neural_geometry_adapter.py',
    'prove/fast_adapter/neural_geometry_visual.py',
    'prove/fast_adapter/p45_neural.py',
    'prove/fast_adapter/p45_visual.py',
    'tools/fast_adapter/p43_train.py',
    'tools/fast_adapter/p45_train.py',
]
PUBLIC_KEYS = set(PUBLIC_ARRAY_KEYS) | {'local_relative'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_manifest(observation_path, task, split):
    path = Path(observation_path).parent / 'MANIFEST.json'
    manifest = json.loads(path.read_text())
    if manifest.get('task') != task or manifest.get('split') != split:
        raise ValueError('Observation manifest task/split mismatch')
    if manifest.get('observations_sha256') != sha(observation_path):
        raise ValueError('Public observations changed after manifest sealing')
    projections = {}
    for kind in VISUAL_REPRESENTATIONS:
        binding = manifest['pca'][kind]
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Training-only projection changed after manifest sealing')
        package = torch.load(binding['path'], map_location='cpu', weights_only=True)
        validate_projection(package, kind, verify_training_file=True)
        projections[kind] = package
    return manifest, projections


def load_data(observation_path, label_path, task, representation, split):
    manifest, projections = read_manifest(observation_path, task, split)
    with np.load(observation_path, allow_pickle=False) as archive:
        if set(archive.files) != PUBLIC_KEYS:
            raise ValueError(f'Unexpected/missing runtime keys: {set(archive.files) ^ PUBLIC_KEYS}')
        data = {key: archive[key] for key in archive.files}
    if manifest['labels']['sha256'] != sha(label_path):
        raise ValueError('Separate supervised labels differ from the sealed data manifest')
    with np.load(label_path, allow_pickle=False) as archive:
        truth = archive['corners'].copy()
    seeds = data['seeds'].tolist()
    if seeds != manifest['seeds'] or len(seeds) != len(set(seeds)):
        raise ValueError('Observation seeds differ from sealed unique seed list')
    shape = data['corners'].shape
    if len(shape) != 4 or shape[-2:] != (4, 2) or truth.shape != shape[:2] + (4, 3):
        raise ValueError('Public corners and separate supervised labels must preserve episode/frame axes')
    if data['corners_valid'].shape != shape[:2] or data['t'].shape != shape[:2]:
        raise ValueError('Malformed public frame validity/time axes')
    if len(seeds) != shape[0]:
        raise ValueError('Seed and episode counts differ')
    episodes = []
    local_key = 'local_relative' if representation == 'relative_fusion' else 'local'
    for episode in range(shape[0]):
        rows, reference = [], None
        if np.any(np.diff(data['t'][episode]) <= 0):
            raise ValueError('Training frames must retain a strictly causal time order')
        for frame in range(shape[1]):
            if not bool(data['corners_valid'][episode, frame]):
                continue
            obs = {key: data[key][episode, frame] for key in (
                'corners', 'camera_position', 'camera_rotation', 'K', 'position', 'rotation',
                'role_bbox', 'role_visible')}
            obs['local'] = data[local_key][episode, frame]
            if reference is None:
                try:
                    reference = initial(obs, TASK_GEOMETRY[task])
                except RuntimeError:
                    continue
            aligned = align_observation(obs, reference)
            target = truth[episode, frame]
            if not np.isfinite(target).all():
                raise ValueError('Nonfinite training-only geometric label')
            rows.append(dict(feature=features(aligned, representation), ray=ray_matrices(aligned),
                             truth=target, t=int(data['t'][episode, frame])))
        if len(rows) >= 4:
            episodes.append(dict(seed=int(seeds[episode]), rows=rows, reference=reference.astype(np.float32)))
    return episodes, manifest, projections


def validate_partitions(train_manifest, dev_manifest, train_pca, dev_pca):
    train_seeds, dev_seeds = set(train_manifest['seeds']), set(dev_manifest['seeds'])
    if train_seeds & dev_seeds:
        raise ValueError('Training/development condition leakage')
    for kind in VISUAL_REPRESENTATIONS:
        if train_manifest['pca'][kind] != dev_manifest['pca'][kind]:
            raise ValueError('Training and development must use the same frozen projection')
        for package in (train_pca[kind], dev_pca[kind]):
            if package['training_binding']['seeds'] != train_manifest['seeds']:
                raise ValueError('Projection fit seeds do not exactly match the training split')


def evaluation(model, episodes, device):
    model.eval()
    rng = np.random.default_rng(91212)
    values = []
    for _ in range(8):
        batch = batch_items(episodes, rng, min(16, len(episodes)), 12, device)
        fixed, adapted, z0, z, inner = meta_prediction(model, batch, False)
        target = batch[-1]
        values.append(dict(
            fixed_corner_rmse=float((fixed - target).square().mean().sqrt().detach()),
            adapted_corner_rmse=float((adapted - target).square().mean().sqrt().detach()),
            latent_delta=float((z - z0).norm(dim=-1).mean().detach()),
            gradient_mean=float(np.mean([row['gradient'] for row in inner])),
        ))
    return {key: float(np.mean([row[key] for row in values])) for key in values[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=list(TASK_GEOMETRY))
    parser.add_argument('--representation', required=True, choices=REPRESENTATIONS)
    parser.add_argument('--train', required=True)
    parser.add_argument('--train-labels', required=True)
    parser.add_argument('--dev', required=True)
    parser.add_argument('--dev-labels', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, choices=[11, 22], default=11)
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.updates != 2000 or args.batch_size != 16:
        raise ValueError('The adopted P45 budget is 2000 updates and batch16 for every representation')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    inputs = {key: dict(path=getattr(args, key), sha256=sha(getattr(args, key)))
              for key in ('train', 'train_labels', 'dev', 'dev_labels')}
    for split in ('train', 'dev'):
        path = Path(getattr(args, split)).parent / 'MANIFEST.json'
        inputs[split + '_manifest'] = dict(path=str(path), sha256=sha(path))
    source_hashes = {source: sha(source) for source in SOURCES}
    write(out / 'FREEZE.json', dict(
        schema='prove.p45-neural-training.v1', args=vars(args), sources=source_hashes, inputs=inputs,
        training_labels_permission=True, online_truth_permission=False, checkpoint_rule='final update',
        width=96, latent_dim=8, feature_dim=95, inner_steps_per_update_event=3, time=time.time(),
    ))
    train, train_manifest, train_pca = load_data(args.train, args.train_labels, args.task, args.representation, 'train')
    dev, dev_manifest, dev_pca = load_data(args.dev, args.dev_labels, args.task, args.representation, 'dev')
    validate_partitions(train_manifest, dev_manifest, train_pca, dev_pca)
    if len(train_manifest['seeds']) != 96 or len(dev_manifest['seeds']) != 12:
        raise ValueError('P45 requires the declared 96 training and 12 development conditions')
    if len(train) != 96 or len(dev) != 12 or any(len(e['rows']) < 13 for e in train + dev):
        raise ValueError('Every condition requires at least 13 valid views; no silent failed-condition removal')
    model = P45NeuralGeometry(args.task, args.representation).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    log, start = [], time.monotonic()
    for step in range(args.updates):
        model.train()
        count = int(rng.choice([3, 6, 12]))
        batch = batch_items(train, rng, args.batch_size, count, args.device)
        fixed, adapted, z0, z, inner = meta_prediction(model, batch, True)
        target = batch[-1]
        fixed_loss = ((fixed - target) / GEOMETRY_SCALE).square().mean()
        adapted_loss = ((adapted - target) / GEOMETRY_SCALE).square().mean()
        loss = .5 * (fixed_loss + adapted_loss) + 1e-4 * z0.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training objective')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 50 == 0 or step + 1 == args.updates:
            row = dict(
                step=step + 1, S=count, loss=float(loss.detach()),
                fixed_rmse=float(fixed_loss.detach().sqrt() * GEOMETRY_SCALE),
                adapted_rmse=float(adapted_loss.detach().sqrt() * GEOMETRY_SCALE),
                z_change=float((z - z0).norm(dim=-1).mean().detach()),
                inner_gradient=float(np.mean([record['gradient'] for record in inner])),
                elapsed=time.monotonic() - start,
            )
            log.append(row)
            print(json.dumps(row), flush=True)
            write(out / 'LOG.json', log)
    package = dict(
        schema='prove.p45-neural-checkpoint.v1', model_config=model.config(), model=model.state_dict(),
        seed=args.seed, updates=args.updates, data_inputs=inputs, sources=source_hashes,
        pca=train_manifest['pca'], representation=args.representation,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    torch.save(package, out / 'checkpoint.pt')
    diagnostic = dict(
        query=evaluation(model, dev, args.device), train_episodes=len(train), development_episodes=len(dev),
        checkpoint_sha256=sha(out / 'checkpoint.pt'), checkpoint_selection='final update',
        elapsed=time.monotonic() - start, parameter_count=package['parameter_count'],
        representation=args.representation, inference_weights_frozen=True,
    )
    write(out / 'DIAGNOSTIC.json', diagnostic)
    print(json.dumps(diagnostic), flush=True)


if __name__ == '__main__':
    main()
