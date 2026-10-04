#!/usr/bin/env python3
"""Finite-budget P61 paired training on public caches and isolated train labels.

This entry point never opens development data. Checkpoints are final-budget
snapshots, not validation-selected weights. The official fusion/query layers
are either frozen or trainable; all new relationship/readout layers train in
both arms, with identical initialization, examples and losses.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from prove.fast_adapter.p60_contour import ContourMeasurement, model_state_sha256
from prove.fast_adapter.p61_adaptation import adapt_z, world_geometry
from prove.task_graph.p59.core import FrozenP50Action, geometry_loss, world_pose_loss


ENCODER_KEYS = ('dino_layers', 'role_patch_masks', 'role_valid', 'K',
                'T_anchor_camera', 'T_anchor_eef')


def encode_frames(model, public, indices, device):
    kwargs = {}
    for key in ENCODER_KEYS:
        value = public[key][indices].to(device)
        kwargs[key] = value if value.dtype == torch.bool else value.float()
    kwargs['image_hw'] = tuple(int(x) for x in public['image_hw'])
    return model(**kwargs).geometry8


def contour_measurements(public, indices):
    transform = public['T_world_anchor'].float()
    measurements = []
    for index in indices:
        if not bool(public['boundary_valid'][index]):
            continue
        camera = transform @ public['T_anchor_camera'][index, 0].float()
        boundary = public['boundary_uv'][index].numpy()
        hull = public.get('goal_hull_vertices', public['boundary_uv'])[index]
        measurements.append(ContourMeasurement(
            boundary=boundary, hull_vertices=np.asarray(hull),
            K=public['K'][index, 0].numpy(), camera_position=camera[:3, 3].numpy(),
            camera_rotation=camera[:3, :3].numpy(),
            image_size=tuple(int(x) for x in public['image_hw']),
            t=int(public['frame_index'][index])))
    return measurements


def supervised_loss(geometry8, z, public, private, indices, phase, p50, device):
    """Train/evaluation-only labels; ordinary adaptation never receives private."""
    transform = public['T_world_anchor'].to(device).float()
    # Teacher-conditioned offline action sensitivity only. This true length
    # never enters encoder inputs, z updates or ordinary runtime predictions.
    held_length = private['geometry12_world'][indices, 11:12].to(device).float()
    prediction = world_geometry(geometry8, z, transform, held_length)
    target = private['geometry12_world'][indices].to(device).float()
    q = public['q'][indices].to(device).float()
    phases = torch.full((len(indices),), phase, device=device, dtype=torch.long)
    predicted_action = p50.decode_world_pose(p50(prediction, q, phases), prediction, q)
    with torch.no_grad():
        target_action = p50.decode_world_pose(p50(target, q, phases), target, q)
    geo = geometry_loss(prediction, target)
    act = world_pose_loss(predicted_action, target_action)
    return geo + act, {'geometry': float(geo.detach()), 'world_action': float(act.detach())}


def episode_loss(model, public, private, *, stage, phase, p50, device,
                 frame_indices=None):
    if stage == 'supervised':
        indices = list(range(len(public['frame_index']))) if frame_indices is None else frame_indices
        geometry = encode_frames(model, public, indices, device)
        return supervised_loss(geometry, torch.zeros_like(geometry), public, private,
                               indices, phase, p50, device)
    # Strictly causal persistent state: current encodings at16 and24; true
    # future query29 is withheld from the measurement history.
    frames = public['frame_index'].tolist()
    lookup = {int(t): i for i, t in enumerate(frames)}
    encoded_indices = [lookup[t] for t in (16, 24, 29)]
    geometry = encode_frames(model, public, encoded_indices, device)
    z = torch.zeros(8, device=device)
    transform = public['T_world_anchor'].to(device).float()
    trace = []
    for i, times in enumerate(((0, 8, 16), (0, 8, 16, 24))):
        observations = contour_measurements(public, [lookup[t] for t in times])
        z, records = adapt_z(geometry[i], z, transform, observations, create_graph=True)
        trace.extend(records)
    query_index = [lookup[29]]
    loss, stats = supervised_loss(geometry[2:3], z[None], public, private,
                                 query_index, phase, p50, device)
    fixed_loss, fixed_stats = supervised_loss(geometry[2:3], torch.zeros_like(geometry[2:3]),
        public, private, query_index, phase, p50, device)
    loss = loss + .25 * fixed_loss
    stats['fixed_query_loss'] = float(fixed_loss.detach())
    stats['z_norm'] = float(z.detach().norm())
    stats['inner_steps'] = len(trace)
    return loss, stats


class TrainCache:
    def __init__(self, episodes, limit=4):
        self.episodes, self.limit, self.cache = episodes, limit, OrderedDict()

    def __getitem__(self, index):
        if index not in self.cache:
            path = Path(self.episodes[index])
            self.cache[index] = (torch.load(path / 'public.pt', weights_only=True, mmap=True),
                                 torch.load(path / 'private.pt', weights_only=True))
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def _lila_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(('fusion.', 'visual_adapter.')):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_arm(model, episodes, p50, output, *, arm, seed=11, supervised_steps=1000,
              meta_steps=300, batch_size=2, lr=1e-4, device='cuda'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('refusing to resume or overwrite a nonempty run directory')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model.to(device).train()
    p50.to(device).eval().requires_grad_(False)
    initial_model = model_state_sha256(model)
    initial_lila = _lila_hash(model)
    sample_schedule = hashlib.sha256()
    initial_p50 = model_state_sha256(p50)
    official = [p for n, p in model.named_parameters()
                if p.requires_grad and n.startswith(('fusion.', 'visual_adapter.'))]
    task = [p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith(('fusion.', 'visual_adapter.'))]
    optimizer = torch.optim.AdamW([{'params': official, 'lr': 1e-5},
                                  {'params': task, 'lr': lr}])
    cache = TrainCache(episodes)
    rng = np.random.default_rng(seed)
    records = []
    max_lila_gradient = 0.
    module_max_gradient = {'fusion': 0., 'visual_adapter': 0.}
    module_initial_hash = {name: model_state_sha256(getattr(model, name)) for name in module_max_gradient}
    start = time.monotonic()
    for stage, count in [('supervised', supervised_steps), ('meta', meta_steps)]:
        for step in range(count):
            optimizer.zero_grad(set_to_none=True)
            totals = []
            indices = rng.integers(len(episodes), size=batch_size).tolist()
            for index in indices:
                public, private = cache[index]
                chosen = sorted(rng.choice(len(public['frame_index']), 2, replace=False).tolist())
                sample_schedule.update(json.dumps({'stage': stage, 'step': step, 'episode': index,
                    'frames': chosen if stage == 'supervised' else [0, 8, 16, 24, 29]}, sort_keys=True).encode())
                loss, stats = episode_loss(model, public, private, stage=stage,
                    phase=step % 4, p50=p50, device=device, frame_indices=chosen)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f'nonfinite {stage} loss at {step}')
                (loss / batch_size).backward()
                totals.append(dict(stats, total=float(loss.detach())))
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(bool(g.isfinite().all()) for g in gradients):
                raise FloatingPointError(f'nonfinite {stage} gradient at {step}')
            lila_sq = sum(float(p.grad.detach().square().sum()) for n, p in model.named_parameters()
                          if n.startswith(('fusion.', 'visual_adapter.')) and p.grad is not None)
            max_lila_gradient = max(max_lila_gradient, lila_sq ** .5)
            for name in module_max_gradient:
                norm = sum(float(p.grad.detach().square().sum())
                           for p in getattr(model, name).parameters() if p.grad is not None) ** .5
                module_max_gradient[name] = max(module_max_gradient[name], norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            record = {'stage': stage, 'step': step + 1, 'episode_indices': indices,
                      'mean_loss': sum(x['total'] for x in totals) / batch_size,
                      'examples': totals, 'lila_gradient_norm': lila_sq ** .5,
                      'elapsed_seconds': time.monotonic() - start}
            records.append(record)
            with (output / 'training.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            if (step + 1) % 10 == 0 or step == 0:
                print(json.dumps({k:v for k,v in record.items() if k != 'examples'}), flush=True)
    final_lila = _lila_hash(model)
    summary = {'initial_model_sha256': initial_model,
               'sample_schedule_sha256': sample_schedule.hexdigest(),
               'training_episode_paths': [str(x) for x in episodes],
               'arm': arm, 'seed': seed, 'supervised_steps': supervised_steps,
               'meta_steps': meta_steps, 'batch_size': batch_size,
               'official_lr': 1e-5, 'task_lr': lr, 'weight_decay': .01,
               'loss_weights': {'geometry': 1., 'world_action': 1., 'meta_online': 1., 'meta_fixed': .25},
               'official_modules': {name: {'initial_sha256': module_initial_hash[name],
                   'final_sha256': model_state_sha256(getattr(model, name)),
                   'max_gradient_norm': module_max_gradient[name]} for name in module_max_gradient},
               'elapsed_seconds': time.monotonic() - start,
               'initial_lila_sha256': initial_lila, 'final_lila_sha256': final_lila,
               'max_lila_gradient_norm': max_lila_gradient,
               'p50_unchanged': model_state_sha256(p50) == initial_p50,
               'all_gradients_finite': True, 'action_loss': 'offline teacher-conditioned true held length', 'selection': 'final fixed budget'}
    if arm == 'frozen_lila' and initial_lila != final_lila:
        raise AssertionError('frozen official LiLa weights changed')
    if arm == 'task_trained_lila' and (max_lila_gradient == 0 or initial_lila == final_lila):
        raise AssertionError('task-trained official LiLa received no effective updates')
    if arm == 'task_trained_lila' and not all(value > 0 for value in module_max_gradient.values()):
        raise AssertionError('one official module received no gradients')
    torch.save({'constructor': {'lila_source': model.lila_binding['source'],
                  'lila_checkpoint': model.lila_binding['checkpoint'],
                  'finetune_lila': model.finetune_lila,
                  'geometry_mean8': model.geometry_mean8.detach().cpu(),
                  'max_views': model.max_views, 'adapter_microbatch': model.adapter_microbatch},
                'lila_binding': model.lila_binding, 'schema': 'prove.p61-relation-checkpoint.v1',
                'state_dict': {n:v.detach().cpu() for n,v in model.state_dict().items()},
                'training': summary}, output / 'final.pt')
    (output / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lila-source', required=True)
    parser.add_argument('--lila-checkpoint', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['frozen_lila', 'task_trained_lila'], required=True)
    parser.add_argument('--p50-checkpoint', default='evaluations/fast_adapter_p50_precision_policy_v01/offline_v01/precision_action.pt')
    parser.add_argument('--supervised-steps', type=int, default=1000)
    parser.add_argument('--meta-steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    from prove.fast_adapter.p61_encoder import P61RelationEncoder
    manifest = json.loads((args.dataset / 'MANIFEST.json').read_text())
    records = manifest.get('episodes', manifest.get('records'))
    episodes = [args.dataset / r['path'] for r in records if r['split'] == 'train']
    if len(episodes) != 96:
        raise ValueError(f'P61 freezes exactly 96 training configurations, got {len(episodes)}')
    torch.manual_seed(args.seed)
    train_geometry = torch.cat([torch.load(path / 'private.pt', weights_only=True)['geometry8']
                                for path in episodes])
    model = P61RelationEncoder(args.lila_source, args.lila_checkpoint, finetune_lila=args.arm == 'task_trained_lila',
                               geometry_mean8=train_geometry.float().mean(0))
    p50 = FrozenP50Action(args.p50_checkpoint)
    result = train_arm(model, episodes, p50, args.output, arm=args.arm,
        seed=args.seed, supervised_steps=args.supervised_steps, meta_steps=args.meta_steps,
        batch_size=args.batch_size, device=args.device)
    result['data_manifest_sha256'] = hashlib.sha256((args.dataset / 'MANIFEST.json').read_bytes()).hexdigest()
    (args.output / 'SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
