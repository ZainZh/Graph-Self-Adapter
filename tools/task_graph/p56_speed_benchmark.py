#!/usr/bin/env python3
"""Local, finite P56 loss/gradient speed comparison on a sealed training view.

This development diagnostic does not write checkpoints or run evaluations.
It deliberately uses float32 eager PyTorch and the original second-order inner
adaptation. Both arms start at the same common warmup state and see the same
four trajectory indices at every update.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from prove.task_graph.p52_stage_policy import Config
from prove.task_graph.p53_short_insert import ShortInsertPolicy
from tools.task_graph.p53_train_short import minibatch_loss, to_device
from tools.task_graph.p56_fast_loss import fast_full_stage_loss
from tools.task_graph.p56_train import validate_bundle


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    'tools/task_graph/p53_train_short.py',
    'tools/task_graph/p56_fast_loss.py',
    'prove/task_graph/p53_short_insert.py',
    'prove/task_graph/p52_stage_policy.py',
    'tools/task_graph/p56_speed_benchmark.py',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def optimized_minibatch_loss(model, episodes, *, horizon, branch):
    if branch not in ('warmup', 'bc', 'joint'):
        raise ValueError(f'Unknown branch: {branch}')
    values = [fast_full_stage_loss(
        model, episode['features'], episode['executed_action'], episode['expert_action'],
        horizon=horizon, meta=branch == 'joint', fixed_bc_weight=.5,
        auxiliary_weight=0 if branch == 'bc' else .1,
    ) for episode in episodes]
    return (torch.stack([value[0] for value in values]).mean(),
            {key: torch.stack([value[1][key] for value in values]).mean()
             for key in ('control', 'online_bc', 'fixed_bc', 'auxiliary')})


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def make_arm(common, device, branch):
    model = ShortInsertPolicy(Config(**common['config']))
    model.load_state_dict(common['model'], strict=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(common['optimizer'])
    if branch == 'bc':
        model.predictor.requires_grad_(False)
    return model, optimizer


def snapshot_grads(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()}


def compare_tensors(left, right):
    squared_difference = 0.0
    squared_reference = 0.0
    max_absolute = 0.0
    max_name = None
    for name, reference in left.items():
        candidate = right[name]
        if reference is None or candidate is None:
            if reference is not None or candidate is not None:
                raise ValueError(f'Missing gradient on one arm: {name}')
            continue
        if reference.shape != candidate.shape:
            raise ValueError(f'Tensor shape differs: {name}')
        difference = candidate.double() - reference.double()
        squared_difference += float(difference.square().sum())
        squared_reference += float(reference.double().square().sum())
        local_max = float(difference.abs().max())
        if local_max > max_absolute:
            max_absolute, max_name = local_max, name
    return {'relative_l2': (squared_difference ** .5) / max(squared_reference ** .5, 1e-30),
            'global_max_absolute': max_absolute, 'global_max_parameter': max_name}


def parameters_cpu(model):
    return {name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()}


def update(model, optimizer, batch, loss_function, horizon, branch, device):
    sync(device)
    baseline_memory = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss, parts = loss_function(model, batch, horizon=horizon, branch=branch)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite objective; comparison stopped')
    loss.backward()
    optimizer.step()
    sync(device)
    seconds = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    return {
        'seconds': seconds,
        'loss': float(loss.detach()),
        'parts': {name: float(value.detach()) for name, value in parts.items()},
        'grads': snapshot_grads(model),
        'peak_allocated_bytes': peak_memory,
        'peak_increment_bytes': None if peak_memory is None else peak_memory - baseline_memory,
    }


def run(args):
    if args.steps < 1 or args.steps > 100:
        raise ValueError('--steps must be between 1 and 100')
    if args.warmup < 0 or args.warmup > 10:
        raise ValueError('--warmup must be between 0 and 10')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    common = torch.load(args.common, map_location='cpu', weights_only=True)
    if common.get('schema') != 'p56_common_warmup_v1' or common.get('common_outer_updates') != 1000:
        raise ValueError('Expected a completed P56 common warmup checkpoint')
    data = torch.load(args.data, map_location='cpu', weights_only=True)
    validate_bundle(data)
    if data['variant'] != common['variant'] or sha256(args.data) != common['data_sha256']:
        raise ValueError('Bundle variant or SHA256 differs from common checkpoint')
    if common['horizon'] != 24:
        raise ValueError('Expected 24-step P56 horizon')
    train = to_device(data['episodes'], device)
    generator = torch.Generator()
    generator.set_state(common['generator_state'])
    # Match p53_train_short.sample_episodes: four independent scalar draws.
    index_batches = [[int(torch.randint(len(train), (), generator=generator)) for _ in range(4)]
                     for _ in range(args.warmup + args.steps)]
    original_model, original_optimizer = make_arm(common, device, args.branch)
    fast_model, fast_optimizer = make_arm(common, device, args.branch)
    arms = ((original_model, original_optimizer, minibatch_loss),
            (fast_model, fast_optimizer, optimized_minibatch_loss))
    for indices in index_batches[:args.warmup]:
        batch = [train[index] for index in indices]
        for model, optimizer, loss_function in arms:
            update(model, optimizer, batch, loss_function, common['horizon'], args.branch, device)
    # Warmup removes one-time CUDA/PyTorch startup from timing; reset all weights
    # and Adam moments so every measured arm begins at the exact common state.
    del arms, original_model, original_optimizer, fast_model, fast_optimizer
    if args.warmup:
        del model, optimizer, loss_function, batch
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    original_model, original_optimizer = make_arm(common, device, args.branch)
    fast_model, fast_optimizer = make_arm(common, device, args.branch)
    records = []
    for indices in index_batches[args.warmup:]:
        batch = [train[index] for index in indices]
        original = update(original_model, original_optimizer, batch, minibatch_loss,
                          common['horizon'], args.branch, device)
        optimized = update(fast_model, fast_optimizer, batch, optimized_minibatch_loss,
                           common['horizon'], args.branch, device)
        loss_abs = abs(original['loss'] - optimized['loss'])
        records.append({
            'sample_indices': indices,
            'sample_episode_ids': [str(train[index]['episode_id']) for index in indices],
            'original_seconds': original['seconds'],
            'optimized_seconds': optimized['seconds'],
            'original_loss': original['loss'],
            'optimized_loss': optimized['loss'],
            'loss_absolute': loss_abs,
            'loss_relative': loss_abs / max(abs(original['loss']), 1e-30),
            'parts_absolute': {name: abs(original['parts'][name] - optimized['parts'][name])
                               for name in original['parts']},
            'grad_difference': compare_tensors(original['grads'], optimized['grads']),
            'original_peak_allocated_bytes': original['peak_allocated_bytes'],
            'optimized_peak_allocated_bytes': optimized['peak_allocated_bytes'],
            'original_peak_increment_bytes': original['peak_increment_bytes'],
            'optimized_peak_increment_bytes': optimized['peak_increment_bytes'],
        })
    parameter_difference = compare_tensors(parameters_cpu(original_model), parameters_cpu(fast_model))
    original_seconds = sum(row['original_seconds'] for row in records)
    optimized_seconds = sum(row['optimized_seconds'] for row in records)
    return {
        'schema': 'p56.loss-speed-benchmark.v1',
        'purpose': 'local development diagnostic; no formal training or evaluation',
        'device': str(device), 'cuda_device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch_version': torch.__version__, 'dtype': 'float32', 'amp': False,
        'tf32': False, 'torch_compile': False, 'branch': args.branch,
        'steps': args.steps, 'warmup_steps_excluded': args.warmup,
        'data_path': str(args.data.resolve()), 'common_path': str(args.common.resolve()),
        'data_sha256': sha256(args.data), 'common_sha256': sha256(args.common),
        'source_sha256': {path: sha256(ROOT / path) for path in SOURCE_PATHS},
        'original_total_seconds': original_seconds,
        'optimized_total_seconds': optimized_seconds,
        'speedup_original_over_optimized': original_seconds / optimized_seconds,
        'max_loss_relative': max(row['loss_relative'] for row in records),
        'max_gradient_relative_l2': max(row['grad_difference']['relative_l2'] for row in records),
        'max_gradient_absolute': max(row['grad_difference']['global_max_absolute'] for row in records),
        'final_parameter_difference': parameter_difference,
        'updates': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--common', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--branch', choices=('warmup', 'bc', 'joint'), default='joint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        report = run(args)
    except Exception as error:
        report = {'schema': 'p56.loss-speed-benchmark.v1', 'status': 'failed',
                  'error_type': type(error).__name__, 'error': str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        raise
    report['status'] = 'completed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'steps', 'branch',
        'original_total_seconds', 'optimized_total_seconds', 'speedup_original_over_optimized',
        'max_loss_relative', 'max_gradient_relative_l2', 'max_gradient_absolute')}))


if __name__ == '__main__':
    main()
