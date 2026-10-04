#!/usr/bin/env python3
"""Optional P56 joint continuation from the unchanged common warmup.

The only implementation change is fast_full_stage_loss. Sampling, optimizer,
initial state, full second-order latent adaptation, and 2000 outer updates
follow p56_train. This entrypoint never runs physics or chooses a checkpoint.
"""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from prove.task_graph.p52_stage_policy import Config
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.p53_train_short import sample_episodes, sha, to_device
from tools.task_graph.p56_fast_loss import fast_full_stage_loss
from tools.task_graph.p56_train import fork_common, validate_plan


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('teacher_only', 'teacher_plus_policy_feedback')


def fast_sources():
    return {name: sha(ROOT / name) for name in source_closure(ROOT,
        ['tools/task_graph/p56_train_fast.py', 'tools/task_graph/p56_fast_loss.py'])}


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def verify_amendment(path, plan_path, data_path, common_path, variant):
    amendment = json.loads(Path(path).read_text())
    if amendment.get('schema') != 'prove.p56-fast-execution-amendment.v1':
        raise ValueError('Explicit P56 fast execution amendment required')
    authorization = amendment.get('authorization')
    if (not isinstance(authorization, dict) or authorization.get('approved') is not True
            or not isinstance(authorization.get('basis'), str) or not authorization['basis'].strip()):
        raise ValueError('User acceleration authorization must be recorded')
    if amendment.get('sources') != fast_sources():
        raise ValueError('Fast implementation source closure differs from amendment')
    if amendment.get('training_plan') != reference(plan_path):
        raise ValueError('Fast amendment binds another training plan')
    expected = amendment.get('variants', {}).get(variant)
    if expected != dict(data_sha256=sha(data_path), common_sha256=sha(common_path)):
        raise ValueError('Fast amendment binds another data/common warmup pair')
    original_plan = json.loads(Path(plan_path).read_text())
    relocation = amendment.get('relocated_inputs', {}).get(variant)
    if not isinstance(relocation, dict):
        raise ValueError('Explicit P56 relocated input references required')
    canonical_data = relocation.get('canonical_dataset')
    local_data = relocation.get('local_dataset')
    canonical_common = relocation.get('canonical_common')
    local_common = relocation.get('local_common')
    if canonical_data != original_plan.get('datasets', {}).get(variant):
        raise ValueError('Canonical dataset reference differs from original plan')
    if local_data != reference(data_path) or local_common != reference(common_path):
        raise ValueError('Relocated data/common bytes or paths differ')
    if (local_data['sha256'] != canonical_data['sha256'] or
            not isinstance(canonical_common, dict) or canonical_common.get('sha256') != local_common['sha256'] or
            canonical_common['sha256'] != expected['common_sha256']):
        raise ValueError('Relocated inputs do not match canonical bytes')
    for key in ('speed_acceptance', 'gradient_acceptance'):
        ref = amendment.get(key)
        if not isinstance(ref, dict) or 'path' not in ref or 'sha256' not in ref:
            raise ValueError('Fast amendment lacks ' + key + ' evidence reference')
        if reference(ref['path']) != ref:
            raise ValueError('Fast amendment evidence changed: ' + key)
    return amendment


def load_relocated_inputs(plan_path, data_path, amendment_path, common_path, variant):
    """Validate a copied dataset against the original plan without editing it."""
    original_plan = json.loads(Path(plan_path).read_text())
    verify_amendment(amendment_path, plan_path, data_path, common_path, variant)
    data = torch.load(data_path, map_location='cpu', weights_only=True)
    local_plan = copy.deepcopy(original_plan)
    local_plan['datasets'][variant] = reference(data_path)
    validate_plan(local_plan, Path(data_path), data, variant, 11)
    return original_plan, data


def minibatch_fast_loss(model, episodes):
    losses = []
    metrics = []
    for episode in episodes:
        loss, parts = fast_full_stage_loss(model, episode['features'], episode['executed_action'],
            episode['expert_action'], horizon=24, meta=True,
            fixed_bc_weight=.5, auxiliary_weight=.1)
        losses.append(loss)
        metrics.append(parts)
    return torch.stack(losses).mean(), {key: torch.stack([part[key] for part in metrics]).mean()
                                      for key in ('control', 'online_bc', 'fixed_bc', 'auxiliary')}


def optimize_fast(model, optimizer, episodes, generator, steps=2000):
    if steps != 2000:
        raise ValueError('P56 joint branch requires exactly 2000 outer updates')
    start = time.monotonic()
    last = {}
    for update in range(steps):
        optimizer.zero_grad(set_to_none=True)
        selected = sample_episodes(episodes, generator, count=4)
        loss, parts = minibatch_fast_loss(model, selected)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite joint objective; stop without retry')
        loss.backward()
        optimizer.step()
        last = {key: float(value.detach()) for key, value in dict(loss=loss, **parts).items()}
        if (update + 1) % 100 == 0:
            print(json.dumps(dict(branch='joint_fast', update=update + 1,
                                  seconds=time.monotonic() - start, **last)), flush=True)
    return dict(seconds=time.monotonic() - start, last=last, updates=steps)


def train_joint_fast(plan_path, data_path, common_path, amendment_path, output, variant, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if variant not in VARIANTS:
        raise ValueError('Unknown P56 data variant')
    _, data = load_relocated_inputs(plan_path, data_path, amendment_path, common_path, variant)
    model, optimizer, generator, common = fork_common(common_path, device)
    if (common['seed'] != 11 or common['variant'] != variant or common['horizon'] != 24
            or common['common_outer_updates'] != 1000 or common['config'] != asdict(Config())
            or common['data_sha256'] != sha(data_path) or common['plan_sha256'] != sha(plan_path)):
        raise ValueError('Fast joint branch differs from common P56 warmup')
    result = optimize_fast(model, optimizer, to_device(data['episodes'], device), generator)
    # Recheck external evidence after the full branch before sealing the final.
    verify_amendment(amendment_path, plan_path, data_path, common_path, variant)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema='p56_short_insert_skill_v1', branch='joint', model=model.state_dict(),
        config=asdict(model.config), common_warmup_sha256=sha(common_path),
        data_sha256=sha(data_path), plan_sha256=sha(plan_path), variant=variant,
        horizon=24, seed=11, common_outer_updates=1000, branch_outer_updates=2000,
        total_outer_updates=3000, result=result,
        objective='P53 .5 online-z BC + .5 fixed-z BC + .1 isolated predictor auxiliary',
        checkpoint_selection='final only',
        deployment='all network weights frozen; only stage z updates',
        execution_amendment=reference(amendment_path), fast_sources=fast_sources(),
        numerical_contract='Mathematically equivalent objective; floating-point operation order may differ'), output)


def main():
    parser = argparse.ArgumentParser()
    for name in ('plan', 'data', 'common', 'amendment', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    train_joint_fast(args.plan, args.data, args.common, args.amendment,
                     args.output, args.variant, args.device)


if __name__ == '__main__':
    main()
