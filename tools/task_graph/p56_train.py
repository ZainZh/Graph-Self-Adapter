#!/usr/bin/env python3
"""P56 offline short-Insert training; no simulator or implicit experiment scope.

P53 policy, losses, optimizer, and update counts are retained. The input bundle
must preserve executed feedback actions separately from offline expert labels.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from prove.task_graph.p52_stage_policy import Config
from prove.task_graph.p53_short_insert import ShortInsertPolicy
from tools.task_graph.p52_train import fit_normalization
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph.p53_train_short import minibatch_loss, optimize, sha, to_device, validate_dataset


VARIANTS = ('teacher_only', 'teacher_plus_policy_feedback')
ROOT = Path(__file__).resolve().parents[2]


def training_sources():
    return {name: sha(ROOT / name) for name in source_closure(ROOT,
        ['tools/task_graph/p56_train.py', 'tools/task_graph/p56_dataset.py'])}


def validate_bundle(data):
    if data.get('schema') != 'p56_short_insert_bundle_v1':
        raise ValueError('P56 sealed bundle required')
    variant = data.get('variant')
    if variant not in VARIANTS:
        raise ValueError('Unknown P56 data variant')
    counts = validate_dataset(data, 24)
    geometry_splits = {}
    condition_splits = {}
    physical_seed_splits = {}
    sources = set()
    for episode in data['episodes']:
        if not all(k in episode for k in ('condition_id', 'geometry_group', 'scale_xyz', 'condition_seed', 'source_policy', 'source_seal_sha256')):
            raise ValueError('Missing condition, geometry, policy, or provenance')
        if episode['source_policy'] not in ('expert', 'policy_fixed'):
            raise ValueError('Unknown source policy')
        if variant == 'teacher_only' and episode['source_policy'] != 'expert':
            raise ValueError('Policy feedback included in teacher-only view')
        group = episode['geometry_group']
        if not isinstance(group, str) or not group:
            raise ValueError('Explicit geometry group required')
        scale = tuple(float(v) for v in episode['scale_xyz'])
        if len(scale) != 3 or any(not torch.isfinite(torch.tensor(v)) or v <= 0 for v in scale):
            raise ValueError('Invalid numerical book scale')
        split = episode['split']
        geometry_splits.setdefault(scale, set()).add(split)
        for key, table in ((episode['condition_id'], condition_splits), (episode['condition_seed'], physical_seed_splits)):
            if table.setdefault(key, split) != split:
                raise ValueError('Condition or physical seed leaks across splits')
        identity = (episode['condition_id'], episode['source_policy'])
        if identity in sources:
            raise ValueError('Duplicate condition/policy trajectory')
        sources.add(identity)
        if episode['source_policy'] == 'policy_fixed' and 'expert_action' not in episode:
            raise ValueError('Policy feedback needs independent expert labels')
    permitted = data.get('permitted_geometry_overlap')
    if permitted != ['1,1,1']:
        raise ValueError('Only explicit original-shape ID overlap is permitted')
    for scale, splits in geometry_splits.items():
        if len(splits) > 1 and not (scale == (1.0, 1.0, 1.0) and splits == {'train', 'test'}):
            raise ValueError('Numerical book geometry leaks across splits')
    if variant == 'teacher_plus_policy_feedback' and not any(e['source_policy'] == 'policy_fixed' for e in data['episodes']):
        raise ValueError('Mixed view has no policy feedback trajectories')
    return counts


def validate_plan(plan, data_path, data, variant, seed):
    if plan.get('schema') != 'p56.training-plan.v1':
        raise ValueError('Frozen P56 training plan required')
    seeds = plan.get('seeds')
    if (not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) and s > 0 for s in seeds)
            or len(seeds) != len(set(seeds)) or plan.get('horizon') != 24):
        raise ValueError('Invalid frozen seed scope or horizon')
    if plan.get('warmup_updates') != 1000 or plan.get('branch_updates') != 2000:
        raise ValueError('Changed common/branch update budget')
    if plan.get('batch_size') != 4 or plan.get('learning_rate') != 3e-4:
        raise ValueError('Changed P53 sampling or optimizer')
    if plan.get('model') != asdict(Config()):
        raise ValueError('Changed P53 architecture, inner rate, or history')
    if plan.get('sources') != training_sources():
        raise ValueError('Frozen P56 trainer/source closure changed')
    if seed not in seeds or variant not in VARIANTS or data['variant'] != variant:
        raise ValueError('Wrong seed or data variant')
    entry = plan.get('datasets', {}).get(variant)
    if not isinstance(entry, dict) or Path(entry.get('path', '')).resolve() != data_path.resolve() or entry.get('sha256') != sha(data_path):
        raise ValueError('Training data differ from frozen plan')
    validate_bundle(data)


def load_inputs(plan_path, data_path, variant, seed):
    plan = json.loads(plan_path.read_text())
    data = torch.load(data_path, map_location='cpu', weights_only=True)
    validate_plan(plan, data_path, data, variant, seed)
    return plan, data


def create_warmup(plan_path, data_path, output, variant, seed, device):
    if output.exists():
        raise FileExistsError(output)
    plan, data = load_inputs(plan_path, data_path, variant, seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = ShortInsertPolicy(Config())
    fit_normalization(model, [e for e in data['episodes'] if e['split'] == 'train'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    result = optimize(model, optimizer, to_device(data['episodes'], device), generator,
                      horizon=24, branch='warmup', steps=1000)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema='p56_common_warmup_v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
                    config=asdict(model.config), generator_state=generator.get_state(),
                    torch_rng_state=torch.get_rng_state(), horizon=24, seed=seed,
                    variant=variant, data_sha256=sha(data_path), plan_sha256=sha(plan_path),
                    common_outer_updates=1000, result=result,
                    objective='P53 all-time fixed-z BC + .1 isolated predictor auxiliary'), output)


def fork_common(common_path, device):
    common = torch.load(common_path, map_location='cpu', weights_only=True)
    if common.get('schema') != 'p56_common_warmup_v1' or common.get('common_outer_updates') != 1000:
        raise ValueError('Wrong common warmup artifact')
    if common.get('config') != asdict(Config()):
        raise ValueError('Changed P53 model configuration')
    model = ShortInsertPolicy(Config())
    model.load_state_dict(common['model'], strict=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(common['optimizer'])
    generator = torch.Generator()
    generator.set_state(common['generator_state'])
    torch.set_rng_state(common['torch_rng_state'])
    return model, optimizer, generator, common


def train_branch(plan_path, data_path, common_path, output, variant, seed, branch, device):
    if output.exists():
        raise FileExistsError(output)
    if branch not in ('bc', 'joint'):
        raise ValueError('Unknown P56 branch')
    _, data = load_inputs(plan_path, data_path, variant, seed)
    model, optimizer, generator, common = fork_common(common_path, device)
    if any((common['seed'] != seed, common['variant'] != variant,
            common['data_sha256'] != sha(data_path), common['plan_sha256'] != sha(plan_path))):
        raise ValueError('Branch does not match common warmup')
    if branch == 'bc':
        model.predictor.requires_grad_(False)
    result = optimize(model, optimizer, to_device(data['episodes'], device), generator,
                      horizon=24, branch=branch, steps=2000)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema='p56_short_insert_skill_v1', branch=branch, model=model.state_dict(),
                    config=asdict(model.config), common_warmup_sha256=sha(common_path),
                    data_sha256=sha(data_path), plan_sha256=sha(plan_path), variant=variant,
                    horizon=24, seed=seed, common_outer_updates=1000,
                    branch_outer_updates=2000, total_outer_updates=3000, result=result,
                    objective=('P53 all-time ordinary BC' if branch == 'bc' else
                               'P53 .5 online-z BC + .5 fixed-z BC + .1 isolated predictor auxiliary'),
                    checkpoint_selection='final only',
                    deployment='all network weights frozen; only stage z updates'), output)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('warmup', 'branch'):
        p = sub.add_parser(name)
        for key in ('plan', 'data', 'output'):
            p.add_argument('--' + key, type=Path, required=True)
        p.add_argument('--variant', choices=VARIANTS, required=True)
        p.add_argument('--seed', type=int, required=True)
        p.add_argument('--device', default='cpu')
        if name == 'branch':
            p.add_argument('--common', type=Path, required=True)
            p.add_argument('--branch', choices=('bc', 'joint'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'warmup':
        create_warmup(args.plan, args.data, args.output, args.variant, args.seed, args.device)
    else:
        train_branch(args.plan, args.data, args.common, args.output, args.variant, args.seed, args.branch, args.device)


if __name__ == '__main__':
    main()
