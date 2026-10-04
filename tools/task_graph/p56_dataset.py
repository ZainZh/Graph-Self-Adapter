#!/usr/bin/env python3
"""Seal-checked P56 public feature bundle with private offline expert labels.

Every predeclared collection cell stays in the catalog. Failed policy episodes
with actual transitions remain eligible; outcome is never a training filter.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from tools.task_graph.p53_train_short import sha
from tools.task_graph.p56_train import validate_bundle, VARIANTS


FEATURE_SHAPES = dict(raw=(2, 3, 3072), boxes=(2, 3, 4), visible=(2, 3), proprio=(18,))


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sealed_file(root, seal, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or seal['files'].get(relative) != sha(path):
        raise ValueError(f'Unsealed or modified file: {relative}')
    return path


def inspect_cell(directory, cell, binding_sha):
    root = Path(directory).resolve()
    result = read(root / 'RESULT.json')
    seal = read(root / 'SEAL.json')
    if result.get('seal_sha256') != sha(root / 'SEAL.json'):
        raise ValueError('Collection result/seal chain mismatch')
    if seal.get('binding_sha256') != sha(root / 'BINDING.json') or sha(root / 'BINDING.json') != binding_sha:
        raise ValueError('Collection binding differs from frozen manifest')
    for relative in seal['files']:
        sealed_file(root, seal, relative)
    if read(root / 'CELL.json') != cell:
        raise ValueError('Cell identity differs from frozen binding')
    for key in ('condition_id', 'geometry_id', 'scale_xyz', 'split', 'source_policy', 'seed'):
        if result.get(key) != cell[key]:
            raise ValueError('Result identity differs from frozen cell')
    qualified = bool(result.get('near_entry_qualified'))
    actions = int(result.get('actions', 0))
    row = dict(cell_index=cell['index'], condition_id=cell['condition_id'], geometry_group=cell['geometry_id'],
               scale_xyz=cell['scale_xyz'], condition_seed=cell['seed'], split=cell['split'],
               source_policy=cell['source_policy'], directory=str(root),
               near_entry_qualified=qualified, actions=actions, outcome=result.get('stop_reason'),
               source_seal_sha256=sha(root / 'SEAL.json'), eligible=False, frames=[])
    if not qualified:
        if actions:
            raise ValueError('Unqualified near entry executed short-stage actions')
        return row
    if not 0 <= actions <= 24:
        raise ValueError('Qualified training cell exceeds short-stage horizon')
    near = read(sealed_file(root, seal, 'NEAR_ENTRY.json'))
    if (near['condition_id'] != cell['condition_id'] or near['geometry_id'] != cell['geometry_id']
            or near['scale_xyz'] != cell['scale_xyz']):
        raise ValueError('Near-entry identity mismatch')
    sealed_file(root, seal, near['frame'])
    if near['frame_sha256'] != sha(root / near['frame']):
        raise ValueError('Near-entry state hash mismatch')
    with np.load(root / near['frame'], allow_pickle=False) as state:
        if 'supervisor_sim_state' not in state or 'supervisor_ctrl' not in state:
            raise ValueError('Near-entry physical state/control missing')
        if hashlib.sha256(state['supervisor_sim_state'].tobytes()).hexdigest() != near['state_sha256']:
            raise ValueError('Near-entry state content hash mismatch')
    row['near_entry_state_sha256'] = near['state_sha256']
    row['scene_xml_sha256'] = near['scene_xml_sha256']
    if actions == 0:
        if result.get('all_executed_expert_labels_valid') is not False:
            raise ValueError('Zero-action entry must declare label exclusion')
        row['exclusion_reason'] = result.get('label_exclusion_reason', 'no_executed_micro_transition')
        return row
    transitions = lines(sealed_file(root, seal, 'public/TRANSITIONS.jsonl'))
    labels = lines(sealed_file(root, seal, 'private/LABELS.jsonl'))
    if len(transitions) != actions or len(labels) != actions + 1:
        raise ValueError('Expected T transitions and T+1 frame-bound labels')
    all_labels_valid = True
    for t, label in enumerate(labels):
        if label['t'] != t:
            raise ValueError('Nonconsecutive expert label')
        frame = label['public_observation']
        if Path(frame).name != frame:
            raise ValueError('Unsafe public frame name')
        relative = 'public/' + frame
        path = sealed_file(root, seal, relative)
        if label['public_observation_sha256'] != sha(path):
            raise ValueError('Expert label not bound to public frame')
        feature_relative = f'public/features_{t:04d}.pt'
        sealed_file(root, seal, feature_relative)
        row['frames'].append(dict(frame=str(path), feature=str(root / feature_relative),
                                  frame_sha256=sha(path), feature_sha256=sha(root / feature_relative)))
        if t == actions:
            continue
        all_labels_valid = all_labels_valid and label.get('expert_label_valid') is True
        transition = transitions[t]
        if transition['t'] != t or transition['observation'] != frame or transition['next_observation'] != labels[t + 1]['public_observation']:
            raise ValueError('Executed transition/frame alignment mismatch')
        if transition['observation_sha256'] != label['public_observation_sha256'] or transition['next_observation_sha256'] != labels[t + 1]['public_observation_sha256']:
            raise ValueError('Executed transition frame hashes mismatch')
        executed = np.asarray(transition['executed_action'], dtype=float)
        expert = np.asarray(label['expert_action'], dtype=float)
        if (executed.shape != (7,) or expert.shape != (7,) or not np.isfinite(executed).all()
                or not np.isfinite(expert).all() or executed[-1] != 1 or expert[-1] != 1):
            raise ValueError('Finite closed-gripper executed and expert OSC7 required')
    if bool(result.get('all_executed_expert_labels_valid')) != all_labels_valid:
        raise ValueError('Result expert-label validity differs from executed labels')
    row['eligible'] = all_labels_valid
    if not all_labels_valid:
        row['exclusion_reason'] = result.get('label_exclusion_reason', 'invalid_executed_expert_label')
    return row


def catalog(manifest_path):
    manifest = read(manifest_path)
    parent = manifest['collection_binding']
    binding_path = Path(parent['path']).resolve()
    if sha(binding_path) != parent['sha256']:
        raise ValueError('Collection binding hash changed')
    binding = read(binding_path)
    if binding.get('schema') != 'prove.p56-collection-binding.v1':
        raise ValueError('P56 collection binding required')
    cells = binding['cells']
    if [c['index'] for c in cells] != list(range(len(cells))):
        raise ValueError('Cell indices must be consecutive')
    if manifest.get('permitted_geometry_overlap') != ['1,1,1']:
        raise ValueError('Explicit original-shape ID overlap declaration required')
    items = manifest['episodes']
    by_index = {item['cell_index']: item for item in items}
    expected = {c['index'] for c in cells if c['split'] in ('train', 'validation') and c['source_policy'] in ('expert', 'policy_fixed')}
    if len(by_index) != len(items) or set(by_index) != expected:
        raise ValueError('Retain every frozen train/development collection cell once')
    groups = {};conditions={};seeds={}
    for cell in cells:
        split = cell['split']
        if split not in ('qualification', 'train', 'validation', 'test'):
            raise ValueError('Explicit collection split required')
        if split == 'qualification':
            continue
        scale = tuple(float(v) for v in cell['scale_xyz'])
        if len(scale) != 3 or any(not np.isfinite(v) or v <= 0 for v in scale):
            raise ValueError('Invalid numerical book geometry')
        groups.setdefault(scale, set()).add(split)
        for key, table in ((cell['condition_id'], conditions), (cell['seed'], seeds)):
            if table.setdefault(key, split) != split:
                raise ValueError('Condition or physical seed leaks across splits')
    for scale, splits in groups.items():
        if len(splits) > 1 and not (scale == (1.0, 1.0, 1.0) and splits == {'train', 'test'}):
            raise ValueError('Numerical book geometry leaks across splits')
    rows = [inspect_cell(by_index[cell['index']]['directory'], cell, parent['sha256']) for cell in cells if cell['index'] in expected]
    by_condition = {}
    for row in rows:
        if not row['near_entry_qualified']:
            continue
        prior = by_condition.setdefault(row['condition_id'], row)
        if prior['near_entry_state_sha256'] != row['near_entry_state_sha256'] or prior['scene_xml_sha256'] != row['scene_xml_sha256']:
            raise ValueError('Expert/policy trajectories do not share the same physical near-entry state and scene')
    return dict(schema='p56.collection-catalog.v1', manifest_sha256=sha(manifest_path),
                collection_binding=parent, cells=rows,
                eligible_counts={split: sum(r['eligible'] and r['split'] == split for r in rows)
                                 for split in ('train', 'validation', 'test')},
                inclusion='all qualified T>=1 teacher and policy trajectories, regardless of success')


def pack(manifest_path, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown data view')
    report = catalog(manifest_path)
    episodes = []
    for row in report['cells']:
        if not row['eligible'] or (variant == 'teacher_only' and row['source_policy'] != 'expert'):
            continue
        root = Path(row['directory'])
        features = {key: [] for key in FEATURE_SHAPES}
        for record in row['frames']:
            cache = torch.load(record['feature'], map_location='cpu', weights_only=True)
            if set(cache) != set(FEATURE_SHAPES):
                raise ValueError('Unexpected public feature keys')
            for key, shape in FEATURE_SHAPES.items():
                tensor = cache[key]
                if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                        or tuple(tensor.shape) != shape or not torch.isfinite(tensor).all()):
                    raise ValueError(f'Invalid public feature: {key}')
                features[key].append(tensor)
        transitions = lines(root / 'public/TRANSITIONS.jsonl')
        labels = lines(root / 'private/LABELS.jsonl')
        episodes.append(dict(episode_id=f"cell_{row['cell_index']:04d}", condition_id=row['condition_id'],
                             geometry_group=row['geometry_group'], scale_xyz=row['scale_xyz'],
                             condition_seed=row['condition_seed'], split=row['split'],
                             source_policy=row['source_policy'], source_seal_sha256=row['source_seal_sha256'],
                             features={key: torch.stack(value) for key, value in features.items()},
                             executed_action=torch.tensor([r['executed_action'][:6] for r in transitions], dtype=torch.float32),
                             expert_action=torch.tensor([r['expert_action'][:6] for r in labels[:-1]], dtype=torch.float32)))
    bundle = dict(schema='p56_short_insert_bundle_v1', variant=variant, stage_scope='near_contact_insert',
                  permitted_geometry_overlap=['1,1,1'],
                  episodes=episodes, catalog=report)
    validate_bundle(bundle)
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = pack(args.manifest, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, args.output)
    print(json.dumps(dict(variant=args.variant, episodes=len(value['episodes']), output=str(args.output))))


if __name__ == '__main__':
    main()
