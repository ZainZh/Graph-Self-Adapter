#!/usr/bin/env python3
"""P56 prepared 16-entry × 9-arm paired local s9 evaluation.

Only a sealed near-entry simulator state is read from private collection data.
Teacher labels are never opened. Policy and z feedback use public observations
and actions actually executed in the current evaluation cell.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from prove.task_graph.p52_stage_policy import Config, StagePolicy, StageRuntime
from tools.task_graph.collect_libero_demo_gt import dump
from tools.task_graph.p52_full_task import InsertJudge, structural_measurement
from tools.task_graph.p52_policy_pilot import Vision, validate_features
from tools.task_graph.p53_train_short import sha
from tools.task_graph.p52_deployment import source_closure
from tools.task_graph import p56_collection as collection


ARMS = ('p53_bc', 'p53_fixed', 'p53_online',
        'teacher_only_bc', 'teacher_only_fixed', 'teacher_only_online',
        'mixed_bc', 'mixed_fixed', 'mixed_online')
CHECKPOINTS = ('p53_bc', 'p53_joint', 'teacher_only_bc', 'teacher_only_joint', 'mixed_bc', 'mixed_joint')
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ARM = dict(p53_bc='p53_bc', p53_joint='p53_online',
                      teacher_only_bc='teacher_only_bc', teacher_only_joint='teacher_only_online',
                      mixed_bc='mixed_bc', mixed_joint='mixed_online')


def read(path):
    return json.loads(Path(path).read_text())


def checkpoint_for_arm(arm):
    prefix, mode = arm.rsplit('_', 1)
    return prefix + ('_bc' if mode == 'bc' else '_joint')


def load_skill(path, arm):
    package = torch.load(path, map_location='cpu', weights_only=True)
    expected = 'p53_short_insert_skill_v01' if arm.startswith('p53_') else 'p56_short_insert_skill_v1'
    if package.get('schema') != expected:
        raise ValueError('Wrong P53/P56 checkpoint schema')
    if package.get('config') != vars(Config()) or package.get('horizon') != 24:
        raise ValueError('Changed short policy architecture or horizon')
    branch = 'bc' if arm.endswith('_bc') else 'joint'
    if package.get('branch') != branch:
        raise ValueError('Checkpoint branch differs from arm')
    if expected == 'p56_short_insert_skill_v1':
        variant = 'teacher_only' if arm.startswith('teacher_only_') else 'teacher_plus_policy_feedback'
        if (package.get('variant') != variant or package.get('total_outer_updates') != 3000
                or package.get('seed') != 11 or package.get('common_outer_updates') != 1000
                or package.get('branch_outer_updates') != 2000):
            raise ValueError('P56 checkpoint data view or training budget mismatch')
    model = StagePolicy(Config(**package['config']))
    model.load_state_dict(package['model'], strict=True)
    return model, package


def verify_entry(entry, collection_binding, model_freeze):
    root = Path(entry['directory']).resolve()
    cell = collection_binding['cells'][entry['cell_index']]
    if cell['split'] != 'test' or cell['source_policy'] != 'entry_only':
        raise ValueError('Evaluation must retain predeclared test entry-only conditions')
    if read(root / 'CELL.json') != cell:
        raise ValueError('Evaluation cell identity differs')
    if read(root / 'BINDING.json') != collection_binding:
        raise ValueError('Evaluation entry uses another collection binding')
    if sha(root / 'BINDING.json') != entry['binding_sha256']:
        raise ValueError('Collection binding hash differs')
    result = read(root / 'RESULT.json')
    seal = read(root / 'SEAL.json')
    if sha(root / 'RESULT.json') != entry['result_sha256'] or sha(root / 'SEAL.json') != entry['seal_sha256']:
        raise ValueError('Collection entry result/seal changed')
    if result['seal_sha256'] != entry['seal_sha256'] or seal['binding_sha256'] != entry['binding_sha256']:
        raise ValueError('Entry result/seal/binding chain mismatch')
    freeze_ref = dict(path=str(Path(model_freeze['path']).resolve()), sha256=model_freeze['sha256'])
    if seal.get('model_freeze') != freeze_ref or seal.get('run_started_unix', -1) < read(freeze_ref['path'])['created_unix']:
        raise ValueError('Test entry generated before or without model freeze')
    execution = read(root / 'EXECUTION.json')
    if seal['files'].get('EXECUTION.json') != sha(root / 'EXECUTION.json'):
        raise ValueError('Unsealed test entry execution receipt')
    if execution.get('model_freeze') != freeze_ref or execution.get('run_started_unix') != seal['run_started_unix']:
        raise ValueError('Test entry execution/freeze chain mismatch')
    if result['cell_index'] != cell['index'] or result['condition_id'] != cell['condition_id']:
        raise ValueError('Entry result identity differs')
    if result['source_policy'] != 'entry_only' or result['actions'] != 0:
        raise ValueError('Evaluation source executed a training trajectory')
    if bool(result['near_entry_qualified']) != entry['near_entry_qualified']:
        raise ValueError('Near-entry qualification changed')
    if not entry['near_entry_qualified']:
        return
    near = read(root / 'NEAR_ENTRY.json')
    if near.get('schema') != collection.ENTRY_SCHEMA or near['source_cell_index'] != cell['index']:
        raise ValueError('Near-entry identity differs')
    if (near['condition_id'] != cell['condition_id'] or near['geometry_id'] != cell['geometry_id']
            or near['scale_xyz'] != cell['scale_xyz']):
        raise ValueError('Near-entry geometry differs')
    if near['frame'] != 'private/near_entry_state.npz':
        raise ValueError('Unexpected near-entry restore path')
    path = root / near['frame']
    if sha(path) != near['frame_sha256'] or seal['files'].get(near['frame']) != near['frame_sha256']:
        raise ValueError('Unsealed near-entry simulator state')
    if seal['files'].get('NEAR_ENTRY.json') != sha(root / 'NEAR_ENTRY.json'):
        raise ValueError('Unsealed near-entry metadata')
    with np.load(path, allow_pickle=False) as state:
        if not all(key in state for key in ('supervisor_sim_state', 'supervisor_ctrl')):
            raise ValueError('Full physical state and control required')
        if hashlib.sha256(state['supervisor_sim_state'].tobytes()).hexdigest() != near['state_sha256']:
            raise ValueError('Near-entry physical state content differs')


def prepare(manifest_path, checkpoints, model_freeze_path):
    manifest = read(manifest_path)
    parent = manifest['collection_binding']
    if sha(parent['path']) != parent['sha256']:
        raise ValueError('Collection binding changed')
    binding = read(parent['path'])
    collection.verify(binding)
    freeze_ref = dict(path=str(model_freeze_path.resolve()), sha256=sha(model_freeze_path))
    freeze = read(model_freeze_path)
    if freeze.get('schema') != 'prove.p56-model-freeze.v1' or freeze.get('test_entry_generation_authorized_after_freeze') is not True:
        raise ValueError('Result-free model freeze before test-entry generation required')
    if not isinstance(freeze.get('created_unix'), (int, float)):
        raise ValueError('Model freeze creation time required')
    if freeze.get('collection_binding_sha256') != parent['sha256']:
        raise ValueError('Model freeze differs from collection scope')
    expected = [c for c in binding['cells'] if c['split'] == 'test' and c['source_policy'] == 'entry_only']
    if len(expected) != 16:
        raise ValueError('Expected 16 predeclared test entries')
    source = {e['cell_index']: e for e in manifest['episodes']}
    if len(source) != len(manifest['episodes']) or set(source) != {c['index'] for c in expected}:
        raise ValueError('Retain all 16 test entry-only results, including failures')
    entries = []
    for cell in expected:
        root = Path(source[cell['index']]['directory']).resolve()
        result = read(root / 'RESULT.json')
        entries.append(dict(cell_index=cell['index'], directory=str(root),
                            binding_sha256=sha(root / 'BINDING.json'),
                            result_sha256=sha(root / 'RESULT.json'), seal_sha256=sha(root / 'SEAL.json'),
                            near_entry_qualified=bool(result['near_entry_qualified'])))
        verify_entry(entries[-1], binding, freeze_ref)
    if set(checkpoints) != set(CHECKPOINTS):
        raise ValueError('Six final checkpoints required for nine arms')
    weights = {name: dict(path=str(Path(path).resolve()), sha256=sha(path)) for name, path in checkpoints.items()}
    packages = {name: load_skill(item['path'], CHECKPOINT_ARM[name])[1]
                for name, item in weights.items()}
    if freeze.get('collection_binding_sha256') != parent['sha256'] or freeze.get('checkpoints') != weights:
        raise ValueError('Model freeze does not bind these checkpoints and collection scope')
    for stem in ('teacher_only', 'mixed'):
        bc, joint = packages[stem + '_bc'], packages[stem + '_joint']
        if bc['common_warmup_sha256'] != joint['common_warmup_sha256'] or bc['data_sha256'] != joint['data_sha256']:
            raise ValueError('BC and joint do not share exact P56 warmup/data')
        if bc['plan_sha256'] != joint['plan_sha256']:
            raise ValueError('BC and joint use different P56 training plans')
    if packages['teacher_only_bc']['plan_sha256'] != packages['mixed_bc']['plan_sha256']:
        raise ValueError('P56 data views use different frozen training plans')
    old_bc, old_joint = packages['p53_bc'], packages['p53_joint']
    if old_bc['common_warmup_sha256'] != old_joint['common_warmup_sha256'] or old_bc['data_sha256'] != old_joint['data_sha256']:
        raise ValueError('P53 historical BC and joint checkpoint pair differs')
    cells = [dict(index=i * len(ARMS) + j, entry=i, arm=arm)
             for i in range(len(entries)) for j, arm in enumerate(ARMS)]
    return dict(schema='p56.micro-eval.v1', collection_binding=parent,
                manifest=dict(path=str(manifest_path.resolve()), sha256=sha(manifest_path)),
                model_freeze=freeze_ref,
                entries=entries, checkpoints=weights, cells=cells, max_actions=24,
                arms=list(ARMS), automatic_retries=0,
                sources={path: sha(ROOT / path) for path in source_closure(ROOT,
                    ['tools/task_graph/p56_evaluate.py', 'tools/task_graph/p52_vision_worker.py'])},
                primary_success='official native goal AND at least one finger touching book jointly hold3',
                denominator='all 16 predeclared test entries for every arm; nonentries retained as system failures')


def verify(binding):
    if binding.get('schema') != 'p56.micro-eval.v1' or binding.get('max_actions') != 24:
        raise ValueError('Wrong P56 micro evaluation scope')
    if binding.get('arms') != list(ARMS) or len(binding.get('entries', [])) != 16:
        raise ValueError('Changed nine-arm or 16-entry denominator')
    cells = [dict(index=i * len(ARMS) + j, entry=i, arm=arm)
             for i in range(16) for j, arm in enumerate(ARMS)]
    if binding.get('cells') != cells:
        raise ValueError('Changed paired evaluation cells')
    for key in ('collection_binding', 'manifest', 'model_freeze'):
        if sha(binding[key]['path']) != binding[key]['sha256']:
            raise ValueError('Frozen evaluation source changed')
    for path, digest in binding['sources'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Frozen evaluator source changed')
    freeze = read(binding['model_freeze']['path'])
    if freeze.get('schema') != 'prove.p56-model-freeze.v1' or freeze.get('checkpoints') != binding['checkpoints']:
        raise ValueError('Model freeze differs from evaluation binding')
    if freeze.get('collection_binding_sha256') != binding['collection_binding']['sha256'] or freeze.get('test_entry_generation_authorized_after_freeze') is not True:
        raise ValueError('Missing pre-entry model freeze declaration')
    collection_binding = read(binding['collection_binding']['path'])
    collection.verify(collection_binding)
    for entry in binding['entries']:
        verify_entry(entry, collection_binding, binding['model_freeze'])
    for checkpoint in binding['checkpoints'].values():
        if sha(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Frozen checkpoint changed')
    return collection_binding


def _append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


def run(binding_path, index, output):
    binding = read(binding_path)
    collection_binding = verify(binding)
    if not 0 <= index < 144:
        raise ValueError('Cell outside 16×9 evaluation scope')
    cell = binding['cells'][index]
    entry = binding['entries'][cell['entry']]
    source_cell = collection_binding['cells'][entry['cell_index']]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'BINDING.json', binding)
    dump(output / 'CELL.json', cell)
    if not entry['near_entry_qualified']:
        source_result = read(Path(entry['directory']) / 'RESULT.json')
        result = dict(cell=cell, condition_id=source_cell['condition_id'], near_entry_qualified=False,
                      micro_executed=False, system_success=False, micro_success=None,
                      official_instant_ever=bool(source_result.get('official_success')),
                      executed_physical_steps=0, z_updates=0, reason='blocked_by_original_near_entry_failure')
        dump(output / 'SEAL.json', dict(binding_sha256=sha(output / 'BINDING.json'), cell=cell,
                                        files={}, executed_physical_steps=0, z_updates=0,
                                        micro_executed=False))
        result['seal_sha256'] = sha(output / 'SEAL.json')
        dump(output / 'RESULT.json', result)
        return result
    for name in ('public', 'private'):
        (output / name).mkdir()
    checkpoint_key = checkpoint_for_arm(cell['arm'])
    model, _ = load_skill(binding['checkpoints'][checkpoint_key]['path'], cell['arm'])
    runtime = StageRuntime(model, online=cell['arm'].endswith('_online'))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    env = vision = renderer = None
    steps = 0
    try:
        _, _, original = collection.verify(collection_binding)
        child, geometry, props = collection.build_child(collection_binding, source_cell, output, original)
        near = read(Path(entry['directory']) / 'NEAR_ENTRY.json')
        scene_sha = hashlib.sha256(child['environment']['relocated_xml'].encode()).hexdigest()
        if scene_sha != near['scene_xml_sha256']:
            raise ValueError('Restored scene differs from sealed near-entry scene')
        env, _ = collection.initialize_child(child, output, geometry['axes'], props['halfsize'],
                                             near_entry=Path(entry['directory']) / near['frame'])
        vision = Vision(original['vision'])
        dump(output / 'VISUAL_BINDING.json', vision.binding)
        renderer = collection.PublicRenderer(env, original['public'])
        names = collection._names(env)
        judge = InsertJudge()
        obs = env._get_observations(force_update=True)
        previous = previous_action = None
        official_hold = 0
        official_ever = False
        official_instant_ever = False
        while True:
            frame = output / 'public' / f'frame_{steps:04d}.npz'
            np.savez_compressed(frame, **renderer.observe(obs))
            features = vision.encode(frame)
            validate_features(features, model.config)
            torch.save(features, output / 'public' / f'features_{steps:04d}.pt')
            if previous is None:
                runtime.enter(features)
            z_before = runtime.z.detach().clone()
            feedback_loss = None
            if previous is not None:
                feedback_loss = runtime.feedback(previous, previous_action, features)
            action = runtime.command(features).numpy()
            if action.shape != (7,) or not np.isfinite(action).all() or action[-1] != 1:
                raise ValueError('Invalid OSC7 command')
            official, fingers, wrong = structural_measurement(env, names)
            judgment = judge.observe(official, fingers, wrong)
            official_hold = official_hold + 1 if official else 0
            official_ever = official_ever or official_hold >= 3
            official_instant_ever = official_instant_ever or bool(official)
            stop = judgment['stop'] or ('micro_budget24' if steps >= 24 else None)
            _append(output / 'public/ACTIONS.jsonl', dict(t=steps, observation=frame.name,
                observation_sha256=sha(frame), z_before=z_before.tolist(), z_after=runtime.z.tolist(),
                z_delta_norm=float(torch.linalg.vector_norm(runtime.z - z_before)),
                feedback_loss=feedback_loss, z_updates=runtime.updates,
                proposed_action=action.tolist(), executed=stop is None,
                executed_physical_steps=steps))
            _append(output / 'private/JUDGMENTS.jsonl', dict(t=steps, official_success=official,
                official_hold=official_hold, finger_contacts=fingers, wrong_target=wrong, **judgment))
            if stop:
                break
            previous, previous_action = features, torch.tensor(action[:6], dtype=torch.float32)
            obs, _, _, _ = env.step(action.copy())
            steps += 1
        if any(not torch.equal(before[key], value) for key, value in model.state_dict().items()):
            raise RuntimeError('Deployment network weights mutated')
        verify(binding)
        files = {str(path.relative_to(output)): sha(path) for name in ('public', 'private')
                 for path in (output / name).iterdir()}
        dump(output / 'SEAL.json', dict(binding_sha256=sha(output / 'BINDING.json'),
                                        cell=cell, files=files, executed_physical_steps=steps,
                                        z_updates=runtime.updates))
        result = dict(cell=cell, condition_id=source_cell['condition_id'], near_entry_qualified=True,
                      micro_executed=True, system_success=bool(judgment['complete']),
                      micro_success=bool(judgment['complete']), official_only_hold3=official_ever,
                      official_success=official, official_instant_ever=official_instant_ever,
                      executed_physical_steps=steps,
                      z_updates=runtime.updates, reason=stop, seal_sha256=sha(output / 'SEAL.json'),
                      full_task_success_evidence=False)
        dump(output / 'RESULT.json', result)
        return result
    except Exception as exc:
        dump(output / 'ABORT.json', dict(cell=cell, reason=str(exc), executed_physical_steps=steps,
                                        automatic_retry=False))
        raise
    finally:
        runtime.exit()
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()
        if vision is not None:
            vision.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--manifest', type=Path, required=True)
    prep.add_argument('--model-freeze', type=Path, required=True)
    for key in CHECKPOINTS:
        prep.add_argument('--' + key.replace('_', '-'), type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    cell = sub.add_parser('run')
    cell.add_argument('--binding', type=Path, required=True)
    cell.add_argument('--index', type=int, required=True)
    cell.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'prepare':
        if args.output.exists():
            raise FileExistsError(args.output)
        checkpoints = {key: getattr(args, key) for key in CHECKPOINTS}
        dump(args.output, prepare(args.manifest, checkpoints, args.model_freeze))
    else:
        print(json.dumps(run(args.binding, args.index, args.output)))


if __name__ == '__main__':
    main()
