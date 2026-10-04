#!/usr/bin/env python3
"""P64 actual LIBERO s9: 12 opened conditions, fixed/online, H24.

No physics in prepare. Restore exact old P56 entry states at their original
asset scale. Position conditions translate the actual caddy, never only its
goal site. Old and shifted entry failures remain in the full denominator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'prove.p64-book-evaluation.v1'
SOURCE_INDICES = (62, 63, 64, 65, 66, 67, 70, 71)
ARMS = ('fixed', 'online')
PUBLIC_KEYS = ('rgb', 'visible_masks', 'camera_K', 'camera_pose', 'eef_pos',
               'eef_quat_xyzw', 'gripper_qpos', 'joint_pos', 'joint_vel')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def check_ref(value):
    if sha(value['path']) != value['sha256']:
        raise ValueError('Frozen file changed: '+value['path'])


def conditions():
    values = []
    for index in (62, 63, 64, 65):
        values.append(dict(id=f'id_{index}', group='id', source_index=index,
                           caddy_shift_world_m=[0., 0., 0.], scale_xyz=[1., 1., 1.]))
    for index in (66, 67, 70, 71):
        values.append(dict(id=f'size_{index}', group='size', source_index=index,
                           caddy_shift_world_m=[0., 0., 0.],
                           scale_xyz=[.90]*3 if index < 70 else [.78, 1., 1.]))
    shifts = ([.004, 0., 0.], [-.004, 0., 0.], [0., .004, 0.], [0., -.004, 0.])
    for index, shift in zip((62, 63, 64, 65), shifts):
        values.append(dict(id=f'position_{index}', group='position_shift', source_index=index,
                           caddy_shift_world_m=list(shift), scale_xyz=[1., 1., 1.]))
    return values


def cells():
    return [dict(index=i*2+j, condition=i, arm=arm)
            for i in range(12) for j, arm in enumerate(ARMS)]


def relocated(value, mapping):
    """Pure declared path substitution; preserve original archived JSON bytes."""
    if isinstance(value, dict):
        return {k: relocated(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [relocated(v, mapping) for v in value]
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            value = value.replace(old, new)
    return value


def verify_entry(entry, collection, legacy):
    """Read only sealed state metadata, never teacher LABELS or micro outcomes."""
    folder = Path(entry['directory'])
    original = next(x for x in legacy['entries'] if x['cell_index'] == entry['cell_index'])
    cell = collection['cells'][entry['cell_index']]
    if cell['index'] not in SOURCE_INDICES or cell['split'] != 'test' or cell['source_policy'] != 'entry_only':
        raise ValueError('Not a predetermined original P56 test entry')
    if read(folder/'CELL.json') != cell or read(folder/'BINDING.json') != collection:
        raise ValueError('P56 entry identity or collection binding changed')
    for name, key in [('RESULT.json', 'result_sha256'), ('SEAL.json', 'seal_sha256'), ('BINDING.json', 'binding_sha256')]:
        if sha(folder/name) != original[key]:
            raise ValueError('P56 entry receipt changed: '+name)
    result, seal = read(folder/'RESULT.json'), read(folder/'SEAL.json')
    if (result['seal_sha256'] != sha(folder/'SEAL.json')
            or seal['binding_sha256'] != original['binding_sha256']
            or result['condition_id'] != cell['condition_id'] or result['actions'] != 0
            or result['source_policy'] != 'entry_only'
            or result['near_entry_qualified'] != original['near_entry_qualified']):
        raise ValueError('P56 entry result/seal chain changed')
    # Preserve the original pre-entry model-freeze receipt verbatim.
    if seal.get('model_freeze') != legacy['model_freeze']:
        raise ValueError('P56 original model freeze chain changed')
    for name in ('EXECUTION.json', 'SUPERVISOR_CALIBRATION.json'):
        if name not in seal['files'] or sha(folder/name) != seal['files'][name]:
            raise ValueError('Unsealed entry calibration/execution')
    if not result['near_entry_qualified']:
        return cell, None
    near = read(folder/'NEAR_ENTRY.json')
    if (near.get('schema') != 'prove.p56-near-entry.v1' or near['source_cell_index'] != cell['index']
            or near['scale_xyz'] != cell['scale_xyz'] or near['frame'] != 'private/near_entry_state.npz'):
        raise ValueError('Entry state identity or original scale changed')
    for name in ('NEAR_ENTRY.json', near['frame']):
        if sha(folder/name) != seal['files'].get(name):
            raise ValueError('Unsealed source near-entry state')
    with np.load(folder/near['frame'], allow_pickle=False) as arrays:
        if 'supervisor_ctrl' not in arrays or hashlib.sha256(arrays['supervisor_sim_state'].tobytes()).hexdigest() != near['state_sha256']:
            raise ValueError('Entry simulator state content changed')
    return cell, near


def prepare(*, legacy_binding, collection_binding, source_scene, geometry_binding, graph_binding, relocation,
            entries_root, encoder_checkpoint, action_checkpoint, a2_head, priors,
            dino_binding, plan):
    """Freeze concrete assets/source states; root review remains a separate step.

    source_scene is the original P53 FULL_TASK_FREEZE, not a rewritten scene.
    relocation JSON maps old absolute roots/files to their local byte copies.
    dino_binding is a JSON map of every DINO source/config/weight file to hash.
    """
    legacy, collection = read(legacy_binding), read(collection_binding)
    if sha(collection_binding) != legacy['collection_binding']['sha256']:
        raise ValueError('Legacy collection digest differs')
    if (sha(graph_binding) != collection['p55_binding']['sha256']
            or sha(geometry_binding) != read(graph_binding)['parent']['sha256']
            or sha(source_scene) != read(geometry_binding)['parent']['sha256']):
        raise ValueError('Original P56/P55/P54/P53 source scene chain changed')
    scene = read(source_scene)
    if scene.get('schema') != 'prove.p53-full-original-task.v1':
        raise ValueError('Exact original P53 scene binding required')
    mapping = read(relocation)
    entries = []
    for index in SOURCE_INDICES:
        original = next(x for x in legacy['entries'] if x['cell_index'] == index)
        relative = Path(original['directory']).relative_to(Path(legacy['collection_binding']['path']).parent)
        entry = dict(cell_index=index, directory=str((Path(entries_root)/relative).resolve()))
        source_cell, _ = verify_entry(entry, collection, legacy)
        expected = next(x for x in conditions() if x['source_index'] == index)['scale_xyz']
        if source_cell['scale_xyz'] != expected:
            raise ValueError('Declared source scale differs from old entry')
        entries.append(entry)
    local_scene = relocated(scene, mapping)
    assets = {}
    for element in ET.fromstring(local_scene['environment']['relocated_xml']).iter():
        file = element.attrib.get('file')
        if file:
            if not Path(file).is_absolute():
                raise ValueError('Scene asset path must resolve absolutely: '+file)
            assets[str(Path(file).resolve())] = sha(file)
    for old_path, expected in scene['environment']['assets'].items():
        local_path = relocated(old_path, mapping)
        if sha(local_path) != expected:
            raise ValueError('Relocated asset differs from original bytes: '+local_path)
        assets[str(Path(local_path).resolve())] = expected
    bddl = local_scene['environment']['bddl']
    if sha(bddl) != scene['environment']['bddl_sha256']:
        raise ValueError('Original LIBERO task BDDL changed')
    assets[str(Path(bddl).resolve())] = sha(bddl)
    calibration = local_scene['prefix']['calibration']
    assets[str(Path(calibration).resolve())] = sha(calibration)
    from tools.task_graph.p52_deployment import source_closure
    sources = {str((ROOT/name).resolve()): sha(ROOT/name) for name in source_closure(ROOT, [
        'tools/task_graph/p64_book_evaluate.py', 'prove/fast_adapter/p64_book_runtime.py',
        'prove/fast_adapter/p64_book_bridge.py', 'prove/fast_adapter/p64_common.py',
        'tools/task_graph/p56_collection.py'])}
    binding = dict(schema=SCHEMA, conditions=conditions(), cells=cells(), arms=list(ARMS),
                   max_actions=24, automatic_retries=0, entries=entries, sources=sources,
                   assets=assets, public_keys=list(PUBLIC_KEYS),
                   position_interpretation='caddy-only position-shift / entry robustness; not certified beyond train support',
                   evidence_scope='opened P56 development entries; local s9, not full Task Graph',
                   denominator='all 12 conditions per arm including original and shifted entry failures')
    for name, path in dict(legacy_binding=legacy_binding, collection_binding=collection_binding,
                           source_scene=source_scene, geometry_binding=geometry_binding, graph_binding=graph_binding,
                           relocation=relocation, encoder_checkpoint=encoder_checkpoint,
                           action_checkpoint=action_checkpoint, a2_head=a2_head, priors=priors,
                           dino_binding=dino_binding, plan=plan).items():
        binding[name] = ref(path)
    verify(binding)
    return binding


def verify(binding):
    if (binding.get('schema') != SCHEMA or binding.get('conditions') != conditions()
            or binding.get('cells') != cells() or binding.get('arms') != list(ARMS)
            or binding.get('max_actions') != 24 or binding.get('automatic_retries') != 0
            or binding.get('public_keys') != list(PUBLIC_KEYS)):
        raise ValueError('Changed P64 12-condition / 24-cell H24 scope')
    for name in ('legacy_binding', 'collection_binding', 'source_scene', 'geometry_binding', 'graph_binding', 'relocation',
                 'encoder_checkpoint', 'action_checkpoint', 'a2_head', 'priors', 'dino_binding', 'plan'):
        check_ref(binding[name])
    dino = read(binding['dino_binding']['path'])
    dino_files = {str(Path(path).resolve()): digest for path, digest in dino['files'].items()}
    for name in ('model.safetensors', 'config.json', 'preprocessor_config.json'):
        if str((Path(dino['model_dir'])/name).resolve()) not in dino_files:
            raise ValueError('DINO runtime model files absent from freeze')
    for path, digest in {**binding['sources'], **binding['assets'], **dino_files}.items():
        if sha(path) != digest:
            raise ValueError('Frozen source/asset changed: '+path)
    legacy = read(binding['legacy_binding']['path'])
    collection = read(binding['collection_binding']['path'])
    if sha(binding['collection_binding']['path']) != legacy['collection_binding']['sha256']:
        raise ValueError('Original collection hash changed')
    if (binding['graph_binding']['sha256'] != collection['p55_binding']['sha256']
            or binding['geometry_binding']['sha256'] != read(binding['graph_binding']['path'])['parent']['sha256']
            or binding['source_scene']['sha256'] != read(binding['geometry_binding']['path'])['parent']['sha256']):
        raise ValueError('Original source scene chain changed')
    if [x['cell_index'] for x in binding['entries']] != list(SOURCE_INDICES):
        raise ValueError('Changed source entry denominator')
    for entry in binding['entries']:
        verify_entry(entry, collection, legacy)
    return collection


def verify_review(binding_path):
    review = read(Path(binding_path).parent/'ROOT_EVALUATION_REVIEW.json')
    if (review.get('approved') is not True or review.get('reviewer') != 'root'
            or review.get('binding_sha256') != sha(binding_path)):
        raise ValueError('Exact root evaluation review required before physics')


def shift_caddy(env, delta):
    """Move the physical caddy body; preserve book, all rotations and qvel."""
    delta = np.asarray(delta, float)
    if delta.shape != (3,) or not np.isfinite(delta).all() or delta[2] != 0 or np.linalg.norm(delta) > .0040000001:
        raise ValueError('Only predeclared caddy XY translation allowed')
    if not delta.any():
        return
    model, data = env.sim.model, env.sim.data
    body = model.body_name2id('desk_caddy_1_main')
    site = model.site_name2id('desk_caddy_1_back_contain_region')
    book = model.body_name2id('black_book_1_main')
    joint = int(model.body_jntadr[body])
    count = int(model.body_jntnum[body])
    if count == 0 and int(model.body_parentid[body]) == 0:
        address = None  # Original LIBERO caddy is fixed directly to world.
    elif count == 1 and joint >= 0 and int(model.jnt_type[joint]) == 0:
        address = int(model.jnt_qposadr[joint])
    else:
        raise ValueError('Caddy must be world-fixed or free-jointed; site-only mutation forbidden')
    before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
    before_site, before_book = data.site_xpos[site].copy(), data.body_xpos[book].copy()
    if address is None:
        model.body_pos[body] += delta
    else:
        data.qpos[address:address+3] += delta
    env.sim.forward()
    expected = before_qpos.copy()
    if address is not None:
        expected[address:address+3] += delta
    if (not np.array_equal(data.qpos, expected) or not np.array_equal(data.qvel, before_qvel)
            or not np.allclose(data.site_xpos[site], before_site+delta, atol=1e-10, rtol=0)
            or not np.allclose(data.body_xpos[book], before_book, atol=1e-10, rtol=0)):
        raise ValueError('Caddy translation changed unintended physical state')
    env._post_process()
    env._update_observables(force=True)


def _append(path, row):
    def convert(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if hasattr(value, 'detach'): return value.detach().cpu().tolist()
        raise TypeError(type(value).__name__)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, default=convert, allow_nan=False)+'\n')


def _seal(output, binding_path, cell, steps):
    files = {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
             if p.is_file() and p.name not in ('SEAL.json', 'RESULT.json', 'ABORT.json')}
    dump(output/'SEAL.json', dict(schema='prove.p64-book-seal.v1', files=files,
                                 binding_sha256=sha(binding_path), cell=cell, executed_physical_steps=steps))


def run(binding_path, index, output):
    binding = read(binding_path)
    collection_binding = verify(binding)
    verify_review(binding_path)
    if type(index) is not int or not 0 <= index < 24:
        raise ValueError('Outside exact P64 24-cell queue')
    cell = binding['cells'][index]
    condition = binding['conditions'][cell['condition']]
    entry = next(x for x in binding['entries'] if x['cell_index'] == condition['source_index'])
    source_cell, near = verify_entry(entry, collection_binding, read(binding['legacy_binding']['path']))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('public', 'private'):
        (output/name).mkdir()
    dump(output/'BINDING.json', binding)
    dump(output/'CELL.json', cell)
    dump(output/'EXECUTION.json', dict(started_unix=time.time(), automatic_retry=False))
    if near is None:
        _seal(output, binding_path, cell, 0)
        result = dict(cell=cell, condition=condition, near_entry_qualified=False,
                      system_success=False, native_hold3=False, executed_physical_steps=0,
                      reason='original_entry_failed', full_task_success_evidence=False,
                      seal_sha256=sha(output/'SEAL.json'))
        dump(output/'RESULT.json', result)
        return result
    import torch
    from tools.task_graph import p56_collection as collection
    from tools.task_graph.p52_full_task import InsertJudge, structural_measurement
    from tools.task_graph.collect_libero_p52_v02 import private_measurement
    from tools.task_graph.p54_full_task import near_gate
    from tools.task_graph.collect_libero_demo_gt import snapshot
    from prove.fast_adapter.p64_common import load_encoder, _state_hash
    from prove.fast_adapter.p64_book_runtime import BookRuntime
    from prove.fast_adapter.p64_book_bridge import BookPriors
    from prove.task_graph.p57_geometry_policy import load_geometry_policy
    env = renderer = runtime = None
    steps = 0
    try:
        scene = read(binding['source_scene']['path'])
        mapping = read(binding['relocation']['path'])
        # Generate original scaled XML first, verifying its old scene identity,
        # then replace only declared asset paths for local execution.
        scene['geometry_properties'] = read(binding['geometry_binding']['path'])['geometry']['properties']
        scene['prefix']['calibration'] = relocated(scene['prefix']['calibration'], mapping)
        child, geometry, props = collection.build_child(collection_binding, source_cell, output, scene)
        if hashlib.sha256(child['environment']['relocated_xml'].encode()).hexdigest() != near['scene_xml_sha256']:
            raise ValueError('Scaled source scene differs from sealed entry')
        if sha(output/'SUPERVISOR_CALIBRATION.json') != sha(Path(entry['directory'])/'SUPERVISOR_CALIBRATION.json'):
            raise ValueError('Restored evaluator calibration differs from sealed source')
        child = relocated(child, mapping)
        env, _ = collection.initialize_child(child, output, geometry['axes'], props['halfsize'],
                                             near_entry=Path(entry['directory'])/near['frame'])
        shift_caddy(env, condition['caddy_shift_world_m'])
        names = collection._names(env)
        obs = env._get_observations(force_update=True)
        gate = near_gate(child['relation'], private_measurement(env, obs, names, child['relation']),
                         dict(near_position_tolerance=.003585, near_gap_max=.00717))
        dump(output/'private/ENTRY_GATE.json', gate)
        arrays, _ = snapshot(env, obs)
        np.savez_compressed(output/'private/INITIAL_STATE.npz',
                            **{k:v for k,v in arrays.items() if k.startswith('supervisor_')})
        qualified = bool(gate['near_ready'])
        reason = 'shifted_or_restored_entry_failed'
        strict = native_ever = native_instant = False
        if qualified:
            device = read(binding['dino_binding']['path']).get('device', 'cuda')
            encoder = load_encoder(binding['encoder_checkpoint']['path'], device=device,
                                   expected_sha256=binding['encoder_checkpoint']['sha256'])
            policy, _, _ = load_geometry_policy(binding['action_checkpoint']['path'], binding['a2_head']['path'], device=device)
            priors = BookPriors(**read(binding['priors']['path'])['values'])
            from prove.fast_adapter.visual_alignment import FrozenDinoROI
            dino = read(binding['dino_binding']['path'])
            vision = FrozenDinoROI(dino['model_dir'], device=device, microbatch=2)
            runtime = BookRuntime(encoder, policy, priors,
                                  'continuous' if cell['arm'] == 'online' else 'fixed',
                                  vision=vision, device=device)
            before = (_state_hash(encoder), _state_hash(policy))
            renderer = collection.PublicRenderer(env, child['public'])
            judge, hold = InsertJudge(), 0
            while True:
                public = renderer.observe(obs)
                public = {key: public[key] for key in PUBLIC_KEYS}
                frame = output/'public'/f'frame_{steps:04d}.npz'
                np.savez_compressed(frame, **public)
                action, telemetry = runtime.step(public)
                action = np.asarray(action, float)
                if action.shape != (7,) or not np.isfinite(action).all() or action[6] != 1:
                    raise ValueError('Invalid frozen neural OSC command')
                # Prediction committed before privileged structural evaluation.
                _append(output/'public/PREDICTIONS.jsonl', dict(t=steps, observation_sha256=sha(frame),
                                                               action=action, telemetry=telemetry))
                official, fingers, wrong = structural_measurement(env, names)
                judgment = judge.observe(official, fingers, wrong)
                hold = hold+1 if official else 0
                native_ever = native_ever or hold >= 3
                native_instant = native_instant or official
                reason = judgment['stop'] or ('micro_budget24' if steps >= 24 else None)
                _append(output/'private/JUDGMENTS.jsonl', dict(t=steps, official=official, fingers=fingers,
                                                              wrong_target=wrong, **judgment))
                arrays, _ = snapshot(env, obs)
                np.savez_compressed(output/'private'/f'frame_{steps:04d}.npz',
                                    **{k:v for k,v in arrays.items() if k.startswith('supervisor_')})
                _append(output/'public/EXECUTED.jsonl', dict(t=steps, executed=reason is None,
                                                            action=action if reason is None else None))
                if reason:
                    strict = bool(judgment['complete'])
                    break
                obs, _, _, _ = env.step(action.copy())
                steps += 1
            after = (_state_hash(encoder), _state_hash(policy))
            dump(output/'public/WEIGHT_AUDIT.json', dict(
                schema='prove.p64-book-weight-audit.v1',
                encoder_before=before[0], encoder_after=after[0],
                policy_before=before[1], policy_after=after[1], unchanged=before == after))
            if before != after:
                raise RuntimeError('Frozen evaluation weights changed')
        _seal(output, binding_path, cell, steps)
        result = dict(cell=cell, condition=condition, near_entry_qualified=qualified,
                      system_success=strict, native_hold3=native_ever, native_instant_ever=native_instant,
                      executed_physical_steps=steps, reason=reason, full_task_success_evidence=False,
                      seal_sha256=sha(output/'SEAL.json'))
        dump(output/'RESULT.json', result)
        return result
    except Exception as exc:
        dump(output/'ABORT.json', dict(cell=cell, reason=str(exc), executed_physical_steps=steps,
                                      automatic_retry=False))
        raise
    finally:
        if runtime is not None: runtime.reset()
        if renderer is not None: renderer.close()
        if env is not None: env.close()


def queue(binding_path, output):
    """Run each frozen cell once; stop at first process failure, never retry."""
    verify(read(binding_path))
    verify_review(binding_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for index in range(24):
        result = subprocess.run([sys.executable, '-m', 'tools.task_graph.p64_book_evaluate', 'run',
                                 '--binding', str(Path(binding_path).resolve()), '--index', str(index),
                                 '--output', str((output/f'cell_{index:03d}').resolve())], cwd=ROOT)
        _append(output/'QUEUE.jsonl', dict(index=index, returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(f'P64 queue stopped at cell {index}; no automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--inputs', type=Path, required=True, help='JSON kwargs for prepare')
    prep.add_argument('--output', type=Path, required=True)
    for name in ('run', 'queue'):
        command = commands.add_parser(name)
        command.add_argument('--binding', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'run': command.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.output.exists(): raise FileExistsError(args.output)
        dump(args.output, prepare(**read(args.inputs)))
    elif args.command == 'run':
        print(json.dumps(run(args.binding, args.index, args.output)))
    else:
        queue(args.binding, args.output)


if __name__ == '__main__':
    main()
