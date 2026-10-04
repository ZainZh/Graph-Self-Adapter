#!/usr/bin/env python3
"""P56 paired geometry and on-policy near-entry data collection.

Preparing a binding does no physics. Running a cell executes the real P55
prefix/Code and, when s9 is reached, records actual actions and isolated
offline teacher labels. Historical P53–P55 files are never written here.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from prove.task_graph.p52_stage_policy import Config, StagePolicy, StageRuntime
from prove.task_graph.p53_micro_insert_v03 import MicroTeacher
from tools.task_graph.collect_libero_demo_gt import dump, sha, snapshot, source_demo
from tools.task_graph.collect_libero_p52_v02 import PublicRenderer, private_measurement
from tools.task_graph.p52_full_task import InsertJudge, structural_measurement
from tools.task_graph.p52_policy_pilot import Vision, validate_features
from tools.task_graph.p53_full_task import weight_digest
from tools.task_graph.p54_book_geometry import BOOK_COLLISION, BOOK_JOINT
from tools.task_graph.p54_full_task import _mass_record, _names, run_prefix
from tools.task_graph.p55_relation_approach import run_code, verify as verify_p55
from tools.task_graph.p56_book_geometry import (bottom_shift, changed_xml,
    supervisor_calibration, validate_scale)

SCHEMA = 'prove.p56-collection-binding.v1'
ENTRY_SCHEMA = 'prove.p56-near-entry.v1'
MODES = ('expert', 'policy_fixed')
SPLITS = ('qualification', 'train', 'validation', 'test')


def _ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def _check_ref(ref):
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('Frozen referenced file changed: '+ref['path'])


def verify_model_freeze(path, binding_path, started):
    if path is None:
        raise ValueError('Test entry generation requires a sealed model-freeze receipt')
    ref = _ref(path)
    receipt = json.loads(Path(path).read_text())
    required = {'teacher_only_bc', 'teacher_only_joint', 'mixed_bc', 'mixed_joint',
        'p53_bc', 'p53_joint'}
    if (receipt.get('schema') != 'prove.p56-model-freeze.v1' or
            receipt.get('collection_binding_sha256') != sha(binding_path) or
            receipt.get('test_entry_generation_authorized_after_freeze') is not True or
            set(receipt.get('checkpoints', {})) != required or
            not isinstance(receipt.get('created_unix'), (int, float)) or
            not np.isfinite(receipt['created_unix']) or
            started < receipt['created_unix']):
        raise ValueError('Invalid or premature P56 model-freeze receipt')
    for checkpoint in receipt['checkpoints'].values():
        _check_ref(checkpoint)
    return ref


def _cells(conditions):
    cells = []
    for condition in conditions:
        for mode in condition['collection_modes']:
            cells.append(dict(index=len(cells), condition_id=condition['condition_id'],
                geometry_id=condition['geometry_id'], scale_xyz=condition['scale_xyz'],
                split=condition['split'],
                source_policy=mode, grasp_world_offset=condition['grasp_world_offset'],
                seed=condition['seed']))
    return cells


def validate_spec(spec):
    if spec.get('schema') != SCHEMA or spec.get('stage_scope') != 'real_prefix_code_near_entry_s9_h24':
        raise ValueError('Explicit P56 real-prefix/near-entry scope required')
    design_ref = spec.get('condition_design')
    if not isinstance(design_ref, dict):
        raise ValueError('P56 condition-design binding required')
    _check_ref(design_ref)
    design = json.loads(Path(design_ref['path']).read_text())
    if design.get('schema') != 'prove.p56-condition-design.v1':
        raise ValueError('Unexpected P56 condition design')
    expected = design['qualification'] + design['conditions']
    if spec.get('conditions') != expected:
        raise ValueError('Binding conditions differ from result-free design')
    if len(expected) != 50 or {s: sum(c['split'] == s for c in expected) for s in SPLITS} != dict(
            qualification=6, train=24, validation=4, test=16):
        raise ValueError('P56 condition cohort changed')
    groups = {name: (axes, split) for name, axes, split in design['geometry']}
    if len(groups) != 12:
        raise ValueError('Expected 12 predeclared geometry identities')
    ids = set()
    for row in expected:
        ident = row['condition_id']
        if not isinstance(ident, str) or not ident or ident in ids or '/' in ident:
            raise ValueError('Invalid or duplicate condition identity')
        ids.add(ident)
        axes, group_split = groups[row['geometry_id']]
        if axes != row['scale_xyz'] or (row['split'] != group_split and row['split'] != 'qualification'):
            raise ValueError('Condition geometry differs from design')
        scale = np.asarray(axes, dtype=float)
        kind = ('isotropic' if np.all(scale == scale[0]) else
                'thickness' if np.array_equal(scale[1:], np.ones(2)) else 'mixed')
        validate_scale(axes, kind)
        offset = np.asarray(row['grasp_world_offset'], dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('Invalid grasp offset')
        if not isinstance(row['seed'], int) or isinstance(row['seed'], bool):
            raise ValueError('Explicit integer seed required')
        required_modes = (['entry_only'] if row['split'] == 'test' else
            ['expert'] if row['split'] == 'qualification' else ['expert', 'policy_fixed'])
        if row['collection_modes'] != required_modes:
            raise ValueError('Collection modes differ from result-free design')
    if any(sum(c['split'] == 'train' and c['geometry_id'] == group for c in expected) != 4
            for group, (_, group_split) in groups.items() if group_split == 'train'):
        raise ValueError('Each train geometry needs four grasp conditions')
    if spec.get('micro_max_actions') != 24 or spec.get('code_max_actions') != 120 or spec.get('prefix_max_actions') != 400:
        raise ValueError('P53/P55 action budgets changed')
    if spec.get('automatic_retries') != 0 or spec.get('later_restores') != 0:
        raise ValueError('No retries or later state restores')
    if spec.get('cells') != _cells(expected):
        raise ValueError('Explicit paired cell identity changed')
    if len(spec['cells']) != 78:
        raise ValueError('Expected six qualification, 48 train, eight validation and 16 test entry cells')
    return spec


def prepare(design_path, p55_binding, p53_joint_checkpoint):
    design = json.loads(Path(design_path).read_text())
    if design.get('schema') != 'prove.p56-condition-design.v1':
        raise ValueError('Expected result-free P56 condition design')
    binding = dict(schema=SCHEMA, stage_scope='real_prefix_code_near_entry_s9_h24',
        condition_design=_ref(design_path), conditions=copy.deepcopy(
            design['qualification'] + design['conditions']),
        p55_binding=_ref(p55_binding), p53_joint_checkpoint=_ref(p53_joint_checkpoint),
        prefix_max_actions=400, code_max_actions=120, micro_max_actions=24,
        automatic_retries=0, later_restores=0)
    binding['cells'] = _cells(binding['conditions'])
    validate_spec(binding)
    p55 = json.loads(Path(binding['p55_binding']['path']).read_text())
    verify_p55(p55)
    binding['task_graph'] = copy.deepcopy(p55['task_graph'])
    ck = torch.load(binding['p53_joint_checkpoint']['path'], map_location='cpu', weights_only=True)
    if ck['schema'] != 'p53_short_insert_skill_v01':
        raise ValueError('Require frozen P53 joint stage checkpoint')
    model = StagePolicy(Config(**ck['config']))
    model.load_state_dict(ck['model'], strict=True)
    binding['sources'] = {str(Path(__file__).resolve()): sha(__file__),
        str(ROOT/'tools/task_graph/p56_book_geometry.py'): sha(ROOT/'tools/task_graph/p56_book_geometry.py')}
    binding['collection_rule'] = ('Real s1–s8a predecessor then actual s9; teacher labels are offline/private. '
        'All failures and nonentries remain in the cohort, with no replacement.')
    return binding


def verify(binding):
    validate_spec(binding)
    for key in ('p55_binding', 'p53_joint_checkpoint'):
        _check_ref(binding[key])
    for path, digest in binding['sources'].items():
        if sha(path) != digest:
            raise ValueError('P56 source changed: '+path)
    p55 = json.loads(Path(binding['p55_binding']['path']).read_text())
    parent, original = verify_p55(p55)
    if binding['task_graph'] != p55['task_graph']:
        raise ValueError('P55 real-prefix/Code task graph changed')
    if binding['p53_joint_checkpoint']['sha256'] != original['checkpoints']['fixed']['sha256']:
        raise ValueError('P56 collection policy differs from frozen P53 joint fixed-z')
    return p55, parent, original


def build_child(binding, cell, output, original=None):
    if original is None:
        _, _, original = verify(binding)
    scale = np.asarray(cell['scale_xyz'], dtype=float)
    kind = ('isotropic' if np.all(scale == scale[0]) else
            'thickness' if np.array_equal(scale[1:], np.ones(2)) else 'mixed')
    validate_scale(scale, kind)
    geometry = dict(id=cell['geometry_id'], axes=cell['scale_xyz'], kind=kind)
    props = original.get('geometry_properties')
    if props is None:
        from tools.task_graph.p54_book_geometry import original_properties
        props = original_properties(original['environment']['relocated_xml'])
    child = copy.deepcopy(original)
    child['environment']['relocated_xml'] = changed_xml(
        original['environment']['relocated_xml'], scale, geometry['kind'], props)
    child['relation']['book_collision_halfsize'] = (
        scale * np.asarray(original['relation']['book_collision_halfsize'], dtype=float)).tolist()
    child['task_graph'] = copy.deepcopy(binding['task_graph'])
    calibration = json.loads(Path(original['prefix']['calibration']).read_text())
    calibration_path = output/'SUPERVISOR_CALIBRATION.json'
    dump(calibration_path, supervisor_calibration(calibration, scale))
    child['prefix']['calibration'] = str(calibration_path)
    return child, geometry, props


def initialize_child(child, output, axes, old_halfsize, near_entry=None):
    """Build the real scene and restore either original task or sealed s9 state."""
    environment = child['environment']
    config = output/'libero_config'
    config.mkdir()
    base = Path(environment['libero_root'])/'libero/libero'
    dump(config/'config.yaml', dict(benchmark_root=str(base),
        bddl_files=str(base/'bddl_files'), init_states=str(base/'init_files'),
        datasets=str(Path(environment['source']).parent), assets=str(base/'assets')))
    os.environ['LIBERO_CONFIG_PATH'] = str(config.resolve())
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, environment['libero_root'])
    from libero.libero.envs import TASK_MAPPING
    kwargs = dict(environment['env_kwargs'])
    kwargs.update(use_camera_obs=False, has_offscreen_renderer=False)
    env = TASK_MAPPING[environment['env_args_original']['problem_name']](**kwargs)
    try:
        env.reset()
        env.reset_from_xml_string(environment['relocated_xml'])
        env.sim.reset()
        if near_entry is None:
            env.sim.set_state_from_flattened(source_demo(environment['source'])['initial'])
            env.sim.forward()
            joint = env.sim.model.joint_name2id(BOOK_JOINT)
            address = int(env.sim.model.jnt_qposadr[joint])
            if int(env.sim.model.jnt_type[joint]) != 0:
                raise ValueError('Book joint must remain free')
            geom = env.sim.model.geom_name2id(BOOK_COLLISION)
            rotation = env.sim.data.geom_xmat[geom].reshape(3, 3).copy()
            z_shift = bottom_shift(rotation, old_halfsize, axes)
            env.sim.data.qpos[address+2] += z_shift
        else:
            with np.load(near_entry, allow_pickle=False) as state:
                env.sim.set_state_from_flattened(state['supervisor_sim_state'])
                env.sim.data.ctrl[:] = state['supervisor_ctrl']
            z_shift = None
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        if near_entry is not None:
            for robot in env.robots:
                ids = [env.sim.model.actuator_name2id(name) for name in robot.gripper.actuators]
                ranges = env.sim.model.actuator_ctrlrange[ids]
                robot.gripper.current_action = np.clip(
                    (env.sim.data.ctrl[ids]-ranges.mean(1))/(.5*(ranges[:,1]-ranges[:,0])), -1, 1)
                robot.controller.update(force=True)
                robot.controller.reset_goal()
        return env, z_shift
    except Exception:
        env.close()
        raise


def _append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')


def _entry(env, obs, output, child, cell):
    arrays, _ = snapshot(env, obs)
    frame = output/'private/near_entry_state.npz'
    np.savez_compressed(frame, **{k: v for k, v in arrays.items() if k.startswith('supervisor_')})
    result = dict(schema=ENTRY_SCHEMA, condition_id=cell['condition_id'],
        geometry_id=cell['geometry_id'], scale_xyz=cell['scale_xyz'],
        source_policy=cell['source_policy'],
        source_cell_index=cell['index'], scene_xml_sha256=hashlib.sha256(
            child['environment']['relocated_xml'].encode()).hexdigest(),
        frame='private/near_entry_state.npz', frame_sha256=sha(frame),
        state_sha256=hashlib.sha256(arrays['supervisor_sim_state'].tobytes()).hexdigest())
    dump(output/'NEAR_ENTRY.json', result)
    return result


def _audit_physics(env, original_xml, props, axes, output):
    import mujoco
    from tools.task_graph.p54_book_geometry import BOOK_BODY, BOOK_COLLISION
    baseline = mujoco.MjModel.from_xml_string(original_xml)
    old_body = baseline.body(BOOK_BODY).id
    old_geom = baseline.geom(BOOK_COLLISION).id
    actual = _mass_record(env)
    expected_size = np.asarray(props['halfsize'], dtype=float) * np.asarray(axes, dtype=float)
    comparisons = dict(mass=(actual['mass'], baseline.body_mass[old_body]),
        ipos=(actual['ipos'], baseline.body_ipos[old_body]),
        iquat=(actual['iquat'], baseline.body_iquat[old_body]),
        inertia=(actual['inertia'], baseline.body_inertia[old_body]),
        collision_friction=(actual['collision_friction'], baseline.geom_friction[old_geom]),
        collision_size=(actual['collision_size'], expected_size))
    for name, (value, expected) in comparisons.items():
        if not np.allclose(value, expected, rtol=1e-10, atol=1e-10):
            raise ValueError('P56 changed physical property: '+name)
    record = dict(schema='prove.p56-book-physics-audit.v1',
        original_mass=float(baseline.body_mass[old_body]),
        expected_collision_size=expected_size.tolist(), actual=actual,
        checks={name: True for name in comparisons})
    dump(output/'PHYSICS_AUDIT.json', record)
    return record


def _micro(env, child, binding, cell, output, names, runtime, vision, renderer):
    teacher = MicroTeacher(child['relation'])
    judge = InsertJudge()
    obs = env._get_observations(force_update=True)
    previous = previous_action = None
    labels = []
    actions = 0
    while True:
        public_path = output/'public'/f'frame_{actions:04d}.npz'
        np.savez_compressed(public_path, **renderer.observe(obs))
        features = vision.encode(public_path)
        validate_features(features, runtime.model.config)
        torch.save(features, output/'public'/f'features_{actions:04d}.pt')
        if previous is None:
            runtime.enter(features)
        else:
            runtime.feedback(previous, previous_action, features)
        # This private call is an offline label. Policy action and feedback have
        # no path to its output or to private geometry.
        measured = private_measurement(env, obs, names, child['relation'])
        expert, label = teacher.command(measured)
        official, fingers, wrong = structural_measurement(env, names)
        judgment = judge.observe(official, fingers, wrong)
        if expert.shape != (7,) or not np.isfinite(expert).all() or expert[6] != 1:
            raise ValueError('Offline teacher label is not a valid closed-gripper OSC7 action')
        action = expert if cell['source_policy'] == 'expert' else runtime.command(features).numpy()
        if action.shape != (7,) or not np.isfinite(action).all() or action[6] != 1:
            raise ValueError('Invalid OSC7 action')
        stop = (judgment['stop'] or label['stop_reason'] or
            ('micro_budget24' if actions >= binding['micro_max_actions'] else None))
        private_path = output/'private'/f'frame_{actions:04d}.npz'
        arrays, _ = snapshot(env, obs)
        np.savez_compressed(private_path, **{k: v for k, v in arrays.items() if k.startswith('supervisor_')})
        label.update(t=actions, stage_id='s9', authority='private_offline_p53_v03_teacher',
            private_state=private_path.name, private_state_sha256=sha(private_path),
            public_observation=public_path.name, public_observation_sha256=sha(public_path),
            runtime_stop_reason=stop, finger_contacts=measured['finger_contacts'],
            expert_label_valid=stop is None)
        _append(output/'private/LABELS.jsonl', label)
        labels.append(label)
        _append(output/'public/ACTIONS.jsonl', dict(t=actions, stage_id='s9',
            observation=public_path.name, observation_sha256=sha(public_path),
            source_policy=cell['source_policy'], executed=stop is None,
            executed_action=action.tolist() if stop is None else None,
            stop_reason=stop))
        _append(output/'private/JUDGMENTS.jsonl', dict(t=actions,
            official_success=official, finger_contacts=fingers,
            wrong_target=wrong, judgment=judgment,
            teacher_domain_stop=label['stop_reason']))
        if actions:
            prior = labels[-2]
            prior_action = np.asarray(prior['_executed_action'], dtype=float)
            _append(output/'public/TRANSITIONS.jsonl', dict(t=actions-1,
                condition_id=cell['condition_id'], stage_id='s9',
                observation=prior['public_observation'],
                observation_sha256=prior['public_observation_sha256'],
                next_observation=public_path.name, next_observation_sha256=sha(public_path),
                executed_action=prior_action.tolist()))
        # Keep the actual training command private only in memory until the
        # following observation is recorded; never serialize it in LABELS.
        if stop:
            break
        labels[-1]['_executed_action'] = action.tolist()
        # The already-written private label does not contain executed command;
        # public transition above is the authoritative executed-action record.
        previous, previous_action = features, torch.tensor(action[:6], dtype=torch.float32)
        obs, _, _, _ = env.step(action.copy())
        actions += 1
    executed = [json.loads(line)['executed_action'] for line in
        (output/'public/TRANSITIONS.jsonl').read_text().splitlines()] if actions else []
    expert_actions = [row['expert_action'] for row in labels[:actions]]
    np.savez_compressed(output/'private/ACTION_PAIRS.npz',
        executed_actions=np.asarray(executed, dtype=np.float32).reshape(-1, 7),
        expert_actions=np.asarray(expert_actions, dtype=np.float32).reshape(-1, 7))
    return dict(actions=actions, frames=actions+1, graph_complete=judgment['complete'],
        official_success=bool(official), stop_reason=stop, updates=runtime.updates,
        label_valid=actions > 0, all_executed_expert_labels_valid=actions > 0,
        label_exclusion_reason=None if actions > 0 else 'no_executed_micro_transition')


def _paired_expert(path, binding_path, cell):
    if path is None:
        raise ValueError('Frozen policy collection requires paired sealed expert entry')
    root = Path(path).resolve()
    result = json.loads((root/'RESULT.json').read_text())
    near = json.loads((root/'NEAR_ENTRY.json').read_text())
    seal = json.loads((root/'SEAL.json').read_text())
    if (result['condition_id'] != cell['condition_id'] or result['source_policy'] != 'expert' or
            not result['near_entry_qualified'] or near['condition_id'] != cell['condition_id'] or
            seal['binding_sha256'] != sha(binding_path) or
            result['seal_sha256'] != sha(root/'SEAL.json')):
        raise ValueError('Policy source must be sealed paired expert near entry')
    for name in ('NEAR_ENTRY.json', near['frame']):
        if seal['files'].get(name) != sha(root/name):
            raise ValueError('Paired entry state is not sealed')
    if near['geometry_id'] != cell['geometry_id'] or near['scale_xyz'] != cell['scale_xyz']:
        raise ValueError('Paired entry geometry changed')
    return root, result, near


def run(binding_path, index, output, paired_entry=None, model_freeze=None):
    binding_path = Path(binding_path)
    binding = json.loads(binding_path.read_text())
    _, parent, original = verify(binding)
    if not 0 <= index < len(binding['cells']):
        raise ValueError('Cell outside frozen P56 scope')
    cell = binding['cells'][index]
    run_started_unix = time.time()
    freeze_ref = verify_model_freeze(model_freeze, binding_path, run_started_unix) if cell['split'] == 'test' else None
    if cell['source_policy'] == 'policy_fixed':
        paired = _paired_expert(paired_entry, binding_path, cell)
    elif paired_entry is not None:
        raise ValueError('Only fixed policy cells may restore a paired expert entry')
    else:
        paired = None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('prefix', 'code', 'public', 'private'):
        (output/name).mkdir()
    child, geometry, props = build_child(binding, cell, output, original)
    dump(output/'BINDING.json', binding)
    dump(output/'CELL.json', cell)
    dump(output/'TASK_GRAPH.json', child['task_graph'])
    execution = dict(schema='prove.p56-cell-execution.v1', run_started_unix=run_started_unix,
        cell_index=index, model_freeze=freeze_ref,
        paired_expert=None if paired is None else dict(directory=str(paired[0]),
            result_sha256=sha(paired[0]/'RESULT.json'),
            near_entry_sha256=sha(paired[0]/'NEAR_ENTRY.json')))
    dump(output/'EXECUTION.json', execution)
    torch.set_num_threads(1)
    torch.manual_seed(cell['seed'])
    np.random.seed(cell['seed'])
    ck = torch.load(binding['p53_joint_checkpoint']['path'], map_location='cpu', weights_only=True)
    model = StagePolicy(Config(**ck['config']))
    model.load_state_dict(ck['model'], strict=True)
    runtime = StageRuntime(model, online=False)
    before = weight_digest(model)
    env = vision = renderer = None
    try:
        env, shift = initialize_child(child, output, geometry['axes'], props['halfsize'],
            near_entry=None if paired is None else paired[0]/paired[2]['frame'])
        _audit_physics(env, original['environment']['relocated_xml'], props, geometry['axes'], output)
        names = _names(env)
        dump(output/'private/NAMES.json', names)
        prefix = (run_prefix(env, child, output/'prefix', output.name,
            dict(grasp_world_offset=cell['grasp_world_offset'])) if paired is None else
            dict(paired_real_predecessor=True, qualified=True, actions=paired[1]['prefix_actions'],
                source_result=execution['paired_expert']))
        dump(output/'PREFIX_RESULT.json', prefix)
        code = (run_code(env, child, parent, output/'code', names) if paired is None and prefix['qualified'] else
            dict(paired_real_predecessor=True, qualified=True, actions=paired[1]['code_actions'],
                next_stage='s9', near_entry_qualified=True, stop_reason='paired_sealed_near_entry') if paired else
            dict(qualified=False, actions=0, stop_reason='prefix_failed', near_entry_qualified=False))
        dump(output/'CODE_RESULT.json', code)
        near = None
        if code['qualified']:
            if code['next_stage'] != 's9':
                raise ValueError('Code did not hand off to s9')
            obs = env._get_observations(force_update=True)
            near = _entry(env, obs, output, child, cell)
            if paired is not None and near['state_sha256'] != paired[2]['state_sha256']:
                raise ValueError('Restored policy entry differs from paired expert physical state')
            if cell['source_policy'] == 'entry_only':
                result = dict(actions=0, frames=0, graph_complete=False,
                    official_success=bool(env._check_success()), stop_reason='entry_only', updates=0,
                    label_valid=False, all_executed_expert_labels_valid=False,
                    label_exclusion_reason='entry_only')
            else:
                vision = Vision(original['vision'])
                dump(output/'VISUAL_BINDING.json', vision.binding)
                renderer = PublicRenderer(env, original['public'])
                result = _micro(env, child, binding, cell, output, names, runtime, vision, renderer)
        else:
            result = dict(actions=0, frames=0, graph_complete=False,
                official_success=bool(env._check_success()), stop_reason=code['stop_reason'], updates=0,
                label_valid=False, all_executed_expert_labels_valid=False,
                label_exclusion_reason='near_entry_not_reached')
        after = weight_digest(model)
        if before != after:
            raise RuntimeError('Frozen P53 weights mutated')
        files = {str(p.relative_to(output)): sha(p) for section in
            ('prefix', 'code', 'public', 'private') for p in sorted((output/section).iterdir()) if p.is_file()}
        for name in ('BINDING.json', 'CELL.json', 'TASK_GRAPH.json', 'EXECUTION.json',
                'PHYSICS_AUDIT.json',
                'SUPERVISOR_CALIBRATION.json', 'PREFIX_RESULT.json', 'CODE_RESULT.json'):
            files[name] = sha(output/name)
        for name in ('NEAR_ENTRY.json', 'VISUAL_BINDING.json'):
            if (output/name).exists():
                files[name] = sha(output/name)
        dump(output/'SEAL.json', dict(files=files, binding_sha256=sha(binding_path),
            cell_index=index, network_before_sha256=before, network_after_sha256=after,
            original_initial_restores=1 if paired is None else 0,
            paired_near_entry_restores=1 if paired is not None else 0,
            later_state_restores=0, run_started_unix=run_started_unix,
            model_freeze=freeze_ref))
        result.update(schema='prove.p56-collection-result.v1', cell_index=index,
            condition_id=cell['condition_id'], geometry_id=cell['geometry_id'],
            scale_xyz=cell['scale_xyz'], split=cell['split'], seed=cell['seed'],
            source_policy=cell['source_policy'],
            near_entry_qualified=bool(code['qualified']), near_entry=near,
            prefix_actions=prefix['actions'], code_actions=code['actions'],
            initial_book_z_shift=shift, failure_stage=(None if result['graph_complete'] else
                's9' if code['qualified'] and cell['source_policy'] != 'entry_only' else
                'entry_only' if code['qualified'] else 'code' if prefix['qualified'] else 'prefix'),
            network_weights_unchanged=True, seal_sha256=sha(output/'SEAL.json'),
            automatic_retry=False, full_task_success_evidence=False)
        dump(output/'RESULT.json', result)
        return result
    except Exception as exc:
        dump(output/'ABORT.json', dict(cell=cell, error=type(exc).__name__,
            reason=str(exc), automatic_retry=False))
        raise
    finally:
        runtime.exit()
        if renderer is not None:
            renderer.close()
        if vision is not None:
            vision.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--design', type=Path, required=True)
    prep.add_argument('--p55-binding', type=Path, required=True)
    prep.add_argument('--p53-joint-checkpoint', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    collect = sub.add_parser('collect')
    collect.add_argument('--binding', type=Path, required=True)
    collect.add_argument('--index', type=int, required=True)
    collect.add_argument('--output', type=Path, required=True)
    collect.add_argument('--paired-entry', type=Path)
    collect.add_argument('--model-freeze', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.output.exists():
            raise FileExistsError(args.output)
        binding = prepare(args.design, args.p55_binding, args.p53_joint_checkpoint)
        dump(args.output, binding)
    else:
        print(json.dumps(run(args.binding, args.index, args.output,
            paired_entry=args.paired_entry, model_freeze=args.model_freeze), indent=2))


if __name__ == '__main__':
    main()
