#!/usr/bin/env python3
"""P54 paired original-task book-size pilot; frozen P53 graph and checkpoints."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from prove.task_graph.p52_stage_policy import Config, StagePolicy, StageRuntime
from prove.task_graph.p53_micro_insert import PublicPosePath, geometry
from tools.task_graph.collect_libero_demo_gt import dump, sha, source_demo
from tools.task_graph.collect_libero_p52_v02 import PublicRenderer, private_measurement
from tools.task_graph.collect_libero_p53_v03 import record_private, run_prefix, validate_graph
from tools.task_graph.p53_full_task import neural_graph, run_neural, verify as verify_p53, weight_digest
from tools.task_graph.p54_book_geometry import (BOOK_BODY, BOOK_COLLISION, BOOK_JOINT, BOOK_MESH,
    BOOK_VISUAL, SCALES, initial_bottom_shift, original_properties, scaled_supervisor_calibration,
    scaled_xml, verify_scaled_xml)

SCHEMA = 'prove.p54-book-scale-full-task.v1'
ARMS = ('bc', 'fixed', 'online')
OLD_TOL = 0.003585
OLD_GAP_MAX = 0.00717


def conditions(collection):
    rows = [dict(id='nominal', grasp_index=None, grasp_world_offset=[0.0, 0.0, 0.0])]
    for g in range(6):
        matches = [c for c in collection['conditions'] if c['grasp_index'] == g]
        if len(matches) != 4 or any(c['grasp_world_offset'] != matches[0]['grasp_world_offset'] for c in matches):
            raise ValueError('P52 grasp condition mismatch')
        rows.append(dict(id=f'grasp_{g}', grasp_index=g, grasp_world_offset=matches[0]['grasp_world_offset']))
    if len({tuple(x['grasp_world_offset']) for x in rows}) != 7:
        raise ValueError('Nominal and six grasp conditions must be distinct')
    return rows


def _cells(conditions_):
    rows = []
    for scale in SCALES:
        for condition in conditions_:
            for arm in ARMS:
                rows.append(dict(index=len(rows), scale=scale, condition=condition['id'], arm=arm,
                    zmode='observed_feedback' if arm == 'online' else 'fixed',
                    grasp_world_offset=condition['grasp_world_offset']))
    return rows


def prepare(parent_path):
    parent_path = Path(parent_path).resolve()
    parent = json.loads(parent_path.read_text())
    verify_p53(parent)
    qualification = json.loads(Path(parent['parent']['path']).read_text())
    collection_path = Path(qualification['parent']['path'])
    collection = json.loads(collection_path.read_text())
    props = original_properties(parent['environment']['relocated_xml'])
    if not np.allclose(props['center'], parent['relation']['book_collision_center'], atol=1e-12) or not np.allclose(props['halfsize'], parent['relation']['book_collision_halfsize'], atol=1e-12):
        raise ValueError('P53 relation differs from compiled book box')
    if not np.isclose(min(props['halfsize']) / 4, OLD_TOL, atol=1e-12):
        raise ValueError('P53 absolute near tolerance differs')
    choices = conditions(collection)
    binding = dict(schema=SCHEMA, parent=dict(path=str(parent_path), sha256=sha(parent_path)),
        p52_collection=dict(path=str(collection_path), sha256=sha(collection_path)),
        geometry=dict(scales=list(SCALES), pivot='original_collision_center', properties=props,
            unchanged=['book mass', 'inertial pose', 'diagonal inertia', 'friction', 'density',
                       'table', 'caddy', 'robot', 'official BDDL and native goal']),
        conditions=choices, cells=_cells(choices), seed=54,
        code_position_tolerance=parent['code_position_tolerance'], code_rotation_tolerance=parent['code_rotation_tolerance'],
        near_position_tolerance=OLD_TOL, near_gap_max=OLD_GAP_MAX,
        prefix_max_actions=400, code_max_actions=120, micro_max_actions=24,
        total_max_actions=544, original_initial_restores=1, later_state_restores=0,
        attempts_per_cell=1, automatic_retries=0,
        source_files={str(Path(__file__).resolve()): sha(__file__), str(ROOT/'tools/task_graph/p54_book_geometry.py'): sha(ROOT/'tools/task_graph/p54_book_geometry.py')},
        scope='42 paired complete-task pilot episodes: 2 book sizes x 7 existing grasp settings x 3 frozen P53 arms; all failures count, no selection or retraining',
        condition_interpretation='Seven existing grasp settings, not seven independent randomized initial states')
    return binding


def verify(binding):
    if binding.get('schema') != SCHEMA or binding.get('geometry', {}).get('scales') != list(SCALES):
        raise ValueError('P54 scope changed')
    if any(binding.get(k) != v for k, v in dict(prefix_max_actions=400, code_max_actions=120,
            micro_max_actions=24, total_max_actions=544, original_initial_restores=1,
            later_state_restores=0, attempts_per_cell=1, automatic_retries=0,
            near_position_tolerance=OLD_TOL, near_gap_max=OLD_GAP_MAX).items()):
        raise ValueError('P54 budget or gate changed')
    parent_path = Path(binding['parent']['path'])
    if sha(parent_path) != binding['parent']['sha256']:
        raise ValueError('P53 parent changed')
    parent = json.loads(parent_path.read_text())
    verify_p53(parent)
    collection_path = Path(binding['p52_collection']['path'])
    if sha(collection_path) != binding['p52_collection']['sha256']:
        raise ValueError('P52 collection changed')
    collection = json.loads(collection_path.read_text())
    choices = conditions(collection)
    if binding['conditions'] != choices or binding['cells'] != _cells(choices) or len(binding['cells']) != 42:
        raise ValueError('P54 condition/cell set changed')
    if binding['geometry']['properties'] != original_properties(parent['environment']['relocated_xml']):
        raise ValueError('P54 book geometry changed')
    for scale in SCALES:
        verify_scaled_xml(parent['environment']['relocated_xml'],
            scaled_xml(parent['environment']['relocated_xml'], scale, binding['geometry']['properties']),
            scale, binding['geometry']['properties'])
    if binding['code_position_tolerance'] != parent['code_position_tolerance'] or binding['code_rotation_tolerance'] != parent['code_rotation_tolerance']:
        raise ValueError('P53 public Code tolerances changed')
    if parent['checkpoints']['fixed'] != parent['checkpoints']['online']:
        raise ValueError('Fixed and online checkpoints differ')
    for path, digest in binding['source_files'].items():
        if sha(path) != digest:
            raise ValueError('P54 source changed: ' + path)
    return parent


def child_binding(parent, scale, calibration_path, properties):
    child = copy.deepcopy(parent)
    original = parent['environment']['relocated_xml']
    child['environment']['relocated_xml'] = scaled_xml(original, scale, properties)
    child['relation']['book_collision_halfsize'] = (scale * np.asarray(parent['relation']['book_collision_halfsize'])).tolist()
    child['prefix']['calibration'] = str(calibration_path)
    if child['reference_poses'] != parent['reference_poses'] or child['task_graph'] != parent['task_graph']:
        raise ValueError('P53 public references or graph changed')
    return child


def initialize_child(binding, output, scale, old_halfsize):
    environment = binding['environment']
    config = output/'libero_config'
    config.mkdir()
    base = Path(environment['libero_root'])/'libero/libero'
    dump(config/'config.yaml', dict(benchmark_root=str(base), bddl_files=str(base/'bddl_files'),
        init_states=str(base/'init_files'), datasets=str(Path(environment['source']).parent), assets=str(base/'assets')))
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
        env.sim.set_state_from_flattened(source_demo(environment['source'])['initial'])
        env.sim.forward()
        j = env.sim.model.joint_name2id(BOOK_JOINT)
        addr = int(env.sim.model.jnt_qposadr[j])
        if int(env.sim.model.jnt_type[j]) != 0:
            raise ValueError('Book joint is not free')
        geom = env.sim.model.geom_name2id(BOOK_COLLISION)
        initial_rotation = env.sim.data.geom_xmat[geom].reshape(3, 3).copy()
        z_shift = initial_bottom_shift(initial_rotation, old_halfsize, scale)
        env.sim.data.qpos[addr+2] += z_shift
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        return env, z_shift
    except Exception:
        env.close()
        raise


def near_gate(relation, private, binding):
    g = geometry(relation, private)
    fingers = private['finger_contacts']
    tol, gap = binding['near_position_tolerance'], binding['near_gap_max']
    return dict(near_ready=bool(fingers is not None and all(fingers)
        and np.all(np.asarray(g['footprint_slack']) >= -tol)
        and 0 <= g['bottom_gap'] <= gap and not private['official_success'] and not private['wrong_target']),
        code_completed_goal=bool(private['official_success']),
        structural_stop=bool(private['wrong_target'] or fingers is None or not any(fingers)))


def run_code_child(env, child, binding, output, names):
    graph = child['task_graph']
    validate_graph(graph)
    nodes = {n['id']: n for n in graph['nodes']}
    stage = 's7'
    active = nodes[stage]
    actions = 0
    obs = env._get_observations(force_update=True)
    initial = np.asarray(obs['robot0_eef_pos']).copy()
    initial[2] = max(p['position'][2] for p in child['reference_poses'])
    targets = [dict(position=initial.tolist(), quaternion_xyzw=np.asarray(obs['robot0_eef_quat']).tolist()),
               child['reference_poses'][41], child['reference_poses'][57]]
    control = PublicPosePath(targets, child['code_position_tolerance'], child['code_rotation_tolerance'])
    dump(output/'PUBLIC_TARGETS.json', dict(targets=targets, origin='P53 frozen public EEF reference path'))
    while True:
        measured = private_measurement(env, obs, names, child['relation'])
        gate = near_gate(child['relation'], measured, binding)
        action, reached, diag = control.command(obs['robot0_eef_pos'], obs['robot0_eef_quat'], active['reference_start'], active['reference_end'])
        transitions = []
        if reached and stage == 's7':
            transitions.append(dict(from_stage='s7', to_stage=active['next']))
            stage = active['next']
            active = nodes[stage]
            action, reached, diag = control.command(obs['robot0_eef_pos'], obs['robot0_eef_quat'], active['reference_start'], active['reference_end'])
        ready = bool(stage == 's8' and reached and gate['near_ready'])
        stop = ('code_completed_goal' if gate['code_completed_goal'] else
                'code_structural_stop' if gate['structural_stop'] else
                'near_ready' if ready else 'code_budget' if actions >= graph['shared_code_budget'] else None)
        path = output/f'frame_{actions:04d}.npz'
        digest = record_private(env, obs, path)
        row = dict(t=actions, stage_id=stage, file=path.name, sha256=digest,
            action=None if stop else action.tolist(), public_controller=diag, supervisor_gate=gate,
            private_geometry=geometry(child['relation'], measured), transitions=transitions, stop_reason=stop)
        with (output/'TRACE.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if stop:
            return dict(qualified=ready, actions=actions, stop_reason=stop,
                        next_stage=active['next'] if ready else None, near_entry_qualified=ready)
        obs, _, _, _ = env.step(action.copy())
        actions += 1


def _names(env):
    return {kind: [getattr(env.sim.model, kind+'_id2name')(i) for i in range(getattr(env.sim.model, count))]
            for kind, count in [('body', 'nbody'), ('site', 'nsite'), ('geom', 'ngeom')]}


def failure_stage(output, prefix, code, result):
    if result['graph_complete']:
        return None
    if code['qualified']:
        return 's9'
    if prefix['qualified']:
        trace = output/'code/TRACE.jsonl'
        if trace.exists():
            last = json.loads(trace.read_text().splitlines()[-1])
            if last.get('stage_id') in ('s7', 's8'):
                return last['stage_id']
        return 's7'
    trace = output/'prefix/TRACE.jsonl'
    if trace.exists():
        last = json.loads(trace.read_text().splitlines()[-1])
        events = last.get('events') or []
        if events and events[-1].get('judge', {}).get('stage_id') in {f's{i}' for i in range(1, 7)}:
            return events[-1]['judge']['stage_id']
    return 'prefix_unresolved'


def _mass_record(env):
    model = env.sim.model
    b = model.body_name2id(BOOK_BODY)
    g = model.geom_name2id(BOOK_COLLISION)
    v = model.geom_name2id(BOOK_VISUAL)
    return dict(mass=float(model.body_mass[b]), ipos=model.body_ipos[b].tolist(),
        iquat=model.body_iquat[b].tolist(), inertia=model.body_inertia[b].tolist(),
        collision_size=model.geom_size[g].tolist(), collision_pos=model.geom_pos[g].tolist(),
        collision_friction=model.geom_friction[g].tolist(), visual_pos=model.geom_pos[v].tolist(),
        visual_mesh_scale=model.mesh_scale[model.geom_dataid[v]].tolist())


def _unchanged_scene_digest(env):
    model = env.sim.model
    omitted_geoms = {model.geom_name2id(BOOK_COLLISION), model.geom_name2id(BOOK_VISUAL)}
    omitted_body = model.body_name2id(BOOK_BODY)
    payload = []
    for g in range(model.ngeom):
        if g not in omitted_geoms:
            payload.extend((np.asarray(model.geom_size[g]), np.asarray(model.geom_pos[g]),
                            np.asarray(model.geom_friction[g])))
    for b in range(model.nbody):
        if b != omitted_body:
            payload.extend((np.asarray(model.body_mass[b]), np.asarray(model.body_inertia[b])))
    return hashlib.sha256(b''.join(np.asarray(a, dtype='<f8').tobytes() for a in payload)).hexdigest()


def qualify(binding_path, output):
    """Compile and inspect both scene variants; zero env.step calls."""
    binding = json.loads(Path(binding_path).read_text())
    parent = verify(binding)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for scale in SCALES:
        folder = output/f'scale_{scale:.2f}'
        folder.mkdir()
        cal = scaled_supervisor_calibration(json.loads(Path(parent['prefix']['calibration']).read_text()), scale)
        cal_path = folder/'SUPERVISOR_CALIBRATION.json'
        dump(cal_path, cal)
        child = child_binding(parent, scale, cal_path, binding['geometry']['properties'])
        verify_scaled_xml(parent['environment']['relocated_xml'], child['environment']['relocated_xml'],
            scale, binding['geometry']['properties'])
        env = None
        try:
            env, shift = initialize_child(child, folder, scale, binding['geometry']['properties']['halfsize'])
            record = dict(scale=scale, z_shift=shift, compiled=_mass_record(env),
                official_at_start=bool(env._check_success()), physics_steps=0,
                unchanged_scene_digest=_unchanged_scene_digest(env))
            g = env.sim.model.geom_name2id(BOOK_COLLISION)
            b = env.sim.model.body_name2id(BOOK_BODY)
            R = env.sim.data.geom_xmat[g].reshape(3,3)
            h = np.asarray(env.sim.model.geom_size[g])
            center_z = float(env.sim.data.geom_xpos[g][2])
            record['collision_bottom_world_z'] = center_z - float(np.abs(R[2]) @ h)
            record['body_origin_world'] = env.sim.data.body_xpos[b].tolist()
            joint = env.sim.model.joint_name2id(BOOK_JOINT)
            book_z_address = int(env.sim.model.jnt_qposadr[joint])+2
            record['other_qpos_sha256'] = hashlib.sha256(np.delete(
                np.asarray(env.sim.data.qpos), book_z_address).astype('<f8').tobytes()).hexdigest()
            record['book_contacts'] = [dict(other_geom=env.sim.model.geom_id2name(
                int(c.geom2 if c.geom1 == g else c.geom1)), distance=float(c.dist))
                for c in (env.sim.data.contact[i] for i in range(env.sim.data.ncon))
                if c.geom1 == g or c.geom2 == g]
            try:
                renderer = PublicRenderer(env, parent['public'])
                observation = renderer.observe(env._get_observations(force_update=True))
                np.savez_compressed(folder/'PUBLIC_INITIAL.npz', **observation)
                renderer.close()
                record['public_initial_sha256'] = sha(folder/'PUBLIC_INITIAL.npz')
            except Exception as exc:
                record['render_error'] = str(exc)
            dump(folder/'GEOMETRY.json', record)
            records.append(record)
        finally:
            if env is not None: env.close()
    old, small = records
    p = binding['geometry']['properties']
    same = all(np.allclose(old['compiled'][k], small['compiled'][k], atol=1e-12, rtol=1e-10)
               for k in ('mass', 'ipos', 'iquat', 'inertia', 'collision_friction'))
    expected = np.asarray(p['halfsize']) * 0.9
    passed = bool(same and np.allclose(small['compiled']['collision_size'], expected, atol=1e-12)
        and np.allclose(old['compiled']['collision_size'], p['halfsize'], atol=1e-12)
        and abs(old['collision_bottom_world_z']-small['collision_bottom_world_z']) < 1e-10
        and np.allclose(old['body_origin_world'][:2], small['body_origin_world'][:2], atol=1e-12)
        and old['other_qpos_sha256'] == small['other_qpos_sha256']
        and old['unchanged_scene_digest'] == small['unchanged_scene_digest']
        and not old['official_at_start'] and not small['official_at_start']
        and 'render_error' not in old and 'render_error' not in small)
    passed = passed and np.allclose(small['compiled']['visual_mesh_scale'],
        0.9*np.asarray(old['compiled']['visual_mesh_scale']), atol=1e-12)
    result = dict(schema='prove.p54-zero-step-geometry-qualification.v1', passed=passed,
        records=records, old_initial_bottom_matches=abs(old['collision_bottom_world_z']-small['collision_bottom_world_z']) < 1e-10,
        mass_inertia_friction_unchanged=same, physics_steps=0, binding_sha256=sha(binding_path))
    dump(output/'RESULT.json', result)
    return result


def run(binding_path, index, output, qualification):
    binding = json.loads(Path(binding_path).read_text())
    parent = verify(binding)
    qualification = Path(qualification)
    q = json.loads(qualification.read_text())
    if (q.get('schema') != 'prove.p54-zero-step-geometry-qualification.v1' or not q.get('passed')
            or q.get('binding_sha256') != sha(binding_path) or q.get('physics_steps') != 0):
        raise ValueError('P54 zero-step geometry qualification required')
    if index not in range(42):
        raise ValueError('Outside P54 frozen 42-cell scope')
    cell = binding['cells'][index]
    output.mkdir(parents=True, exist_ok=False)
    for name in ('prefix', 'code', 'public', 'private'):
        (output/name).mkdir()
    cal = scaled_supervisor_calibration(json.loads(Path(parent['prefix']['calibration']).read_text()), cell['scale'])
    cal_path = output/'SUPERVISOR_CALIBRATION.json'
    dump(cal_path, cal)
    child = child_binding(parent, cell['scale'], cal_path, binding['geometry']['properties'])
    dump(output/'BINDING.json', binding)
    dump(output/'CELL.json', cell)
    dump(output/'TASK_GRAPH.json', child['task_graph'])
    torch.set_num_threads(1)
    torch.manual_seed(binding['seed'])
    np.random.seed(binding['seed'])
    checkpoint = torch.load(parent['checkpoints'][cell['arm']]['path'], map_location='cpu', weights_only=True)
    model = StagePolicy(Config(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'], strict=True)
    runtime = StageRuntime(model, online=cell['zmode'] == 'observed_feedback')
    before = weight_digest(model)
    env = vision = renderer = None
    try:
        env, z_shift = initialize_child(child, output, cell['scale'], binding['geometry']['properties']['halfsize'])
        names = _names(env)
        dump(output/'private/NAMES.json', names)
        prefix = run_prefix(env, child, output/'prefix', output.name,
            dict(grasp_world_offset=cell['grasp_world_offset']))
        dump(output/'PREFIX_RESULT.json', prefix)
        code = (run_code_child(env, child, binding, output/'code', names) if prefix['qualified'] else
            dict(qualified=False, actions=0, near_entry_qualified=False, stop_reason='prefix_failed'))
        dump(output/'CODE_RESULT.json', code)
        transitions = []
        if prefix['qualified']:
            transitions.append(dict(from_stage='s6', to_stage='s7', after_actions=prefix['actions']))
        if (output/'code/TRACE.jsonl').exists():
            for line in (output/'code/TRACE.jsonl').read_text().splitlines():
                row = json.loads(line)
                for transition in row['transitions']:
                    transitions.append(dict(**transition, after_actions=prefix['actions']+row['t']))
        if code['qualified']:
            if code['next_stage'] != 's9':
                raise ValueError('Code handoff differs from frozen graph')
            transitions.append(dict(from_stage='s8', to_stage='s9', after_actions=prefix['actions']+code['actions']))
            from tools.task_graph.p52_policy_pilot import Vision
            vision = Vision(parent['vision'])
            dump(output/'VISUAL_BINDING.json', vision.binding)
            renderer = PublicRenderer(env, parent['public'])
            result = run_neural(env, child, runtime, vision, renderer, output, names)
        else:
            result = dict(actions=0, graph_complete=False, official_success=bool(env._check_success()),
                          stop_reason=code['stop_reason'], updates=0)
        if result['graph_complete']:
            transitions.append(dict(from_stage='s9', to_stage=None,
                after_actions=prefix['actions']+code['actions']+result['actions']))
        after = weight_digest(model)
        if before != after:
            raise RuntimeError('Frozen network weights mutated')
        verify(binding)
        dump(output/'STAGE_TRANSITIONS.json', transitions)
        files = {str(p.relative_to(output)): sha(p) for d in ('prefix', 'code', 'public', 'private')
                 for p in sorted((output/d).iterdir()) if p.is_file()}
        files.update({n: sha(output/n) for n in ('BINDING.json', 'CELL.json', 'TASK_GRAPH.json',
            'SUPERVISOR_CALIBRATION.json', 'PREFIX_RESULT.json', 'CODE_RESULT.json', 'STAGE_TRANSITIONS.json')})
        if (output/'VISUAL_BINDING.json').exists():
            files['VISUAL_BINDING.json'] = sha(output/'VISUAL_BINDING.json')
        dump(output/'SEAL.json', dict(files=files, binding_sha256=sha(output/'BINDING.json'),
            qualification_sha256=sha(qualification), cell=cell, original_initial_restores=1,
            later_state_restores=0, network_before_sha256=before, network_after_sha256=after))
        result.update(cell=cell, initial_book_z_shift=z_shift, prefix_actions=prefix['actions'],
            code_actions=code['actions'], near_entry_qualified=code['qualified'],
            total_actions=prefix['actions']+code['actions']+result['actions'],
            network_weights_unchanged=before==after, seal_sha256=sha(output/'SEAL.json'),
            automatic_retry=False, independent_statistical_evidence=False,
            failure_stage=failure_stage(output, prefix, code, result))
        dump(output/'RESULT.json', result)
        return result
    except Exception as exc:
        dump(output/'ABORT.json', dict(cell=cell, error=type(exc).__name__, reason=str(exc), automatic_retry=False))
        raise
    finally:
        runtime.exit()
        if renderer is not None: renderer.close()
        if env is not None: env.close()
        if vision is not None: vision.close()


def queue_manifest(binding_path, qualification_path, python, cwd, timeout_seconds=1800):
    binding_path, qualification_path = Path(binding_path).resolve(), Path(qualification_path).resolve()
    binding = json.loads(binding_path.read_text())
    verify(binding)
    qualified = json.loads(qualification_path.read_text())
    if not qualified.get('passed') or qualified.get('binding_sha256') != sha(binding_path):
        raise ValueError('P54 geometry qualification has not passed for this binding')
    if not 0 < timeout_seconds <= 1800:
        raise ValueError('Finite per-cell timeout must be <=1800 seconds')
    jobs = []
    for cell in binding['cells']:
        jobs.append(dict(id=f"cell_{cell['index']:02d}_{cell['condition']}_{cell['arm']}_s{int(100*cell['scale'])}",
            seed=binding['seed'], timeout_seconds=timeout_seconds,
            argv=[str(python), str(Path(__file__).resolve()), 'run', '--binding', str(binding_path),
                  '--qualification', str(qualification_path), '--index', str(cell['index'])]))
    return dict(schema='p53.explicit-queue.v1', cwd=str(Path(cwd).resolve()), workers=3,
        attempts_per_job=1, bound_files={str(binding_path): sha(binding_path),
            str(qualification_path): sha(qualification_path), str(Path(__file__).resolve()): sha(__file__)},
        jobs=jobs, scope='P54 42 predeclared paired complete original-task cells, one attempt each, no retries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('qualify')
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('queue-manifest')
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--cwd', type=Path, default=ROOT)
    p.add_argument('--timeout-seconds', type=int, default=1800)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.output.exists(): raise FileExistsError(args.output)
        binding = prepare(args.parent)
        verify(binding)
        dump(args.output, binding)
    elif args.command == 'qualify':
        print(json.dumps(qualify(args.binding, args.output), indent=2))
    elif args.command == 'queue-manifest':
        if args.output.exists(): raise FileExistsError(args.output)
        dump(args.output, queue_manifest(args.binding, args.qualification,
            args.python, args.cwd, args.timeout_seconds))
    else:
        print(json.dumps(run(args.binding, args.index, args.output, args.qualification), indent=2))
