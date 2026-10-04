#!/usr/bin/env python3
"""P55: frozen P54 full task with a public EEF-axis s8a handoff continuation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from prove.task_graph.p52_stage_policy import Config, StagePolicy, StageRuntime
from prove.task_graph.p53_micro_insert import PublicPosePath, geometry
from prove.task_graph.p55_relation_approach import PublicRelationApproach
from tools.task_graph.collect_libero_demo_gt import dump, sha, snapshot
from tools.task_graph.collect_libero_p52_v02 import PublicRenderer, private_measurement
from tools.task_graph.collect_libero_p53_v03 import record_private
from tools.task_graph.p52_full_task import InsertJudge, structural_measurement
from tools.task_graph.p52_policy_pilot import Vision, validate_features
from tools.task_graph.p53_full_task import weight_digest
from tools.task_graph.p54_full_task import (child_binding as p54_child_binding,
    initialize_child, near_gate, run_prefix, verify as verify_p54, _names)
from tools.task_graph.p54_book_geometry import SCALES, scaled_supervisor_calibration

SCHEMA = 'prove.p55-public-relation-approach.v1'
QUAL_SCHEMA = 'prove.p55-two-nominal-real-prefix-code-qualification.v1'


def graph_from_p54(graph):
    result = copy.deepcopy(graph)
    if len(result['nodes']) != 3 or [n['id'] for n in result['nodes']] != ['s7', 's8', 's9']:
        raise ValueError('P54 three-node Code/neural graph required')
    if result['nodes'][1]['next'] != 's9' or result['nodes'][2]['backend'] != 'frozen_stage_policy':
        raise ValueError('P54 handoff changed')
    result['schema'] = 'prove.p55-relation-taskgraph.v1'
    result['graph_id'] = result['graph_id']+'_public_relation_s8a'
    result['nodes'][1]['next'] = 's8a'
    result['nodes'][1]['termination'] = 'public_endpoint_reached'
    result['nodes'].insert(2, dict(id='s8a', skill='Approach', backend='public_ref41_ref57_axis',
        next='s9', termination='original_near_ready',
        reference_axis='unit_vector_public_ref41_to_ref57',
        lateral_alignment='hold_public_ref57_axis', orientation='hold_public_ref57'))
    result['code_supervisor_gates'] = ['near_ready', 'structural_stop', 'code_completed_goal']
    return result


def prepare(parent_path, zero_qualification_path):
    parent_path = Path(parent_path).resolve()
    zero_qualification_path = Path(zero_qualification_path).resolve()
    parent = json.loads(parent_path.read_text())
    verify_p54(parent)
    zero = json.loads(zero_qualification_path.read_text())
    if not zero.get('passed') or zero.get('binding_sha256') != sha(parent_path) or zero.get('physics_steps') != 0:
        raise ValueError('P54 zero-step geometry qualification required')
    original = json.loads(Path(parent['parent']['path']).read_text())
    return dict(schema=SCHEMA, parent=dict(path=str(parent_path), sha256=sha(parent_path)),
        zero_qualification=dict(path=str(zero_qualification_path), sha256=sha(zero_qualification_path)),
        task_graph=graph_from_p54(original['task_graph']), cells=copy.deepcopy(parent['cells']),
        seed=parent['seed'], prefix_max_actions=400, code_max_actions=120,
        micro_max_actions=24, total_max_actions=544, initial_restores=1, later_restores=0,
        automatic_retries=0, attempts_per_cell=1,
        sources={str(Path(__file__).resolve()): sha(__file__),
                 str(ROOT/'prove/task_graph/p55_relation_approach.py'): sha(ROOT/'prove/task_graph/p55_relation_approach.py')},
        graph_change='Insert public EEF-only s8a Approach after s8 endpoint; retain P53/P54 prefix, model, references, thresholds, budgets and three Boolean supervisor gates',
        runtime_boundary='No object scale or continuous private geometry enters s8a command; private geometry is diagnostic only',
        qualification='Two nominal original-initial-state real prefix+Code runs (scale1.0 and0.9), no teacher/neural s9; both near_ready required before queue')


def verify(binding):
    if binding.get('schema') != SCHEMA or len(binding.get('cells', [])) != 42:
        raise ValueError('Wrong P55 42-cell scope')
    for name, value in dict(seed=54, prefix_max_actions=400, code_max_actions=120,
            micro_max_actions=24, total_max_actions=544, initial_restores=1,
            later_restores=0, automatic_retries=0, attempts_per_cell=1).items():
        if binding.get(name) != value:
            raise ValueError('P55 scope or budget changed: '+name)
    parent_path = Path(binding['parent']['path'])
    if sha(parent_path) != binding['parent']['sha256']:
        raise ValueError('P54 binding changed')
    parent = json.loads(parent_path.read_text())
    original = verify_p54(parent)
    zero = binding['zero_qualification']
    if sha(zero['path']) != zero['sha256']:
        raise ValueError('P54 zero qualification changed')
    outcome = json.loads(Path(zero['path']).read_text())
    if not outcome.get('passed') or outcome.get('binding_sha256') != sha(parent_path) or outcome.get('physics_steps') != 0:
        raise ValueError('P54 geometry qualification invalid')
    if binding['cells'] != parent['cells'] or binding['task_graph'] != graph_from_p54(original['task_graph']):
        raise ValueError('P55 graph/cells changed')
    if [n['id'] for n in binding['task_graph']['nodes']] != ['s7','s8','s8a','s9']:
        raise ValueError('P55 node order changed')
    for path, digest in binding['sources'].items():
        if sha(path) != digest:
            raise ValueError('P55 source changed: '+path)
    return parent, original


def child_binding(parent, original, scale, calibration_path, graph):
    child = p54_child_binding(original, scale, calibration_path, parent['geometry']['properties'])
    child['task_graph'] = copy.deepcopy(graph)
    return child


def run_code(env, child, gate_binding, output, names):
    graph = child['task_graph']
    if [n['id'] for n in graph['nodes']] != ['s7','s8','s8a','s9'] or graph['shared_code_budget'] != 120:
        raise ValueError('P55 Code graph changed')
    nodes = {n['id']: n for n in graph['nodes']}
    obs = env._get_observations(force_update=True)
    initial = np.asarray(obs['robot0_eef_pos']).copy()
    initial[2] = max(p['position'][2] for p in child['reference_poses'])
    targets = [dict(position=initial.tolist(), quaternion_xyzw=np.asarray(obs['robot0_eef_quat']).tolist()),
               child['reference_poses'][41], child['reference_poses'][57]]
    path_control = PublicPosePath(targets, child['code_position_tolerance'], child['code_rotation_tolerance'])
    relation_control = PublicRelationApproach(targets[1], targets[2])
    dump(output/'PUBLIC_TARGETS.json', dict(targets=targets,
        origin='Current public EEF + frozen P53 public references 41 and 57',
        continuation='Public ref41->ref57 unit EEF axis and ref57 pose; no scale or private geometry'))
    stage = 's7'
    actions = 0
    while True:
        measured = private_measurement(env, obs, names, child['relation'])
        gate = near_gate(child['relation'], measured, gate_binding)
        transitions = []
        reached = False
        if stage in ('s7', 's8'):
            active = nodes[stage]
            action, reached, diagnostic = path_control.command(obs['robot0_eef_pos'],
                obs['robot0_eef_quat'], active['reference_start'], active['reference_end'])
            if reached and stage == 's7':
                transitions.append(dict(from_stage='s7', to_stage='s8'))
                stage = 's8'
                active = nodes[stage]
                action, reached, diagnostic = path_control.command(obs['robot0_eef_pos'],
                    obs['robot0_eef_quat'], active['reference_start'], active['reference_end'])
            if reached and stage == 's8':
                transitions.append(dict(from_stage='s8', to_stage='s8a'))
                stage = 's8a'
        if stage == 's8a':
            action, diagnostic = relation_control.command(obs['robot0_eef_pos'], obs['robot0_eef_quat'])
            diagnostic['public_reference_index'] = 57
        ready = bool(stage == 's8a' and gate['near_ready'])
        stop = ('code_completed_goal' if gate['code_completed_goal'] else
                'code_structural_stop' if gate['structural_stop'] else
                'near_ready' if ready else 'code_budget' if actions >= graph['shared_code_budget'] else None)
        frame = output/f'frame_{actions:04d}.npz'
        digest = record_private(env, obs, frame)
        row = dict(t=actions, stage_id=stage, file=frame.name, sha256=digest,
            action=None if stop else action.tolist(), public_controller=diagnostic,
            supervisor_gate=gate, private_geometry=geometry(child['relation'], measured),
            transitions=transitions, endpoint_reached=reached, stop_reason=stop)
        with (output/'TRACE.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if stop:
            return dict(qualified=ready, actions=actions, stop_reason=stop,
                        next_stage='s9' if ready else None, near_entry_qualified=ready,
                        completed_goal_before_s9=stop == 'code_completed_goal')
        obs, _, _, _ = env.step(action.copy())
        actions += 1


def run_neural(env, binding, runtime, vision, renderer, output, names):
    node = next(n for n in binding['task_graph']['nodes'] if n['id'] == 's9')
    if node['backend'] != 'frozen_stage_policy' or node['max_actions'] != 24:
        raise ValueError('P55 frozen neural s9 changed')
    judge = InsertJudge()
    obs = env._get_observations(force_update=True)
    previous = previous_action = None
    steps = 0
    while True:
        path = output/'public'/f'frame_{steps:04d}.npz'
        np.savez_compressed(path, **renderer.observe(obs))
        features = vision.encode(path)
        validate_features(features, runtime.model.config)
        torch.save(features, output/'public'/f'features_{steps:04d}.pt')
        if previous is None: runtime.enter(features)
        before_z = runtime.z.clone()
        loss = None
        if previous is not None: loss = runtime.feedback(previous, previous_action, features)
        action = runtime.command(features).numpy()
        if action.shape != (7,) or not np.isfinite(action).all() or action[-1] != 1:
            raise ValueError('Invalid frozen neural action')
        official, fingers, wrong = structural_measurement(env, names)
        judgment = judge.observe(official, fingers, wrong)
        stop = judgment['stop'] or ('micro_budget24' if steps >= 24 else None)
        row = dict(t=steps, stage_id='s9', backend=node['backend'], observation=path.name,
            sha256=sha(path), z_before=before_z.tolist(), z_after=runtime.z.tolist(),
            inner_loss=loss, updates=runtime.updates, action=action.tolist(), executed=stop is None)
        with (output/'public/ACTIONS.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        private, truth = snapshot(env, obs)
        np.savez_compressed(output/'private'/f'frame_{steps:04d}.npz',
            **{k:v for k,v in private.items() if k.startswith('supervisor_')})
        with (output/'private/JUDGMENTS.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(t=steps, official_success=official, finger_contacts=fingers,
                wrong_target=wrong, contacts=truth['contacts'], **judgment), allow_nan=False)+'\n')
        if stop:
            return dict(actions=steps, graph_complete=judgment['complete'], official_success=official,
                        final_judgment=judgment, stop_reason=stop, updates=runtime.updates)
        previous = features
        previous_action = torch.tensor(action[:6], dtype=torch.float32)
        obs, _, _, _ = env.step(action.copy())
        steps += 1


def _calibration(original, scale, output):
    path = output/'SUPERVISOR_CALIBRATION.json'
    base = json.loads(Path(original['prefix']['calibration']).read_text())
    dump(path, scaled_supervisor_calibration(base, scale))
    return path


def _seal(output, file_names, binding_hash, cell, before=None, after=None):
    files = {str(p.relative_to(output)): sha(p) for d in ('prefix','code','public','private')
             for p in sorted((output/d).iterdir()) if p.is_file()}
    files.update({name: sha(output/name) for name in file_names})
    seal = dict(files=files, binding_sha256=binding_hash, cell=cell,
                original_initial_restores=1, later_state_restores=0)
    if before is not None:
        seal.update(network_before_sha256=before, network_after_sha256=after)
    dump(output/'SEAL.json', seal)
    return sha(output/'SEAL.json')


def qualify(binding_path, output):
    binding_path = Path(binding_path)
    binding = json.loads(binding_path.read_text())
    parent, original = verify(binding)
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'BINDING.json', binding)
    outcomes = []
    for scale in SCALES:
        cell = next(c for c in binding['cells'] if c['scale'] == scale and c['condition'] == 'nominal' and c['arm'] == 'bc')
        folder = output/f'scale_{scale:.2f}'
        folder.mkdir()
        for name in ('prefix','code','public','private'): (folder/name).mkdir()
        cal = _calibration(original, scale, folder)
        child = child_binding(parent, original, scale, cal, binding['task_graph'])
        dump(folder/'BINDING.json', binding)
        dump(folder/'CELL.json', cell)
        dump(folder/'TASK_GRAPH.json', child['task_graph'])
        env = None
        try:
            np.random.seed(binding['seed'])
            env, z_shift = initialize_child(child, folder, scale, parent['geometry']['properties']['halfsize'])
            names = _names(env)
            dump(folder/'private/NAMES.json', names)
            prefix = run_prefix(env, child, folder/'prefix', folder.name,
                dict(grasp_world_offset=cell['grasp_world_offset']))
            dump(folder/'PREFIX_RESULT.json', prefix)
            code = (run_code(env, child, parent, folder/'code', names) if prefix['qualified'] else
                    dict(qualified=False, actions=0, near_entry_qualified=False, stop_reason='prefix_failed'))
            dump(folder/'CODE_RESULT.json', code)
            seal_hash = _seal(folder, ('BINDING.json','CELL.json','TASK_GRAPH.json','SUPERVISOR_CALIBRATION.json',
                'PREFIX_RESULT.json','CODE_RESULT.json'), sha(binding_path), cell)
            result = dict(scale=scale, condition='nominal', arm='bc', prefix_qualified=prefix['qualified'],
                near_entry_qualified=code['qualified'], prefix_actions=prefix['actions'],
                code_actions=code['actions'], official_success_at_stop=bool(env._check_success()),
                micro_actions=0, teacher_actions=0, initial_book_z_shift=z_shift,
                stop_reason=code['stop_reason'], seal_sha256=seal_hash,
                original_initial_restores=1, later_state_restores=0, automatic_retry=False)
            dump(folder/'RESULT.json', result)
            outcomes.append(result)
        except Exception as exc:
            dump(folder/'ABORT.json', dict(error=type(exc).__name__, reason=str(exc), automatic_retry=False))
            raise
        finally:
            if env is not None: env.close()
    passed = bool(len(outcomes) == 2 and all(r['prefix_qualified'] and r['near_entry_qualified']
        and not r['official_success_at_stop'] and r['micro_actions'] == 0 for r in outcomes))
    summary = dict(schema=QUAL_SCHEMA, passed=passed, binding_sha256=sha(binding_path),
        outcomes=outcomes, scale_count=2, source='original initial state, real prefix and Code only',
        teacher_executed=False, neural_executed=False, automatic_retries=0)
    dump(output/'RESULT.json', summary)
    return summary


def run(binding_path, qualification_path, index, output):
    binding_path, qualification_path = Path(binding_path), Path(qualification_path)
    binding = json.loads(binding_path.read_text())
    parent, original = verify(binding)
    qualification = json.loads(qualification_path.read_text())
    if qualification.get('schema') != QUAL_SCHEMA or not qualification.get('passed') or qualification.get('binding_sha256') != sha(binding_path):
        raise ValueError('Two-scale real Code qualification required')
    if index not in range(42): raise ValueError('Outside P55 42-cell scope')
    cell = binding['cells'][index]
    output.mkdir(parents=True, exist_ok=False)
    for name in ('prefix','code','public','private'): (output/name).mkdir()
    cal = _calibration(original, cell['scale'], output)
    child = child_binding(parent, original, cell['scale'], cal, binding['task_graph'])
    dump(output/'BINDING.json', binding)
    dump(output/'CELL.json', cell)
    dump(output/'TASK_GRAPH.json', child['task_graph'])
    torch.set_num_threads(1)
    torch.manual_seed(binding['seed'])
    np.random.seed(binding['seed'])
    ck = torch.load(original['checkpoints'][cell['arm']]['path'], map_location='cpu', weights_only=True)
    model = StagePolicy(Config(**ck['config']))
    model.load_state_dict(ck['model'], strict=True)
    runtime = StageRuntime(model, online=cell['zmode'] == 'observed_feedback')
    before = weight_digest(model)
    env = vision = renderer = None
    try:
        env, z_shift = initialize_child(child, output, cell['scale'], parent['geometry']['properties']['halfsize'])
        names = _names(env)
        dump(output/'private/NAMES.json', names)
        prefix = run_prefix(env, child, output/'prefix', output.name,
            dict(grasp_world_offset=cell['grasp_world_offset']))
        dump(output/'PREFIX_RESULT.json', prefix)
        code = (run_code(env, child, parent, output/'code', names) if prefix['qualified'] else
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
            transitions.append(dict(from_stage='s8a', to_stage='s9', after_actions=prefix['actions']+code['actions']))
            vision = Vision(original['vision'])
            dump(output/'VISUAL_BINDING.json', vision.binding)
            renderer = PublicRenderer(env, original['public'])
            result = run_neural(env, child, runtime, vision, renderer, output, names)
        else:
            result = dict(actions=0, graph_complete=False, official_success=bool(env._check_success()),
                          stop_reason=code['stop_reason'], updates=0)
        if result['graph_complete']:
            transitions.append(dict(from_stage='s9', to_stage=None,
                after_actions=prefix['actions']+code['actions']+result['actions']))
        after = weight_digest(model)
        if before != after: raise RuntimeError('Frozen network weights mutated')
        verify(binding)
        dump(output/'STAGE_TRANSITIONS.json', transitions)
        names_to_seal = ['BINDING.json','CELL.json','TASK_GRAPH.json','SUPERVISOR_CALIBRATION.json',
            'PREFIX_RESULT.json','CODE_RESULT.json','STAGE_TRANSITIONS.json']
        if (output/'VISUAL_BINDING.json').exists(): names_to_seal.append('VISUAL_BINDING.json')
        seal_hash = _seal(output, names_to_seal, sha(binding_path), cell, before, after)
        result.update(cell=cell, initial_book_z_shift=z_shift, prefix_actions=prefix['actions'],
            code_actions=code['actions'], near_entry_qualified=code['qualified'],
            total_actions=prefix['actions']+code['actions']+result['actions'],
            network_weights_unchanged=before == after, seal_sha256=seal_hash,
            qualification_sha256=sha(qualification_path), automatic_retry=False,
            independent_statistical_evidence=False,
            failure_stage=None if result['graph_complete'] else 's9' if code['qualified'] else
                (json.loads((output/'code/TRACE.jsonl').read_text().splitlines()[-1])['stage_id']
                 if prefix['qualified'] else 'prefix'))
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
    if qualified.get('schema') != QUAL_SCHEMA or not qualified.get('passed') or qualified.get('binding_sha256') != sha(binding_path):
        raise ValueError('P55 two-scale qualification has not passed')
    if not 0 < timeout_seconds <= 1800: raise ValueError('Finite <=1800 s per-cell timeout required')
    jobs = [dict(id=f"cell_{c['index']:02d}_{c['condition']}_{c['arm']}_s{int(c['scale']*100)}",
        seed=binding['seed'], timeout_seconds=timeout_seconds,
        argv=[str(python), str(Path(__file__).resolve()), 'run', '--binding', str(binding_path),
              '--qualification', str(qualification_path), '--index', str(c['index'])]) for c in binding['cells']]
    return dict(schema='p53.explicit-queue.v1', cwd=str(Path(cwd).resolve()), workers=3,
        attempts_per_job=1, bound_files={str(binding_path):sha(binding_path),
            str(qualification_path):sha(qualification_path), str(Path(__file__).resolve()):sha(__file__),
            str(ROOT/'prove/task_graph/p55_relation_approach.py'):sha(ROOT/'prove/task_graph/p55_relation_approach.py')},
        jobs=jobs, scope='P55 fixed 42-cell complete original-task comparison after two real near-entry qualifications')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--zero-qualification', type=Path, required=True)
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
        binding = prepare(args.parent, args.zero_qualification)
        verify(binding)
        dump(args.output, binding)
    elif args.command == 'qualify':
        print(json.dumps(qualify(args.binding, args.output), indent=2))
    elif args.command == 'queue-manifest':
        if args.output.exists(): raise FileExistsError(args.output)
        dump(args.output, queue_manifest(args.binding, args.qualification,
            args.python, args.cwd, args.timeout_seconds))
    else:
        print(json.dumps(run(args.binding, args.qualification, args.index, args.output), indent=2))
