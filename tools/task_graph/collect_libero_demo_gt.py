#!/usr/bin/env python3
"""Freeze then replay one original LIBERO demo with explicitly privileged GT.

prepare is simulator-free. collect requires the resulting binding, constructs
one environment, restores only the initial state, and plays each action once.
This is teacher data, never a runtime observer or independent success trial.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASK = 'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy'
SCHEMA = 'prove.libero-one-demo-gt-binding.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def source_demo(path):
    with h5py.File(path, 'r') as handle:
        data = handle['data']
        group = data['demo_0']
        actions, states = group['actions'][()], group['states'][()]
        initial = np.asarray(group.attrs['init_state'])
        if actions.ndim != 2 or actions.shape[1] != 7 or not 0 < len(actions) <= 1000:
            raise ValueError('Expected one bounded 7D original demo')
        if states.ndim != 2 or len(states) != len(actions) or initial.shape != states.shape[1:]:
            raise ValueError('State/action alignment mismatch')
        if not all(np.isfinite(v).all() for v in (actions, states, initial)):
            raise ValueError('Nonfinite demo')
        if not np.array_equal(initial, states[0]):
            raise ValueError('Initial-state attribute must agree with recorded first pre-action state')
        if Path(str(data.attrs['bddl_file_name'])).stem != TASK:
            raise ValueError('Only the root-selected book/caddy demo is in scope')
        return dict(actions=actions, states=states, initial=initial,
                    xml=str(group.attrs['model_file']),
                    env_args=json.loads(data.attrs['env_args']))


def relocate_xml(xml, assets_root, robosuite_root):
    """Change asset file locations only; freeze every referenced asset byte."""
    tree = ET.fromstring(xml)
    bindings = {}
    for element in tree.iter():
        old = element.get('file')
        if old is None:
            continue
        if '/robosuite/' in old:
            target = robosuite_root / old.rsplit('/robosuite/', 1)[1]
        elif '/assets/' in old:
            target = assets_root / old.split('/assets/', 1)[1]
        else:
            raise ValueError(f'Unknown model asset path: {old}')
        target = target.resolve(strict=True)
        bindings[str(target)] = sha(target)
        element.set('file', str(target))
    return ET.tostring(tree, encoding='unicode'), bindings


def source_files(libero_root, robosuite_root):
    result = {}
    for root in (libero_root / 'libero', robosuite_root):
        for path in sorted(root.rglob('*.py')):
            if '__pycache__' not in path.parts:
                result[str(path.resolve())] = sha(path)
    return result


def prepare(source, libero_root, robosuite_root):
    source, libero_root, robosuite_root = map(lambda p: Path(p).resolve(),
                                            (source, libero_root, robosuite_root))
    demo = source_demo(source)
    bddl = libero_root / 'libero/libero/bddl_files/libero_10' / (TASK + '.bddl')
    assets = libero_root / 'libero/libero/assets'
    xml, asset_bindings = relocate_xml(demo['xml'], assets, robosuite_root)
    kwargs = dict(demo['env_args']['env_kwargs'])
    kwargs['bddl_file_name'] = str(bddl)
    if kwargs['controller_configs']['type'] != 'OSC_POSE' or not kwargs['controller_configs']['control_delta']:
        raise ValueError('Original delta OSC_POSE required')
    # Original kwargs are retained, including cameras, resolution and controller gains.
    return dict(schema=SCHEMA, authority='supervisor_demo_only', source=str(source),
                source_sha256=sha(source), group='data/demo_0', task=TASK,
                max_actions=len(demo['actions']), state_alignment='states[t] precedes actions[t]',
                libero_root=str(libero_root), robosuite_root=str(robosuite_root),
                bddl=str(bddl), bddl_sha256=sha(bddl), env_args_original=demo['env_args'],
                env_kwargs=kwargs, relocated_xml=xml, assets=asset_bindings,
                implementation={str(Path(__file__).resolve()): sha(__file__),
                                **source_files(libero_root, robosuite_root)},
                packages={name: importlib.metadata.version(name) for name in ('mujoco', 'robosuite', 'numpy', 'h5py')},
                scope=dict(episodes=1, initial_state_restores=1, action_replays=1,
                           automatic_retries=0, policy_services=False, learned_models=False,
                           physics_state_restoration_during_replay=False))


def verify(binding):
    if binding.get('schema') != SCHEMA or binding.get('group') != 'data/demo_0' or binding.get('task') != TASK:
        raise ValueError('Invalid one-demo binding')
    files = {binding['source']: binding['source_sha256'], binding['bddl']: binding['bddl_sha256'],
             **binding['assets'], **binding['implementation']}
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    for name, version in binding['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Frozen package changed: {name}')
    regenerated = prepare(binding['source'], binding['libero_root'], binding['robosuite_root'])
    if regenerated != binding:
        raise ValueError('Binding content differs from frozen source/configuration')


def snapshot(env, obs):
    data = env.sim.data
    arrays = {'supervisor_sim_state': env.sim.get_state().flatten().copy()}
    for field in ('qpos', 'qvel', 'qacc', 'act', 'ctrl', 'body_xpos', 'body_xquat',
                  'body_xmat', 'site_xpos', 'site_xmat', 'geom_xpos', 'geom_xmat'):
        arrays['supervisor_' + field] = np.asarray(getattr(data, field)).copy()
    for key in ('agentview_image', 'robot0_eye_in_hand_image', 'robot0_joint_pos',
                'robot0_joint_vel', 'robot0_gripper_qpos', 'robot0_eef_pos', 'robot0_eef_quat'):
        if key in obs:
            arrays['observation_' + key] = np.asarray(obs[key]).copy()
    contacts = [dict(geom1=int(data.contact[i].geom1), geom2=int(data.contact[i].geom2),
                     distance=float(data.contact[i].dist), position=data.contact[i].pos.tolist())
                for i in range(data.ncon)]
    goals = {json.dumps(state): bool(env._eval_predicate(state))
             for state in env.parsed_problem['goal_state']}
    return arrays, dict(goal_predicates=goals, success=bool(env._check_success()), contacts=contacts)


def replay(env, demo, output, capture=snapshot):
    """One pass only; caller has already restored exact initial state."""
    actions, states = demo['actions'], demo['states']
    if len(actions) != len(states):
        raise ValueError('Replay length mismatch')
    output = Path(output)
    rows, previous_goals = [], None
    obs = env._get_observations(force_update=True)
    for t in range(len(actions) + 1):
        arrays, supervisor = capture(env, obs)
        reference = states[t] if t < len(states) else None
        delta = None if reference is None else arrays['supervisor_sim_state'] - reference
        divergence = None if delta is None else dict(l2=float(np.linalg.norm(delta)),
                                                    max_abs=float(np.max(np.abs(delta))))
        goals = supervisor['goal_predicates']
        events = {key: value for key, value in goals.items()
                  if previous_goals is None or previous_goals.get(key) != value}
        arrays['action_next'] = actions[t] if t < len(actions) else np.empty((0,), dtype=actions.dtype)
        file = output / f'frame_{t:04d}.npz'
        np.savez_compressed(file, **arrays)
        row = dict(frame=t, file=file.name, sha256=sha(file), supervisor=supervisor,
                   supervisor_predicate_events=events, state_divergence=divergence)
        rows.append(row)
        with (output / 'FRAMES.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        previous_goals = goals
        if t < len(actions):
            obs, reward, done, info = env.step(actions[t].copy())
    divergences = [r['state_divergence']['l2'] for r in rows if r['state_divergence'] is not None]
    return dict(schema='prove.libero-one-demo-gt-result.v1', authority='supervisor_demo_only',
                mode='original_action_physics_replay', actions_executed=len(actions),
                frames=len(rows), success=rows[-1]['supervisor']['success'],
                max_recorded_state_l2=max(divergences),
                state_comparison='pre-action states[t]; last post-action frame has no source state',
                note='A replay may diverge; source demo success is not a claim of replay success')


def collect(binding_path, output):
    binding = json.loads(Path(binding_path).read_text())
    verify(binding)
    demo = source_demo(binding['source'])
    if len(demo['actions']) != binding['max_actions']:
        raise ValueError('Frozen action budget mismatch')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'BINDING.json', binding)
    # Scope/config are on disk before environment construction. No automatic retry.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    config_dir = output / 'libero_config'
    config_dir.mkdir()
    base = Path(binding['libero_root']) / 'libero/libero'
    # JSON is a valid YAML mapping and avoids interactive LIBERO first-run setup.
    dump(config_dir / 'config.yaml', dict(benchmark_root=str(base), bddl_files=str(base / 'bddl_files'),
         init_states=str(base / 'init_files'), datasets=str(Path(binding['source']).parent), assets=str(base / 'assets')))
    os.environ['LIBERO_CONFIG_PATH'] = str(config_dir.resolve())
    sys.path.insert(0, binding['libero_root'])
    from libero.libero.envs import TASK_MAPPING
    env = None
    try:
        env = TASK_MAPPING[binding['env_args_original']['problem_name']](**binding['env_kwargs'])
        # Call raw reset once, avoiding ControlEnv's unbounded randomization retry.
        env.reset()
        env.reset_from_xml_string(binding['relocated_xml'])
        env.sim.reset()
        env.sim.set_state_from_flattened(demo['initial'])
        env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        names = {kind: [getattr(env.sim.model, kind + '_id2name')(i) for i in range(getattr(env.sim.model, count))]
                 for kind, count in (('body', 'nbody'), ('site', 'nsite'), ('geom', 'ngeom'), ('joint', 'njnt'))}
        dump(output / 'SUPERVISOR_NAMES.json', names)
        result = replay(env, demo, output)
        result['binding_sha256'] = sha(output / 'BINDING.json')
        dump(output / 'RESULT.json', result)
        return result
    except Exception as exc:
        dump(output / 'FAILURE.json', dict(error_type=type(exc).__name__, error=str(exc),
             status='failed_partial_teacher_capture_not_success_evidence', automatic_retry=False))
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pre = sub.add_parser('prepare')
    pre.add_argument('--source', type=Path, required=True)
    pre.add_argument('--libero-root', type=Path, default=ROOT / 'data/sources/LIBERO-PRO')
    pre.add_argument('--robosuite-root', type=Path, required=True)
    pre.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('collect')
    run.add_argument('--binding', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.output.exists():
            raise FileExistsError(args.output)
        dump(args.output, prepare(args.source, args.libero_root, args.robosuite_root))
        print(args.output)
    else:
        print(json.dumps(collect(args.binding, args.output), indent=2))


if __name__ == '__main__':
    main()
