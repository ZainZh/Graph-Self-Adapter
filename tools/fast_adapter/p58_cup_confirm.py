#!/usr/bin/env python3
"""Result-free P58 cup confirmation: 32 new conditions x eight frozen arms."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from prove.fast_adapter.p45_neural import load_model as load_p47_model
from prove.fast_adapter.p47_neural import P47Latent
from prove.fast_adapter.p58_cup_runtime import load_runtime
from prove.fast_adapter.robot_geometry_env_p45_v02 import configuration, source_assets
from tools.fast_adapter.p45_common import js, make_controller, make_env, observe_roles, sha
from tools.fast_adapter.p47_evaluate import evaluate_sealed, parameters_sha, snapshot
from tools.fast_adapter.p58_cup_evaluate import ENVIRONMENT, SOURCES as PILOT_SOURCES, STEPS


SCHEMA = 'prove.p58-cup-confirmation-evaluation.v1'
ID_SEEDS = tuple(range(5858000, 5858016))
GEOMETRY_SEEDS = tuple(range(5858100, 5858116))
NEW_REPS = ('position_only', 'functional_frame')
NEW_ARMS = ('fixed', 'online', 'direct')
REFERENCE_REP = 'p47_geometry_reference'
REFERENCE_ARMS = ('fixed', 'continuous')
SOURCES = tuple(dict.fromkeys((*PILOT_SOURCES, 'tools/fast_adapter/p58_cup_confirm.py')))


def conditions():
    return [dict(condition_seed=seed, split=split,
                 configuration=configuration('cup', seed, split))
            for split, seeds in (('id', ID_SEEDS), ('geometry', GEOMETRY_SEEDS))
            for seed in seeds]


def cells():
    result = []
    for condition in conditions():
        common = dict(condition_seed=condition['condition_seed'], split=condition['split'])
        result.extend(dict(**common, representation=rep, arm=arm)
                      for rep in NEW_REPS for arm in NEW_ARMS)
        result.extend(dict(**common, representation=REFERENCE_REP, arm=arm)
                      for arm in REFERENCE_ARMS)
    return result


def _verify_file(reference, label):
    if sha(reference['path']) != reference['sha256']:
        raise ValueError(f'Frozen {label} changed: {reference["path"]}')


def validate(path):
    value = json.loads(Path(path).read_text())
    if (value.get('schema') != SCHEMA or value.get('status') != 'frozen'
            or value.get('root_authorized') is not True or value.get('steps') != STEPS
            or value.get('conditions') != conditions() or value.get('cells') != cells()):
        raise ValueError('Invalid P58 cup confirmation freeze')
    if value.get('environment') != ENVIRONMENT or value.get('robot_assets') != source_assets():
        raise ValueError('Panda environment/assets changed')
    for name, digest in value['sources'].items():
        if sha(name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for name, item in value['models'].items():
        _verify_file(item, 'model ' + name)
    _verify_file(value['plan'], 'plan')
    _verify_file(value['engineering_gate'], 'engineering gate')
    _verify_file(value['pilot']['freeze'], 'pilot freeze')
    _verify_file(value['pilot']['report'], 'pilot report')
    _verify_file(value['seed_audit'], 'seed audit')
    return value


def freeze(output, pilot_freeze, pilot_report, seed_audit):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pilot = json.loads(Path(pilot_freeze).read_text())
    if pilot.get('schema') != 'prove.p58-cup-development-evaluation.v1':
        raise ValueError('Confirmation requires the sealed P58 cup development freeze')
    if len(pilot.get('cells', ())) != 32:
        raise ValueError('Development provenance must contain all 32 pilot cells')
    for name, digest in pilot['sources'].items():
        if sha(name) != digest:
            raise ValueError('Pilot source changed before confirmation: ' + name)
    for name, item in pilot['models'].items():
        _verify_file(item, 'pilot model ' + name)
    report = json.loads(Path(pilot_report).read_text())
    if report.get('schema') != 'prove.p58-cup-development-report.v1' or report.get('attempts') != 32:
        raise ValueError('Confirmation requires the complete cup development audit')
    audit = json.loads(Path(seed_audit).read_text())
    expected_seeds = list(ID_SEEDS + GEOMETRY_SEEDS)
    if (audit.get('schema') != 'prove.p58-cup-confirmation-seed-audit.v1'
            or audit.get('proposed_seeds') != expected_seeds or audit.get('exact_matches') != []):
        raise ValueError('Confirmation seed non-overlap audit is invalid')
    value = dict(
        schema=SCHEMA, status='frozen', root_authorized=True, phase='confirmation',
        result_free_at_freeze=True, steps=STEPS, attempts=256,
        conditions=conditions(), cells=cells(), models=copy.deepcopy(pilot['models']),
        plan=copy.deepcopy(pilot['plan']), sources={name: sha(name) for name in SOURCES},
        engineering_gate=copy.deepcopy(pilot['engineering_gate']),
        environment=copy.deepcopy(ENVIRONMENT), robot_assets=source_assets(),
        pilot=dict(
            freeze=dict(path=str(Path(pilot_freeze)), sha256=sha(pilot_freeze)),
            report=dict(path=str(Path(pilot_report)), sha256=sha(pilot_report)),
            identity='opened development only; never pooled with confirmation outcomes'),
        seed_audit=dict(path=str(Path(seed_audit)), sha256=sha(seed_audit)),
        representation_contracts=dict(
            position_only='predicted grasp center plus training-mean orientation; not a complete pose',
            functional_frame='predicted center plus two axes with Procrustes pose recovery',
            p47_geometry_reference='historical visible-corner model'),
        input_boundary='P58 uses visible masks, calibration and EEF only; P47 reference retains visible corners',
        success='unchanged P45 sustained last10 bilateral contact and lift, with mechanical validity',
        reporting='ID and geometry OOD separate; all 256 attempts retained; program errors separate')
    output.parent.mkdir(parents=True, exist_ok=True)
    js(output, value)
    return value


def rollout(index, cell, frozen, out, freeze_path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model_ref = frozen['models'][cell['representation']]
    js(out / 'PRE_RUN.json', dict(index=index, cell=cell, freeze_sha256=sha(freeze_path),
        model=model_ref, steps=STEPS, timestamp=time.time(), runtime_truth=False))
    env = make_env('cup', cell['condition_seed'], cell['split'], backend=ENVIRONMENT)
    try:
        if env.cfg != next(item['configuration'] for item in frozen['conditions']
                           if item['condition_seed'] == cell['condition_seed'] and item['split'] == cell['split']):
            raise ValueError('Runtime condition differs from frozen configuration')
        controller = make_controller(env, backend=dict(
            controller_module='prove.fast_adapter.robot_geometry_env_p45_v02'))
        if cell['representation'] == REFERENCE_REP:
            model, _ = load_p47_model(model_ref['path'])
            runtime = P47Latent(model, cell['arm'])
            new = False
        else:
            runtime = load_runtime(model_ref['path'], cell['arm'])
            model = runtime.model
            new = True
        before = parameters_sha(model)
        trace, observations = [], []
        for t in range(STEPS):
            obs, _, masks = observe_roles(env)
            if not controller.adaptation_frozen:
                if new:
                    estimate = runtime.update(obs, masks, t)
                else:
                    runtime.update(dict(obs, local=np.zeros(48, np.float32)), t)
                    estimate = runtime.estimate()
            else:
                estimate = runtime.estimate()
            p, rotation, closed, stage = controller.command(t, estimate, env.proprio())
            row = dict(t=t, state=snapshot(env), command_p=p.copy(), command_R=rotation.copy(),
                closed=bool(closed), stage=stage, estimate=copy.deepcopy(estimate),
                adaptation=copy.deepcopy(runtime.records[-1]) if runtime.records else None,
                controls_before=env.data.ctrl.copy())
            observations.append(dict(t=t, corners=copy.deepcopy(obs['corners']),
                camera_position=obs['camera_position'].copy(), camera_rotation=obs['camera_rotation'].copy(),
                K=obs['K'].copy(), position=obs['position'].copy(), rotation=obs['rotation'].copy()))
            env.step(p, rotation, closed)
            row['controls_after'] = env.data.ctrl.copy()
            row['robot_tracking'] = env.robot_tracking_diagnostics()
            trace.append(row)
        terminal = snapshot(env)
        np.save(out / 'TRACE.npy', np.asarray(trace, dtype=object), allow_pickle=True)
        np.save(out / 'OBS.npy', np.asarray(observations, dtype=object), allow_pickle=True)
        np.save(out / 'TERMINAL.npy', terminal, allow_pickle=True)
        after = parameters_sha(model)
        js(out / 'SEAL.json', dict(schema='prove.p58-cup-confirmation-trajectory-seal.v1',
            files={name: sha(out / name) for name in ('TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'PRE_RUN.json')},
            neural_parameters_before=before, neural_parameters_after=after))
        result = evaluate_sealed(env, trace, terminal, out)
        result.update(cell=cell, index=index, neural_weights_unchanged=before == after,
            adaptation_updates=getattr(runtime, 'update_events', 0))
        js(out / 'RESULT.json', result)
        return result
    finally:
        env.close()


def run_one(index, freeze_path, output):
    output = Path(output)
    try:
        frozen = validate(freeze_path)
        result = rollout(index, frozen['cells'][index], frozen, output, freeze_path)
        return 0, {key: result[key] for key in
                   ('index', 'validated_success', 'outcome_class', 'neural_weights_unchanged')}
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        payload = dict(schema='prove.p58-cup-confirmation-cell-error.v1', index=index,
            outcome_class='program_error', error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), task_success_denominator_included=False,
            confirmation_attempt_denominator_included=True, timestamp=time.time())
        js(output / 'ERROR.json', payload)
        return 2, {key: payload[key] for key in ('index', 'outcome_class', 'error_type', 'error')}


def cell_output(root, index, cell):
    return Path(root) / f"{index:03d}_{cell['split']}_{cell['condition_seed']}_{cell['representation']}_{cell['arm']}"


def worker(freeze_path, output_root, worker_index, workers):
    frozen = validate(freeze_path)
    failures = 0
    for index, cell in enumerate(frozen['cells']):
        if index % workers != worker_index:
            continue
        command = [sys.executable, '-m', 'tools.fast_adapter.p58_cup_confirm', 'run',
            '--freeze', freeze_path, '--index', str(index),
            '--output', str(cell_output(output_root, index, cell))]
        failures += subprocess.run(command, env=os.environ.copy()).returncode != 0
    return int(failures > 0)


def queue(freeze_path, output_root, workers):
    if workers != 3:
        raise ValueError('The frozen confirmation queue uses exactly three workers')
    validate(freeze_path)
    processes = []
    for worker_index in range(workers):
        environment = os.environ.copy()
        environment['MUJOCO_GL'] = 'egl'
        environment['EGL_DEVICE_ID'] = str(worker_index)
        environment['MUJOCO_EGL_DEVICE_ID'] = str(worker_index)
        command = [sys.executable, '-m', 'tools.fast_adapter.p58_cup_confirm', 'worker',
            '--freeze', freeze_path, '--output-root', output_root,
            '--worker-index', str(worker_index), '--workers', str(workers)]
        processes.append(subprocess.Popen(command, env=environment))
    codes = [process.wait() for process in processes]
    return int(any(codes)), codes


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('freeze')
    make.add_argument('--output', required=True)
    make.add_argument('--pilot-freeze', required=True)
    make.add_argument('--pilot-report', required=True)
    make.add_argument('--seed-audit', required=True)
    run = commands.add_parser('run')
    run.add_argument('--freeze', required=True)
    run.add_argument('--index', type=int, required=True)
    run.add_argument('--output', required=True)
    work = commands.add_parser('worker')
    work.add_argument('--freeze', required=True)
    work.add_argument('--output-root', required=True)
    work.add_argument('--worker-index', type=int, required=True)
    work.add_argument('--workers', type=int, default=3)
    launch = commands.add_parser('queue')
    launch.add_argument('--freeze', required=True)
    launch.add_argument('--output-root', required=True)
    launch.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.command == 'freeze':
        value = freeze(args.output, args.pilot_freeze, args.pilot_report, args.seed_audit)
        print(json.dumps(dict(schema=value['schema'], conditions=len(value['conditions']),
                              cells=len(value['cells'])), indent=2))
    elif args.command == 'run':
        code, result = run_one(args.index, args.freeze, args.output)
        print(json.dumps(result, indent=2))
        raise SystemExit(code)
    elif args.command == 'worker':
        raise SystemExit(worker(args.freeze, args.output_root, args.worker_index, args.workers))
    else:
        code, codes = queue(args.freeze, args.output_root, args.workers)
        print(json.dumps(dict(worker_exit_codes=codes), indent=2))
        raise SystemExit(code)


if __name__ == '__main__':
    main()
