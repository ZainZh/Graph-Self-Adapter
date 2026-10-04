#!/usr/bin/env python3
"""P50 bounded development pilot. Prepare/review first; never auto-run physics.

Thirty-two paired insert episodes compare neural/scripted actions under the
same frozen geometry head and fixed/continuous z. Evaluator-only configurations
and states are saved outside controller observations. Native success/facts are
read only after full trace sealing, using the unchanged P47 evaluator.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from prove.task_graph.precision_action_policy import (
    PrecisionActionController, encode_state, load_frozen_head, training_contract,
)
from tools.task_graph.train_precision_action_policy import SOURCES as OFFLINE_SOURCES, dump_new, sha

ADAPTER_SHA = '4c195b345a6a147bdeaa4bb5ec8614600f599196c6919a451616e99d964e3d25'
HEAD_SHA = '96fdcf6a8ed49e53615bdbff8d3f9cf69ffed85353c18f0584b31e024b4b4fc7'
BACKEND = dict(module='prove.fast_adapter.robot_geometry_env_p45_v02', class_name='RobotGeometrySkill',
               configuration_module='prove.fast_adapter.geometry3d_env',
               controller_module='prove.fast_adapter.robot_geometry_env_p45')
SOURCE_ROOTS = [
    'tools/task_graph/p50_precision_pilot.py', 'tests/test_p50_precision_pilot.py',
    'prove/fast_adapter/robot_geometry_env_p45_v02.py',
    'prove/fast_adapter/robot_geometry_env_p45.py',
    'prove/fast_adapter/p47_neural.py', 'tools/fast_adapter/p47_evaluate.py',
    'tools/fast_adapter/p45_common.py',
]


def cells():
    return [dict(task='insert', split=split, condition_seed=base + i,
                 controller=controller, arm=arm, rep='geometry', model_seed=11,
                 video=controller == 'neural' and i == 0)
            for split, base in [('id', 505100), ('geometry', 505200)]
            for i in range(4) for controller in ('neural', 'scripted')
            for arm in ('fixed', 'continuous')]


def cell_name(index, cell):
    return f"{index:02d}_{cell['split']}_{cell['condition_seed']}_{cell['controller']}_{cell['arm']}"


def source_closure():
    """Conservative recursive repository import closure, including packages."""
    pending, result = list(SOURCE_ROOTS) + list(OFFLINE_SOURCES), {}
    while pending:
        relative = pending.pop()
        if relative in result:
            continue
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f'Required pilot source missing: {relative}')
        result[relative] = sha(path)
        tree = ast.parse(path.read_text())
        modules = []
        path_parts = Path(relative).with_suffix('').parts
        package = list(path_parts[:-1])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if node.level > len(package):
                        raise ValueError(f'Relative import escapes source package: {relative}')
                    base = package[:len(package) - node.level + 1]
                    module = '.'.join(base + (node.module.split('.') if node.module else []))
                else:
                    module = node.module or ''
                if module:
                    modules.append(module)
                    modules.extend(module + '.' + alias.name for alias in node.names)
        for module in modules:
            if not module.startswith(('prove.', 'tools.')):
                continue
            parts = module.split('.')
            candidate = '/'.join(parts) + '.py'
            if (ROOT / candidate).is_file():
                pending.append(candidate)
            for i in range(1, len(parts) + 1):
                package = '/'.join(parts[:i]) + '/__init__.py'
                if (ROOT / package).is_file():
                    pending.append(package)
    return dict(sorted(result.items()))


def package_versions():
    import cv2
    import mujoco
    import scipy
    return dict(python=sys.version.split()[0], numpy=np.__version__, torch=str(torch.__version__),
                scipy=scipy.__version__, mujoco=mujoco.__version__, opencv=cv2.__version__)


def validate_offline_evidence(head, offline_query):
    """Bind the qualified mapping back to its immutable training provenance."""
    offline_query = Path(offline_query)
    paths = dict(offline_query=offline_query,
                 offline_training_seal=offline_query.parent / 'TRAINING_SEAL.json',
                 offline_freeze=offline_query.parent / 'OFFLINE_FREEZE.json',
                 offline_training_start=offline_query.parent / 'TRAINING_START.json')
    query, seal, freeze, start = [json.loads(paths[name].read_text()) for name in
                                 ('offline_query', 'offline_training_seal', 'offline_freeze', 'offline_training_start')]
    if (query.get('schema') != 'prove.p50-offline-action-query.v1'
            or query.get('checkpoint_sha256') != HEAD_SHA or sha(head) != HEAD_SHA
            or query.get('offline_action_approximation_qualified') is not True
            or query.get('robot_rollouts') != 0
            or query.get('training_seal_sha256') != sha(paths['offline_training_seal'])
            or seal.get('schema') != 'prove.p50-offline-training-seal.v1'
            or seal.get('checkpoint_sha256') != HEAD_SHA
            or seal.get('offline_freeze_sha256') != sha(paths['offline_freeze'])
            or seal.get('training_start_sha256') != sha(paths['offline_training_start'])
            or start.get('offline_freeze_sha256') != sha(paths['offline_freeze'])
            or freeze.get('schema') != 'prove.p50-offline-freeze.v1'
            or freeze.get('contract') != training_contract()):
        raise ValueError('Offline query/training seal/freeze/checkpoint provenance chain mismatch')
    if set(freeze['source_sha256']) != set(OFFLINE_SOURCES):
        raise ValueError('Offline qualified source inventory mismatch')
    for name, expected in freeze['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Offline-qualified mapping source changed: {name}')
    return paths


def freeze_pilot(output, adapter, head, offline_query, plan):
    """Simulator-free binding. Root approval is a separate hash-bound receipt."""
    from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
    from prove.fast_adapter.geometry3d_env import configuration
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if sha(adapter) != ADAPTER_SHA or sha(head) != HEAD_SHA:
        raise ValueError('Wrong frozen adapter or action checkpoint')
    offline_paths = validate_offline_evidence(head, offline_query)
    artifact_paths = dict(adapter=adapter, action_head=head, plan=plan, **offline_paths)
    frozen = dict(schema='prove.p50-physical-development-freeze.v1', status='frozen',
                  root_approval_required=True, phase='development', task='insert', cells=cells(),
                  steps_per_episode=240, automatic_retries=0,
                  backend=BACKEND, model_device='cpu', model_threads=1,
                  renderer_gpu_ids=[0, 1], physics_workers_maximum=2,
                  sources=source_closure(), packages=package_versions(), robot_assets=source_assets(),
                  files={name: dict(path=str(Path(path).resolve().relative_to(ROOT)), sha256=sha(path))
                         for name, path in artifact_paths.items()},
                  conditions=[dict(split=split, condition_seed=seed, evaluator_configuration=configuration('insert', seed, split))
                              for split, base in [('id', 505100), ('geometry', 505200)] for seed in range(base, base + 4)],
                  policy_contract=dict(input='public corners/masks/calibration/robot pose; estimated slot geometry and calibrated held length',
                                       geometry='unchanged P47Latent G seed11', neural_action='frozen P50 MLP at t>=12 with valid geometry',
                                       missing_geometry='neural holds measured EEF pose; scripted preserves original last-target hold',
                                       phase_judges='same .003 m/.045 rad, three consecutive arrivals',
                                       first12='same public observation motion', gripper='closed preheld peg'),
                  evaluation_contract=dict(method='unchanged P47 evaluate_sealed', facts_access='only after SEAL.json',
                                           denominator='all 32 prespecified cells including ordinary failures and explicit aborts',
                                           offline_query_is_task_success=False, independent_success_rate_claim=False),
                  counterfactual='deepcopy controller before issued command; same robot state/current observation/phase with pre-update decoded points',
                  head_domain='log all excursions; no filtering, retuning, retry, or replacement')
    dump_new(output / 'PHYSICAL_FREEZE.json', frozen)
    snapshot_dir = output / 'source_snapshot'
    snapshot_dir.mkdir(exist_ok=False)
    for name, digest in frozen['sources'].items():
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        if sha(target) != digest:
            raise ValueError('Source changed while creating freeze snapshot')
    return frozen


def validate_freeze(path, approval=None, check_runtime=True):
    frozen = json.loads(Path(path).read_text())
    if (frozen.get('schema') != 'prove.p50-physical-development-freeze.v1'
            or frozen.get('cells') != cells() or frozen.get('backend') != BACKEND
            or frozen.get('steps_per_episode') != 240 or frozen.get('automatic_retries') != 0
            or frozen.get('phase') != 'development' or frozen.get('model_device') != 'cpu'
            or frozen.get('model_threads') != 1 or frozen.get('renderer_gpu_ids') != [0, 1]
            or frozen.get('physics_workers_maximum') != 2):
        raise ValueError('Pilot scope differs from the reviewed fixed 32-cell development design')
    if approval is not None:
        receipt = json.loads(Path(approval).read_text())
        if (receipt.get('schema') != 'prove.p50-root-physical-approval.v1'
                or receipt.get('root_reviewed') is not True or receipt.get('physical_execution_authorized') is not True
                or receipt.get('freeze_sha256') != sha(path)):
            raise ValueError('Physical execution requires root approval bound to this exact freeze')
    if check_runtime:
        if frozen['sources'] != source_closure():
            raise ValueError('Pilot runtime source closure changed')
        if frozen['packages'] != package_versions():
            raise ValueError('Pilot runtime package versions changed')
        from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
        if frozen['robot_assets'] != source_assets():
            raise ValueError('Robot assets changed')
        for name, item in frozen['files'].items():
            if sha(ROOT / item['path']) != item['sha256']:
                raise ValueError(f'Frozen pilot artifact changed: {name}')
        validate_offline_evidence(ROOT / frozen['files']['action_head']['path'], ROOT / frozen['files']['offline_query']['path'])
        if frozen['files']['adapter']['sha256'] != ADAPTER_SHA or frozen['files']['action_head']['sha256'] != HEAD_SHA:
            raise ValueError('Pilot checkpoint binding changed')
    return frozen


def domain_excursions(features):
    """Report physical feature values; never reject an OOD episode by them."""
    x = np.asarray(features, float)
    if x.shape != (13,) or not np.isfinite(x).all():
        raise ValueError('Finite 13D head features required for domain logging')
    domain = training_contract()['domain']
    p, size, length = x[:3] * .2, x[6:8] * .04, x[8] * .1
    angle = float(np.linalg.norm(x[3:6]))
    violated = []
    for i, label in enumerate(('relative_x', 'relative_y', 'relative_z')):
        if p[i] < domain['translation_lower'][i] - 1e-7 or p[i] > domain['translation_upper'][i] + 1e-7:
            violated.append(label)
    if angle > domain['rotation_angle'][1] + 1e-7:
        violated.append('relative_rotation_angle')
    for i, label in enumerate(('estimated_half_size_x', 'estimated_half_size_y')):
        if size[i] < domain['half_size_lower'][i] - 1e-7 or size[i] > domain['half_size_upper'][i] + 1e-7:
            violated.append(label)
    if length < domain['held_length'][0] - 1e-7 or length > domain['held_length'][1] + 1e-7:
        violated.append('calibrated_held_length')
    return dict(outside_training_domain=bool(violated), violated_fields=violated,
                relative_position=p.tolist(), rotation_angle=angle,
                estimated_half_size=size.tolist(), calibrated_held_length=float(length))


def controller_state(controller):
    return copy.deepcopy({k: v for k, v in controller.__dict__.items() if k not in ('model', 'records')})


def shadow_command(controller, t, estimate, proprio):
    # Frozen model is shared; history is not controller dynamics. Avoid O(T^2)
    # copying while independently cloning every mutable phase/state field.
    memo = {}
    if hasattr(controller, 'model'):
        memo[id(controller.model)] = controller.model
    if hasattr(controller, 'records'):
        memo[id(controller.records)] = []
    shadow = copy.deepcopy(controller, memo)
    return shadow.command(t, estimate, proprio)


def seal_trajectory(output, filenames, model_before, model_after, freeze_binding):
    seal = dict(schema='prove.p50-trajectory-seal.v1', freeze=freeze_binding,
                files={name: sha(Path(output) / name) for name in filenames},
                neural_parameters_before=model_before, neural_parameters_after=model_after)
    dump_new(Path(output) / 'SEAL.json', seal)
    if model_before != model_after:
        raise RuntimeError('Frozen neural weights changed; evaluator must not run')
    return seal


def run_cell(freeze_path, approval_path, index, gpu):
    if approval_path is None:
        raise ValueError('Root approval receipt required before any simulator construction')
    if gpu not in (0, 1):
        raise ValueError('Only frozen renderer GPUs 0 and 1 are allowed; GPU3 is excluded')
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
    torch.set_num_threads(1)
    frozen = validate_freeze(freeze_path, approval_path)
    if not 0 <= index < len(frozen['cells']):
        raise ValueError('Cell index outside frozen scope')
    cell = frozen['cells'][index]
    output = Path(freeze_path).parent / 'cells' / cell_name(index, cell)
    output.mkdir(parents=True, exist_ok=False)  # Existing attempt cannot be rerun.
    freeze_binding = dict(path=str(freeze_path), sha256=sha(freeze_path))
    dump_new(output / 'PRE_RUN.json', dict(cell_index=index, cell=cell, freeze=freeze_binding,
                                         approval_sha256=sha(approval_path), renderer_gpu=gpu,
                                         steps=240, source_snapshot='../../source_snapshot',
                                         evaluator_only_state='saved but not supplied to policy',
                                         insertion_secure_grasp_assumed=True))
    from prove.fast_adapter.geometry3d_env import configuration
    from prove.fast_adapter.geometry3d_adapter import pose
    from prove.fast_adapter.p47_neural import P47Latent
    from prove.fast_adapter.p45_neural import load_model
    from prove.fast_adapter.robot_geometry_env_p45 import GeometryTaskController
    from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
    from tools.fast_adapter.p45_common import observe_roles
    from tools.fast_adapter.p47_evaluate import PUBLIC_KEYS, evaluate_sealed, parameters_sha, snapshot
    env = None
    traces, observations, rgb_frames, mask_frames, video_frames, detail_frames = [], [], [], [], [], []
    started = time.monotonic()
    try:
        model, package = load_model(ROOT / frozen['files']['adapter']['path'], 'cpu')
        if model.task != 'insert' or model.representation != 'geometry' or package['seed'] != 11:
            raise ValueError('Wrong frozen adapter identity')
        model.eval().requires_grad_(False)
        head = load_frozen_head(ROOT / frozen['files']['action_head']['path'])
        before_hash = dict(adapter=parameters_sha(model), action_head=parameters_sha(head))
        cfg = configuration('insert', cell['condition_seed'], cell['split'])
        expected = next(row['evaluator_configuration'] for row in frozen['conditions']
                        if row['split'] == cell['split'] and row['condition_seed'] == cell['condition_seed'])
        if cfg != expected:
            raise ValueError('Environment configuration changed after freeze')
        env = RobotGeometrySkill(cfg, render=True)
        if env.model.nmocap != 0 or env.model.nu < 7:
            raise ValueError('Pilot requires articulated actuated Panda without mocap')
        env.calibrate_held_geometry()
        length = float(env.tool_calibration['length'])
        controller = (PrecisionActionController(head, env.start_pos, env.start_R, length=length)
                      if cell['controller'] == 'neural' else
                      GeometryTaskController('insert', env.start_pos, env.start_R, length=length))
        adapter = P47Latent(model, cell['arm'], device='cpu')
        p, R = env.proprio()
        previous_action = np.r_[p, R.ravel(), 1.]
        for t in range(240):
            observed, rgb, masks = observe_roles(env)
            obs = {key: copy.deepcopy(observed[key]) for key in PUBLIC_KEYS}
            obs['local'] = np.zeros(48, np.float32)
            if obs['corners'] is not None and not controller.adaptation_frozen:
                adapter.update(obs, t)
            estimate, proprio = adapter.estimate(), env.proprio()
            observations.append(dict(obs, t=t, previous_action=previous_action.copy(),
                                     adaptation_frozen=controller.adaptation_frozen))
            if cell['video'] or t == 0:
                rgb_frames.append(rgb.copy())
                mask_frames.append(masks.copy())
            before_state = controller_state(controller)
            update = copy.deepcopy(adapter.records[-1]) if adapter.records and adapter.records[-1]['t'] == t else None
            shadow = None
            if update is not None:
                shadow = shadow_command(controller, t, pose(np.asarray(update['pre_update_points'])), proprio)
            phase = 'align' if controller.phase == 'observe' else controller.phase
            domain = domain_excursions(encode_state(estimate, proprio, length, phase)) if t >= 12 and estimate is not None else None
            p, R, closed, stage = controller.command(t, estimate, proprio)
            if update is not None:
                p0, R0, closed0, stage0 = shadow
                update.update(same_state_action_position_delta=float(np.linalg.norm(p - p0)),
                              same_state_action_rotation_delta=float(np.linalg.norm(Rotation.from_matrix(R @ R0.T).as_rotvec())),
                              same_state_closed_changed=bool(closed != closed0), same_state_stage_changed=bool(stage != stage0),
                              shadow_command_position=np.asarray(p0).copy(), shadow_command_rotation=np.asarray(R0).copy())
            neural_record = copy.deepcopy(controller.records[-1]) if cell['controller'] == 'neural' and controller.records and controller.records[-1]['t'] == t else None
            row = dict(t=t, state=snapshot(env), command_p=p.copy(), command_R=R.copy(), closed=bool(closed), stage=stage,
                       estimate=copy.deepcopy(estimate), proprio=copy.deepcopy(proprio), controller_before=before_state,
                       controller_after=controller_state(controller), adaptation=update, head_domain=domain,
                       neural_action=neural_record, controls_before=env.data.ctrl.copy())
            if cell['video']:
                video_frames.append(np.concatenate([env.render_global(), rgb], axis=1))
                detail_frames.append(env.render_detail())
            env.step(p, R, closed)
            row['controls_after'] = env.data.ctrl.copy()
            row['robot_tracking'] = env.robot_tracking_diagnostics()
            traces.append(row)
            previous_action = np.r_[p, R.ravel(), float(closed)]
        terminal = snapshot(env)
        if cell['video']:
            _, rgb, _ = observe_roles(env)
            video_frames.append(np.concatenate([env.render_global(), rgb], axis=1))
            detail_frames.append(env.render_detail())
        np.save(output / 'TRACE.npy', np.array(traces, dtype=object), allow_pickle=True)
        np.save(output / 'OBS.npy', np.array(observations, dtype=object), allow_pickle=True)
        np.save(output / 'TERMINAL.npy', terminal, allow_pickle=True)
        np.savez_compressed(output / 'PUBLIC_PIXELS.npz', rgb=np.asarray(rgb_frames), visible_masks=np.asarray(mask_frames))
        filenames = ['TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'PRE_RUN.json', 'PUBLIC_PIXELS.npz']
        if cell['video']:
            np.savez_compressed(output / 'FRAMES.npz', frames=np.asarray(video_frames))
            np.savez_compressed(output / 'DETAIL_FRAMES.npz', frames=np.asarray(detail_frames))
            filenames += ['FRAMES.npz', 'DETAIL_FRAMES.npz']
        # Recheck all runtime sources/assets/checkpoints before evaluator access.
        validate_freeze(freeze_path, approval_path)
        after_hash = dict(adapter=parameters_sha(model), action_head=parameters_sha(head))
        seal_trajectory(output, filenames, before_hash, after_hash, freeze_binding)
        facts = evaluate_sealed(env, traces, terminal, output)
        changes = [r['adaptation'] for r in traces if r['adaptation'] is not None]
        precision_changes = [r['adaptation'] for r in traces if r['t'] >= 12 and r['adaptation'] is not None]
        result = dict(cell=cell, cell_index=index, phase='development', **facts, steps=240,
                      seal_sha256=sha(output / 'SEAL.json'), frozen_neural_weights_unchanged=True,
                      gradient_steps=sum(r['U'] for r in changes), adaptation_events=adapter.update_events,
                      valid_corner_frames=sum(r['corners'] is not None for r in observations),
                      neural_precision_commands=sum(r['neural_action'] is not None for r in traces),
                      head_domain_excursion_steps=sum(bool(r['head_domain'] and r['head_domain']['outside_training_domain']) for r in traces),
                      max_precision_same_state_position_delta=max([r['same_state_action_position_delta'] for r in precision_changes] + [0.]),
                      max_precision_same_state_rotation_delta=max([r['same_state_action_rotation_delta'] for r in precision_changes] + [0.]),
                      final_controller_phase=controller.phase,
                      tool_calibration={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in env.tool_calibration.items()},
                      elapsed_seconds=time.monotonic() - started,
                      interpretation='bounded physical development; offline approximation gate is not task performance')
        dump_new(output / 'RESULT.json', result)
        return result
    except Exception as exc:
        partial = {}
        for name, values in [('TRACE_PARTIAL.npy', traces), ('OBS_PARTIAL.npy', observations)]:
            np.save(output / name, np.array(values, dtype=object), allow_pickle=True)
            partial[name] = sha(output / name)
        if rgb_frames:
            np.savez_compressed(output / 'PUBLIC_PIXELS_PARTIAL.npz', rgb=np.asarray(rgb_frames), visible_masks=np.asarray(mask_frames))
            partial['PUBLIC_PIXELS_PARTIAL.npz'] = sha(output / 'PUBLIC_PIXELS_PARTIAL.npz')
        if env is not None:
            try:
                np.save(output / 'STATE_AT_ABORT.npy', snapshot(env), allow_pickle=True)
                partial['STATE_AT_ABORT.npy'] = sha(output / 'STATE_AT_ABORT.npy')
            except Exception:
                pass
        dump_new(output / 'ABORT.json', dict(cell=cell, type=type(exc).__name__, reason=str(exc), completed_steps=len(traces),
                                            partial_files=partial, automatic_retry=False,
                                            interpretation='explicit retained execution abort; never silently excluded'))
        raise
    finally:
        if env is not None:
            env.close()


def summarize(freeze_path):
    frozen = validate_freeze(freeze_path, check_runtime=False)
    rows = []
    for index, cell in enumerate(frozen['cells']):
        output = Path(freeze_path).parent / 'cells' / cell_name(index, cell)
        result, abort = output / 'RESULT.json', output / 'ABORT.json'
        if result.exists():
            record = json.loads(result.read_text())
            if record['seal_sha256'] != sha(output / 'SEAL.json'):
                raise ValueError('Result trajectory seal changed')
            seal = json.loads((output / 'SEAL.json').read_text())
            if seal['neural_parameters_before'] != seal['neural_parameters_after']:
                raise ValueError('Changed model in completed cell')
            for name, digest in seal['files'].items():
                if sha(output / name) != digest:
                    raise ValueError(f'Sealed trace changed: {output / name}')
            rows.append(dict(cell=cell, status='complete', result=record))
        elif abort.exists():
            rows.append(dict(cell=cell, status='abort', abort=json.loads(abort.read_text())))
        else:
            rows.append(dict(cell=cell, status='not_complete'))
    return dict(schema='prove.p50-development-summary.v1', expected_cells=32,
                completed=sum(r['status'] == 'complete' for r in rows), aborted=sum(r['status'] == 'abort' for r in rows),
                cells=rows, independent_success_rate_claim=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    for name in ('output', 'adapter', 'head', 'offline-query', 'plan'):
        prep.add_argument('--' + name, type=Path, required=True)
    run = sub.add_parser('run-cell')
    run.add_argument('--freeze', type=Path, required=True)
    run.add_argument('--approval', type=Path, required=True)
    run.add_argument('--index', type=int, required=True)
    run.add_argument('--gpu', type=int, choices=(0, 1), default=0)
    summary = sub.add_parser('summarize')
    summary.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        value = freeze_pilot(args.output, args.adapter, args.head, args.offline_query, args.plan)
    elif args.mode == 'run-cell':
        value = run_cell(args.freeze, args.approval, args.index, args.gpu)
    else:
        value = summarize(args.freeze)
    print(json.dumps(value, indent=2, allow_nan=False))
