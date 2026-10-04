#!/usr/bin/env python3
"""P60 bounded Panda insert development with three public-geometry update arms.

The public P45 G/seed11 geometry model and P50 precision action head are frozen.
Each condition compares fixed z, bounded continuous z updates, and the P47
same-information direct solver.  All arms share the first 12 observation
motions, the 240-command budget, public held-tool calibration, and the sealed
P47 evaluator.  This module prepares a result-free freeze; physical execution
requires a root receipt bound to that exact freeze.
"""
from __future__ import annotations

import argparse
import copy
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
    PrecisionActionController, encode_state, load_frozen_head,
)
from tools.task_graph.p50_precision_pilot import (
    ADAPTER_SHA, BACKEND, HEAD_SHA, controller_state, domain_excursions,
    package_versions, shadow_command, source_closure as p50_source_closure,
    validate_offline_evidence,
)
from tools.task_graph.train_precision_action_policy import dump_new, sha

SCHEMA = 'prove.p60-precision-development-freeze.v1'
APPROVAL_SCHEMA = 'prove.p60-root-physical-review.v1'
RESULT_SCHEMA = 'prove.p60-cell-result.v1'
SEAL_SCHEMA = 'prove.p60-trajectory-seal.v1'
SUMMARY_SCHEMA = 'prove.p60-development-summary.v1'
RUNNER = 'tools/task_graph/p60_precision_validation.py'
TEST = 'tests/test_p60_precision_validation.py'
STEPS = 240
ARMS = ('corner_fixed', 'corner_continuous', 'corner_direct')
ARM_MODES = dict(corner_fixed='fixed', corner_continuous='continuous', corner_direct='direct')
SPLITS = (('id', 605100), ('geometry', 605200))


def cells():
    """Eight prespecified conditions x three geometry-update arms."""
    return [dict(task='insert', split=split, condition_seed=base + offset,
                 controller='p50_frozen_neural', arm=arm,
                 adapter_mode=ARM_MODES[arm], rep='geometry', model_seed=11,
                 video=offset == 0)
            for split, base in SPLITS for offset in range(4) for arm in ARMS]


def cell_name(index, cell):
    return f"{index:02d}_{cell['split']}_{cell['condition_seed']}_{cell['arm']}"


def source_closure():
    """Bind P50's reviewed closure plus the P60-owned entrypoint and test."""
    result = p50_source_closure()
    for relative in (RUNNER, TEST):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f'Required P60 source missing: {relative}')
        result[relative] = sha(path)
    return dict(sorted(result.items()))


def _binding(path):
    """Use a repository-relative path when portable, otherwise retain absolute."""
    resolved = Path(path).resolve()
    try:
        stored = str(resolved.relative_to(ROOT))
    except ValueError:
        stored = str(resolved)
    return dict(path=stored, sha256=sha(resolved))


def _bound_path(item):
    path = Path(item['path'])
    return path if path.is_absolute() else ROOT / path


def _validate_p50_freeze(path, adapter, head):
    record = json.loads(Path(path).read_text())
    if (record.get('schema') != 'prove.p50-physical-development-freeze.v1'
            or record.get('steps_per_episode') != STEPS
            or record.get('automatic_retries') != 0
            or record.get('backend') != BACKEND
            or record.get('files', {}).get('adapter', {}).get('sha256') != ADAPTER_SHA
            or record.get('files', {}).get('action_head', {}).get('sha256') != HEAD_SHA
            or sha(adapter) != ADAPTER_SHA or sha(head) != HEAD_SHA):
        raise ValueError('P60 must reuse the exact reviewed P50 backend and checkpoints')
    return record


def _validate_p59_contact_diagnostic(path):
    record = json.loads(Path(path).read_text())
    failed = {(row.get('split'), row.get('condition_seed')) for row in record.get('cells', [])}
    if (record.get('schema') != 'prove.p59-oracle-terminal-contact-diagnostic.v1'
            or record.get('phase') != 'post-seal_read_only'
            or failed != {('id', 505100), ('id', 505104), ('id', 505107)}
            or record.get('shared_trace_findings', {}).get('terminal_finger_to_slot_side_wall_contact') is not True
            or record.get('shared_trace_findings', {}).get('strict_depth_requirement') != 'depth > 0.034 m'):
        raise ValueError('P60 must retain the sealed P59 three-cell execution-limit diagnosis')
    return record


def freeze_pilot(output, adapter, head, offline_query, p50_freeze, p59_contact_diagnostic,
                 plan, renderer_gpu_ids):
    """Create a result-free source/model/config freeze; run no physics."""
    from prove.fast_adapter.geometry3d_env import configuration
    from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets

    gpu_ids = [int(value) for value in renderer_gpu_ids]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids) < 0:
        raise ValueError('Renderer GPU ids must be a nonempty unique list of nonnegative integers')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('P60 freeze output must be empty or new')
    _validate_p50_freeze(p50_freeze, adapter, head)
    _validate_p59_contact_diagnostic(p59_contact_diagnostic)
    offline_paths = validate_offline_evidence(head, offline_query)
    artifacts = dict(adapter=adapter, action_head=head, p50_physical_freeze=p50_freeze,
                     p59_contact_diagnostic=p59_contact_diagnostic, plan=plan, **offline_paths)
    frozen = dict(
        schema=SCHEMA, status='frozen', phase='development', fresh_test_claim=False,
        root_review_required=True, task='insert', cells=cells(), steps_per_episode=STEPS,
        automatic_retries=0, backend=BACKEND, model_device='cpu', model_threads=1,
        renderer_gpu_ids=gpu_ids, physics_workers_maximum=len(gpu_ids),
        sources=source_closure(), packages=package_versions(), robot_assets=source_assets(),
        files={name: _binding(path) for name, path in artifacts.items()},
        conditions=[dict(split=split, condition_seed=seed,
                         evaluator_configuration=configuration('insert', seed, split))
                    for split, base in SPLITS for seed in range(base, base + 4)],
        model_contract=dict(
            geometry='unchanged P45 G representation seed11; public observations only',
            action='same frozen P50 PrecisionActionController in all arms',
            arms=ARM_MODES, weights='geometry and action weights immutable',
            first12='identical public observation motion in every arm'),
        input_contract=dict(
            geometry='visible corners/masks, camera calibration, robot pose and history',
            held_length='env.calibrate_held_geometry(); cfg length and geom size forbidden',
            evaluator_truth='never supplied to geometry adapter or action controller'),
        control_contract=dict(
            command_budget=STEPS, gripper='closed preheld peg',
            phase_judges='.003 m/.045 rad and three consecutive arrivals',
            missing_geometry='hold measured EEF pose', no_force_change=True),
        evaluation_contract=dict(
            method='unchanged P47 evaluate_sealed after trajectory/model seal',
            strict='last 10 native successes and full-trajectory mechanical validity',
            denominator='all 24 prespecified development cells including failures and aborts',
            retries='ordinary physical failures are never retried',
            independent_success_rate_claim=False),
        inherited_limit_evidence=dict(
            source='P59 sealed read-only contact diagnosis',
            known_conditions=[505100, 505104, 505107],
            interpretation='accurate geometry can still stop below 34 mm when Panda fingers contact opposing slot walls',
            policy='retain as an execution ceiling; do not relabel as geometry-readout failure or change the gate'))
    dump_new(output / 'PHYSICAL_FREEZE.json', frozen)
    snapshot_dir = output / 'source_snapshot'
    snapshot_dir.mkdir(exist_ok=False)
    for name, digest in frozen['sources'].items():
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        if sha(target) != digest:
            raise ValueError('Source changed while creating P60 snapshot')
    return frozen


def validate_freeze(path, approval=None, check_runtime=True):
    frozen = json.loads(Path(path).read_text())
    gpu_ids = frozen.get('renderer_gpu_ids')
    if (frozen.get('schema') != SCHEMA or frozen.get('status') != 'frozen'
            or frozen.get('phase') != 'development' or frozen.get('fresh_test_claim') is not False
            or frozen.get('cells') != cells() or frozen.get('backend') != BACKEND
            or frozen.get('steps_per_episode') != STEPS or frozen.get('automatic_retries') != 0
            or frozen.get('model_device') != 'cpu' or frozen.get('model_threads') != 1
            or not isinstance(gpu_ids, list) or not gpu_ids
            or len(set(gpu_ids)) != len(gpu_ids) or any(type(value) is not int or value < 0 for value in gpu_ids)
            or frozen.get('physics_workers_maximum') != len(gpu_ids)):
        raise ValueError('P60 scope differs from the fixed 24-cell development design')
    if approval is not None:
        receipt = json.loads(Path(approval).read_text())
        if (receipt.get('schema') != APPROVAL_SCHEMA
                or receipt.get('root_reviewed') is not True
                or receipt.get('physical_execution_authorized') is not True
                or receipt.get('freeze_sha256') != sha(path)):
            raise ValueError('P60 physics requires root review bound to this exact freeze')
    if check_runtime:
        if frozen['sources'] != source_closure() or frozen['packages'] != package_versions():
            raise ValueError('P60 runtime source or package closure changed')
        from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
        if frozen['robot_assets'] != source_assets():
            raise ValueError('P60 Panda assets changed')
        for name, item in frozen['files'].items():
            if sha(_bound_path(item)) != item['sha256']:
                raise ValueError(f'Frozen P60 artifact changed: {name}')
        validate_offline_evidence(_bound_path(frozen['files']['action_head']),
                                  _bound_path(frozen['files']['offline_query']))
        _validate_p50_freeze(_bound_path(frozen['files']['p50_physical_freeze']),
                             _bound_path(frozen['files']['adapter']),
                             _bound_path(frozen['files']['action_head']))
        _validate_p59_contact_diagnostic(_bound_path(frozen['files']['p59_contact_diagnostic']))
    return frozen


def _seal(output, filenames, before, after, freeze_binding):
    seal = dict(schema=SEAL_SCHEMA, freeze=freeze_binding,
                files={name: sha(Path(output) / name) for name in filenames},
                neural_parameters_before=before, neural_parameters_after=after)
    dump_new(Path(output) / 'SEAL.json', seal)
    if before != after:
        raise RuntimeError('Frozen neural weights changed; evaluator must not run')
    return seal


def _adaptation_counts(changes, mode):
    """Keep iterative gradient work separate from direct linear solves."""
    if mode not in ARM_MODES.values():
        raise ValueError('Unknown P60 adapter mode')
    gradient_steps = sum(int(row['U']) for row in changes) if mode == 'continuous' else 0
    direct_solves = (sum(row.get('solver') is not None for row in changes)
                     if mode == 'direct' else 0)
    return gradient_steps, direct_solves


def run_cell(freeze_path, approval_path, index, gpu):
    """Execute one create-once cell after exact-freeze root review."""
    if approval_path is None:
        raise ValueError('P60 root review receipt required before simulator construction')
    torch.set_num_threads(1)
    # Reject an unreviewed scope/device before importing or constructing physics.
    frozen = validate_freeze(freeze_path, approval_path, check_runtime=False)
    if gpu not in frozen['renderer_gpu_ids']:
        raise ValueError('Requested GPU is outside the frozen renderer device list')
    if not 0 <= index < len(frozen['cells']):
        raise ValueError('Cell index outside frozen P60 scope')
    frozen = validate_freeze(freeze_path, approval_path)
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
    cell = frozen['cells'][index]
    output = Path(freeze_path).parent / 'cells' / cell_name(index, cell)
    output.mkdir(parents=True, exist_ok=False)
    freeze_binding = dict(path=str(Path(freeze_path)), sha256=sha(freeze_path))
    dump_new(output / 'PRE_RUN.json', dict(
        schema='prove.p60-pre-run.v1', cell_index=index, cell=cell, freeze=freeze_binding,
        approval_sha256=sha(approval_path), renderer_gpu=gpu, steps=STEPS,
        source_snapshot='../../source_snapshot', automatic_retry=False,
        evaluator_only_state='saved but never supplied to runtime',
        held_length_permission='public RGB visible-mask calibration only; no truth length'))

    from prove.fast_adapter.geometry3d_adapter import pose
    from prove.fast_adapter.geometry3d_env import configuration
    from prove.fast_adapter.p45_neural import load_model
    from prove.fast_adapter.p47_neural import P47Latent
    from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
    from tools.fast_adapter.p45_common import observe_roles
    from tools.fast_adapter.p47_evaluate import PUBLIC_KEYS, evaluate_sealed, parameters_sha, snapshot

    env = None
    traces, observations, rgb_frames, mask_frames, video_frames, detail_frames = [], [], [], [], [], []
    started = time.monotonic()
    try:
        model, package = load_model(_bound_path(frozen['files']['adapter']), 'cpu')
        if model.task != 'insert' or model.representation != 'geometry' or package['seed'] != 11:
            raise ValueError('Wrong frozen P60 geometry adapter identity')
        model.eval().requires_grad_(False)
        head = load_frozen_head(_bound_path(frozen['files']['action_head']))
        before_hash = dict(geometry_model=parameters_sha(model), action_head=parameters_sha(head))
        cfg = configuration('insert', cell['condition_seed'], cell['split'])
        expected = next(row['evaluator_configuration'] for row in frozen['conditions']
                        if row['split'] == cell['split'] and row['condition_seed'] == cell['condition_seed'])
        if cfg != expected:
            raise ValueError('P60 environment configuration changed after freeze')
        env = RobotGeometrySkill(cfg, render=True)
        if env.model.nmocap != 0 or env.model.nu < 7:
            raise ValueError('P60 requires articulated actuated Panda without mocap')
        calibration = env.calibrate_held_geometry()
        if calibration.get('source') != 'RGB visible-mask bbox plus known own-tool pose':
            raise ValueError('P60 held length must use public visual calibration')
        length = float(calibration['length'])
        controller = PrecisionActionController(head, env.start_pos, env.start_R, length=length)
        adapter = P47Latent(model, cell['adapter_mode'], device='cpu')
        p, rotation = env.proprio()
        previous_action = np.r_[p, rotation.ravel(), 1.]
        for t in range(STEPS):
            observed, rgb, masks = observe_roles(env)
            obs = {key: copy.deepcopy(observed[key]) for key in PUBLIC_KEYS}
            obs['local'] = np.zeros(48, np.float32)
            if obs['corners'] is not None and not controller.adaptation_frozen:
                adapter.update(obs, t)
            estimate, proprio = adapter.estimate(), env.proprio()
            observations.append(dict(obs, t=t, previous_action=previous_action.copy(),
                                     geometry_source=cell['arm'], adaptation_frozen=controller.adaptation_frozen))
            if cell['video'] or t == 0:
                rgb_frames.append(rgb.copy()); mask_frames.append(masks.copy())
            before_state = controller_state(controller)
            update = (copy.deepcopy(adapter.records[-1])
                      if adapter.records and adapter.records[-1]['t'] == t else None)
            shadow = None
            if update is not None:
                shadow = shadow_command(controller, t, pose(np.asarray(update['pre_update_points'])), proprio)
            phase = 'align' if controller.phase == 'observe' else controller.phase
            domain = (domain_excursions(encode_state(estimate, proprio, length, phase))
                      if t >= 12 and estimate is not None else None)
            p, rotation, closed, stage = controller.command(t, estimate, proprio)
            if update is not None:
                p0, r0, closed0, stage0 = shadow
                update.update(
                    same_state_action_position_delta=float(np.linalg.norm(p - p0)),
                    same_state_action_rotation_delta=float(np.linalg.norm(
                        Rotation.from_matrix(rotation @ r0.T).as_rotvec())),
                    same_state_closed_changed=bool(closed != closed0),
                    same_state_stage_changed=bool(stage != stage0),
                    shadow_command_position=np.asarray(p0).copy(),
                    shadow_command_rotation=np.asarray(r0).copy())
            neural_record = (copy.deepcopy(controller.records[-1])
                             if controller.records and controller.records[-1]['t'] == t else None)
            row = dict(t=t, state=snapshot(env), command_p=p.copy(), command_R=rotation.copy(),
                       closed=bool(closed), stage=stage, estimate=copy.deepcopy(estimate),
                       proprio=copy.deepcopy(proprio), controller_before=before_state,
                       controller_after=controller_state(controller), adaptation=update,
                       head_domain=domain, neural_action=neural_record,
                       controls_before=env.data.ctrl.copy())
            if cell['video']:
                video_frames.append(np.concatenate([env.render_global(), rgb], axis=1))
                detail_frames.append(env.render_detail())
            env.step(p, rotation, closed)
            row['controls_after'] = env.data.ctrl.copy()
            row['robot_tracking'] = env.robot_tracking_diagnostics()
            traces.append(row)
            previous_action = np.r_[p, rotation.ravel(), float(closed)]
        terminal = snapshot(env)
        if cell['video']:
            _, rgb, _ = observe_roles(env)
            video_frames.append(np.concatenate([env.render_global(), rgb], axis=1))
            detail_frames.append(env.render_detail())
        np.save(output / 'TRACE.npy', np.array(traces, dtype=object), allow_pickle=True)
        np.save(output / 'OBS.npy', np.array(observations, dtype=object), allow_pickle=True)
        np.save(output / 'TERMINAL.npy', terminal, allow_pickle=True)
        np.savez_compressed(output / 'PUBLIC_PIXELS.npz', rgb=np.asarray(rgb_frames),
                            visible_masks=np.asarray(mask_frames))
        filenames = ['TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'PRE_RUN.json', 'PUBLIC_PIXELS.npz']
        if cell['video']:
            np.savez_compressed(output / 'FRAMES.npz', frames=np.asarray(video_frames))
            np.savez_compressed(output / 'DETAIL_FRAMES.npz', frames=np.asarray(detail_frames))
            filenames += ['FRAMES.npz', 'DETAIL_FRAMES.npz']
        validate_freeze(freeze_path, approval_path)
        after_hash = dict(geometry_model=parameters_sha(model), action_head=parameters_sha(head))
        _seal(output, filenames, before_hash, after_hash, freeze_binding)
        facts = evaluate_sealed(env, traces, terminal, output)
        changes = [row['adaptation'] for row in traces if row['adaptation'] is not None]
        precision_changes = [row['adaptation'] for row in traces
                             if row['t'] >= 12 and row['adaptation'] is not None]
        gradient_steps, direct_solves = _adaptation_counts(changes, cell['adapter_mode'])
        result = dict(
            schema=RESULT_SCHEMA, cell=cell, cell_index=index, phase='development',
            fresh_test_claim=False, **facts, steps=STEPS,
            seal_sha256=sha(output / 'SEAL.json'), frozen_neural_weights_unchanged=True,
            gradient_steps=gradient_steps, direct_solves=direct_solves,
            adaptation_events=adapter.update_events,
            valid_corner_frames=sum(row['corners'] is not None for row in observations),
            neural_precision_commands=sum(row['neural_action'] is not None for row in traces),
            head_domain_excursion_steps=sum(bool(row['head_domain'] and row['head_domain']['outside_training_domain'])
                                            for row in traces),
            max_precision_same_state_position_delta=max(
                [row['same_state_action_position_delta'] for row in precision_changes] + [0.]),
            max_precision_same_state_rotation_delta=max(
                [row['same_state_action_rotation_delta'] for row in precision_changes] + [0.]),
            final_controller_phase=controller.phase,
            tool_calibration={key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in calibration.items()},
            elapsed_seconds=time.monotonic() - started,
            interpretation='bounded P60 development; public geometry arms share one frozen action head')
        dump_new(output / 'RESULT.json', result)
        return result
    except Exception as exc:
        partial = {}
        for name, values in [('TRACE_PARTIAL.npy', traces), ('OBS_PARTIAL.npy', observations)]:
            np.save(output / name, np.array(values, dtype=object), allow_pickle=True)
            partial[name] = sha(output / name)
        if rgb_frames:
            np.savez_compressed(output / 'PUBLIC_PIXELS_PARTIAL.npz', rgb=np.asarray(rgb_frames),
                                visible_masks=np.asarray(mask_frames))
            partial['PUBLIC_PIXELS_PARTIAL.npz'] = sha(output / 'PUBLIC_PIXELS_PARTIAL.npz')
        if env is not None:
            try:
                np.save(output / 'STATE_AT_ABORT.npy', snapshot(env), allow_pickle=True)
                partial['STATE_AT_ABORT.npy'] = sha(output / 'STATE_AT_ABORT.npy')
            except Exception:
                pass
        dump_new(output / 'ABORT.json', dict(
            schema='prove.p60-cell-abort.v1', cell=cell, type=type(exc).__name__, reason=str(exc),
            completed_steps=len(traces), partial_files=partial, automatic_retry=False,
            interpretation='retained execution abort; never silently excluded or automatically retried'))
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
            if (record.get('schema') != RESULT_SCHEMA or record.get('cell_index') != index
                    or record.get('cell') != cell
                    or record['seal_sha256'] != sha(output / 'SEAL.json')):
                raise ValueError('P60 result or trajectory seal changed')
            seal = json.loads((output / 'SEAL.json').read_text())
            if (seal.get('schema') != SEAL_SCHEMA
                    or seal.get('freeze', {}).get('sha256') != sha(freeze_path)
                    or seal['neural_parameters_before'] != seal['neural_parameters_after']):
                raise ValueError('Changed model in completed P60 cell')
            for name, digest in seal['files'].items():
                if sha(output / name) != digest:
                    raise ValueError(f'Sealed P60 trace changed: {output / name}')
            rows.append(dict(cell=cell, status='complete', result=record))
        elif abort.exists():
            record = json.loads(abort.read_text())
            if record.get('schema') != 'prove.p60-cell-abort.v1' or record.get('cell') != cell:
                raise ValueError('P60 abort does not match its frozen arm and condition')
            rows.append(dict(cell=cell, status='abort', abort=record))
        else:
            rows.append(dict(cell=cell, status='not_complete'))
    return dict(schema=SUMMARY_SCHEMA, expected_cells=len(cells()),
                completed=sum(row['status'] == 'complete' for row in rows),
                aborted=sum(row['status'] == 'abort' for row in rows), cells=rows,
                fresh_test_claim=False, independent_success_rate_claim=False)


def _gpu_ids(value):
    try:
        result = [int(item) for item in value.split(',') if item != '']
    except ValueError as exc:
        raise argparse.ArgumentTypeError('renderer GPUs must be comma-separated integers') from exc
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError('renderer GPUs must be unique nonnegative integers')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    for name in ('output', 'adapter', 'head', 'offline-query', 'p50-freeze',
                 'p59-contact-diagnostic', 'plan'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--renderer-gpus', type=_gpu_ids, required=True,
                      help='comma-separated renderer device ids frozen for this deployment')
    run = sub.add_parser('run-cell')
    run.add_argument('--freeze', type=Path, required=True)
    run.add_argument('--approval', type=Path, required=True)
    run.add_argument('--index', type=int, required=True)
    run.add_argument('--gpu', type=int, required=True)
    summary = sub.add_parser('summarize')
    summary.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        value = freeze_pilot(args.output, args.adapter, args.head, args.offline_query,
                             args.p50_freeze, args.p59_contact_diagnostic, args.plan,
                             args.renderer_gpus)
    elif args.mode == 'run-cell':
        value = run_cell(args.freeze, args.approval, args.index, args.gpu)
    else:
        value = summarize(args.freeze)
    print(json.dumps(value, indent=2, allow_nan=False))
