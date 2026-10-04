#!/usr/bin/env python3
"""P59 Oracle-geometry control-utility development pilot.

Thirty-two Panda preheld-insert cells compare explicit simulator Oracle slot
geometry with the historical P50 fixed-z geometry estimate.  Both arms use the
same frozen P50 neural action head, controller state machine, public held-tool
calibration, 240-step budget, and sealed P47 evaluator.  Oracle access is an
explicit diagnostic permission in this runner and is never a normal runtime.

``prepare`` is simulator-free apart from package/asset identity inspection.
``run-cell`` refuses to construct an environment without a root review receipt
bound to the exact P59 freeze.
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
import torch

from prove.task_graph.precision_action_policy import (
    PrecisionActionController, encode_state, load_frozen_head,
)
from tools.task_graph.p50_precision_pilot import (
    ADAPTER_SHA, BACKEND, HEAD_SHA, domain_excursions, package_versions,
    source_closure as p50_source_closure, validate_offline_evidence,
)
from tools.task_graph.train_precision_action_policy import dump_new, sha

SCHEMA = 'prove.p59-oracle-geometry-development-freeze.v1'
APPROVAL_SCHEMA = 'prove.p59-root-physical-review.v1'
RUNNER = 'tools/task_graph/p59_oracle_precision_pilot.py'
TEST = 'tests/test_p59_oracle_precision_pilot.py'
ARMS = ('p50_fixed', 'oracle')
SPLITS = (('id', 505100), ('geometry', 505200))
STEPS = 240


def cells():
    """The exact 16-condition x two-arm P59 development matrix."""
    return [dict(task='insert', split=split, condition_seed=base + offset,
                 arm=arm, controller='p50_frozen_neural', geometry_source=arm,
                 rep='geometry', model_seed=11,
                 previously_opened_by_p50=offset < 4,
                 video=offset == 0)
            for split, base in SPLITS for offset in range(8) for arm in ARMS]


def cell_name(index, cell):
    return f"{index:02d}_{cell['split']}_{cell['condition_seed']}_{cell['arm']}"


def source_closure():
    """Reuse P50's conservative closure and add the P59-owned entrypoints."""
    result = p50_source_closure()
    for relative in (RUNNER, TEST):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f'Required P59 source missing: {relative}')
        result[relative] = sha(path)
    return dict(sorted(result.items()))


def _binding(path):
    path = Path(path)
    return dict(path=str(path.resolve().relative_to(ROOT)), sha256=sha(path))


def _validate_p50_binding(path, adapter, head):
    """Bind P59 to the completed historical P50 source/model identity."""
    record = json.loads(Path(path).read_text())
    if (record.get('schema') != 'prove.p50-physical-development-freeze.v1'
            or record.get('steps_per_episode') != STEPS
            or record.get('automatic_retries') != 0
            or record.get('backend') != BACKEND
            or record.get('files', {}).get('adapter', {}).get('sha256') != ADAPTER_SHA
            or record.get('files', {}).get('action_head', {}).get('sha256') != HEAD_SHA
            or sha(adapter) != ADAPTER_SHA or sha(head) != HEAD_SHA):
        raise ValueError('P59 must reuse the exact historical P50 backend and frozen models')
    return record


def freeze_pilot(output, adapter, head, offline_query, p50_freeze, plan):
    """Create a result-free P59 source/model/config freeze; run no physics."""
    from prove.fast_adapter.geometry3d_env import configuration
    from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('P59 freeze output must be an existing empty or new directory')
    _validate_p50_binding(p50_freeze, adapter, head)
    offline_paths = validate_offline_evidence(head, offline_query)
    artifacts = dict(adapter=adapter, action_head=head, p50_physical_freeze=p50_freeze,
                     plan=plan, **offline_paths)
    conditions = [dict(split=split, condition_seed=seed,
                       development_status='opened_by_p50' if seed < base + 4 else 'new_p59_development',
                       evaluator_configuration=configuration('insert', seed, split))
                  for split, base in SPLITS for seed in range(base, base + 8)]
    frozen = dict(
        schema=SCHEMA, status='frozen', root_review_required=True,
        phase='development', fresh_test_claim=False, task='insert', cells=cells(),
        steps_per_episode=STEPS, automatic_retries=0, backend=BACKEND,
        model_device='cpu', model_threads=1, renderer_gpu_ids=[0, 1],
        physics_workers_maximum=2, sources=source_closure(), packages=package_versions(),
        robot_assets=source_assets(), conditions=conditions,
        files={name: _binding(path) for name, path in artifacts.items()},
        model_contract=dict(
            geometry_checkpoint_sha256=ADAPTER_SHA, action_head_sha256=HEAD_SHA,
            p50_fixed='historical P45 insert/G/seed11, fixed z; current public observation re-encoding unchanged',
            oracle='explicit env.oracle() slot center/orientation/half-size; diagnostic only',
            action='same frozen P50 PrecisionActionController in both arms',
            weights='geometry and action weights immutable in both arms'),
        input_contract=dict(
            common='public corners/masks/calibration/robot pose plus public visible-mask held-length calibration',
            held_length='env.calibrate_held_geometry(); cfg length and geom size forbidden',
            p50_fixed='public observation only; no evaluator truth',
            oracle='slot truth only through explicit env.oracle(); never an ordinary runtime',
            first12='same P50 public observation motion; action state machine ignores geometry'),
        control_contract=dict(
            controller='P50 frozen neural action head and phase state machine',
            phase_judges='.003 m/.045 rad and three consecutive arrivals',
            gripper='closed preheld peg', missing_fixed_geometry='hold measured EEF pose',
            command_budget=STEPS),
        evaluation_contract=dict(
            method='unchanged P47 evaluate_sealed after trajectory/model seal',
            strict='last 10 native successes and full-trajectory mechanical validity',
            denominator='all 32 prespecified development cells including ordinary failures and explicit aborts',
            retries='ordinary physical failures are never retried',
            claims_excluded=['fresh test', 'deployable Oracle', 'visual geometry solved',
                             'online adaptation benefit', 'independent success rate']),
        advancement_gate=dict(
            purpose='development gate for proceeding to public-information representation work',
            oracle_validated_success_minimum=14, oracle_denominator=16,
            oracle_mechanical_anomalies_maximum=0,
            oracle_vs_p50_fixed_pairwise_harms_maximum=0,
            oracle_vs_p50_fixed_net_rescues_minimum=2,
            statistical_significance_claim=False))
    dump_new(output / 'PHYSICAL_FREEZE.json', frozen)
    snapshot_dir = output / 'source_snapshot'
    snapshot_dir.mkdir(exist_ok=False)
    for name, digest in frozen['sources'].items():
        target = snapshot_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        if sha(target) != digest:
            raise ValueError('Source changed while creating P59 freeze snapshot')
    return frozen


def validate_freeze(path, approval=None, check_runtime=True):
    frozen = json.loads(Path(path).read_text())
    if (frozen.get('schema') != SCHEMA or frozen.get('status') != 'frozen'
            or frozen.get('phase') != 'development' or frozen.get('fresh_test_claim') is not False
            or frozen.get('cells') != cells() or frozen.get('backend') != BACKEND
            or frozen.get('steps_per_episode') != STEPS or frozen.get('automatic_retries') != 0
            or frozen.get('model_device') != 'cpu' or frozen.get('model_threads') != 1
            or frozen.get('renderer_gpu_ids') != [0, 1]
            or frozen.get('physics_workers_maximum') != 2):
        raise ValueError('P59 scope differs from the reviewed 32-cell Oracle diagnostic')
    if approval is not None:
        receipt = json.loads(Path(approval).read_text())
        if (receipt.get('schema') != APPROVAL_SCHEMA
                or receipt.get('root_reviewed') is not True
                or receipt.get('physical_execution_authorized') is not True
                or receipt.get('freeze_sha256') != sha(path)):
            raise ValueError('P59 physics requires root review bound to this exact freeze')
    if check_runtime:
        if frozen['sources'] != source_closure() or frozen['packages'] != package_versions():
            raise ValueError('P59 runtime source or package closure changed')
        from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
        if frozen['robot_assets'] != source_assets():
            raise ValueError('P59 Panda assets changed')
        for name, item in frozen['files'].items():
            if sha(ROOT / item['path']) != item['sha256']:
                raise ValueError(f'Frozen P59 artifact changed: {name}')
        if (frozen['files']['adapter']['sha256'] != ADAPTER_SHA
                or frozen['files']['action_head']['sha256'] != HEAD_SHA):
            raise ValueError('P59 frozen model binding changed')
        validate_offline_evidence(ROOT / frozen['files']['action_head']['path'],
                                  ROOT / frozen['files']['offline_query']['path'])
        _validate_p50_binding(ROOT / frozen['files']['p50_physical_freeze']['path'],
                              ROOT / frozen['files']['adapter']['path'],
                              ROOT / frozen['files']['action_head']['path'])
    return frozen


def _controller_state(controller):
    return copy.deepcopy({key: value for key, value in controller.__dict__.items()
                          if key not in ('model', 'records')})


def _json_geometry(estimate):
    return dict(center=np.asarray(estimate[0]).tolist(), rotation=np.asarray(estimate[1]).tolist(),
                half_size=np.asarray(estimate[2]).tolist())


def _seal(output, filenames, before, after, freeze_binding):
    seal = dict(schema='prove.p59-oracle-trajectory-seal.v1', freeze=freeze_binding,
                files={name: sha(Path(output) / name) for name in filenames},
                neural_parameters_before=before, neural_parameters_after=after)
    dump_new(Path(output) / 'SEAL.json', seal)
    if before != after:
        raise RuntimeError('Frozen P50 neural weights changed; evaluator must not run')


def run_cell(freeze_path, approval_path, index, gpu):
    """Execute exactly one create-once cell after exact-freeze root review."""
    if approval_path is None:
        raise ValueError('P59 root review receipt required before simulator construction')
    if gpu not in (0, 1):
        raise ValueError('Only frozen renderer GPUs 0 and 1 are allowed')
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
    torch.set_num_threads(1)
    frozen = validate_freeze(freeze_path, approval_path)
    if not 0 <= index < len(frozen['cells']):
        raise ValueError('Cell index outside frozen P59 scope')
    cell = frozen['cells'][index]
    output = Path(freeze_path).parent / 'cells' / cell_name(index, cell)
    output.mkdir(parents=True, exist_ok=False)
    freeze_binding = dict(path=str(Path(freeze_path)), sha256=sha(freeze_path))
    dump_new(output / 'PRE_RUN.json', dict(
        schema='prove.p59-pre-run.v1', cell_index=index, cell=cell, freeze=freeze_binding,
        approval_sha256=sha(approval_path), renderer_gpu=gpu, steps=STEPS,
        source_snapshot='../../source_snapshot', automatic_retry=False,
        evaluator_only_state='saved but not supplied except explicit Oracle slot geometry in oracle arm',
        oracle_permission='diagnostic slot center/orientation/half-size only' if cell['arm'] == 'oracle' else None,
        held_length_permission='public RGB visible-mask calibration only; no truth length'))

    from prove.fast_adapter.geometry3d_env import configuration
    from prove.fast_adapter.p45_neural import load_model
    from prove.fast_adapter.p47_neural import P47Latent
    from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
    from tools.fast_adapter.p45_common import observe_roles
    from tools.fast_adapter.p47_evaluate import PUBLIC_KEYS, evaluate_sealed, parameters_sha, snapshot

    env = None
    traces, observations, rgb_frames, mask_frames, video_frames, detail_frames = [], [], [], [], [], []
    oracle_access_count = 0
    started = time.monotonic()
    try:
        model, package = load_model(ROOT / frozen['files']['adapter']['path'], 'cpu')
        if model.task != 'insert' or model.representation != 'geometry' or package['seed'] != 11:
            raise ValueError('Wrong frozen P50 geometry model identity')
        model.eval().requires_grad_(False)
        head = load_frozen_head(ROOT / frozen['files']['action_head']['path'])
        before_hash = dict(geometry_model=parameters_sha(model), action_head=parameters_sha(head))
        cfg = configuration('insert', cell['condition_seed'], cell['split'])
        expected = next(row['evaluator_configuration'] for row in frozen['conditions']
                        if row['split'] == cell['split'] and row['condition_seed'] == cell['condition_seed'])
        if cfg != expected:
            raise ValueError('P59 environment configuration changed after freeze')
        env = RobotGeometrySkill(cfg, render=True)
        if env.model.nmocap != 0 or env.model.nu < 7:
            raise ValueError('P59 requires articulated actuated Panda without mocap')
        calibration = env.calibrate_held_geometry()
        if calibration.get('source') != 'RGB visible-mask bbox plus known own-tool pose':
            raise ValueError('P59 held length must use the original public visual calibration')
        length = float(calibration['length'])
        controller = PrecisionActionController(head, env.start_pos, env.start_R, length=length)
        adapter = P47Latent(model, 'fixed', device='cpu') if cell['arm'] == 'p50_fixed' else None
        p, rotation = env.proprio()
        previous_action = np.r_[p, rotation.ravel(), 1.]
        for t in range(STEPS):
            observed, rgb, masks = observe_roles(env)
            obs = {key: copy.deepcopy(observed[key]) for key in PUBLIC_KEYS}
            obs['local'] = np.zeros(48, np.float32)
            if adapter is not None and obs['corners'] is not None:
                adapter.update(obs, t)
            if cell['arm'] == 'oracle' and t >= 12:
                estimate = tuple(np.asarray(value).copy() for value in env.oracle())
                oracle_access_count += 1
                geometry_record = dict(source='explicit_env_oracle_diagnostic', **_json_geometry(estimate))
            elif cell['arm'] == 'oracle':
                estimate = None
                geometry_record = dict(source='shared_observation_prefix', oracle_not_accessed=True)
            else:
                estimate = adapter.estimate()
                geometry_record = (dict(source='historical_p50_fixed_z', **_json_geometry(estimate))
                                   if estimate is not None else dict(source='historical_p50_fixed_z', unavailable=True))
            proprio = env.proprio()
            observations.append(dict(obs, t=t, previous_action=previous_action.copy(),
                                     geometry_source=cell['arm'], oracle_access=cell['arm'] == 'oracle'))
            if cell['video'] or t == 0:
                rgb_frames.append(rgb.copy())
                mask_frames.append(masks.copy())
            phase = 'align' if controller.phase == 'observe' else controller.phase
            domain = (domain_excursions(encode_state(estimate, proprio, length, phase))
                      if t >= 12 and estimate is not None else None)
            before_state = _controller_state(controller)
            p, rotation, closed, stage = controller.command(t, estimate, proprio)
            neural_record = (copy.deepcopy(controller.records[-1])
                             if controller.records and controller.records[-1]['t'] == t else None)
            row = dict(t=t, state=snapshot(env), command_p=p.copy(), command_R=rotation.copy(),
                       closed=bool(closed), stage=stage, estimate=copy.deepcopy(estimate),
                       geometry=geometry_record, proprio=copy.deepcopy(proprio),
                       controller_before=before_state, controller_after=_controller_state(controller),
                       adaptation=(copy.deepcopy(adapter.records[-1])
                                   if adapter is not None and adapter.records and adapter.records[-1]['t'] == t else None),
                       head_domain=domain, neural_action=neural_record, controls_before=env.data.ctrl.copy())
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
        np.savez_compressed(output / 'PUBLIC_PIXELS.npz', rgb=np.asarray(rgb_frames), visible_masks=np.asarray(mask_frames))
        filenames = ['TRACE.npy', 'OBS.npy', 'TERMINAL.npy', 'PRE_RUN.json', 'PUBLIC_PIXELS.npz']
        if cell['video']:
            np.savez_compressed(output / 'FRAMES.npz', frames=np.asarray(video_frames))
            np.savez_compressed(output / 'DETAIL_FRAMES.npz', frames=np.asarray(detail_frames))
            filenames += ['FRAMES.npz', 'DETAIL_FRAMES.npz']
        validate_freeze(freeze_path, approval_path)
        after_hash = dict(geometry_model=parameters_sha(model), action_head=parameters_sha(head))
        _seal(output, filenames, before_hash, after_hash, freeze_binding)
        facts = evaluate_sealed(env, traces, terminal, output)
        result = dict(
            schema='prove.p59-cell-result.v1', cell=cell, cell_index=index, phase='development',
            fresh_test_claim=False, **facts, steps=STEPS, seal_sha256=sha(output / 'SEAL.json'),
            frozen_neural_weights_unchanged=True, geometry_source=cell['arm'],
            oracle_access_count=oracle_access_count,
            oracle_is_diagnostic_only=cell['arm'] == 'oracle',
            gradient_steps=0, adaptation_events=0 if adapter is None else adapter.update_events,
            valid_corner_frames=sum(row['corners'] is not None for row in observations),
            neural_precision_commands=sum(row['neural_action'] is not None for row in traces),
            head_domain_excursion_steps=sum(bool(row['head_domain'] and row['head_domain']['outside_training_domain']) for row in traces),
            final_controller_phase=controller.phase,
            tool_calibration={key: value.tolist() if isinstance(value, np.ndarray) else value
                              for key, value in calibration.items()},
            elapsed_seconds=time.monotonic() - started,
            interpretation='bounded P59 development Oracle geometry control-utility diagnostic')
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
            schema='prove.p59-cell-abort.v1', cell=cell, type=type(exc).__name__, reason=str(exc),
            completed_steps=len(traces), oracle_access_count=oracle_access_count,
            partial_files=partial, automatic_retry=False,
            interpretation='retained P59 execution abort; never silently excluded'))
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
                raise ValueError('P59 result trajectory seal changed')
            seal = json.loads((output / 'SEAL.json').read_text())
            if seal['neural_parameters_before'] != seal['neural_parameters_after']:
                raise ValueError('P59 completed cell changed frozen weights')
            for name, digest in seal['files'].items():
                if sha(output / name) != digest:
                    raise ValueError(f'P59 sealed trace changed: {output / name}')
            rows.append(dict(cell=cell, status='complete', result=record))
        elif abort.exists():
            rows.append(dict(cell=cell, status='abort', abort=json.loads(abort.read_text())))
        else:
            rows.append(dict(cell=cell, status='not_complete'))
    complete = sum(row['status'] == 'complete' for row in rows)
    assessment = dict(evaluable=False, passed=False, reason='all 32 cells must complete without abort')
    if complete == 32:
        paired = {}
        for row in rows:
            key = (row['cell']['split'], row['cell']['condition_seed'])
            paired.setdefault(key, {})[row['cell']['arm']] = row['result']
        if len(paired) != 16 or any(set(pair) != set(ARMS) for pair in paired.values()):
            raise ValueError('P59 completed results do not form the frozen 16-condition pairing')
        oracle_successes = sum(pair['oracle']['validated_success'] for pair in paired.values())
        fixed_successes = sum(pair['p50_fixed']['validated_success'] for pair in paired.values())
        rescues = sum(pair['oracle']['validated_success'] and not pair['p50_fixed']['validated_success']
                      for pair in paired.values())
        harms = sum(pair['p50_fixed']['validated_success'] and not pair['oracle']['validated_success']
                    for pair in paired.values())
        oracle_anomalies = sum(not pair['oracle']['mechanical_valid'] for pair in paired.values())
        assessment = dict(
            evaluable=True, oracle_validated_successes=oracle_successes,
            p50_fixed_validated_successes=fixed_successes,
            oracle_mechanical_anomalies=oracle_anomalies,
            paired_rescues=rescues, paired_harms=harms, net_rescues=rescues - harms,
            criteria=frozen['advancement_gate'],
            passed=bool(oracle_successes >= 14 and oracle_anomalies == 0 and harms == 0
                        and rescues - harms >= 2))
    return dict(schema='prove.p59-development-summary.v1', expected_cells=32,
                completed=complete, aborted=sum(row['status'] == 'abort' for row in rows),
                cells=rows, advancement_assessment=assessment,
                fresh_test_claim=False, oracle_deployment_claim=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    for name in ('output', 'adapter', 'head', 'offline-query', 'p50-freeze', 'plan'):
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
        value = freeze_pilot(args.output, args.adapter, args.head, args.offline_query,
                             args.p50_freeze, args.plan)
    elif args.mode == 'run-cell':
        value = run_cell(args.freeze, args.approval, args.index, args.gpu)
    else:
        value = summarize(args.freeze)
    print(json.dumps(value, indent=2, allow_nan=False))
