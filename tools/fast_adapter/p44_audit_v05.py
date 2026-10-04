"""Replay sealed P44 neural decisions and, when requested, recorded physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prove.fast_adapter.neural_geometry_adapter import NeuralGeometryLatent, load_model
from tools.fast_adapter.p44_evaluate_v05 import evaluate_sealed, js, restore_physical, sha


def check_seal(cell):
    cell = Path(cell)
    pre = json.loads((cell / 'PRE_RUN.json').read_text())
    seal = json.loads((cell / 'SEAL.json').read_text())
    for key, name in [('trace', 'TRACE.npy'), ('observations', 'OBS.npy'),
                      ('terminal', 'TERMINAL.npy'), ('pre_run', 'PRE_RUN.json')]:
        if sha(cell / name) != seal[key]:
            raise ValueError(f'Sealed evidence changed: {cell / name}')
    if 'frames' in seal and sha(cell / 'FRAMES.npz') != seal['frames']:
        raise ValueError('Sealed original RGB changed')
    if 'detail_frames' in seal and sha(cell / 'DETAIL_FRAMES.npz') != seal['detail_frames']:
        raise ValueError('Sealed original detail RGB changed')
    return pre, seal


def audit(cell, physics=False):
    from prove.fast_adapter.robot_geometry_env_v05 import RobotGeometrySkill, robot_command, source_assets

    cell = Path(cell)
    pre, seal = check_seal(cell)
    result = json.loads((cell / 'RESULT.json').read_text())
    for path, digest in pre['sources'].items():
        if sha(path) != digest or sha(cell / 'source' / path) != digest:
            raise ValueError(f'Runtime source mismatch: {path}')
    if source_assets() != pre['robot_assets']:
        raise ValueError('Robot asset inventory changed')
    checkpoint = pre['bindings']['checkpoint']
    if sha(checkpoint['path']) != checkpoint['sha256']:
        raise ValueError('Checkpoint changed')
    protocol = pre['bindings']['freeze']
    if sha(protocol['path']) != protocol['sha256']:
        raise ValueError('Pilot protocol changed')
    frozen = json.loads(Path(protocol['path']).read_text())
    traces = np.load(cell / 'TRACE.npy', allow_pickle=True)
    observations = np.load(cell / 'OBS.npy', allow_pickle=True)
    terminal = np.load(cell / 'TERMINAL.npy', allow_pickle=True).item()
    if len(traces) != pre['steps'] or len(observations) != pre['steps']:
        raise ValueError('Incomplete physical trajectory')
    model, _ = load_model(checkpoint['path'])
    adapter = NeuralGeometryLatent(model, pre['arm'])
    env = RobotGeometrySkill(pre['cfg'], render=False)
    task = pre['cfg']['task']
    close_step = frozen.get('cup_close_step', 105)
    length = result['tool_calibration'].get('length', 0.)
    maxima = dict(action_position=0., action_rotation=0., latent=0.,
                  physical_qpos=0., physical_qvel=0., actuator_controls=0.,
                  initial_integration_restore=0.)
    anchor = None
    try:
        if physics:
            # Insertion's visible-mask calibration runs mj_forward before the
            # first action. Replay that sealed integration state, including
            # controls/warmstart, instead of using the constructor's earlier one.
            restore_physical(env, traces[0])
            spec = env.mj.mjtState.mjSTATE_INTEGRATION
            # restore_physical refreshes derived kinematics with mj_forward;
            # reapply integration fields in case that refresh altered warmstart.
            env.mj.mj_setState(env.model, env.data, traces[0]['integration'], spec)
            restored = np.empty_like(traces[0]['integration'])
            env.mj.mj_getState(env.model, env.data, restored, spec)
            maxima['initial_integration_restore'] = float(np.max(np.abs(
                restored - traces[0]['integration'])))
        for t, (obs, row) in enumerate(zip(observations, traces)):
            if row['t'] != t:
                raise ValueError('Nonconsecutive action sequence')
            if task != 'cup' or t < close_step:
                adapter.update(obs, t)
            estimate = adapter.estimate()
            if task == 'cup' and t == close_step and estimate is not None:
                p, rotation, size = estimate
                anchor = (p + rotation[:, 0] * (size[0] - .003) - rotation[:, 2] * .004,
                          rotation.copy())
            p, rotation, closed, stage = robot_command(
                task, t, estimate, env.start_pos, env.start_R, length, anchor)
            maxima['action_position'] = max(maxima['action_position'], float(np.max(np.abs(p - row['command_p']))))
            maxima['action_rotation'] = max(maxima['action_rotation'], float(np.max(np.abs(rotation - row['command_R']))))
            if bool(closed) != row['closed'] or stage != row['stage']:
                raise ValueError('Replayed gripper command or phase differs')
            if row['adaptation'] is not None:
                maxima['latent'] = max(maxima['latent'], float(np.max(np.abs(
                    adapter.z.numpy()[0] - np.asarray(row['adaptation']['z'])))))
            if physics:
                maxima['physical_qpos'] = max(maxima['physical_qpos'], float(np.max(np.abs(env.data.qpos - row['qpos']))))
                maxima['physical_qvel'] = max(maxima['physical_qvel'], float(np.max(np.abs(env.data.qvel - row['qvel']))))
                env.step(row['command_p'], row['command_R'], row['closed'])
                maxima['actuator_controls'] = max(maxima['actuator_controls'], float(np.max(np.abs(
                    env.data.ctrl - row['controls_after']))))
        if physics:
            maxima['physical_qpos'] = max(maxima['physical_qpos'], float(np.max(np.abs(env.data.qpos - terminal['qpos']))))
            maxima['physical_qvel'] = max(maxima['physical_qvel'], float(np.max(np.abs(env.data.qvel - terminal['qvel']))))
        recomputed = evaluate_sealed(env, traces, terminal)
        metric_fields = ('success', 'last10_successes', 'max_last10_penetration',
                         'max_last10_all_contact_penetration')
        metric_equal = all(np.isclose(recomputed[k], result[k], atol=1e-8, rtol=0) for k in metric_fields)
        controls = np.array([row['controls_after'] for row in traces])
        control_variation = float(np.max(np.ptp(controls[:, :7], axis=0)))
        passed = bool(max(maxima.values()) < 1e-6 and metric_equal
                      and seal['neural_parameters_before'] == seal['neural_parameters_after']
                      and env.model.nmocap == 0 and control_variation > 0.)
        return dict(cell=str(cell), passed=passed, maximum_error=maxima,
                    evaluator_recomputed=metric_equal, full_physics_replay=physics,
                    physics_initialization='Sealed first pre-action mjSTATE_INTEGRATION' if physics else None,
                    actual_robot_actuator_variation=control_variation, nmocap=int(env.model.nmocap),
                    result_sha256=sha(cell / 'RESULT.json'),
                    scope='Existing sealed development trajectory replay; no new condition or adaptation tuning')
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True)
    parser.add_argument('--physics', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    record = audit(args.cell, physics=args.physics)
    js(args.out, record)
    print(json.dumps(record), flush=True)
    if not record['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
