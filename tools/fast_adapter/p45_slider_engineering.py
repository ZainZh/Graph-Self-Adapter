"""Pre-result frozen Oracle engineering for the P45 Panda slider task.

Runtime mechanism evaluations must use the unified P45 evaluator instead.
This tool explicitly reads target geometry to qualify physical feasibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from prove.fast_adapter.p45_slider_env import SliderGeometrySkill, configuration
from prove.fast_adapter.robot_geometry_env_p45 import GeometryTaskController, source_assets


SOURCES = (
    'prove/fast_adapter/p45_slider_env.py',
    'prove/fast_adapter/robot_geometry_env_p45.py',
    'prove/fast_adapter/robot_geometry_env_v05.py',
    'prove/fast_adapter/geometry3d_env.py',
    'tools/fast_adapter/p45_slider_engineering.py',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert) + '\n')


def run(seed, split, out, render=True, steps=240):
    if steps < 10:
        raise ValueError('Engineering qualification needs at least 10 terminal states')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = configuration('slider', seed, split)
    js(out / 'PRE_RUN.json', dict(
        scope='p45_slider_oracle_engineering', task='slider', geometry_family='cup',
        config=cfg, steps=steps, render=render, timestamp=time.time(),
        sources={path: sha(path) for path in SOURCES}, assets=source_assets(),
        input_boundary='Oracle engineering may use target geometry; not model evidence',
        motion='physical bilateral handle grasp, passive rail translation, no robot attachment',
        success_rule='last 10 states: bilateral contact, displacement > .060m, abs(speed) < .01m/s; max penetration < .002m',
        qualification_gates=dict(sustained_terminal_states=10,
            max_contact_penetration=.002, max_arm_limit_rad=.001,
            max_finger_limit_m=.0001, valid_prefix_frames=8,
            median_prefix_corner_rms_px=4.),
        static_geometry_adaptation='ends on close; pulling uses the frozen grasp anchor'))
    for source in SOURCES:
        target = out / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
    env = SliderGeometrySkill(cfg, render=render)
    control = GeometryTaskController('slider', env.start_pos, env.start_R)
    trace, frames = [], []
    try:
        js(out / 'RESET.json', dict(proprio=env.proprio()[0].tolist(), robot=env.robot_tracking_diagnostics()))
        if render:
            from PIL import Image
            Image.fromarray(env.render_global()).save(out / 'start.png')
        for t in range(steps):
            estimate = env.oracle()
            controller_before = control.state_dict()
            p, R, closed, stage = control.command(t, estimate, env.proprio())
            record = dict(t=t, state=env.snapshot(), command_p=p, command_R=R,
                          closed=closed, stage=stage, proprio=env.proprio(),
                          oracle_estimate=estimate, controller_before=controller_before,
                          controller_after=control.state_dict())
            if render:
                obs, _, _ = env.observe()
                record['observation'] = obs
                frames.append(np.concatenate([env.render_global(), env.render_detail()], axis=1))
            env.step(p, R, closed)
            record['tracking'] = env.robot_tracking_diagnostics()
            trace.append(record)
        terminal = env.snapshot()
        np.save(out / 'TRACE.npy', np.array(trace, dtype=object), allow_pickle=True)
        np.save(out / 'TERMINAL.npy', terminal, allow_pickle=True)
        sealed = ['TRACE.npy', 'TERMINAL.npy']
        if render:
            np.savez_compressed(out / 'FRAMES.npz', frames=np.array(frames))
            Image.fromarray(env.render_global()).save(out / 'end.png')
            sealed.append('FRAMES.npz')
        js(out / 'SEAL.json', {name: sha(out / name) for name in sealed})
        # Object joint state and contact truth are opened only after sealing.
        terminal_facts = env.facts()
        held, penetration = [], []
        for state in [row['state'] for row in trace[-9:]] + [terminal]:
            env.restore(state)
            held.append(env.facts()['success'])
            penetration.append(max([max(0., -float(c.dist)) for c in env.data.contact] + [0.]))
        result = dict(task='slider', seed=seed, split=split,
                      scope='p45_slider_oracle_engineering', **terminal_facts)
        result.update(success=bool(all(held) and max(penetration) < .002),
                      last10_successes=sum(held), max_last10_penetration=max(penetration),
                      robot=env.robot_tracking_diagnostics(),
                      controller_terminal=control.state_dict(),
                      max_position_tracking_error=max(row['tracking']['position_error'] for row in trace),
                      max_angle_tracking_error=max(row['tracking']['angle_error'] for row in trace),
                      max_arm_joint_limit_violation_rad=max(row['tracking']['max_substep_arm_limit_violation_rad'] for row in trace),
                      max_finger_joint_limit_violation_m=max(row['tracking']['max_substep_finger_limit_violation_m'] for row in trace),
                      max_full_trajectory_contact_penetration=max(row['tracking']['max_substep_penetration_m'] for row in trace),
                      final_controller_phase=control.phase)
        result['physical_qualified'] = bool(
            result['success'] and result['max_arm_joint_limit_violation_rad'] <= .001
            and result['max_finger_joint_limit_violation_m'] <= .0001
            and result['max_full_trajectory_contact_penetration'] < .002)
        if render:
            errors = []
            for row in trace[:12]:
                obs = row['observation']
                if obs['corners'] is None:
                    continue
                center, target_R, half = row['oracle_estimate']
                corners = np.array([[-half[0], half[1], 0], [half[0], half[1], 0],
                                    [half[0], -half[1], 0], [-half[0], -half[1], 0]]) @ target_R.T + center
                cam = (corners - obs['camera_position']) @ obs['camera_rotation']
                K = obs['K']
                uv = np.c_[K[0, 2] + K[0, 0] * cam[:, 0] / (-cam[:, 2]),
                           K[1, 2] - K[1, 1] * cam[:, 1] / (-cam[:, 2])]
                errors.append(min(float(np.sqrt(np.square(np.roll(obs['corners'], k, axis=0) - uv).sum(-1).mean()))
                                  for k in range(4)))
            result['prefix_valid_observations'] = len(errors)
            result['prefix_corner_rms_median_px'] = float(np.median(errors)) if errors else None
            result['observer_qualified'] = bool(len(errors) >= 8 and np.median(errors) <= 4.)
        else:
            result['prefix_valid_observations'] = None
            result['prefix_corner_rms_median_px'] = None
            result['observer_qualified'] = None
        result['qualified'] = bool(result['physical_qualified'] and result['observer_qualified'])
        js(out / 'RESULT.json', result)
        if render:
            import imageio.v2 as imageio
            imageio.mimsave(out / 'raw.mp4', frames, fps=round(1 / .06),
                           quality=8, macro_block_size=16)
        print((out / 'RESULT.json').read_text(), flush=True)
        return result
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--split', choices=['id', 'geometry'], default='id')
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    run(args.seed, args.split, args.out, not args.no_render, args.steps)
