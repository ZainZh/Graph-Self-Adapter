"""Zero-physics package-3 mounting check on opened package-2 slider states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prove.fast_adapter.p45_slider_env_v02 import SliderGeometrySkill as ParentSlider
from prove.fast_adapter.p45_slider_env_v03 import SliderGeometrySkill


SOURCES = [
    'prove/fast_adapter/p45_slider_env_v03.py',
    'prove/fast_adapter/p45_slider_env_v02.py',
    'prove/fast_adapter/robot_geometry_env_p45_v02.py',
    'prove/fast_adapter/geometry3d_env.py',
    'prove/fast_adapter/p45_handle_surfaces.py',
    'prove/fast_adapter/p45_visible_frame.py',
    'tools/fast_adapter/p45_slider_static_v03.py',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def run(base, out):
    import mujoco
    from PIL import Image
    base, out = Path(base), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cells = [base / f'slider_{split}_{seed}' for split, start in [('id', 452000), ('geometry', 452100)]
             for seed in range(start, start + 4)]
    js(out / 'PRE_RENDER.json', dict(
        scope='opened_package2_states_static_v03_mount_diagnosis_not_new_rollouts',
        sources={p: sha(p) for p in SOURCES}, cells=[str(cell) for cell in cells],
        fixed_change='panel front meets handle rear; continuous short carriage connector',
        unchanged='task target, camera, observer, controller, fixed mass/inertia and slider joint',
        observer_gate=dict(min_valid_prefix=8, max_median_rms_px=4.)))
    for source in SOURCES:
        target = out / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
    results = []
    for cell in cells:
        cfg = json.loads((cell / 'PRE_RUN.json').read_text())['config']
        trace = np.load(cell / 'TRACE.npy', allow_pickle=True)
        seal = json.loads((cell / 'SEAL.json').read_text())
        if sha(cell / 'TRACE.npy') != seal['TRACE.npy']:
            raise ValueError(f'Sealed trace changed: {cell}')
        folder = out / cell.name
        folder.mkdir()
        original_step = mujoco.mj_step
        skipped = 0
        def no_step(*args, **kwargs):
            nonlocal skipped
            skipped += 1
        mujoco.mj_step = no_step
        try:
            parent = ParentSlider(cfg, render=True)
            env = SliderGeometrySkill(cfg, render=True)
        finally:
            mujoco.mj_step = original_step
        try:
            fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                      'jnt_type', 'jnt_range', 'dof_damping', 'dof_frictionloss',
                      'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange')
            same_physics = {name: bool(np.array_equal(getattr(parent.model, name), getattr(env.model, name)))
                            for name in fields}
            if not all(same_physics.values()):
                raise AssertionError(f'Unexpected physics difference: {same_physics}')
            if not all(np.array_equal(a, b) for a, b in zip(parent.camera_local, env.camera_local)):
                raise AssertionError('The slider mounting repair must not change the camera')
            observations, rgb_frames, masks = [], [], []
            for row in trace[:12]:
                env.restore(row['state'])
                obs, rgb, mask = env.observe()
                observations.append(obs)
                rgb_frames.append(rgb)
                masks.append(mask)
            parent.restore(trace[0]['state'])
            _, old_rgb, _ = parent.observe()
            Image.fromarray(np.concatenate([old_rgb, rgb_frames[0]], axis=1)).save(folder / 'BEFORE_AFTER.png')
            np.save(folder / 'OBS.npy', np.array(observations, dtype=object), allow_pickle=True)
            np.savez_compressed(folder / 'PUBLIC_PIXELS.npz', rgb=np.array(rgb_frames), mask=np.array(masks))
            js(folder / 'PREDICTION_SEAL.json', {name: sha(folder / name) for name in ['OBS.npy', 'PUBLIC_PIXELS.npz']})
            # Privileged target geometry enters only this post-seal evaluator.
            errors = []
            for row, obs in zip(trace[:12], observations):
                if obs['corners'] is None:
                    continue
                env.restore(row['state'])
                p, R, half = env.oracle()
                points = np.array([[-half[0], half[1], 0], [half[0], half[1], 0],
                                   [half[0], -half[1], 0], [-half[0], -half[1], 0]]) @ R.T + p
                camera = (points - obs['camera_position']) @ obs['camera_rotation']
                K = obs['K']
                uv = np.c_[K[0, 2] + K[0, 0] * camera[:, 0] / -camera[:, 2],
                           K[1, 2] - K[1, 1] * camera[:, 1] / -camera[:, 2]]
                errors.append(min(float(np.sqrt(np.square(np.roll(obs['corners'], k, axis=0) - uv).sum(1).mean()))
                                  for k in range(4)))
            median = float(np.median(errors)) if errors else None
            result = dict(cell=cell.name, config_sha256=sha(cell / 'PRE_RUN.json'),
                          trace_sha256=sha(cell / 'TRACE.npy'), physics_steps_executed=0,
                          constructor_steps_suppressed=skipped, same_physics_arrays=same_physics,
                          valid_prefix=len(errors), corner_rms_median_px=median,
                          observer_qualified_static=bool(len(errors) >= 8 and median <= 4.),
                          reasons=[obs.get('reason') for obs in observations],
                          prediction_seal_sha256=sha(folder / 'PREDICTION_SEAL.json'))
            js(folder / 'RESULT.json', result)
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            parent.close()
            env.close()
    js(out / 'RESULTS.json', dict(scope='opened_state_static_observer_check_not_physical_qualification',
                                qualified_observers=sum(row['observer_qualified_static'] for row in results),
                                total=len(results), results=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='evaluations/fast_adapter_p45_robot_mechanism_v01/engineering/package2')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.base, args.out)
