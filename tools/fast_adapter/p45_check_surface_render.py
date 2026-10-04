"""Compare surface partition at identical sealed states; zero physics steps.

Run on the rendering server. This checks RGB/inertia changes independently of
the package-2 robot-base, IK or camera changes. It also exports public pixels
of the truly rendered front-face semantic parts for offline observer checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prove.fast_adapter.geometry3d_env import RES
from prove.fast_adapter.p45_handle_surfaces import FRONT_FACE_PREFIX, split_handle_visible_faces
from prove.fast_adapter.p45_visible_frame import visible_frame_corners
from prove.fast_adapter.robot_geometry_env_p45 import RobotGeometrySkill
from prove.fast_adapter.p45_slider_env import SliderGeometrySkill


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def run(cell, out, frames=12):
    import mujoco
    from PIL import Image
    cell, out = Path(cell), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((cell / 'PRE_RUN.json').read_text())['config']
    if cfg['task'] not in ('cup', 'slider'):
        raise ValueError('Surface comparison applies only to cup-family handles')
    trace = np.load(cell / 'TRACE.npy', allow_pickle=True)
    if frames > len(trace):
        raise ValueError('Requested frames exceed the sealed trace')
    # Constructors include settling loops. Disable only mj_step temporarily;
    # all states below come from the sealed source, with forward kinematics.
    original_step = mujoco.mj_step
    skipped = 0
    def no_step(*args, **kwargs):
        nonlocal skipped
        skipped += 1
    mujoco.mj_step = no_step
    try:
        env = (SliderGeometrySkill if cfg['task'] == 'slider' else RobotGeometrySkill)(cfg, render=False)
    finally:
        mujoco.mj_step = original_step
    changed_xml = split_handle_visible_faces(env.xml)
    model = mujoco.MjModel.from_xml_string(changed_xml)
    data = mujoco.MjData(model)
    for attr in ('nq', 'nv', 'nu', 'nbody'):
        if getattr(env.model, attr) != getattr(model, attr):
            raise AssertionError(f'Surface partition changed {attr}')
    physics_fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
                      'jnt_type', 'jnt_range', 'dof_damping', 'dof_frictionloss',
                      'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange')
    physics_equal = {name: bool(np.array_equal(getattr(env.model, name), getattr(model, name)))
                     for name in physics_fields}
    if not all(physics_equal.values()):
        raise AssertionError(f'Surface partition changed physics arrays: {physics_equal}')
    before_renderer = mujoco.Renderer(env.model, RES, RES)
    after_renderer = mujoco.Renderer(model, RES, RES)
    front_ids = [i for i in range(model.ngeom) if (model.geom(i).name or '').startswith(FRONT_FACE_PREFIX)]
    if len(front_ids) != 4:
        raise AssertionError('Expected four visible semantic front parts')
    arrays = {key: [] for key in ('rgb', 'mask', 'parts', 'camera_position', 'camera_rotation', 'K', 'corners')}
    comparisons = []
    try:
        for t in range(frames):
            state = trace[t]['state']
            env.restore(state)
            for key in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart'):
                if key in state:
                    getattr(data, key)[:] = state[key]
            data.time = state.get('time', 0.)
            mujoco.mj_forward(model, data)
            before_renderer.update_scene(env.data, camera='wrist')
            original_rgb = before_renderer.render().copy()
            after_renderer.disable_segmentation_rendering()
            after_renderer.update_scene(data, camera='wrist')
            rgb = after_renderer.render().copy()
            after_renderer.enable_segmentation_rendering()
            segmentation = after_renderer.render().copy()
            after_renderer.disable_segmentation_rendering()
            rendered = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            parts = np.stack([(segmentation[:, :, 0] == i) & rendered for i in front_ids])
            mask = parts.any(0).astype(np.uint8)
            corners, _ = visible_frame_corners(mask)
            p, R = env.proprio()
            cp, cr = env.camera_local
            f = (RES / 2) / np.tan(np.radians(65 / 2))
            arrays['rgb'].append(rgb)
            arrays['mask'].append(mask)
            arrays['parts'].append(parts.astype(np.uint8))
            arrays['camera_position'].append(p + R @ cp)
            arrays['camera_rotation'].append(R @ cr)
            arrays['K'].append(np.array([[f, 0, RES / 2], [0, f, RES / 2], [0, 0, 1.]]))
            arrays['corners'].append(np.full((4, 2), np.nan) if corners is None else corners)
            diff = np.abs(rgb.astype(np.int16) - original_rgb.astype(np.int16))
            comparisons.append(dict(frame=t, max_channel_difference=int(diff.max()),
                                    mean_absolute_difference=float(diff.mean()),
                                    changed_pixel_fraction=float((diff.max(2) > 0).mean()),
                                    changed_over_1_pixel_fraction=float((diff.max(2) > 1).mean())))
            if t == 0:
                Image.fromarray(np.concatenate([original_rgb, rgb], axis=1)).save(out / 'RGB_BEFORE_AFTER.png')
        path = out / 'PUBLIC_PIXELS.npz'
        np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})
        js(out / 'MANIFEST.json', dict(source=str(cell), source_trace_sha256=sha(cell / 'TRACE.npy'),
                                      output_sha256=sha(path), frames=frames,
                                      suppressed_constructor_steps=skipped, physics_steps_executed=0,
                                      no_target_truth_in_pixels=True,
                                      goal_mask_semantics='visible front-surface parts only',
                                      source_hashes={name: sha(name) for name in [
                                          'prove/fast_adapter/p45_handle_surfaces.py',
                                          'prove/fast_adapter/p45_visible_frame.py',
                                          'prove/fast_adapter/robot_geometry_env_p45.py',
                                          'prove/fast_adapter/p45_slider_env.py',
                                          'tools/fast_adapter/p45_check_surface_render.py']}))
        js(out / 'RGB_COMPARISON.json', dict(scope='same_sealed_state_surface_partition_only',
                                            no_physics_steps=True, exact_physics_arrays=physics_equal,
                                            frame_comparisons=comparisons))
        print(json.dumps(dict(out=str(out), frames=frames, comparisons=comparisons)), flush=True)
    finally:
        before_renderer.close()
        after_renderer.close()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--frames', type=int, default=12)
    args = parser.parse_args()
    run(args.cell, args.out, args.frames)
