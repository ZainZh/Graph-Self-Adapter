"""Freeze or materialize all 16 P62 trails, with zero physics integration."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'evaluations/fast_adapter_p62_observable_update_v01/physical_v01'
DEFAULT_OUTPUT = ROOT/'evaluations/fast_adapter_p63_multiview_feedback_v01/views_v04'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def inventory():
    from prove.fast_adapter.p63_views import TIMES, VIEWS
    from prove.fast_adapter.robot_geometry_env_p45_v02 import source_assets
    acceptance = json.loads((SOURCE/'ACCEPTANCE_REPORT.json').read_text())
    if acceptance['completed'] != 16 or not acceptance['all_complete_trace_audits_pass']:
        raise ValueError('P62 completeness audit unavailable')
    cells = sorted((SOURCE/'cells').iterdir())
    if len(cells) != 16:
        raise ValueError('exactly 16 trajectories required')
    bindings = {}
    for cell in cells:
        seal = json.loads((cell/'SEAL.json').read_text())
        for name, expected in seal['files'].items():
            actual = sha(cell/name)
            if actual != expected:
                raise ValueError(f'source seal mismatch: {cell/name}')
            bindings[str((cell/name).relative_to(ROOT))] = actual
        for name in ('SEAL.json', 'RESULT.json'):
            bindings[str((cell/name).relative_to(ROOT))] = sha(cell/name)
    plan = ROOT/'evaluations/fast_adapter_p63_multiview_feedback_v01/PLAN.md'
    bindings[str(plan.relative_to(ROOT))] = sha(plan)
    amendment = plan.parent/'RENDER_AMENDMENT_SENSOR_TIME.md'
    bindings[str(amendment.relative_to(ROOT))] = sha(amendment)
    for name in ('RESTORE_LAG_DIAGNOSTIC.json', 'RESTORE_LAG_DIAGNOSTIC_SCRIPT.py'):
        diagnostic = plan.parent/'views_v02'/name
        bindings[str(diagnostic.relative_to(ROOT))] = sha(diagnostic)
    for name in ('PHYSICAL_FREEZE.json', 'ACCEPTANCE_REPORT.json'):
        bindings[str((SOURCE/name).relative_to(ROOT))] = sha(SOURCE/name)
    prior = json.loads((SOURCE/'PHYSICAL_FREEZE.json').read_text())
    code = dict(prior['sources'])
    for relative, expected in code.items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'frozen P62 source drift: {relative}')
    for relative in ('prove/fast_adapter/p63_views.py', 'tools/fast_adapter/p63_materialize.py',
                     'tests/test_p63_views.py'):
        code[relative] = sha(ROOT/relative)
    assets = source_assets()
    if assets != prior['robot_assets']:
        raise ValueError('P62 model assets changed')
    return dict(schema='prove.p63-render-freeze.v1', times=list(TIMES), views=list(VIEWS),
                resolution=448, trajectories=[c.name for c in cells], states=224,
                sources=bindings, code=code, robot_assets=assets, physics_steps=0,
                renderer_workers=1, renderer_gpu=0, threads=1, fresh_test_claim=False,
                geometry_source='same-frame sealed P62 uncorrected geometry8, no new model computation',
                sensor_time='t0 current after calibration mj_forward; t>0 integration_time-.001 with qpos-dt*qvel',
                camera_convention='OpenGL world camera: +x right,+y up,-z forward; top-left raster',
                projection_diagnostic='post-public-seal private projected goal geometry against visible mask',
                parity='all available saved wrist frames at fixed requested times; pose/calibration all states')


def materialize(output):
    import numpy as np
    from prove.fast_adapter.p63_views import (TIMES, VIEWS, render_only_environment,
        restore_sensor_time, render_public, visibility_record, PUBLIC_ARRAY_KEYS)
    freeze = json.loads((output/'FREEZE.json').read_text())
    if freeze != inventory():
        raise ValueError('render source/configuration freeze drift')
    review = json.loads((output/'ROOT_REVIEW.json').read_text())
    if not review.get('approved') or review.get('freeze_sha256') != sha(output/'FREEZE.json'):
        raise ValueError('root implementation review must bind current freeze')
    if (output/'public').exists():
        raise FileExistsError('public cache already exists; never overwrite')
    p62 = json.loads((SOURCE/'PHYSICAL_FREEZE.json').read_text())
    visibility, parity, files = [], [], {}
    for name in freeze['trajectories']:
        source = SOURCE/'cells'/name
        pre = json.loads((source/'PRE_RUN.json').read_text())
        result = json.loads((source/'RESULT.json').read_text())
        cfg = next(c['evaluator_configuration'] for c in p62['conditions']
                   if c['condition_seed'] == pre['cell']['condition_seed'])
        trace = np.load(source/'TRACE.npy', allow_pickle=True)
        obs = np.load(source/'OBS.npy', allow_pickle=True)
        predictions = np.load(source/'PREDICTIONS.npy', allow_pickle=True)
        if len(trace) != 240 or len(predictions) != 240:
            raise ValueError('incomplete source')
        saved = np.load(source/'PUBLIC_PIXELS.npz')
        rows, checks, reconstruction = [], [], []
        with render_only_environment(cfg) as (env, suppressed):
            xml_hash = hashlib.sha256(env.xml.encode()).hexdigest()
            for t in TIMES:
                if trace[t]['t'] != t or predictions[t]['t'] != t:
                    raise ValueError('source time mismatch')
                reconstruction.append(dict(t=t, **restore_sensor_time(env, trace[t])))
                public = render_public(env)
                expected = obs[t]
                np.testing.assert_allclose(public['K'][0], expected['K'], rtol=0, atol=1e-9)
                np.testing.assert_allclose(public['T_world_camera'][0, :3, 3], expected['camera_position'], rtol=0, atol=1e-8)
                np.testing.assert_allclose(public['T_world_camera'][0, :3, :3], expected['camera_rotation'], rtol=0, atol=1e-8)
                np.testing.assert_allclose(public['T_world_eef'][:3, 3], expected['position'], rtol=0, atol=1e-10)
                np.testing.assert_allclose(public['T_world_eef'][:3, :3], expected['rotation'], rtol=0, atol=1e-10)
                if t < len(saved['rgb']):
                    np.testing.assert_array_equal(public['rgb'][0], saved['rgb'][t])
                    np.testing.assert_array_equal(public['visible_masks'][0], saved['visible_masks'][t].astype(bool))
                    checks.append(t)
                public.update(t=t, simulation_time=float(env.data.time),
                              p62_geometry8=predictions[t]['geometry8'], T_world_anchor=predictions[t]['T_world_anchor'])
                if set(public) != PUBLIC_ARRAY_KEYS:
                    raise ValueError('public input allowlist violation')
                rows.append(public)
                visibility.extend(dict(cell=name, **r) for r in visibility_record(public['visible_masks'], t))
        folder = output/'public'/name
        folder.mkdir(parents=True)
        arrays = {key:np.asarray([r[key] for r in rows]) for key in rows[0]}
        calibration = result['tool_calibration']
        if calibration['source'] != 'RGB visible-mask bbox plus known own-tool pose':
            raise ValueError('unsupported held length source')
        arrays['public_held_length'] = np.asarray(calibration['length'])
        np.savez_compressed(folder/'VIEWS.npz', **arrays)
        write_new(folder/'MANIFEST.json', dict(cell=pre['cell'], times=list(TIMES), views=list(VIEWS),
            source_seal=sha(source/'SEAL.json'), source_predictions=sha(source/'PREDICTIONS.npy'),
            source_result=sha(source/'RESULT.json'), geometry_source=freeze['geometry_source'],
            public_held_length_source=calibration['source']))
        parity.append(dict(cell=name, wrist_pixel_exact_times=checks, pose_calibration_all_times=True,
                           constructor_steps_suppressed=suppressed, xml_sha256=xml_hash, physics_steps=0, reconstruction=reconstruction))
        for path in folder.iterdir():
            files[str(path.relative_to(output))] = sha(path)
        print(f'{name}: {len(rows)} states; {len(checks)} exact wrist pixel checks', flush=True)
    write_new(output/'VISIBILITY.json', dict(records=visibility, views=list(VIEWS), states=224))
    write_new(output/'RESTORE_PARITY.json', dict(records=parity, physics_steps=0))
    for name in ('VISIBILITY.json', 'RESTORE_PARITY.json'):
        files[name] = sha(output/name)
    write_new(output/'PUBLIC_SEAL.json', dict(schema='prove.p63-public-view-seal.v1', files=files,
        freeze_sha256=sha(output/'FREEZE.json'), states=224, images=448, physics_steps=0))



def private_labels(output):
    """Isolated post-public-seal evaluator truth and projection diagnostics."""
    import numpy as np
    import cv2
    from scipy.spatial.transform import Rotation
    from prove.fast_adapter.p63_views import project_world, TIMES, VIEWS
    seal = json.loads((output/'PUBLIC_SEAL.json').read_text())
    for relative, expected in seal['files'].items():
        if sha(output/relative) != expected:
            raise ValueError('public seal mismatch')
    freeze = json.loads((output/'FREEZE.json').read_text())
    prior = json.loads((SOURCE/'PHYSICAL_FREEZE.json').read_text())
    diagnostics, files = [], {}
    for name in freeze['trajectories']:
        meta = json.loads((output/'public'/name/'MANIFEST.json').read_text())
        cfg = next(c['evaluator_configuration'] for c in prior['conditions']
                   if c['condition_seed'] == meta['cell']['condition_seed'])
        # Same explicit scene geometry used by GeometrySkill. No private labels
        # are returned to the public renderer or any candidate constructor.
        center = np.asarray(cfg['center'])
        rotation = Rotation.from_euler('xyz', cfg['angles'], degrees=True).as_matrix()
        half = np.asarray(cfg['half']) + np.asarray(cfg['clearance'])
        labels = dict(center_world=center.tolist(), rotation_world=rotation.tolist(),
            opening_half=half.tolist(), evaluator_held_length=cfg['length'],
            source_config=dict(path=str(SOURCE/'PHYSICAL_FREEZE.json'),
                               sha256=sha(SOURCE/'PHYSICAL_FREEZE.json'), condition_seed=cfg['seed']),
            public_seal_sha256=sha(output/'PUBLIC_SEAL.json'))
        path = output/'private'/name/'LABELS.json'
        write_new(path, labels)
        files[str(path.relative_to(output))] = sha(path)
        data = np.load(output/'public'/name/'VIEWS.npz')
        # Pixel projection sanity: projected 3-D outer-rim samples compared
        # to visible goal pixels, reported separately for each fixed view/time.
        outer = half + .006
        corners = np.array([[-outer[0],-outer[1],0],[outer[0],-outer[1],0],
                            [outer[0],outer[1],0],[-outer[0],outer[1],0]])
        alpha = np.arange(64)/64
        local = np.concatenate([corners[i] + alpha[:,None]*(corners[(i+1)%4]-corners[i]) for i in range(4)])
        world = local @ rotation.T + center + rotation[:, 2] * .0006
        for i,t in enumerate(TIMES):
            for v,view in enumerate(VIEWS):
                uv = project_world(world,data['K'][i,v],data['T_world_camera'][i,v])
                mask = data['visible_masks'][i,v,2]
                dt = cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
                xy = np.rint(uv).astype(int)
                within = (xy[:,0]>=0)&(xy[:,0]<448)&(xy[:,1]>=0)&(xy[:,1]<448)
                distances = dt[xy[within,1],xy[within,0]]
                diagnostics.append(dict(cell=name,t=t,view=view, projected_in_frame=int(within.sum()),
                    visible_goal_pixels=int(mask.sum()),
                    projected_outerrim_distance_to_visible_goal_px_median=float(np.median(distances)) if len(distances) else None,
                    note='Postseal private diagnostic; occlusion may increase distance; no camera selection'))
    write_new(output/'private'/'PROJECTION_DIAGNOSTIC.json',dict(records=diagnostics))
    files['private/PROJECTION_DIAGNOSTIC.json']=sha(output/'private'/'PROJECTION_DIAGNOSTIC.json')
    write_new(output/'PRIVATE_SEAL.json',dict(files=files, public_seal_sha256=sha(output/'PUBLIC_SEAL.json')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--private-labels', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        write_new(args.output/'FREEZE.json', inventory())
    elif args.private_labels:
        private_labels(args.output)
    else:
        materialize(args.output)


if __name__ == '__main__':
    # Set before MuJoCo/OpenGL/torch imports; one render worker and one CPU thread.
    for key, value in dict(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', CUDA_VISIBLE_DEVICES='0',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1').items():
        os.environ[key] = value
    main()
