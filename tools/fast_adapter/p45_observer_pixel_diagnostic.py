"""Opened-state, pixel-only P45 observer diagnosis; never a rollout runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from prove.fast_adapter.p45_visible_frame import visible_frame_corners


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def cyclic_rms(corners, target):
    return min(float(np.sqrt(np.square(np.roll(corners, k, axis=0) - target).sum(1).mean()))
               for k in range(4))


def run(base, out):
    base, out = Path(base), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = ['prove/fast_adapter/p45_visible_frame.py',
               'tools/fast_adapter/p45_observer_pixel_diagnostic.py']
    packages = sorted((base / 'diagnostics/public_masks').glob('*/PUBLIC_PIXELS.npz'))
    packages = [p for p in packages if p.parent.name.startswith(('cup_', 'slider_'))]
    predictions = {}
    # Predict only from exported public pixels before reading object truth.
    for package in packages:
        with np.load(package) as data:
            values = [visible_frame_corners(mask) for mask in data['mask']]
        corners = np.array([np.full((4, 2), np.nan) if value[0] is None else value[0] for value in values])
        path = out / (package.parent.name + '_CORNERS.npz')
        np.savez_compressed(path, corners=corners, valid=np.isfinite(corners).all((1, 2)))
        predictions[package.parent.name] = dict(path=path, values=values, corners=corners)
    write(out / 'PREDICTION_SEAL.json', dict(
        scope='opened_package1_observer_diagnosis_not_fresh_or_new_rollouts',
        sources={p: sha(p) for p in sources},
        public_pixels={str(p): sha(p) for p in packages},
        predictions={str(x['path']): sha(x['path']) for x in predictions.values()}))
    results = []
    for package in packages:
        manifest = json.loads((package.parent / 'MANIFEST.json').read_text())
        cell = Path(manifest['source'])
        if sha(cell / 'TRACE.npy') != manifest['source_trace_sha256']:
            raise ValueError(f'Trace hash mismatch: {cell}')
        cfg = json.loads((cell / 'PRE_RUN.json').read_text())['config']
        trace = np.load(cell / 'TRACE.npy', allow_pickle=True)
        old = np.load(cell / 'OBS.npy', allow_pickle=True) if (cell / 'OBS.npy').exists() else [row['observation'] for row in trace]
        r, w, hh, hz = [cfg[k] for k in ('radius', 'handle_width', 'handle_height', 'handle_z')]
        local = np.array([[r-.003, -.004, hz+hh/2+.003], [r+w+.003, -.004, hz+hh/2+.003],
                          [r+w+.003, -.004, hz-hh/2-.003], [r-.003, -.004, hz-hh/2-.003]])
        current_errors, old_errors = [], []
        with np.load(package) as data:
            for t, corners in enumerate(predictions[package.parent.name]['corners']):
                q = trace[t]['state']['qpos']
                if cfg['task'] == 'cup':
                    R = Rotation.from_quat(np.roll(q[3:7], -1)).as_matrix()
                    p = q[:3]
                else:
                    R = Rotation.from_euler('z', cfg['yaw'], degrees=True).as_matrix()
                    p = np.array(cfg['center']) + R[:, 0] * q[0]
                cam = (local @ R.T + p - data['camera_position'][t]) @ data['camera_rotation'][t]
                K = data['K'][t]
                target = np.c_[K[0, 2] + K[0, 0] * cam[:, 0] / -cam[:, 2],
                               K[1, 2] - K[1, 1] * cam[:, 1] / -cam[:, 2]]
                if np.isfinite(corners).all():
                    current_errors.append(cyclic_rms(corners, target))
                if old[t]['corners'] is not None:
                    old_errors.append(cyclic_rms(old[t]['corners'], target))
        median = float(np.median(current_errors)) if current_errors else None
        results.append(dict(
            cell=package.parent.name, valid_frames=len(current_errors),
            new_rms_median_px=median, old_rms_median_px=float(np.median(old_errors)) if old_errors else None,
            observer_gate_pass=bool(len(current_errors) >= 8 and median <= 4.),
            per_frame_valid_rms_px=current_errors,
            source_config_sha256=sha(cell / 'PRE_RUN.json'),
            reasons=[x[1]['reason'] for x in predictions[package.parent.name]['values']]))
    report = dict(scope='opened_package1_observer_diagnosis_not_fresh_or_new_rollouts',
                  method='whole-bar visible mask; four exterior pixel lines; adjacent intersections',
                  prediction_seal_sha256=sha(out / 'PREDICTION_SEAL.json'),
                  limits=['Only opened configurations; not qualification of a new executed package',
                          'Re-forwarded sealed state may differ slightly from original end-of-step derived camera values',
                          'Whole 8mm-thick bar silhouettes can mix front/back surfaces and violate a planar ray loss'],
                  results=results)
    write(out / 'RESULTS.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='evaluations/fast_adapter_p45_robot_mechanism_v01')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.base, args.out)
