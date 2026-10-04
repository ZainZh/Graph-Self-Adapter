#!/usr/bin/env python3
"""Offline source/replay divergence and named teacher-state review. No simulator."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spans(indices):
    result = []
    for value in indices:
        if result and value == result[-1][1] + 1:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


def stats(values):
    return dict(max=float(max(values)), final=float(values[-1]), mean=float(np.mean(values)))


def review(source, capture, output):
    source, capture, output = map(Path, (source, capture, output))
    output.mkdir(parents=True, exist_ok=False)
    binding = json.loads((capture / 'BINDING.json').read_text())
    if sha(source) != binding['source_sha256']:
        raise ValueError('Source differs from replay binding')
    names = json.loads((capture / 'SUPERVISOR_NAMES.json').read_text())
    rows = [json.loads(line) for line in (capture / 'FRAMES.jsonl').read_text().splitlines()]
    book_body = names['body'].index('black_book_1_main')
    back_site = names['site'].index('desk_caddy_1_back_contain_region')
    book_geoms = {i for i, name in enumerate(names['geom']) if name and name.startswith('black_book_1_g')}
    fingers = {finger: {i for i, name in enumerate(names['geom'])
                       if name and name.startswith(f'gripper0_finger{finger}_') and 'collision' in name}
               for finger in (1, 2)}
    qpos, qvel, eef, main_mae, wrist_mae = [], [], [], [], []
    named = []
    with h5py.File(source) as f:
        demo = f['data/demo_0']
        states = demo['states'][()]
        for t, row in enumerate(rows):
            path = capture / row['file']
            if sha(path) != row['sha256']:
                raise ValueError('Replay frame hash mismatch')
            with np.load(path) as z:
                nq = len(z['supervisor_qpos'])
                nv = len(z['supervisor_qvel'])
                if t < len(states):
                    qpos.append(float(np.linalg.norm(z['supervisor_qpos'] - states[t, 1:1+nq])))
                    qvel.append(float(np.linalg.norm(z['supervisor_qvel'] - states[t, 1+nq:1+nq+nv])))
                if t > 0:
                    eef.append(float(np.linalg.norm(z['observation_robot0_eef_pos'] - demo['obs/ee_pos'][t-1])))
                    for target, field, original in ((main_mae, 'agentview_image', 'agentview_rgb'),
                                                    (wrist_mae, 'robot0_eye_in_hand_image', 'eye_in_hand_rgb')):
                        target.append(float(np.abs(z['observation_' + field].astype(float) -
                                                   demo['obs/' + original][t-1]).mean()))
                touch = {}
                for finger, geoms in fingers.items():
                    touch[f'finger{finger}_book_contact'] = any(
                        (contact['geom1'] in geoms and contact['geom2'] in book_geoms) or
                        (contact['geom2'] in geoms and contact['geom1'] in book_geoms)
                        for contact in row['supervisor']['contacts'])
                named.append(dict(frame=t, book_body_position=z['supervisor_body_xpos'][book_body].tolist(),
                                  book_body_quaternion_wxyz=z['supervisor_body_xquat'][book_body].tolist(),
                                  caddy_back_site_position=z['supervisor_site_xpos'][back_site].tolist(),
                                  caddy_back_site_rotation_matrix=z['supervisor_site_xmat'][back_site].reshape(3,3).tolist(),
                                  eef_position=z['observation_robot0_eef_pos'].tolist(),
                                  gripper_qpos=z['observation_robot0_gripper_qpos'].tolist(),
                                  live_goal=bool(row['supervisor']['success']), **touch))
        summary = dict(schema='prove.libero-teacher-replay-diagnostic.v1', authority='supervisor_demo_only',
                       source_sha256=sha(source), binding_sha256=sha(capture / 'BINDING.json'),
                       initial_full_state_error=rows[0]['state_divergence'],
                       recorded_pre_action_qpos_l2=stats(qpos), recorded_pre_action_qvel_l2=stats(qvel),
                       post_action_eef_position_error_m=stats(eef),
                       post_action_agentview_rgb_mae_0_255=stats(main_mae),
                       post_action_wrist_rgb_mae_0_255=stats(wrist_mae),
                       first_live_goal_frame=next((r['frame'] for r in named if r['live_goal']), None),
                       final_live_goal=named[-1]['live_goal'],
                       finger1_book_contact_spans=spans([r['frame'] for r in named if r['finger1_book_contact']]),
                       finger2_book_contact_spans=spans([r['frame'] for r in named if r['finger2_book_contact']]),
                       both_fingers_book_contact_spans=spans([r['frame'] for r in named if
                           r['finger1_book_contact'] and r['finger2_book_contact']]),
                       limits=['one teacher physics replay, not model success',
                               'not bitwise or semantic source identity',
                               'qpos combines joint translation/rotation/quaternion units; qvel combines velocity units',
                               'final pre-action state error is frame 233; post-action EEF/image final error is frame 234',
                               'source terminal reward/done are synthetic and unused as success evidence',
                               'contact alone does not certify stable grasp'],
                       alignment='replay frame t precedes action t; compare image/EEF against source observation t-1 for t>=1')
        (output / 'DIAGNOSTIC.json').write_text(json.dumps(summary, indent=2) + '\n')
        (output / 'NAMED_SUPERVISOR_STATES.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in named))
        # Scientific evidence contact sheet, display-only vertical flip for OpenGL images.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        selected = [0, 30, 57, 68, 85, 130, 180, 200, 220, 234]
        fig, axes = plt.subplots(len(selected), 4, figsize=(10, 24), layout='constrained')
        for r, t in enumerate(selected):
            with np.load(capture / f'frame_{t:04d}.npz') as z:
                images = [z['observation_agentview_image'], z['observation_robot0_eye_in_hand_image'],
                          demo['obs/agentview_rgb'][t-1] if t else None,
                          demo['obs/eye_in_hand_rgb'][t-1] if t else None]
                labels = ['replay main', 'replay wrist', 'source main', 'source wrist']
                for c, (img, label) in enumerate(zip(images, labels)):
                    ax = axes[r,c]
                    if img is not None:
                        ax.imshow(img[::-1])
                    else:
                        ax.text(.5,.5,'No source image\nfor initial pre-action state',ha='center',va='center')
                    ax.set_title(f'{label} | frame {t}' if c < 2 else f'{label} | obs {t-1}', fontsize=9)
                    ax.axis('off')
        fig.suptitle('One official demo: physics replay and original frames\nDisplay flip only; no state restoration during replay', fontsize=13)
        fig.savefig(output / 'CONTACT_SHEET.png', dpi=120)
        plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.source, args.capture, args.output), indent=2))
