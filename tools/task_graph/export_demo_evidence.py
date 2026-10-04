"""Export deterministic offline demo evidence, never a learned stage decomposition.

Actions, proprioception and selected RGB frames are exported. Simulator state is
inventoried but never copied into runtime observations. Provenance and coordinate
conventions must be declared by the caller, not inferred from the file name.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from .inspect_demo import sha256_file

SOURCE_KINDS = ('human_teleoperation', 'policy_rollout', 'simulator_gt_demonstration', 'unknown')
PROPRIO_KEYS = ('ee_pos', 'ee_ori', 'eef_pos', 'eef_quat', 'robot0_eef_pos',
                'robot0_eef_quat', 'gripper_states', 'gripper_qpos', 'robot0_gripper_qpos', 'joint_states')


def action_events(actions: np.ndarray, *, min_translation: float = 0.05,
                  direction_cosine: float = 0.0) -> list[dict]:
    """Suggest frame indices from sign switches and adjacent translation turns.

    These are kinematic review hints, not contact, success, or semantic labels.
    Threshold units are those of the stored action, not metres.
    """
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] < 4 or not np.isfinite(actions).all():
        raise ValueError('actions must be finite [T,D] with D >= 4')
    if min_translation <= 0 or not -1 <= direction_cosine <= 1:
        raise ValueError('invalid event thresholds')
    result = []
    for frame in range(1, len(actions)):
        reasons = []
        if np.sign(actions[frame, -1]) != np.sign(actions[frame - 1, -1]):
            reasons.append('gripper_command_sign_switch')
        before, after = actions[frame - 1, :3], actions[frame, :3]
        bn, an = np.linalg.norm(before), np.linalg.norm(after)
        if min(bn, an) >= min_translation and float(np.dot(before, after) / (bn * an)) <= direction_cosine:
            reasons.append('translation_direction_change')
        if reasons:
            result.append({'frame': frame, 'reasons': reasons})
    return result


def export_demo(dataset: Path, output: Path, *, group: str, source_kind: str,
                provenance: str, coordinate_conventions: str, frames: list[int],
                image_key: str = 'obs/agentview_rgb', flip_vertical: bool = False,
                start: int = 0, stop: int | None = None,
                observation_timing: str = 'unspecified; stored indices preserved') -> dict:
    if source_kind not in SOURCE_KINDS or not provenance.strip() or not coordinate_conventions.strip():
        raise ValueError('explicit source kind, provenance and coordinate conventions required')
    if output.exists():
        raise FileExistsError(f'refusing to overwrite evidence: {output}')
    with h5py.File(dataset, 'r') as stream:
        demo = stream[group]
        all_actions = demo['actions']
        count = len(all_actions)
        stop = count if stop is None else stop
        if not 0 <= start < stop <= count:
            raise ValueError('invalid half-open frame range')
        frames = sorted(set(frames))
        if any(frame < start or frame >= stop for frame in frames):
            raise ValueError('requested image frame outside exported range')
        actions = np.asarray(all_actions[start:stop])
        events = action_events(actions)
        for event in events:
            event['frame'] += start
        proprio = {}
        if 'obs' in demo:
            for key in PROPRIO_KEYS:
                if key in demo['obs']:
                    arr = demo['obs'][key]
                    if len(arr) < stop:
                        raise ValueError(f'proprioception length mismatch: {key}')
                    proprio[key] = np.asarray(arr[start:stop]).tolist()
        images = []
        if frames:
            rgb = demo[image_key]
            for frame in frames:
                array = np.asarray(rgb[frame])
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                    raise ValueError('image dataset must be uint8 [T,H,W,3] RGB')
                images.append(Image.fromarray(array[::-1] if flip_vertical else array))
        inventory = {}
        demo.visititems(lambda name, obj: inventory.update({name: {'shape': list(obj.shape), 'dtype': str(obj.dtype)}})
                        if isinstance(obj, h5py.Dataset) else None)
    manifest = {'format': 'prove.offline-demo-evidence.v1', 'source': {
        'path': str(dataset), 'sha256': sha256_file(dataset), 'group': group,
        'source_kind': source_kind, 'provenance': provenance, 'sample_count': count},
        'frame_range': {'start_inclusive': start, 'stop_exclusive': stop},
        'coordinate_conventions': coordinate_conventions,
        'alignment': observation_timing,
        'scope': 'offline development evidence; no stage graph or success inference; no runtime GT channel',
        'heuristic_events': events,
        'event_method': {'translation_columns': [0, 1, 2], 'gripper_column': -1,
                         'min_translation': 0.05, 'direction_cosine': 0.0,
                         'claim': 'kinematic review hints, not learned Agent parsing'},
        'image_evidence': {'dataset_key': image_key, 'frames': frames, 'flip_vertical': flip_vertical},
        'dataset_inventory': inventory, 'artifacts': {}}
    output.mkdir(parents=True)
    payload = {'frame_indices': list(range(start, stop)), 'actions': actions.tolist(), 'proprioception': proprio}
    payload_path = output / 'ACTION_PROPRIO.json'
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    manifest['artifacts'][payload_path.name] = sha256_file(payload_path)
    if images:
        width = max(im.width for im in images)
        height = max(im.height for im in images) + 24
        cols = min(4, len(images))
        sheet = Image.new('RGB', (cols * width, math.ceil(len(images) / cols) * height), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, (frame, im) in enumerate(zip(frames, images)):
            x, y = (index % cols) * width, (index // cols) * height
            sheet.paste(im, (x, y + 24))
            draw.text((x + 4, y + 4), f'frame {frame}', fill='black')
        sheet_path = output / 'CONTACT_SHEET.png'
        sheet.save(sheet_path)
        manifest['artifacts'][sheet_path.name] = sha256_file(sheet_path)
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--group', required=True)
    parser.add_argument('--source-kind', required=True, choices=SOURCE_KINDS)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--coordinate-conventions', required=True)
    parser.add_argument('--frames', default='', help='Comma-separated root-selected frame indices')
    parser.add_argument('--image-key', default='obs/agentview_rgb')
    parser.add_argument('--flip-vertical', action='store_true')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--observation-timing', default='unspecified; stored indices preserved')
    args = vars(parser.parse_args())
    args['frames'] = [int(value) for value in args['frames'].split(',') if value.strip()]
    manifest = export_demo(**args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
