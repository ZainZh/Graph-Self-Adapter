"""Bounded P43 training/dev collection and frozen RGB features; never final test.

Privileged oracle labels and teacher commands live only in this collector.
The runtime archive has a strict public-observation key allowlist.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from prove.fast_adapter.geometry3d_env import GeometrySkill, configuration
from prove.fast_adapter.geometry3d_adapter import command
from prove.fast_adapter.neural_geometry_visual import (
    FrozenRoleVision, PUBLIC_ARRAY_KEYS, observe_roles, fit_role_pca)

ROOT = Path('evaluations/fast_adapter_p43_neural_geometry_v01')
TIMES = np.r_[np.arange(12), np.rint(np.linspace(12, 90, 36)).astype(int)]
BASE = {'cup': 430000, 'insert': 431000}
SOURCE_PATHS = ['prove/fast_adapter/geometry3d_env.py',
                'prove/fast_adapter/geometry3d_adapter.py',
                'prove/fast_adapter/neural_geometry_visual.py',
                'prove/fast_adapter/lila_frontend.py',
                'prove/fast_adapter/visual_alignment.py', 'tools/fast_adapter/p43_data.py']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def js(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n')


def serial_obs(obs, t, previous_action):
    return dict(corners=np.full((4, 2), np.nan) if obs['corners'] is None else obs['corners'],
                corners_valid=obs['corners'] is not None,
                **{k: obs[k] for k in ('camera_position', 'camera_rotation', 'K', 'position', 'rotation', 'role_bbox', 'role_visible')},
                t=t, previous_action=previous_action)


def training_label(env):
    """Privileged labels for the visible OUTER functional rectangle only."""
    center, rotation, size = env.oracle()
    outer = size + (.006 if env.task == 'insert' else 0.)
    a, b = outer
    local = np.array([[-a, b, 0], [a, b, 0], [a, -b, 0], [-a, -b, 0]])
    # Insert rim caps have their upper visible face 0.6 mm above mouth frame.
    visible_center = center + rotation[:, 2]*.0006 if env.task == 'insert' else center
    return dict(corners=local@rotation.T+visible_center,
                center=center, rotation=rotation, size=size, outer_size=outer)


def collect_episode(task, split, index):
    seed = BASE[task] + (100 if split == 'dev' else 0) + index
    distribution = 'geometry' if split == 'dev' and index >= 4 else 'id'
    out = ROOT/'data_raw'/task/split/f'{seed}'
    if (out/'RECEIPT.json').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    env = GeometrySkill(configuration(task, seed, distribution))
    observations, images, masks, labels, trace = [], [], [], [], []
    start = time.monotonic()
    try:
        # Collection teacher may use known held dimensions; runtime calibration
        # remains the original visible-mask estimation, outside this teacher.
        length = env.cfg['length'] if task == 'insert' else 0.
        p, rotation = env.proprio()
        previous_action = np.r_[p, rotation.ravel(), 0.]
        for t in range(int(TIMES[-1])+1):
            if t in TIMES:
                obs, rgb, role_masks = observe_roles(env)
                observations.append(serial_obs(obs, t, previous_action.copy()))
                images.append(rgb); masks.append(role_masks); labels.append(training_label(env))
            if t == TIMES[-1]:
                break
            p, rotation, closed, _ = command(task, t, env.oracle(), env.start_pos, env.start_R, length)
            trace.append(dict(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                              command_p=p, command_R=rotation, closed=closed))
            env.step(p, rotation, closed)
            previous_action = np.r_[p, rotation.ravel(), float(closed)]
        arrays = {k: np.asarray([o[k] for o in observations]) for k in observations[0]}
        np.savez_compressed(out/'public_observations.npz', **arrays)
        np.savez_compressed(out/'visible_images.npz', rgb=np.asarray(images),
                            role_masks_packed=np.packbits(np.asarray(masks), axis=-1))
        label_dir = ROOT/'data_labels'/task/split
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir/f'{seed}.npz'
        np.savez_compressed(label_path, **{k: np.asarray([o[k] for o in labels]) for k in labels[0]})
        audit_dir = ROOT/'data_collector_audit'/task/split
        audit_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(audit_dir/f'{seed}.npz', **{k: np.asarray([o[k] for o in trace]) for k in trace[0]})
        receipt = dict(task=task, split=split, seed=seed, distribution=distribution,
                       frames=len(observations), physics_actions=len(trace), times=TIMES.tolist(),
                       corner_valid=int(arrays['corners_valid'].sum()),
                       role_visible=arrays['role_visible'].sum(0).tolist(),
                       public_sha256=sha(out/'public_observations.npz'),
                       images_sha256=sha(out/'visible_images.npz'), label_sha256=sha(label_path),
                       elapsed_seconds=time.monotonic()-start, source_hashes={p: sha(p) for p in SOURCE_PATHS})
        js(out/'RECEIPT.json', receipt)
        print(json.dumps(receipt), flush=True)
    finally:
        env.close()
    return out


def preflight(model_dir, device):
    start = time.monotonic()
    vision = FrozenRoleVision(model_dir=model_dir, device=device)
    env = GeometrySkill(configuration('cup', 430000))
    try:
        obs, rgb, masks = observe_roles(env)
        images = np.stack([rgb, np.zeros_like(rgb), rgb])
        role_masks = np.stack([masks, masks, np.zeros_like(masks)])
        encoded = vision.encode(images, role_masks).cpu().numpy()
        delta = float(np.linalg.norm(encoded[0]-encoded[1]))
        absent = float(np.max(np.abs(encoded[2])))
        params = list(vision.net.model.parameters()) + list(vision.net.fusion.parameters()) + list(vision.net.adapter.parameters())
        weights = {'dino': str(Path(model_dir)/'model.safetensors'),
                   'lila': 'data/sources/lila-wam-reference-20260923/VISUAL_WEIGHTS.pt',
                   'lila_source': 'data/sources/lila-wam-reference-20260923/models/vla_model_fm.py'}
        record = dict(passed=bool(delta > 1e-3 and absent == 0 and not any(p.requires_grad for p in params)),
                      rgb_shape=list(rgb.shape), masks_shape=list(masks.shape), raw_feature_shape=list(encoded.shape),
                      rgb_black_difference_norm=delta, empty_mask_feature_absmax=absent,
                      all_visual_parameters_frozen=not any(p.requires_grad for p in params),
                      no_mask_truth_fallback=True, native_resolution=448, layers=[13, 17, 21],
                      feature_contract='Actor/Object/Goal each appearance,x-moment,y-moment x 1024; 3072/role',
                      weights={k: dict(path=str(Path(v).resolve()), sha256=sha(v)) for k, v in weights.items()},
                      torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(torch.device(device)), elapsed_seconds=time.monotonic()-start)
        js(ROOT/'visual/PREFLIGHT.json', record)
        from PIL import Image
        Image.fromarray(rgb).save(ROOT/'visual/PREFLIGHT_RGB.png')
        overlay=rgb.copy()
        colors=np.array([[40, 170, 235], [185, 100, 215], [245, 185, 40]])
        for i in range(3):
            m=masks[i].astype(bool); overlay[m]=(overlay[m].astype(float)*.5+colors[i]*.5).astype(np.uint8)
        Image.fromarray(overlay).save(ROOT/'visual/PREFLIGHT_ROLES.png')
        print(json.dumps(record), flush=True)
        if not record['passed']:
            raise RuntimeError('Frozen visual qualification failed')
    finally:
        env.close()


def collect(task, split):
    for i in range(48 if split == 'train' else 8):
        collect_episode(task, split, i)


def encode(task, model_dir, device):
    vision = FrozenRoleVision(model_dir=model_dir, device=device)
    for split in ('train', 'dev'):
        for episode in sorted((ROOT/'data_raw'/task/split).glob('*')):
            if not (episode/'RECEIPT.json').exists():
                raise RuntimeError(f'Incomplete episode {episode}')
            target = episode/'raw_visual.npy'
            if target.exists():
                continue
            start = time.monotonic()
            with np.load(episode/'visible_images.npz', allow_pickle=False) as arrays:
                masks = np.unpackbits(arrays['role_masks_packed'], axis=-1)
                values = vision.encode(arrays['rgb'], masks).cpu().numpy()
            np.save(target, values, allow_pickle=False)
            js(episode/'ENCODE.json', dict(sha256=sha(target), shape=list(values.shape), seconds=time.monotonic()-start))
            print(f'encoded {task}/{split}/{episode.name} {time.monotonic()-start:.2f}s', flush=True)
    packs = {}
    for split in ('train', 'dev'):
        episodes = sorted((ROOT/'data_raw'/task/split).glob('*'))
        expected = 48 if split == 'train' else 8
        if len(episodes) != expected:
            raise RuntimeError(f'Expected {expected} episodes, got {len(episodes)}')
        rows = []
        for episode in episodes:
            with np.load(episode/'public_observations.npz', allow_pickle=False) as arr:
                row = {k: arr[k].copy() for k in arr.files}
            row['local'] = np.load(episode/'raw_visual.npy', allow_pickle=False)
            rows.append(row)
        packs[split] = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
        packs[split]['seeds'] = np.asarray([int(episode.name) for episode in episodes])
    pca = fit_role_pca(packs['train']['local'], packs['train']['role_visible'], device=device)
    pca_path = ROOT/'data'/task/'role_pca.pt'; pca_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pca, pca_path); vision.set_pca(pca)
    for split, pack in packs.items():
        shape = pack['local'].shape[:2]
        raw = torch.as_tensor(pack['local'].reshape(-1, 3, 3072), device=device)
        visible = torch.as_tensor(pack['role_visible'].reshape(-1, 3), device=device)
        pack['local'] = vision.project(raw, visible).reshape(*shape, 48).detach().cpu().numpy()
        if set(pack) != set(PUBLIC_ARRAY_KEYS):
            raise RuntimeError(f'Unexpected runtime fields: {set(pack) ^ set(PUBLIC_ARRAY_KEYS)}')
        dest = ROOT/'data'/task/split; dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest/'observations.npz', **pack)
        labels = []
        for seed in pack['seeds']:
            with np.load(ROOT/'data_labels'/task/split/f'{seed}.npz', allow_pickle=False) as label:
                labels.append({k: label[k].copy() for k in label.files})
        label_path = ROOT/'data_labels'/task/split/'labels.npz'
        np.savez_compressed(label_path, **{k: np.stack([l[k] for l in labels]) for k in labels[0]})
        js(dest/'MANIFEST.json', dict(task=task, split=split, episodes=int(shape[0]), frames_per_episode=int(shape[1]),
              seeds=pack['seeds'].tolist(), schema={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in pack.items()},
              public_keys=list(PUBLIC_ARRAY_KEYS), privileged_arrays_in_runtime=False,
              observations_sha256=sha(dest/'observations.npz'), labels_path=str(label_path), labels_sha256=sha(label_path),
              pca_path=str(pca_path), pca_sha256=sha(pca_path), pca_fit_split='train',
              role_visibility_counts=pack['role_visible'].sum((0, 1)).tolist(),
              corner_valid=int(pack['corners_valid'].sum()),
              source_hashes={p: sha(p) for p in SOURCE_PATHS}))
        print(f'packed {dest}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'collect', 'encode'))
    parser.add_argument('--task', choices=('cup', 'insert'))
    parser.add_argument('--split', choices=('train', 'dev'))
    parser.add_argument('--model-dir', default='assets/dinov3')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.mode == 'preflight': preflight(args.model_dir, args.device)
    elif args.mode == 'collect': collect(args.task, args.split)
    else: encode(args.task, args.model_dir, args.device)
