"""P45 bounded Panda collection and train-only frozen visual projection.

Teacher geometry is privileged and labels are stored separately. Collection
stops approaching before contact; it cannot count as task-success evidence.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from tools.fast_adapter.p45_common import (
    ROOT, SOURCES, TASK_BASE, js, make_env, observe_roles, privileged_training_label,
    public_row, sha, make_controller,
)
from prove.fast_adapter.neural_geometry_visual import FrozenRoleVision
from prove.fast_adapter.p45_visual import fit_projections, project_features

TIMES = np.r_[np.arange(24), np.rint(np.linspace(24, 127, 40)).astype(int)]
COUNTS = {'train': 96, 'dev': 12}


def validate_freeze(freeze, task):
    d = json.loads(Path(freeze).read_text())
    assert d['status'] == 'frozen' and d['scope'] == 'p45_training_collection'
    assert task in d['tasks'] and d['counts'] == COUNTS
    assert d['times'] == TIMES.tolist()
    for p, digest in d['sources'].items():
        if sha(p) != digest:
            raise ValueError(f'Frozen collection component changed: {p}')
    gate = d['engineering_gate'][task]
    if sha(gate['path']) != gate['sha256'] or not json.loads(Path(gate['path']).read_text())['passed']:
        raise ValueError('Engineering gate missing or changed')
    return d


def collect(task, split, index, freeze):
    frozen = validate_freeze(freeze, task)
    if not 0 <= index < COUNTS[split]:
        raise ValueError('Outside the declared collection range')
    seed = TASK_BASE[task] + (400 if split == 'dev' else 0) + index
    distribution = 'geometry' if split == 'dev' and index >= 6 else 'id'
    out = ROOT / 'data_raw' / task / split / str(seed)
    if (out / 'RECEIPT.json').exists():
        r = json.loads((out / 'RECEIPT.json').read_text())
        assert r['freeze_sha256'] == sha(freeze)
        return out
    if out.exists():
        raise RuntimeError(f'Incomplete attempt must be retained and reviewed: {out}')
    out.mkdir(parents=True)
    js(out / 'PRE_RUN.json', dict(task=task, split=split, seed=seed, distribution=distribution,
        freeze=dict(path=str(freeze), sha256=sha(freeze)), source_hashes=frozen['sources'],
        scope='privileged training/development collection only', timestamps=time.time(),
        times=TIMES.tolist(), physical_action_limit=127))
    backend = frozen['environments'][task]
    env = make_env(task, seed, distribution, backend=backend)
    try:
        # This collector may know the held length; runtime must visually calibrate it.
        controller = make_controller(env, length=env.cfg.get('length', 0.), backend=backend)
        p, R = env.proprio(); previous = np.r_[p, R.ravel(), float(task == 'insert')]
        observations, images, masks, labels, trace = [], [], [], [], []
        hold = None; started = time.monotonic()
        for t in range(128):
            if t in TIMES:
                obs, rgb, role_masks = observe_roles(env)
                observations.append(public_row(obs, t, previous))
                images.append(rgb); masks.append(role_masks); labels.append(privileged_training_label(env))
            if t == 127:
                break
            if 12 <= t < 24:
                # A second public, smaller observation circuit increases the
                # number of distinct pre-contact training views. It is common
                # to all representations and never uses the teacher geometry.
                waypoints = np.array([[.009, 0, 0], [0, .009, 0], [-.009, 0, 0], [0, -.009, .005]])
                p = env.start_pos + env.start_R @ waypoints[(t - 12) // 3]
                R, closed, stage = env.start_R, task == 'insert', 'second common training observation circuit'
            else:
                p, R, closed, stage = controller.command(t if t < 12 else t - 12, env.oracle(), env.proprio())
            approaching_contact = ((task != 'insert' and closed)
                                   or (task == 'insert' and controller.phase in ('insert', 'hold')))
            if hold is None and approaching_contact:
                hold = tuple(x.copy() for x in env.proprio())
            if hold is not None:
                p, R = hold
                stage = 'training only: hold pre-contact observation pose'
            closed = task == 'insert'
            trace.append(dict(t=t, command_p=p.copy(), command_R=R.copy(), closed=closed,
                              stage=stage, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy()))
            env.step(p, R, closed)
            previous = np.r_[p, R.ravel(), float(closed)]
        pack = {k: np.asarray([r[k] for r in observations]) for k in observations[0]}
        np.savez_compressed(out / 'public_observations.npz', **pack)
        np.savez_compressed(out / 'visible_images.npz', rgb=np.asarray(images),
                            role_masks_packed=np.packbits(np.asarray(masks), axis=-1))
        label_path = ROOT / 'data_labels' / task / split / f'{seed}.npz'
        label_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(label_path, **{k: np.asarray([r[k] for r in labels]) for k in labels[0]})
        np.save(out / 'COLLECTOR_TRACE.npy', np.asarray(trace, dtype=object), allow_pickle=True)
        camera = np.round(np.c_[pack['camera_position'], pack['camera_rotation'].reshape(64, 9)], 5)
        label_movement = float(np.max(np.linalg.norm(np.asarray([r['center'] for r in labels]) - labels[0]['center'], axis=1)))
        receipt = dict(task=task, split=split, seed=seed, distribution=distribution,
            freeze_sha256=sha(freeze), frames=len(images), actions=len(trace),
            corner_valid=int(pack['corners_valid'].sum()), role_visible=pack['role_visible'].sum(0).tolist(),
            distinct_camera_poses_to_1e5=len(np.unique(camera, axis=0)),
            target_translation_during_collection=label_movement,
            public_sha256=sha(out / 'public_observations.npz'), images_sha256=sha(out / 'visible_images.npz'),
            labels=dict(path=str(label_path), sha256=sha(label_path)),
            seconds=time.monotonic()-started, task_outcome_evaluated=False)
        receipt['collection_quality_qualified'] = bool(
            receipt['corner_valid'] >= 13
            and receipt['distinct_camera_poses_to_1e5'] >= 4
            and label_movement <= .001)
        js(out / 'RECEIPT.json', receipt)
        print(json.dumps(receipt), flush=True)
    finally:
        env.close()
    return out


def encode(task, model_dir, device, freeze):
    frozen = json.loads(Path(freeze).read_text())
    if (frozen.get('status') != 'frozen' or frozen.get('scope') != 'p45_visual_encoding'
            or task not in frozen['tasks'] or frozen['model_dir'] != str(model_dir)):
        raise ValueError('Encoding requires a result-free visual-weights/source freeze')
    for path, digest in {**frozen['sources'], **frozen['visual_assets']}.items():
        if sha(path) != digest:
            raise ValueError(f'Frozen visual component changed: {path}')
    encoding_binding = dict(path=str(freeze), sha256=sha(freeze))
    vision = FrozenRoleVision(model_dir=model_dir, device=device, microbatch=4)
    packs, episodes_by_split = {}, {}
    for split, count in COUNTS.items():
        episodes = sorted((ROOT / 'data_raw' / task / split).glob('*'))
        assert len(episodes) == count and all((e / 'RECEIPT.json').exists() for e in episodes)
        rows = []
        for episode in episodes:
            receipt = json.loads((episode / 'RECEIPT.json').read_text())
            if not receipt['collection_quality_qualified']:
                raise ValueError(f'Unqualified collection condition retained; no silent removal: {episode}')
            for filename, field in [('public_observations.npz', 'public_sha256'), ('visible_images.npz', 'images_sha256')]:
                assert sha(episode / filename) == receipt[field]
            raw_file = episode / 'raw_visual.npy'
            with np.load(episode / 'public_observations.npz', allow_pickle=False) as a:
                row = {k: a[k].copy() for k in a.files}
            if not raw_file.exists():
                with np.load(episode / 'visible_images.npz', allow_pickle=False) as a:
                    role_masks = np.unpackbits(a['role_masks_packed'], axis=-1)
                    raw = vision.encode(a['rgb'], role_masks).cpu().numpy()
                np.save(raw_file, raw, allow_pickle=False)
                js(episode / 'ENCODE.json', dict(raw_sha256=sha(raw_file), shape=list(raw.shape),
                    encoding_freeze=encoding_binding,
                    vision_sources={s: sha(s) for s in SOURCES if 'visual' in s or 'lila' in s}))
            else:
                encoded = json.loads((episode / 'ENCODE.json').read_text())
                assert encoded['encoding_freeze'] == encoding_binding
                assert sha(raw_file) == encoded['raw_sha256']
            row['local'] = np.load(raw_file, allow_pickle=False)
            rows.append(row)
            print(f'encoded {task}/{split}/{episode.name}', flush=True)
        packs[split] = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        packs[split]['seeds'] = np.asarray([int(e.name) for e in episodes])
        episodes_by_split[split] = episodes
    task_root = ROOT / 'data' / task; task_root.mkdir(parents=True, exist_ok=True)
    inputs = task_root / 'TRAIN_INPUTS.json'
    js(inputs, dict(task=task, split='train', seeds=packs['train']['seeds'].tolist(), encoding_freeze=encoding_binding,
        inputs=[dict(path=str(e), raw_sha256=sha(e / 'raw_visual.npy'),
                     public_sha256=sha(e / 'public_observations.npz')) for e in episodes_by_split['train']]))
    binding = dict(path=str(inputs), sha256=sha(inputs), split='train', seeds=packs['train']['seeds'].tolist())
    projections = fit_projections(packs['train']['local'], packs['train']['role_visible'],
                                  packs['train']['role_bbox'], binding, seed=450, device=device)
    projection_bindings = {}
    for kind, package in projections.items():
        path = task_root / f'{kind}_pca.pt'; torch.save(package, path)
        projection_bindings[kind] = dict(path=str(path), sha256=sha(path))
    for split, pack in packs.items():
        raw = pack.pop('local')
        for kind, key in [('absolute_fusion', 'local'), ('relative_fusion', 'local_relative')]:
            pack[key] = project_features(raw, pack['role_visible'], pack['role_bbox'], projections[kind], device).cpu().numpy()
        out = task_root / split; out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'observations.npz', **pack)
        label_rows = []
        for seed in pack['seeds']:
            with np.load(ROOT / 'data_labels' / task / split / f'{seed}.npz', allow_pickle=False) as a:
                label_rows.append({k: a[k].copy() for k in a.files})
        label_path = task_root / f'{split}_labels.npz'
        np.savez_compressed(label_path, **{k: np.stack([r[k] for r in label_rows]) for k in label_rows[0]})
        js(out / 'MANIFEST.json', dict(task=task, split=split, seeds=pack['seeds'].tolist(),
            observations_sha256=sha(out / 'observations.npz'), pca=projection_bindings,
            encoding_freeze=encoding_binding,
            labels=dict(path=str(label_path), sha256=sha(label_path)), input_fields=sorted(pack),
            source_hashes={s: sha(s) for s in SOURCES + ['prove/fast_adapter/p45_visual.py', 'tools/fast_adapter/p45_data.py']}))
    js(task_root / 'READY.json', dict(task=task, train=96, dev=12, representations=list(projections)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('operation', choices=['collect', 'encode'])
    p.add_argument('--task', required=True, choices=list(TASK_BASE))
    p.add_argument('--split', choices=list(COUNTS)); p.add_argument('--freeze')
    p.add_argument('--start', type=int, default=0); p.add_argument('--end', type=int)
    p.add_argument('--model-dir', default='assets/dinov3'); p.add_argument('--device', default='cuda')
    args = p.parse_args(); torch.set_num_threads(2)
    if args.operation == 'collect':
        if not args.split or not args.freeze:
            p.error('collection requires split and result-free frozen configuration')
        for i in range(args.start, COUNTS[args.split] if args.end is None else args.end):
            collect(args.task, args.split, i, args.freeze)
    else:
        if not args.freeze:
            p.error('encoding requires result-free source and visual-weight freeze')
        encode(args.task, args.model_dir, args.device, args.freeze)


if __name__ == '__main__':
    main()
