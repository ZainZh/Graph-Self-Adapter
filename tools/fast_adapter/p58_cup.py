#!/usr/bin/env python3
"""P58 cup zero-physics export, ridge qualification and meta training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from prove.fast_adapter.p58_cup_functional import (AdapterConfig, FunctionalFrameAdapter,
    REPRESENTATIONS, camera_pose, camera_to_world, functional_points, project,
    mean_rotation, points_to_pose, public_features, ray_system, world_to_camera)
from prove.task_graph.p57_public_geometry import mask_token


DIMS = (8, 16, 32)
LAMBDAS = (.01, .1, 1., 10.)
ROOT = Path(__file__).resolve().parents[2]
SOURCES = ('prove/fast_adapter/p58_geometry_model.py',
           'prove/fast_adapter/p58_cup_functional.py',
           'prove/task_graph/p57_public_geometry.py',
           'tools/fast_adapter/p58_cup.py')
TRAINING = dict(seed=11, updates=500, batch=16, optimizer='Adam', learning_rate=3e-4,
    outer='0.5 fixed geometry MSE + 0.5 adapted geometry MSE; units 0.01m',
    inner='three higher-order GN steps, alpha .65, damping 1e-4, direction cap 1',
    history='all up to 32; then first12+latest20', final_checkpoint_only=True)
READOUT = dict(pca_dims=list(DIMS), ridge=list(LAMBDAS), folds=5,
    grouping='condition seed; all 64 frames stay in one fold',
    uv='goal-mask-bbox-relative', xyz='camera frame divided by 0.01m',
    selection_score='0.5*(mean point pixel error/5px + mean point position error/5mm)',
    dev_used_for_selection=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def export_split(source, output, split):
    """Compress P45 RGB/masks to public tokens; keep labels in another file."""
    source, output = Path(source), Path(output)
    episodes = sorted((source/'data_raw/cup'/split).glob('*'), key=lambda p: int(p.name))
    if len(episodes) != (96 if split == 'train' else 12):
        raise ValueError(f'Unexpected P45 cup {split} count: {len(episodes)}')
    public_rows, label_rows, seeds, bindings = [], [], [], []
    for episode in episodes:
        seed = int(episode.name)
        receipt_path = episode/'RECEIPT.json'
        receipt = json.loads(receipt_path.read_text())
        public_path, image_path = episode/'public_observations.npz', episode/'visible_images.npz'
        label_path = source/'data_labels/cup'/split/f'{seed}.npz'
        if (receipt.get('task') != 'cup' or receipt.get('split') != split or receipt.get('seed') != seed
                or receipt.get('frames') != 64 or receipt.get('collection_quality_qualified') is not True
                or sha(public_path) != receipt.get('public_sha256')
                or sha(image_path) != receipt.get('images_sha256')
                or sha(label_path) != receipt.get('labels', {}).get('sha256')):
            raise ValueError(f'P45 source receipt mismatch: {episode}')
        with np.load(image_path, allow_pickle=False) as d:
            masks = np.unpackbits(d['role_masks_packed'], axis=-1)[..., :448]
        with np.load(public_path, allow_pickle=False) as d:
            obs = {k: d[k] for k in d.files}
        with np.load(label_path, allow_pickle=False) as d:
            labels = {k: d[k] for k in d.files}
        tokens = np.zeros((*masks.shape[:2], 55), np.float32)
        valid = np.zeros(masks.shape[:2], bool)
        for t in range(len(masks)):
            for role in range(3):
                tokens[t, role], valid[t, role] = mask_token(masks[t, role].astype(np.uint8))
        cam = camera_pose(obs['camera_position'], obs['camera_rotation'])
        eef = camera_pose(obs['position'], obs['rotation'])
        points = functional_points(labels['center'], labels['rotation'], labels['size'])
        uv, uv_valid = project(points, obs['K'], cam)
        xyz_camera = world_to_camera(points, cam)
        features = public_features(tokens, valid, obs['K'], cam, eef)
        public_rows.append(dict(features=features, mask_valid=valid, K=obs['K'],
                                camera_pose=cam, eef_transform=eef))
        label_rows.append(dict(points_world=points, points_uv=uv, uv_valid=uv_valid,
                               points_camera=xyz_camera))
        seeds.append(seed)
        bindings.append(dict(seed=seed, receipt_sha256=sha(receipt_path),
            public_sha256=receipt['public_sha256'], images_sha256=receipt['images_sha256'],
            labels_sha256=receipt['labels']['sha256'], collection_freeze_sha256=receipt['freeze_sha256']))
    output.mkdir(parents=True, exist_ok=True)
    public_path, label_path = output/f'{split}_public.npz', output/f'{split}_labels.npz'
    np.savez_compressed(public_path, seeds=np.asarray(seeds),
                        **{k: np.stack([r[k] for r in public_rows]) for k in public_rows[0]})
    np.savez_compressed(label_path, seeds=np.asarray(seeds),
                        **{k: np.stack([r[k] for r in label_rows]) for k in label_rows[0]})
    return dict(split=split, conditions=len(seeds), frames=int(sum(len(r['features']) for r in public_rows)),
                public=dict(path=str(public_path), sha256=sha(public_path)),
                labels=dict(path=str(label_path), sha256=sha(label_path)), sources=bindings)


def export(source, output):
    output = Path(output)
    if (output/'MANIFEST.json').exists():
        raise FileExistsError(output/'MANIFEST.json')
    splits = [export_split(source, output, split) for split in ('train', 'dev')]
    manifest = dict(schema='prove.p58-cup-mask-pack.v1', physics_steps=0,
                    source=str(Path(source).resolve()), splits=splits,
                    semantics=dict(points='grasp_center, +.03 R[:,0], +.03 R[:,1]',
                                   runtime='mask/calibration/EEF only; labels isolated'))
    _write_json(output/'MANIFEST.json', manifest)
    return manifest


def _load(pack, split):
    pack = Path(pack)
    manifest=json.loads((pack/'MANIFEST.json').read_text())
    if manifest.get('schema')!='prove.p58-cup-mask-pack.v1' or manifest.get('physics_steps')!=0:
        raise ValueError('Invalid P58 cup pack manifest')
    entry=next((row for row in manifest.get('splits',[]) if row.get('split')==split),None)
    public_path,label_path=pack/f'{split}_public.npz',pack/f'{split}_labels.npz'
    if entry is None or sha(public_path)!=entry['public']['sha256'] or sha(label_path)!=entry['labels']['sha256']:
        raise ValueError('P58 compact pack changed after export')
    with np.load(public_path, allow_pickle=False) as d:
        public = {k: d[k] for k in d.files}
    with np.load(label_path, allow_pickle=False) as d:
        labels = {k: d[k] for k in d.files}
    if not np.array_equal(public.pop('seeds'), labels.pop('seeds')):
        raise ValueError('Public and label condition identities differ')
    return public, labels


def _targets(public, labels, representation):
    n = 1 if representation == 'position_only' else 3
    # P57-style bbox-relative UV. The goal bbox is public token role2[0:4].
    bbox = public['features'][..., 110:114]
    center = (bbox[..., :2] + bbox[..., 2:]) * .5
    extent = (bbox[..., 2:] - bbox[..., :2]).clip(1/448.)
    relative_uv = (labels['points_uv'][..., :n, :] / 448. - center[..., None, :]) / extent[..., None, :]
    return np.concatenate((relative_uv.reshape(*labels['points_uv'].shape[:2], -1),
                           labels['points_camera'][..., :n, :].reshape(*labels['points_camera'].shape[:2], -1) / .01), -1)


def _fit_readout(x, y, train_conditions, dim, ridge):
    rows = x[train_conditions].reshape(-1, x.shape[-1]).astype(np.float64)
    target = y[train_conditions].reshape(-1, y.shape[-1]).astype(np.float64)
    mean, scale = rows.mean(0), rows.std(0).clip(.01)
    normalized = (rows-mean)/scale
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    components = vt[:dim]
    z = normalized @ components.T
    design = np.c_[np.ones(len(z)), z]
    penalty = np.eye(dim+1) * ridge; penalty[0, 0] = 0
    coef = np.linalg.solve(design.T@design + penalty, design.T@target)
    return dict(mean=mean.astype(np.float32), scale=scale.astype(np.float32),
                components=components.astype(np.float32), coef=coef.astype(np.float32),
                dim=dim, ridge=ridge)


def _predict(model, x):
    z = ((x-model['mean'])/model['scale']) @ model['components'].T
    return np.c_[np.ones(len(z)), z] @ model['coef']


def _cv_score(pred, truth, bbox, point_count):
    pred_uv = pred[..., :2*point_count].reshape(-1, point_count, 2)
    true_uv = truth[..., :2*point_count].reshape(-1, point_count, 2)
    extent = (bbox[..., 2:]-bbox[..., :2]).clip(1/448.).reshape(-1, 1, 2)
    pixel_error = np.linalg.norm((pred_uv-true_uv)*extent*448., axis=-1).mean()
    pred_xyz = pred[..., 2*point_count:].reshape(-1, point_count, 3)
    true_xyz = truth[..., 2*point_count:].reshape(-1, point_count, 3)
    position_error = np.linalg.norm((pred_xyz-true_xyz)*.01, axis=-1).mean()
    return float(.5*(pixel_error/5. + position_error/.005)), float(pixel_error), float(position_error)


def choose_readout(x, y, bbox=None, point_count=None):
    point_count = point_count or (y.shape[-1]//5)
    bbox = x[..., 110:114] if bbox is None else bbox
    folds = np.arange(len(x)) % 5
    candidates = []
    for dim in DIMS:
        for ridge in LAMBDAS:
            losses = []
            for fold in range(5):
                model = _fit_readout(x, y, np.flatnonzero(folds != fold), dim, ridge)
                pred = _predict(model, x[folds == fold].reshape(-1, x.shape[-1]))
                truth = y[folds == fold].reshape(-1, y.shape[-1])
                losses.append(_cv_score(pred, truth, bbox[folds == fold], point_count))
            candidates.append(dict(dim=dim, ridge=ridge,
                grouped_cv_score=float(np.mean([v[0] for v in losses])),
                pixel_error=float(np.mean([v[1] for v in losses])),
                position_error_m=float(np.mean([v[2] for v in losses]))))
    selected = min(candidates, key=lambda row: (row['grouped_cv_score'], row['dim'], row['ridge']))
    return _fit_readout(x, y, np.arange(len(x)), selected['dim'], selected['ridge']), candidates


def _readout_predictions(readout, public, representation):
    n = 1 if representation == 'position_only' else 3
    flat = public['features'].reshape(-1, public['features'].shape[-1])
    pred = _predict(readout, flat).reshape(*public['features'].shape[:2], -1)
    bbox = public['features'][..., 110:114]
    center = (bbox[..., :2]+bbox[..., 2:])*.5
    extent = (bbox[..., 2:]-bbox[..., :2]).clip(1/448.)
    uv = (center[..., None, :] + extent[..., None, :]
          * pred[..., :2*n].reshape(*pred.shape[:2], n, 2))*448.
    xyz = pred[..., 2*n:].reshape(*pred.shape[:2], n, 3) * .01
    return uv, camera_to_world(xyz, public['camera_pose'])


def freeze(pack, plan, output):
    pack, plan, output = Path(pack).resolve(), Path(plan).resolve(), Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    manifest = pack/'MANIFEST.json'
    value = dict(schema='prove.p58-cup-training-freeze.v1', status='frozen', physics_steps=0,
        pack_manifest=dict(path=str(manifest),sha256=sha(manifest)),
        plan=dict(path=str(plan),sha256=sha(plan)), feature_dim=196,
        representations={name:dict(point_count=1 if name=='position_only' else 3,
            geometry_dim=3 if name=='position_only' else 9,
            latent_dim=3 if name=='position_only' else 6)
            for name in REPRESENTATIONS}, readout=READOUT, training=TRAINING,
        sources={name:sha(ROOT/name) for name in SOURCES})
    output.parent.mkdir(parents=True,exist_ok=True);_write_json(output,value);return value


def verify_freeze(path, pack, representation, updates, seed):
    value=json.loads(Path(path).read_text());manifest=Path(pack)/'MANIFEST.json'
    if (value.get('schema')!='prove.p58-cup-training-freeze.v1' or value.get('physics_steps')!=0
            or value.get('feature_dim')!=196 or value.get('readout')!=READOUT
            or value.get('training')!=TRAINING or sha(manifest)!=value['pack_manifest']['sha256']
            or representation not in value.get('representations',{})
            or updates!=TRAINING['updates'] or seed!=TRAINING['seed']):
        raise ValueError('P58 cup training differs from frozen contract')
    for name,digest in value['sources'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Frozen P58 source changed: '+name)
    if sha(value['plan']['path'])!=value['plan']['sha256']: raise ValueError('P58 plan changed')
    return value


def train(pack, representation, output, freeze_path, updates=500, seed=11, device='cpu'):
    if representation not in REPRESENTATIONS:
        raise ValueError('Unknown representation')
    verify_freeze(freeze_path,pack,representation,updates,seed)
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    torch.manual_seed(seed); np.random.seed(seed)
    public, labels = _load(pack, 'train')
    target = _targets(public, labels, representation)
    n = 1 if representation == 'position_only' else 3
    readout, candidates = choose_readout(public['features'], target, point_count=n)
    pred_uv, base = _readout_predictions(readout, public, representation)
    truth = labels['points_world'][..., :n, :]
    residual = ((truth-base)/.01).reshape(-1, n*3)
    _, _, vt = np.linalg.svd(residual-residual.mean(0), full_matrices=False)
    latent_dim = 3 if representation == 'position_only' else 6
    basis = np.zeros((n*3, latent_dim), np.float32)
    rank = min(n*3, latent_dim); basis[:, :rank] = vt[:rank].T
    model = FunctionalFrameAdapter(basis, AdapterConfig(representation=representation,
                                   latent_dim=latent_dim)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    A, C = ray_system(pred_uv, public['K'], public['camera_pose'])
    valid=np.repeat(public['mask_valid'][...,2,None],n,axis=-1)
    base_t = torch.tensor(base, dtype=torch.float32, device=device)
    truth_t = torch.tensor(truth, dtype=torch.float32, device=device)
    A_t, C_t = (torch.tensor(v, dtype=torch.float32, device=device) for v in (A, C))
    valid_t=torch.tensor(valid,dtype=torch.bool,device=device)
    rng = np.random.default_rng(seed); losses = []
    for step in range(updates):
        ids = rng.integers(0, len(base), 16)
        end = int(rng.integers(3, base.shape[1]+1))
        b = base_t[ids, end-1]
        gt = truth_t[ids, end-1]
        support=np.arange(end) if end<=32 else np.r_[np.arange(12),np.arange(end-20,end)]
        rays_A = A_t[ids][:,support]
        rays_C = C_t[ids][:,support]
        rays_valid=valid_t[ids][:,support]
        z0 = model.z0[None].expand(len(ids), -1)
        z, _ = model.adapt(b, z0, rays_A, rays_C, rays_valid, create_graph=True)
        fixed = ((model.decode(b, z0)-gt)/.01).square().mean()
        adapted = ((model.decode(b, z)-gt)/.01).square().mean()
        loss = .5*(fixed+adapted)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss.detach()))
    dev_public,dev_labels=_load(pack,'dev');dev_target=_targets(dev_public,dev_labels,representation)
    dev_flat=dev_public['features'].reshape(-1,dev_public['features'].shape[-1])
    dev_pred=_predict(readout,dev_flat).reshape(*dev_target.shape)
    dev_score,dev_px,dev_m=_cv_score(dev_pred,dev_target,dev_public['features'][...,110:114],n)
    rotations=points_to_pose(labels['points_world'])[:,:,:3,:3]
    mean_R=mean_rotation(rotations)
    if not all(np.isfinite(v) for v in losses) or not np.isfinite(dev_score+dev_px+dev_m):
        raise ValueError('Nonfinite P58 training or development metric')
    package = dict(schema='prove.p58-cup-functional.v1', representation=representation,
                   seed=seed, updates=updates, readout={k: torch.as_tensor(v) if isinstance(v,np.ndarray) else v for k,v in readout.items()},
                   readout_candidates=candidates, model=model.state_dict(), config=vars(model.config),
                   mean_rotation=torch.tensor(mean_R,dtype=torch.float32) if n == 1 else torch.empty(0),
                   train_freeze=dict(path=str(Path(freeze_path)),sha256=sha(freeze_path)),
                   dev_readout=dict(score=dev_score,pixel_error=dev_px,position_error_m=dev_m),
                   training_diagnostics=dict(final_loss=losses[-1],frames=int(np.prod(base.shape[:2])),
                       valid_rays=int(valid.sum()),source_hashes={name:sha(ROOT/name) for name in SOURCES},
                       meta_scope='each sampled causal prefix starts from learned z0; not full-sequence carried-z outer training'),
                   physics_steps=0)
    output.parent.mkdir(parents=True, exist_ok=True); torch.save(package, output)
    return package


def queue(output, checkpoints):
    cells = []
    for condition in range(455406, 455410):
        for representation in REPRESENTATIONS:
            for arm in ('fixed','online','direct'):
                cells.append(dict(condition_seed=condition, representation=representation, arm=arm))
        for arm in ('fixed','continuous'):
            cells.append(dict(condition_seed=condition, representation='p47_geometry_reference', arm=arm))
    value = dict(schema='prove.p58-cup-development-queue.v1', status='draft_requires_root_freeze',
                 physics_steps=32, cells=cells,
                 checkpoints={name:dict(path=str(Path(path)),sha256=sha(path)) for name,path in checkpoints.items()})
    _write_json(output, value); return value


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--source',required=True);e.add_argument('--output',required=True)
    f=sub.add_parser('freeze');f.add_argument('--pack',required=True);f.add_argument('--plan',required=True);f.add_argument('--output',required=True)
    t=sub.add_parser('train');t.add_argument('--pack',required=True);t.add_argument('--representation',choices=REPRESENTATIONS,required=True);t.add_argument('--output',required=True);t.add_argument('--freeze',required=True);t.add_argument('--device',default='cpu');t.add_argument('--updates',type=int,default=500)
    q=sub.add_parser('queue');q.add_argument('--output',required=True);q.add_argument('--position-only',required=True);q.add_argument('--functional-frame',required=True)
    a=p.parse_args()
    if a.command=='export': result=export(a.source,a.output)
    elif a.command=='freeze': result=freeze(a.pack,a.plan,a.output)
    elif a.command=='train': result=train(a.pack,a.representation,a.output,a.freeze,a.updates,device=a.device)
    else: result=queue(a.output,{'position_only':a.position_only,'functional_frame':a.functional_frame})
    summary={'command':a.command,'schema':result.get('schema')}
    if a.command=='export': summary.update(splits=[dict(split=x['split'],conditions=x['conditions'],frames=x['frames']) for x in result['splits']])
    elif a.command=='train': summary.update(representation=result['representation'],updates=result['updates'],dev_readout=result['dev_readout'],training_diagnostics=result['training_diagnostics'])
    elif a.command=='freeze': summary.update(feature_dim=result['feature_dim'],representations=result['representations'])
    else: summary.update(physics_steps=result['physics_steps'],cells=len(result['cells']))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
