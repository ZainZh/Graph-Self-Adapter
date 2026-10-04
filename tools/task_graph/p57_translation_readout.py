#!/usr/bin/env python3
"""P57 A2: train-only selected PCA/ridge readout of two virtual frame origins.

This is an offline geometry qualification. It never creates a simulator or
issues robot commands. Runtime prediction accepts only the P57 public input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from prove.task_graph.p57_measurement_head import GeometryMeasurementHead, MeasurementConfig, MODES
from tools.task_graph.p57_train_measurement import (fit_bbox_relative_mean, fit_pose_prior,
                                                    load_bundle, pose_prior_pixels, prior_pixels,
                                                    public_input, validate_bundle)


ROOT = Path(__file__).resolve().parents[2]
DIMS = (0, 8, 16, 32)
LAMBDAS = (.01, .1, 1., 10., 100., 1000.)
FLOOR = .01
SOURCE_FILES = (ROOT/'tools/task_graph/p57_translation_readout.py',
                ROOT/'tools/task_graph/p57_train_measurement.py',
                ROOT/'prove/task_graph/p57_measurement_head.py',
                ROOT/'prove/task_graph/p57_public_geometry.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def public_matrix(episode, mode):
    """One row per current frame; no label, action, condition or outcome input."""
    cached = episode.setdefault('_a2_public_cache', {})
    if mode in cached:
        return cached[mode]
    obs = public_input(episode)
    # Reuse the A1 public geometric auxiliary construction; it has no weights.
    geometry = GeometryMeasurementHead._auxiliary(
        SimpleNamespace(config=MeasurementConfig(mode=mode), aux_dim=47), obs)
    blocks = []
    if mode in ('mask_geometry', 'fusion'):
        blocks.append(obs['mask_tokens'].flatten(1))
    if mode in ('dino_lila', 'fusion'):
        blocks.append(obs['raw'].flatten(1))
    blocks.append(geometry.flatten(1))
    result = (torch.cat(blocks, -1).float(), obs)
    cached[mode] = result
    return result


def relative_origins(episode, image_size=448):
    cached = episode.setdefault('_a2_label_cache', {})
    if image_size in cached:
        return cached[image_size]
    obs = public_input(episode)
    uv, valid = origin_targets(episode)
    valid = valid & obs['mask_valid'][..., 1:3]
    box = obs['boxes'][..., 1:3, :] * image_size
    center = (box[..., :2] + box[..., 2:]) * .5
    extent = (box[..., 2:] - box[..., :2]).clamp_min(1.)
    relative = (uv - center) / extent
    result = (relative.flatten(1), valid[..., None].expand_as(relative).flatten(1), center, extent)
    cached[image_size] = result
    return result


def origin_targets(episode):
    labels = episode['labels']
    required = {'book_uv', 'desired_book_uv', 'book_uv_valid', 'desired_book_uv_valid'}
    if not required.issubset(labels):
        raise ValueError('Missing isolated virtual-origin labels')
    uv = torch.stack((labels['book_uv'][..., 0, :], labels['desired_book_uv'][..., 0, :]), -2)
    valid = torch.stack((labels['book_uv_valid'][..., 0], labels['desired_book_uv_valid'][..., 0]), -1)
    return uv.float(), valid.bool()


def _weighted_mean_scale(x, frame_weight, floor):
    w = frame_weight / frame_weight.sum()
    mean = (x * w[:, None]).sum(0)
    scale = ((x - mean).square() * w[:, None]).sum(0).sqrt().clamp_min(floor)
    return mean, scale


def assemble(episodes, indices, mode):
    parts = []; targets = []; valid = []; weight = []; slices = []
    start = 0
    for index in indices:
        episode = episodes[index]
        x, _ = public_matrix(episode, mode)
        y, mask, _, _ = relative_origins(episode)
        n = len(x)
        parts.append(x); targets.append(y); valid.append(mask)
        weight.append(torch.full((n,), 1. / n))
        slices.append((index, slice(start, start + n)))
        start += n
    return torch.cat(parts), torch.cat(targets), torch.cat(valid), torch.cat(weight), slices


def prepare_pca(x, weight, max_rank, *, device):
    """Deterministic thin PCA from the frame Gram matrix; train fold only."""
    x = x.to(device)
    weight = weight.to(device)
    mean, scale = _weighted_mean_scale(x, weight, FLOOR)
    centered = (x - mean) / scale
    n = len(x)
    rank = min(max_rank, n - 1, x.shape[1])
    if rank <= 0:
        return mean.cpu(), scale.cpu(), torch.empty(x.shape[1], 0), torch.empty(0)
    gram = centered @ centered.T
    eigenvalue, eigenvector = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalue, descending=True)[:rank]
    value = eigenvalue[order].clamp_min(0)
    threshold = max(1e-6, float(value[0]) * 1e-7)
    actual = int((value > threshold).sum())
    if actual == 0:
        return mean.cpu(), scale.cpu(), torch.empty(x.shape[1], 0), torch.empty(0)
    value = value[:actual]
    u = eigenvector[:, order[:actual]]
    # Signed eigenvectors are arbitrary; fix their signs for reproducible bytes.
    peak = u.abs().argmax(0)
    sign = torch.sign(u[peak, torch.arange(actual, device=device)]).clamp(min=-1, max=1)
    u = u * torch.where(sign == 0, 1, sign)
    singular = value.sqrt().clamp_min(1e-6)
    components = centered.T @ (u / singular)
    score = centered @ components
    _, score_scale = _weighted_mean_scale(score, weight, 1e-3)
    return mean.cpu(), scale.cpu(), components.cpu(), score_scale.cpu()


def transform(x, model, rank):
    centered = (x - model['feature_mean']) / model['feature_scale']
    if rank == 0:
        return x.new_zeros((len(x), 0))
    return (centered @ model['components'][:, :rank]) / model['score_scale'][:rank]


def fit_ridge(model, x, y, valid, weight, rank, ridge, *, score_full=None):
    score = transform(x, model, rank) if score_full is None else score_full[:, :rank]
    value = torch.empty(rank + 1, y.shape[-1])
    for column in range(y.shape[-1]):
        selected = valid[:, column]
        if not selected.any():
            raise ValueError('No train-visible label in one origin coordinate')
        w = weight[selected]
        w = w / w.sum()
        mean_y = (y[selected, column] * w).sum()
        value[0, column] = mean_y
        if rank:
            z = score[selected]
            residual = y[selected, column] - mean_y
            gram = z.T @ (z * w[:, None]) + ridge * torch.eye(rank)
            value[1:, column] = torch.linalg.solve(gram, z.T @ (w * residual))
    return value


def predict_relative(model, coef, x):
    rank = len(coef) - 1
    return coef[0] + transform(x, model, rank) @ coef[1:]


def predict_pixels(relative, obs, image_size=448):
    box = obs['boxes'][..., 1:3, :] * image_size
    center = (box[..., :2] + box[..., 2:]) * .5
    extent = (box[..., 2:] - box[..., :2]).clamp_min(1.)
    return center + extent * relative.reshape(-1, 2, 2, 2)


def fit_representation(x, weight, *, device):
    mean, scale, components, score_scale = prepare_pca(x, weight, 32, device=device)
    return dict(feature_mean=mean, feature_scale=scale,
                components=components, score_scale=score_scale,
                actual_rank=components.shape[1])


def score_episodes(episodes, indices, mode, model, coef):
    rows = []
    for index in indices:
        episode = episodes[index]
        x, obs = public_matrix(episode, mode)
        _, valid, _, _ = relative_origins(episode)
        truth, _ = origin_targets(episode)
        pred = predict_pixels(predict_relative(model, coef, x), obs)
        admissible = valid.reshape(-1, 2, 2, 2)[..., 0]
        error = torch.linalg.vector_norm(pred - truth, dim=-1)
        rows.append(dict(index=index, condition_id=episode['condition_id'],
                         source_policy=episode['source_policy'],
                         pixel_mean=float(error[admissible].mean()) if admissible.any() else None,
                         valid_origins=int(admissible.sum()), total_origins=int(admissible.numel())))
    return rows


def choose_candidate(episodes, train_indices, mode, *, device):
    groups = sorted({episodes[i]['condition_id'] for i in train_indices})
    if any(sum(episodes[i]['condition_id'] == group for i in train_indices) != 2 for group in groups):
        raise ValueError('A2 requires paired expert/policy train conditions')
    scores = {(rank, ridge): [] for rank in DIMS for ridge in LAMBDAS}
    for group in groups:
        fitting = [i for i in train_indices if episodes[i]['condition_id'] != group]
        held = [i for i in train_indices if episodes[i]['condition_id'] == group]
        x, y, valid, weight, _ = assemble(episodes, fitting, mode)
        representation = fit_representation(x, weight, device=device)
        train_scores = transform(x, representation, representation['actual_rank'])
        held_data = []
        for index in held:
            episode = episodes[index]
            test_x, obs = public_matrix(episode, mode)
            _, test_valid, _, _ = relative_origins(episode)
            truth, _ = origin_targets(episode)
            held_data.append((index, transform(test_x, representation, representation['actual_rank']),
                              obs, test_valid.reshape(-1, 2, 2, 2)[..., 0], truth))
        for rank in DIMS:
            if rank > representation['actual_rank']:
                continue
            for ridge in LAMBDAS:
                coef = fit_ridge(representation, x, y, valid, weight, rank, ridge,
                                 score_full=train_scores)
                for _, test_score, obs, test_valid, truth in held_data:
                    relative = coef[0] + test_score[:, :rank] @ coef[1:]
                    prediction = predict_pixels(relative, obs)
                    error = torch.linalg.vector_norm(prediction - truth, dim=-1)
                    scores[rank, ridge].append(float(error[test_valid].mean()))
    candidates = [dict(dim=rank, ridge=ridge, train_loco_pixel_macro=float(np.mean(values)),
                       episodes=len(values)) for (rank, ridge), values in scores.items() if values]
    candidates.sort(key=lambda row: (row['train_loco_pixel_macro'], row['dim'], -row['ridge']))
    if not candidates or candidates[0]['episodes'] != len(train_indices):
        raise ValueError('Incomplete train-only leave-one-condition-out selection')
    return candidates[0], candidates


def _macro(rows):
    def group(items, name):
        numbers = [item[name] for item in items if item[name] is not None]
        return float(np.mean(numbers)) if numbers else None
    return dict(episodes=len(rows), pixel_mean=group(rows, 'pixel_mean'),
                bbox_relative_mean_pixel=group(rows, 'bbox_relative_mean_pixel'),
                pose_prior_pixel=group(rows, 'pose_prior_pixel'),
                by_source_policy={source: dict(episodes=sum(r['source_policy'] == source for r in rows),
                                               pixel_mean=group([r for r in rows if r['source_policy'] == source], 'pixel_mean'))
                                  for source in sorted({r['source_policy'] for r in rows})})


def fit_equal_episode_priors(episodes, train_indices):
    bbox_each = []
    pose_each = []
    for index in train_indices:
        episode = episodes[index]
        relative, valid, _, _ = relative_origins(episode)
        rel = relative.reshape(-1, 2, 2, 2)
        mask = valid.reshape(-1, 2, 2, 2)[..., 0]
        summed = (rel * mask[..., None]).sum(0)
        count = mask.sum(0)[..., None].clamp_min(1)
        bbox_each.append(summed / count)
        pose_each.append(fit_pose_prior(episodes, [index]))
    bbox = torch.stack(bbox_each).mean(0).unsqueeze(-2)
    pose = {key: np.stack([row[key] for row in pose_each]).mean(0) for key in pose_each[0]}
    return bbox, pose


def evaluate_once(episodes, indices, mode, model, coef, bbox_prior, pose_prior):
    rows = []; predictions = []
    for index in indices:
        episode = episodes[index]
        x, obs = public_matrix(episode, mode)
        _, valid, _, _ = relative_origins(episode)
        truth, _ = origin_targets(episode)
        pred = predict_pixels(predict_relative(model, coef, x), obs)
        bbox = prior_pixels(obs, bbox_prior, 448)[..., 0, :]
        pose = pose_prior_pixels(obs, pose_prior)[..., 0, :]
        admissible = valid.reshape(-1, 2, 2, 2)[..., 0]
        err = lambda p: torch.linalg.vector_norm(p - truth, dim=-1)
        row = dict(condition_id=episode['condition_id'], source_policy=episode['source_policy'],
                   frames=len(pred), valid_origins=int(admissible.sum()), total_origins=int(admissible.numel()),
                   pixel_mean=float(err(pred)[admissible].mean()) if admissible.any() else None,
                   bbox_relative_mean_pixel=float(err(bbox)[admissible].mean()) if admissible.any() else None,
                   pose_prior_pixel=float(err(pose)[admissible].mean()) if admissible.any() else None)
        rows.append(row)
        predictions.append(dict(condition_id=episode['condition_id'], split=episode['split'],
                                source_policy=episode['source_policy'],
                                source_seal_sha256=episode['source_seal_sha256'],
                                public_frame_sha256=[r['sha256'] for r in episode['input']['public_frames']],
                                pred_uv=pred, role_valid=obs['mask_valid'][..., 1:3],
                                camera_K=obs['camera_K'], camera_pose=obs['camera_pose'],
                                eef_transform=obs['eef_transform']))
    return rows, predictions


def freeze(manifest_path, plan_path, output):
    manifest_path = Path(manifest_path).resolve();plan_path = Path(plan_path).resolve();output = Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    data = json.loads(manifest_path.read_text())
    if data.get('schema') != 'p57.geometry-pack-manifest.v1' or data.get('cohort_counts') != dict(
            train=34, validation=8, transitions=573, frames=615) or len(data.get('episodes',[]))!=42:
        raise ValueError('Require exact opened P57 pack')
    split_hash = {split: _digest([dict(condition_id=r['condition_id'], source_policy=r['source_policy'],
                                      bundle_sha256=r['bundle']['sha256']) for r in data['episodes'] if r['split'] == split])
                  for split in ('train', 'validation')}
    value = dict(schema='p57.a2-translation-training-freeze.v1',
                 pack_manifest=dict(path=str(manifest_path), sha256=sha(manifest_path)),
                 plan=dict(path=str(plan_path), sha256=sha(plan_path)),
                 sources={str(path): sha(path) for path in SOURCE_FILES}, splits=split_hash,
                 grid=dict(pca_dims=list(DIMS), ridge_lambdas=list(LAMBDAS), std_floor=FLOOR,
                           group='condition_id paired source policies', score='per-trajectory mean origin pixel error',
                           tie_break='lower dim then stronger ridge'),
                 train=dict(device='GPU or CPU deterministic eigh', refit='all train after LOCO',
                            development='one final readout only', physics_steps=0),
                 arms=[dict(mode=mode, output_relative=f'runs_a2/{mode}/HEAD.pt') for mode in MODES])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    return value


def verify_freeze(freeze_path, manifest_path, mode, output):
    freeze_path=Path(freeze_path).resolve();manifest_path=Path(manifest_path).resolve();output=Path(output).resolve()
    value=json.loads(freeze_path.read_text())
    if value.get('schema')!='p57.a2-translation-training-freeze.v1' or mode not in MODES:
        raise ValueError('Invalid A2 training freeze')
    if sha(manifest_path)!=value['pack_manifest']['sha256'] or sha(value['plan']['path'])!=value['plan']['sha256']:
        raise ValueError('A2 pack or plan changed after freeze')
    for path in SOURCE_FILES:
        if value['sources'].get(str(path))!=sha(path):raise ValueError('A2 source changed after freeze')
    if value['grid']!=dict(pca_dims=list(DIMS),ridge_lambdas=list(LAMBDAS),std_floor=FLOOR,
                           group='condition_id paired source policies',score='per-trajectory mean origin pixel error',
                           tie_break='lower dim then stronger ridge'):
        raise ValueError('A2 search grid changed')
    if value['train']!=dict(device='GPU or CPU deterministic eigh',refit='all train after LOCO',
                            development='one final readout only',physics_steps=0):
        raise ValueError('A2 train contract changed')
    if value['arms']!=[dict(mode=name,output_relative=f'runs_a2/{name}/HEAD.pt') for name in MODES]:
        raise ValueError('A2 arm queue changed')
    arm=next((arm for arm in value['arms'] if arm['mode']==mode),None)
    if arm is None or output!=(freeze_path.parent/arm['output_relative']).resolve():
        raise ValueError('A2 output outside frozen arms')
    return value


def run(manifest_path, freeze_path, output, mode, *, device='cpu'):
    verify_freeze(freeze_path,manifest_path,mode,output)
    output=Path(output);prediction_path=output.parent/'PREDICTIONS.pt';report_path=output.parent/'REPORT.json'
    for path in (output,prediction_path,report_path):
        if path.exists():raise FileExistsError(path)
    episodes=load_bundle(manifest_path)['episodes'];train_indices,dev_indices=validate_bundle(dict(schema='p57.geometry-bundle.v1',episodes=episodes))
    selected,candidates=choose_candidate(episodes,train_indices,mode,device=device)
    x,y,valid,weight,_=assemble(episodes,train_indices,mode)
    model=fit_representation(x,weight,device=device)
    coef=fit_ridge(model,x,y,valid,weight,selected['dim'],selected['ridge'])
    bbox_prior,pose_prior=fit_equal_episode_priors(episodes,train_indices)
    historical_bbox=fit_bbox_relative_mean(episodes,train_indices,448)
    historical_pose=fit_pose_prior(episodes,train_indices)
    rows,predictions=evaluate_once(episodes,dev_indices,mode,model,coef,bbox_prior,pose_prior)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='p57.a2-public-origin-predictions.v1',role_order=['book','desired_book_frame'],
                    predictions=predictions,labels_included=False),prediction_path)
    parameters=int(model['components'][:,:selected['dim']].numel()+coef.numel()+model['feature_mean'].numel()*2+model['score_scale'][:selected['dim']].numel())
    checkpoint=dict(schema='p57.a2-translation-readout.v1',mode=mode,selected=selected,
                    candidates=candidates,model=model,coef=coef,
                    baseline_bbox_relative_mean=bbox_prior,
                    baseline_pose_prior={k:torch.from_numpy(v) for k,v in pose_prior.items()},
                    historical_a1_frame_weighted_bbox_relative_mean=historical_bbox,
                    historical_a1_frame_weighted_pose_prior={k:torch.from_numpy(v) for k,v in historical_pose.items()},
                    parameter_count=parameters,pack_manifest_sha256=sha(manifest_path),
                    freeze_sha256=sha(freeze_path),source_sha256={str(p):sha(p) for p in SOURCE_FILES},
                    public_predictions_sha256=sha(prediction_path),validation_per_episode=rows,
                    validation_macro=_macro(rows),claim='offline origin translation readout; no adapter or physics')
    torch.save(checkpoint,output)
    report=dict(schema='p57.a2-translation-report.v1',mode=mode,selected=selected,candidates=candidates,
                readout_identity=('train_bbox_relative_mean_only' if selected['dim']==0 else 'pca_ridge_origin_residual'),
                baseline_weighting='each train episode has one vote',
                parameter_count=parameters,checkpoint_sha256=sha(output),predictions_sha256=sha(prediction_path),
                pack_manifest_sha256=sha(manifest_path),freeze_sha256=sha(freeze_path),
                validation_per_episode=rows,validation_macro=_macro(rows))
    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(mode=mode,selected=selected,output=str(output),report=str(report_path))),flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--manifest',type=Path,required=True);f.add_argument('--plan',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    t=sub.add_parser('train');t.add_argument('--manifest',type=Path,required=True);t.add_argument('--freeze',type=Path,required=True);t.add_argument('--mode',choices=MODES,required=True);t.add_argument('--output',type=Path,required=True);t.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(1)
    if a.command=='freeze':print(json.dumps(dict(arms=len(freeze(a.manifest,a.plan,a.output)['arms']),output=str(a.output))))
    else:run(a.manifest,a.freeze,a.output,a.mode,device=a.device)


if __name__=='__main__':main()
