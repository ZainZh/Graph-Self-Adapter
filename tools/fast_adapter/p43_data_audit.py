"""Offline P43 dataset audit; this file may read separated oracle labels."""
import json
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.geometry3d_adapter import project
from prove.fast_adapter.neural_geometry_visual import PUBLIC_ARRAY_KEYS
from tools.fast_adapter.p43_data import ROOT, SOURCE_PATHS, sha, js


def main():
    torch.set_num_threads(2)
    records = []
    for task in ('cup', 'insert'):
        pca = torch.load(ROOT/'data'/task/'role_pca.pt', map_location='cpu', weights_only=True)
        train_raw, train_visible = [], []
        for split in ('train', 'dev'):
            pack = np.load(ROOT/'data'/task/split/'observations.npz', allow_pickle=False)
            labels = np.load(ROOT/'data_labels'/task/split/'labels.npz', allow_pickle=False)
            assert set(pack.files) == set(PUBLIC_ARRAY_KEYS)
            assert pca['fit_split'] == 'train'
            frame_counts, raw_receipts, reprojection, displacements, spans = [], [], [], [], []
            projected_error = 0.
            for i, seed in enumerate(pack['seeds']):
                episode = ROOT/'data_raw'/task/split/str(seed)
                receipt = json.loads((episode/'RECEIPT.json').read_text())
                assert receipt['public_sha256'] == sha(episode/'public_observations.npz')
                assert receipt['images_sha256'] == sha(episode/'visible_images.npz')
                assert receipt['label_sha256'] == sha(ROOT/'data_labels'/task/split/f'{seed}.npz')
                assert receipt['source_hashes'] == {p: sha(p) for p in SOURCE_PATHS}
                raw = np.load(episode/'public_observations.npz', allow_pickle=False)
                for k in raw.files:
                    assert np.array_equal(pack[k][i], raw[k], equal_nan=True), (task, split, seed, k)
                per_label = np.load(ROOT/'data_labels'/task/split/f'{seed}.npz', allow_pickle=False)
                for k in labels.files:
                    assert np.array_equal(labels[k][i], per_label[k], equal_nan=True)
                features = torch.as_tensor(np.load(episode/'raw_visual.npy', allow_pickle=False))
                visible = torch.as_tensor(raw['role_visible'])
                projected = torch.einsum('brd,rdk->brk', features-pca['mean'], pca['basis'])/pca['scale']
                projected = (projected*visible[..., None]).flatten(1).numpy()
                projected_error = max(projected_error, float(np.max(np.abs(projected-pack['local'][i]))))
                assert not pack['local'][i].reshape(-1, 3, 16)[~raw['role_visible']].any()
                if split == 'train':
                    train_raw.append(features); train_visible.append(visible)
                camera = raw['camera_position']
                displacements.append(float(np.linalg.norm(camera-camera[0], axis=1).max()))
                spans.append(np.ptp(camera, axis=0).tolist())
                frame_counts.append(int(raw['corners_valid'].sum()))
                for j in np.flatnonzero(raw['corners_valid']):
                    obs = {k: raw[k][j] for k in ('camera_position', 'camera_rotation', 'K')}
                    uv = project(labels['corners'][i, j], obs)
                    reprojection.append(min(float(np.square(uv-np.roll(raw['corners'][j], -s, axis=0)).mean()**.5) for s in range(4)))
                raw_receipts.append(dict(seed=int(seed), receipt_sha256=sha(episode/'RECEIPT.json'),
                     public_sha256=receipt['public_sha256'], images_sha256=receipt['images_sha256'],
                     labels_sha256=receipt['label_sha256'], visual_sha256=sha(episode/'raw_visual.npy'),
                     action_trace_sha256=sha(ROOT/'data_collector_audit'/task/split/f'{seed}.npz')))
            assert projected_error < 3e-4, projected_error
            assert min(displacements) > .015
            records.append(dict(task=task, split=split, episodes=len(pack['seeds']),
                sampled_frames=int(np.prod(pack['corners_valid'].shape)), physics_actions=len(pack['seeds'])*90,
                valid_corners_per_episode=frame_counts, role_visible_frames=pack['role_visible'].sum((0,1)).tolist(),
                camera_displacement_from_start_min_max=[min(displacements), max(displacements)],
                camera_axis_spans=spans, runtime_pack_has_only_allowlisted_public_arrays=True,
                runtime_pack_equals_raw_public_observations=True, separated_label_pack_equals_raw_labels=True,
                raw_sha256_checks_passed=True, pca_projection_cpu_gpu_absmax=projected_error,
                absent_role_projected_features_exact_zero=True,
                oracle_outer_corner_reprojection_px_quantiles=np.quantile(reprojection,[0,.5,.9,1]).tolist(),
                observations_sha256=sha(ROOT/'data'/task/split/'observations.npz'),
                labels_sha256=sha(ROOT/'data_labels'/task/split/'labels.npz'),
                pca_sha256=sha(ROOT/'data'/task/'role_pca.pt'), raw_receipts=raw_receipts))
        train_raw = torch.cat(train_raw); train_visible = torch.cat(train_visible)
        mean_errors, scale_errors = [], []
        for role in range(3):
            x = train_raw[:,role][train_visible[:,role]]
            mean_errors.append(float((x.mean(0)-pca['mean'][role]).abs().max()))
            scale = ((x-pca['mean'][role])@pca['basis'][role]).std(0).clamp_min(.01)
            scale_errors.append(float((scale-pca['scale'][role]).abs().max()))
        assert max(mean_errors) < 3e-5
        assert max(scale_errors) < 3e-5
        js(ROOT/'data'/task/'PCA_AUDIT.json', dict(passed=True, fitted_split='train',
           training_visible_frames=train_visible.sum(0).tolist(), mean_recomputed_cpu_absmax=mean_errors,
           scale_recomputed_cpu_absmax=scale_errors, no_development_or_test_frames_used=True))
    result = dict(passed=True, configurations=sum(r['episodes'] for r in records),
                  sampled_frames=sum(r['sampled_frames'] for r in records),
                  physics_actions=sum(r['physics_actions'] for r in records),
                  p41_source_hashes={p:sha(p) for p in SOURCE_PATHS[:2]}, datasets=records)
    js(ROOT/'data/DATA_AUDIT.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='datasets'}), flush=True)


if __name__ == '__main__':
    main()
