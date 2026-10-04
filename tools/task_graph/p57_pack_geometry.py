#!/usr/bin/env python3
"""P57 Stage A offline geometry bundle from sealed P56 train/development data.

Public inputs and private training labels remain separate. No simulator is
created or stepped, and frozen visual features are loaded from P56 caches.
Virtual anchors are coordinate conventions, never visible surface corners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from prove.task_graph.p57_public_geometry import public_geometry
from tools.task_graph import p56_dataset
from tools.task_graph.collect_libero_demo_gt import dump, sha

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'p57.geometry-bundle.v1'
ANCHOR_OFFSET = 0.03
ANCHOR_LOCAL = np.array([[0., 0., 0., 1.],
                         [ANCHOR_OFFSET, 0., 0., 1.],
                         [0., 0., ANCHOR_OFFSET, 1.]], dtype=np.float64)
FEATURE_SHAPES = dict(raw=(2, 3, 3072), boxes=(2, 3, 4),
                      visible=(2, 3), proprio=(18,))


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def frozen_final_relation(manifest_path):
    """Follow the sealed P56→P55→P54→P53 chain to its fixed task goal."""
    manifest = read(manifest_path)
    links = []
    ref = manifest['collection_binding']
    for next_key in ('p55_binding', 'parent', 'parent'):
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Historical relation-binding chain changed')
        links.append(dict(path=str(Path(ref['path']).resolve()), sha256=ref['sha256']))
        current = read(ref['path'])
        ref = current[next_key]
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('Frozen P53 final relation source changed')
    links.append(dict(path=str(Path(ref['path']).resolve()), sha256=ref['sha256']))
    original = read(ref['path'])
    relation = original['relation']
    final = _pose(relation['final_caddy_book'], 'frozen_final_caddy_book')
    relation_bytes = json.dumps(relation, sort_keys=True, separators=(',', ':'),
                                allow_nan=False).encode()
    provenance = dict(binding_chain=links, p53_relation_sha256=hashlib.sha256(relation_bytes).hexdigest(),
        source_frames=relation.get('source_frames', []),
        goal_semantics='frozen P53 final_caddy_book; not phase-dependent micro teacher target')
    return final, provenance


def _pose(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Invalid pose: '+name)
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('Invalid pose homogeneous row: '+name)
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError('Invalid pose rotation: '+name)
    return matrix


def _world_pose(position, rotation):
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return _pose(result, 'supervisor_world_pose')


def world_anchors(world_frame):
    """Three virtual frame axes: origin, +3cm local X, +3cm local Z."""
    pose = _pose(world_frame, 'anchor_frame')
    return (pose @ ANCHOR_LOCAL.T).T[:, :3]


def project_anchors(world_points, camera_K, camera_pose, width=448, height=448):
    """MuJoCo camera: local +X right, +Y up, -Z optical forward."""
    points = np.asarray(world_points, dtype=np.float64)
    K = np.asarray(camera_K, dtype=np.float64)
    poses = np.asarray(camera_pose, dtype=np.float64)
    if points.shape != (3, 3) or K.shape != (2, 3, 3) or poses.shape != (2, 4, 4):
        raise ValueError('Expected three anchors and two calibrated cameras')
    if not np.isfinite(points).all() or not np.isfinite(K).all() or not np.isfinite(poses).all():
        raise ValueError('Nonfinite anchor or camera calibration')
    uv = np.zeros((2, 3, 2), dtype=np.float64)
    depth = np.zeros((2, 3), dtype=np.float64)
    valid = np.zeros((2, 3), dtype=bool)
    homogeneous = np.c_[points, np.ones(3)]
    for view in range(2):
        camera = _pose(poses[view], 'public_camera')
        xyz = (np.linalg.inv(camera) @ homogeneous.T).T[:, :3]
        depth[view] = -xyz[:, 2]
        positive = depth[view] > 1e-8
        safe = np.where(positive, depth[view], 1.)
        uv[view, :, 0] = K[view, 0, 0] * xyz[:, 0] / safe + K[view, 0, 2]
        uv[view, :, 1] = K[view, 1, 1] * -xyz[:, 1] / safe + K[view, 1, 2]
        valid[view] = (positive & (uv[view, :, 0] >= 0) &
            (uv[view, :, 0] < width) & (uv[view, :, 1] >= 0) &
            (uv[view, :, 1] < height))
        uv[view, ~positive] = 0.
    if not np.isfinite(uv).all() or not np.isfinite(depth).all():
        raise ValueError('Nonfinite virtual-anchor projection')
    return uv, depth, valid


def private_geometry(label, private_state, names, public, fixed_caddy_book, *, check_camera=False):
    """Reconstruct GT frames only for labels; never return them in input."""
    E_B = _pose(label['actual_eef_book'], 'book_in_eef')
    C_B = _pose(label['actual_caddy_book'], 'book_in_caddy')
    C_G = _pose(fixed_caddy_book, 'frozen_desired_book_in_caddy')
    phase_target = _pose(label['desired_caddy_book'], 'teacher_phase_target')
    if label.get('teacher_phase') == 'insert' and not np.allclose(phase_target, C_G, atol=1e-8):
        raise ValueError('Insert teacher target differs from frozen final relation')
    W_E = _pose(public['eef_transform'], 'public_eef')
    W_B = _pose(W_E @ E_B, 'world_book_reconstructed')
    W_C = _pose(W_B @ np.linalg.inv(C_B), 'world_caddy_reconstructed')
    W_G = _pose(W_C @ C_G, 'desired_book_world')
    G_B = _pose(np.linalg.inv(W_G) @ W_B, 'book_in_desired_book')
    body_index = names['body'].index('black_book_1_main')
    site_index = names['site'].index('desk_caddy_1_back_contain_region')
    actual_book = _world_pose(private_state['supervisor_body_xpos'][body_index],
                              private_state['supervisor_body_xmat'][body_index])
    actual_caddy = _world_pose(private_state['supervisor_site_xpos'][site_index],
                               private_state['supervisor_site_xmat'][site_index])
    book_error = float(np.max(np.abs(W_B - actual_book)))
    caddy_error = float(np.max(np.abs(W_C - actual_caddy)))
    if book_error > 2e-5 or caddy_error > 2e-5:
        raise ValueError('Teacher pose reconstruction differs from sealed simulator transforms')
    book = world_anchors(W_B)
    desired = world_anchors(W_G)
    image = public['image_shape']
    book_uv, book_depth, book_valid = project_anchors(
        book, public['camera_K'], public['camera_pose'], image[1], image[0])
    desired_uv, desired_depth, desired_valid = project_anchors(
        desired, public['camera_K'], public['camera_pose'], image[1], image[0])
    if check_camera:
        # A visible book geom cannot have its body origin behind the camera.
        visible = public['book_mask_pixels'] > 0
        if np.any(visible & (book_depth[:, 0] <= 0)):
            raise ValueError('Public camera sign conflicts with visible book mask')
    return dict(book_world_anchors=book, desired_book_world_anchors=desired,
        book_uv=book_uv, desired_book_uv=desired_uv,
        book_uv_depth=book_depth, desired_book_uv_depth=desired_depth,
        book_uv_valid=book_valid, desired_book_uv_valid=desired_valid,
        book_in_eef=E_B, book_in_caddy=C_B, desired_book_in_caddy=C_G,
        world_book=W_B, world_caddy=W_C, desired_book_world=W_G,
        book_in_desired=G_B, reconstruction_error=np.array([book_error, caddy_error]))


def _stack(records, key, dtype=None):
    value = np.stack([record[key] for record in records])
    if dtype is not None:
        value = value.astype(dtype)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite packed '+key)
    return torch.from_numpy(value)


def audit_teacher_targets(rows, source_manifest):
    """Preserve evidence that teacher's phase target is not the fixed goal."""
    phases = {}
    changed = []
    per_episode = []
    largest_z_range = 0.
    for row in rows:
        root = Path(row['directory'])
        labels = lines(root/'private/LABELS.jsonl')
        targets = np.asarray([label['desired_caddy_book'] for label in labels], dtype=np.float64)
        if targets.shape != (row['actions']+1, 4, 4) or not np.isfinite(targets).all():
            raise ValueError('Invalid sealed teacher phase target sequence')
        distinct = 1 + int(np.any(np.max(np.abs(targets-targets[0]), axis=(1, 2)) > 1e-8))
        z_range = float(np.ptp(targets[:, 2, 3]))
        largest_z_range = max(largest_z_range, z_range)
        local_phases = sorted({label['teacher_phase'] for label in labels})
        for label in labels:
            phase = label['teacher_phase']
            phases[phase] = phases.get(phase, 0)+1
        if distinct > 1:
            changed.append(row['cell_index'])
        per_episode.append(dict(cell_index=row['cell_index'], distinct_teacher_targets=distinct,
                                teacher_phases=local_phases, target_z_range_m=z_range))
    audit = dict(schema='p57.teacher-phase-target-audit.v1',
        source_manifest=reference(source_manifest), episodes=len(rows),
        varying_target_episodes=len(changed), varying_cell_indices=changed,
        teacher_phase_frames=phases, max_teacher_target_z_range_m=largest_z_range,
        per_episode=per_episode,
        interpretation=('P56 teacher action labels are valid per-step corrections; its desired_caddy_book '
                        'is a phase-dependent action target, not the fixed desired-book-frame label.'))
    if (audit['episodes'] != 42 or audit['varying_target_episodes'] != 21 or
            phases != dict(local_align=85, insert=530) or
            not np.isclose(largest_z_range, .12594861221714843, atol=1e-8)):
        raise ValueError('Opened P56 teacher target audit differs from pre-export evidence')
    return audit


def pack_episode(row, fixed_caddy_book, relation_provenance):
    if not row['eligible'] or not row['near_entry_qualified'] or row['actions'] < 1:
        raise ValueError('Only sealed, valid entered P56 episodes can become P57 training data')
    root = Path(row['directory']).resolve()
    seal = read(root/'SEAL.json')
    names = read(p56_dataset.sealed_file(root, seal, 'private/NAMES.json'))
    labels = lines(p56_dataset.sealed_file(root, seal, 'private/LABELS.jsonl'))
    transitions = lines(p56_dataset.sealed_file(root, seal, 'public/TRANSITIONS.jsonl'))
    T = row['actions']
    if len(labels) != T+1 or len(transitions) != T or len(row['frames']) != T+1:
        raise ValueError('P56 trajectory T/T+1 alignment changed')
    public_frames = []
    features = {key: [] for key in FEATURE_SHAPES}
    public_records = []
    private_records = []
    for t, (source, label) in enumerate(zip(row['frames'], labels)):
        if label['t'] != t or Path(source['frame']).name != label['public_observation']:
            raise ValueError('P56 frame/label alignment changed')
        if sha(source['frame']) != source['frame_sha256']:
            raise ValueError('Sealed public frame changed after catalog verification')
        if sha(source['feature']) != source['feature_sha256']:
            raise ValueError('Sealed P56 feature changed after catalog verification')
        frame = p56_dataset.sealed_file(root, seal, 'public/'+label['public_observation'])
        private = p56_dataset.sealed_file(root, seal, 'private/'+label['private_state'])
        if sha(private) != label['private_state_sha256']:
            raise ValueError('Teacher label/private state binding changed')
        with np.load(frame, allow_pickle=False) as observation:
            geometry = public_geometry(observation)
            masks = np.asarray(observation['visible_masks'])
            rgb = np.asarray(observation['rgb'])
            if masks.shape != (2, 3, 448, 448) or rgb.shape != (2, 448, 448, 3):
                raise ValueError('Original P56 dual 448 RGB/mask interface changed')
            geometry['image_shape'] = (448, 448)
            geometry['book_mask_pixels'] = masks[:, 1].sum(axis=(1, 2))
        cached = torch.load(source['feature'], map_location='cpu', weights_only=True)
        if set(cached) != set(FEATURE_SHAPES):
            raise ValueError('Unexpected frozen P56 feature keys')
        for key, shape in FEATURE_SHAPES.items():
            value = cached[key]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                    tuple(value.shape) != shape or not torch.isfinite(value).all()):
                raise ValueError('Invalid frozen P56 '+key+' feature')
            features[key].append(value)
        with np.load(private, allow_pickle=False) as snapshot:
            if not all(key in snapshot for key in ('supervisor_sim_state', 'supervisor_ctrl',
                    'supervisor_body_xpos', 'supervisor_body_xmat',
                    'supervisor_site_xpos', 'supervisor_site_xmat')):
                raise ValueError('Full sealed P56 per-frame simulator state required')
            derived = private_geometry(label, snapshot, names, geometry, fixed_caddy_book,
                check_camera=t in (0, T))
        public_frames.append(dict(path=str(frame), sha256=sha(frame),
            feature_path=str(Path(source['feature']).resolve()),
            feature_sha256=source['feature_sha256']))
        public_records.append(geometry)
        private_records.append(derived)
    executed = np.asarray([record['executed_action'][:6] for record in transitions], dtype=np.float32)
    expert = np.asarray([record['expert_action'][:6] for record in labels[:-1]], dtype=np.float32)
    if executed.shape != (T, 6) or expert.shape != (T, 6) or not np.isfinite(executed).all() or not np.isfinite(expert).all():
        raise ValueError('Invalid P56 executed/expert action alignment')
    if not all(record.get('expert_label_valid') is True for record in labels[:-1]):
        raise ValueError('Invalid offline teacher correction in executed trajectory')
    public_keys = dict(mask_tokens=np.float32, mask_valid=None,
        camera_K=np.float64, camera_pose=np.float64, eef_transform=np.float64)
    private_keys = dict(book_world_anchors=np.float32, desired_book_world_anchors=np.float32,
        book_uv=np.float32, desired_book_uv=np.float32,
        book_uv_depth=np.float32, desired_book_uv_depth=np.float32,
        book_uv_valid=None, desired_book_uv_valid=None,
        book_in_eef=np.float32, book_in_caddy=np.float32,
        desired_book_in_caddy=np.float32, world_book=np.float32,
        world_caddy=np.float32, desired_book_world=np.float32,
        book_in_desired=np.float32, reconstruction_error=np.float32)
    result = dict(schema=SCHEMA, cell_index=row['cell_index'], condition_id=row['condition_id'],
        geometry_id=row['geometry_group'], scale_xyz=row['scale_xyz'], split=row['split'],
        source_policy=row['source_policy'], source_seal_sha256=row['source_seal_sha256'],
        relation_provenance=relation_provenance,
        anchor_definition=dict(frames=['book', 'desired_book_frame'],
            local_points=['origin', 'positive_x_0.03m', 'positive_z_0.03m'],
            semantics='virtual coordinate anchors, not visible surface corners or actual mask mouth'),
        input=dict(features={key: torch.stack(values) for key, values in features.items()},
            public_frames=public_frames,
            **{key: _stack(public_records, key, dtype) for key, dtype in public_keys.items()}),
        labels={key: _stack(private_records, key, dtype) for key, dtype in private_keys.items()},
        executed_actions=torch.from_numpy(executed), expert_actions=torch.from_numpy(expert))
    if len(result['input']['public_frames']) != T+1 or result['executed_actions'].shape[0] != T:
        raise AssertionError('Packed P57 action/frame length mismatch')
    return result


def pack(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    output = Path(output).resolve()
    fixed_goal, relation_provenance = frozen_final_relation(manifest_path)
    catalog = p56_dataset.catalog(manifest_path)
    source_rows = catalog['cells']
    selected = [row for row in source_rows if row['eligible'] and row['split'] in ('train', 'validation')]
    if (len(source_rows) != 56 or len(selected) != 42 or
            sum(row['split'] == 'train' for row in selected) != 34 or
            sum(row['split'] == 'validation' for row in selected) != 8 or
            sum(row['actions'] for row in selected) != 573 or
            sum(row['actions']+1 for row in selected) != 615):
        raise ValueError('P57 Stage A requires the exact opened P56 42-trajectory cohort')
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'SOURCE_CATALOG.json', catalog)
    dump(output/'LABEL_TARGET_AUDIT.json', audit_teacher_targets(selected, manifest_path))
    records = []
    for row in selected:
        episode = pack_episode(row, fixed_goal, relation_provenance)
        destination = output/f"cell_{row['cell_index']:03d}.pt"
        torch.save(episode, destination)
        records.append(dict(cell_index=row['cell_index'], condition_id=row['condition_id'],
            geometry_id=row['geometry_group'], split=row['split'], source_policy=row['source_policy'],
            actions=row['actions'], frames=row['actions']+1, bundle=reference(destination),
            source_seal=dict(path=str(Path(row['directory'])/'SEAL.json'),
                             sha256=row['source_seal_sha256'])))
    source_all = [dict(cell_index=row['cell_index'], split=row['split'],
        source_policy=row['source_policy'], near_entry_qualified=row['near_entry_qualified'],
        eligible=row['eligible'], source_seal=dict(path=str(Path(row['directory'])/'SEAL.json'),
            sha256=row['source_seal_sha256'])) for row in source_rows]
    result = dict(schema='p57.geometry-pack-manifest.v1',
        plan=reference(ROOT/'evaluations/fast_adapter_p57_geometry_tokens_v01/PLAN.md'),
        source_collection_manifest=reference(manifest_path),
        source_catalog=reference(output/'SOURCE_CATALOG.json'),
        teacher_target_audit=reference(output/'LABEL_TARGET_AUDIT.json'),
        sources={str(Path(__file__).resolve()): sha(__file__),
            str(ROOT/'prove/task_graph/p57_public_geometry.py'):
                sha(ROOT/'prove/task_graph/p57_public_geometry.py')},
        anchor_definition=dict(frames=['book', 'desired_book_frame'], offset_m=ANCHOR_OFFSET,
            runtime_permission='No private geometry, desired-book pose or virtual-anchor label is a runtime input'),
        fixed_goal_provenance=relation_provenance,
        all_collection_cells=source_all, episodes=records,
        cohort_counts=dict(train=34, validation=8, transitions=573, frames=615),
        split_rule='P56 frozen episode-level split; no frame-level reassignment or test trajectories',
        simulator_steps=0, vision_feature_extractions=0)
    dump(output/'MANIFEST.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = pack(args.manifest, args.output)
    print(json.dumps(dict(output=str(args.output), cohort_counts=result['cohort_counts'])))


if __name__ == '__main__':
    main()
