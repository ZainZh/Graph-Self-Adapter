"""P61 native RGB cache; labels are opt-in and never runtime inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from prove.fast_adapter.p59_measurement import rotation_from_6d
from prove.fast_adapter.p60_contour import outer_hull_boundary

FRAME_INDICES = (0, 4, 8, 12, 16, 20, 24, 29)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def source_episode(path):
    """Load only public, hash-bound, original pixels and calibrated poses."""
    path = Path(path)
    manifest = json.loads((path/'MANIFEST.json').read_text())
    if sha256(path/'public.npz') != manifest['output']['public.npz']:
        raise ValueError('upstream public digest mismatch')
    with np.load(path/'public.npz', allow_pickle=False) as a:
        indices = np.asarray(a['frame_index'])
        rows = np.array([np.flatnonzero(indices == t).item() for t in FRAME_INDICES])
        rgb = a['rgb'][rows].copy()
        masks = np.unpackbits(a['role_masks_packed'][rows], axis=-1, count=448).astype(bool)
        p, R = a['eef_position'][rows], a['eef_rotation'][rows]
        K, camera = a['camera_K'][rows], a['camera_pose'][rows]
    if rgb.shape != (8,448,448,3) or masks.shape != (8,3,448,448):
        raise ValueError('native image/mask shape drift')
    world_anchor = np.eye(4, dtype=np.float32)
    world_anchor[:3,:3], world_anchor[:3,3] = R[0], p[0]
    anchor_world = np.linalg.inv(world_anchor)
    world_eef = np.broadcast_to(np.eye(4, dtype=np.float32), (8,4,4)).copy()
    world_eef[:,:3,:3], world_eef[:,:3,3] = R,p
    boundary, valid, hulls = [], [], []
    for mask in masks[:,2]:
        points, info = outer_hull_boundary(mask)
        valid.append(points is not None)
        boundary.append(np.zeros((32,2), np.float32) if points is None else points)
        hulls.append(torch.tensor(info.get('hull_vertices', []),dtype=torch.float32).reshape(-1,2))
    patch = torch.nn.functional.avg_pool2d(torch.from_numpy(masks.copy()).float(),16,16)
    public = dict(role_patch_masks=patch[:,None], role_valid=torch.from_numpy(masks.reshape(8,3,-1).any(-1))[:,None],
        K=torch.from_numpy(K.copy()).float()[:,None], T_anchor_camera=torch.from_numpy(anchor_world@camera).float()[:,None],
        T_anchor_eef=torch.from_numpy(anchor_world@world_eef).float(), T_world_anchor=torch.from_numpy(world_anchor),
        q=torch.from_numpy(np.concatenate((p,R[:,:,0],R[:,:,1]),-1)).float(),
        frame_index=torch.tensor(FRAME_INDICES), boundary_uv=torch.tensor(np.asarray(boundary),dtype=torch.float32),
        boundary_valid=torch.tensor(valid), goal_hull_vertices=hulls, image_hw=torch.tensor([448,448]),
        episode_id=str(manifest['seed']), source_split=manifest['split'], distribution=manifest['distribution'])
    return rgb, public, rows, manifest

def offline_labels(path, rows, T_world_anchor, *, allow_private=False):
    if allow_private is not True:
        raise PermissionError('offline label access requires explicit opt-in')
    path = Path(path)
    manifest = json.loads((path/'MANIFEST.json').read_text())
    if sha256(path/'private_labels.npz') != manifest['output']['private_labels.npz']:
        raise ValueError('upstream private digest mismatch')
    with np.load(path/'private_labels.npz',allow_pickle=False) as a:
        geometry = a['control_geometry'][rows].copy()
    anchor = np.asarray(T_world_anchor)
    out=[]
    for g in geometry:
        R=rotation_from_6d(g[3:9]); opening_center=g[:3]
        center=anchor[:3,:3].T@(opening_center-anchor[:3,3])
        rotvec=Rotation.from_matrix(anchor[:3,:3].T@R).as_rotvec()
        out.append(np.r_[center,rotvec,np.log(g[9:11])])
    return dict(geometry8=torch.tensor(np.asarray(out),dtype=torch.float32),
                geometry12_world=torch.from_numpy(geometry).float())

def load_episode(path, *, allow_private=False):
    path=Path(path)
    public=torch.load(path/'public.pt',map_location='cpu',weights_only=True)
    if allow_private:
        return public,torch.load(path/'private.pt',map_location='cpu',weights_only=True)
    return public
