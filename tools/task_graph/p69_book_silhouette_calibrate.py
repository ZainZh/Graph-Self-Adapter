"""Demo-only reference construction and separate fixed train-frame diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from prove.task_graph.p69_book_silhouette import calibrate,fit_book_pose
from prove.task_graph.p69_public_grounding import mask_plane_points,plane_pose_from_points


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def frame(p):
    with np.load(p,allow_pickle=False) as x:return {k:x[k].copy() for k in x.files}


def build(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    seal=read(root/'SEAL.json')
    def checked(relative):
        p=root/relative
        if sha(p)!=seal['files'][relative]:raise ValueError('Changed captured source '+relative)
        return p
    public_path=checked('demo_initial/public/frame_0000.npz');private_path=checked('demo_initial/private/geometry_0000.json')
    demo=frame(public_path);g=read(private_path)['book'];source=np.eye(4);source[:3,:3]=g['world_rotation'];source[:3,3]=g['world_position']
    center=[-.00047,-.01215,.06937];half=[.01434,.05519,.0672]
    ref=calibrate(demo['visible_masks'][0,1],demo['camera_K'][0],demo['camera_pose'][0],source,center,half)
    ref['source_bindings']={str(public_path):sha(public_path),str(private_path):sha(private_path)}
    # Initialization is visual planar delta only; no current private geometry enters the fit.
    z=g['collision_aabb_max'][2]
    def planar(f):return plane_pose_from_points(mask_plane_points(f['visible_masks'][0,1],f['camera_K'][0],f['camera_pose'][0],z),nominal_yaw=0.)
    dp=planar(demo);records=[]
    for name,t in [('demo_initial',0),('train_20',0),('train_20',12),('train_2',0),('train_2',12)]:
        p=checked(f'{name}/public/frame_{t:04d}.npz');f=frame(p);coarse=planar(f)
        start=np.r_[np.asarray(coarse.xy)-dp.xy,((coarse.yaw-dp.yaw+np.pi/2)%np.pi)-np.pi/2]
        prediction=fit_book_pose(f['visible_masks'][0,1],f['camera_K'][0],f['camera_pose'][0],ref,start)
        # Only after prediction: private current pose used for diagnosis, never calibration/fit.
        truth_path=checked(f'{name}/private/geometry_{t:04d}.json');truth=read(truth_path)['book'];position=np.asarray(truth['world_position']);rotation=np.asarray(truth['world_rotation']);yaw=np.arctan2(rotation[1,0],rotation[0,0])
        error=((prediction['body_yaw']-yaw+np.pi)%(2*np.pi))-np.pi
        records.append(dict(episode=name,t=t,public_sha256=sha(p),private_diagnostic_sha256=sha(truth_path),prediction=prediction,
            diagnostic=dict(xy_error_mm=float(np.linalg.norm(np.asarray(prediction['body_xy'])-position[:2])*1000),yaw_error_deg=float(abs(np.degrees(error))),truth_z=float(position[2]),reference_z=float(source[2,3])),prediction_before_truth=True))
    (output/'BOOK_REFERENCE.json').write_text(json.dumps(ref,indent=2)+'\n')
    (output/'TRAIN_DIAGNOSTIC.json').write_text(json.dumps(dict(schema='prove.p69-book-silhouette-diagnostic.v1',records=records,physics_steps=0,parameter_search=False,reference_sha256=sha(output/'BOOK_REFERENCE.json')),indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('episode','t','diagnostic')} for r in records],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.capture,a.output)
