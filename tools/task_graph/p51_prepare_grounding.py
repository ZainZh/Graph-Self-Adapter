#!/usr/bin/env python3
"""Offline physical-plane reference binding from opened teacher + static design."""
from pathlib import Path
import json,sys,hashlib
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prove.fast_adapter.geometry3d_adapter import project
out=ROOT/'evaluations/fast_adapter_p51_libero_complete_v01/grounding_reference_v01'
base=ROOT/'evaluations/fast_adapter_p48_task_graph_v01/gt_replay_v01'
render=json.loads((out/'RENDER.json').read_text());names=json.loads((base/'SUPERVISOR_NAMES.json').read_text())
f=np.load(base/'frame_0000.npz');bid=names['body'].index('desk_caddy_1_main');sid=names['site'].index('desk_caddy_1_back_contain_region')
body_p=f['supervisor_body_xpos'][bid];body_R=f['supervisor_body_xmat'][bid].reshape(3,3)
T=np.eye(4);T[:3,:3]=body_R@np.array([[0,0,-1],[-1,0,0],[0,1,0]]);T[:3,3]=body_p+body_R@np.array([-.07088454,-.14,.085])
goal=np.eye(4);goal[:3,3]=f['supervisor_site_xpos'][sid];goal[:3,:3]=f['supervisor_site_xmat'][sid].reshape(3,3)
patch=np.array([[-.019,.016,0],[.019,.016,0],[.019,-.016,0],[-.019,-.016,0]])
calib={k:np.asarray(render[k]) for k in ('K','camera_position','camera_rotation')}
pixels=project(patch@T[:3,:3].T+T[:3,3],calib)
# Manually annotated real brown front-wall pixels, strictly inside its visible
# physical face; this is an offline image annotation, not goal-site projection.
polygon=np.array([[196,198],[257,198],[257,247],[196,247]],np.int32)
mask=np.zeros((448,448),np.uint8);cv2.fillConvexPoly(mask,polygon,1)
caddy=np.load(out/'reference_masks.npz')['caddy'].astype(bool);face=mask.astype(bool)&caddy
np.savez_compressed(out/'reference_masks.npz',caddy=caddy,front_wall=face)
mesh=ROOT/'data/sources/LIBERO-PRO/libero/libero/assets/turbosquid_objects/desk_caddy/visual/desk_caddy_vis.msh'
reference=dict(schema='prove.p51-physical-plane-reference.v1',T_world_reference=T.tolist(),T_reference_target=(np.linalg.inv(T)@goal).tolist(),
 reference_patch_pixels=pixels.tolist(),patch_local_points=patch.tolist(),front_wall_polygon_pixels=polygon.tolist(),
 axes='x=front-wall image right, y=physical up, z=outward toward agentview; virtual reference frame gives nominal R=identity',
 point_order='[-a,+b],[+a,+b],[+a,-b],[-a,-b]; a=.019m,b=.016m',
 point_semantics='nominal physical plane reference points, NOT four natural corners',
 plane_prior=dict(object_local_x=-.07088454,region='right raised section of actual brown front outer wall',
                  mesh_area_weighted_x_std_m=.00023799499,qualification='nominal planar approximation to mildly uneven physical mesh'),
 mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),source_teacher_frame=render['source_frame'],
 source_teacher_frame_sha256=render['source_frame_sha256'],
 target_relation_permission='back-site is read ONLY from opened teacher frame0 to freeze reference-to-target transform; never projected as runtime observation',
 runtime_inputs='RGB plus GT visible role masks, calibrated camera, own robot pose only',
 camera=render,multiview_identifiability_claim=False)
(out/'REFERENCE.json').write_text(json.dumps(reference,indent=2)+'\n')
(out/'nominal_reference.json').write_text(json.dumps(dict(T_world_reference=T.tolist(),T_reference_target=(np.linalg.inv(T)@goal).tolist(),T_world_target=goal.tolist()),indent=2)+'\n')
print(json.dumps(dict(output=str(out),reference_patch_pixels=pixels.tolist()),indent=2))
