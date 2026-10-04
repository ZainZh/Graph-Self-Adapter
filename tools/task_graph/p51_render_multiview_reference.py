#!/usr/bin/env python3
"""Two calibrated native reference RGB channels, one opened state, zero actions."""
from pathlib import Path
import hashlib,json,os,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,cv2,mujoco
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prove.fast_adapter.geometry3d_adapter import project
base=ROOT/'evaluations/fast_adapter_p48_task_graph_v01/gt_replay_v01'
original=ROOT/'evaluations/fast_adapter_p51_libero_complete_v01/grounding_reference_v01'
out=ROOT/'evaluations/fast_adapter_p51_libero_complete_v01/grounding_multiview_reference_v01'
out.mkdir(parents=True,exist_ok=False)
binding=json.loads((base/'BINDING.json').read_text());frame=np.load(base/'frame_0000.npz')
ref=json.loads((original/'REFERENCE.json').read_text());T=np.asarray(ref['T_world_reference']);patch=np.asarray(ref['patch_local_points'])
model=mujoco.MjModel.from_xml_string(binding['relocated_xml']);data=mujoco.MjData(model)
for field in ('qpos','qvel','act'):getattr(data,field)[:]=frame['supervisor_'+field]
mujoco.mj_forward(model,data)
option=mujoco.MjvOption();option.geomgroup[0]=0;option.geomgroup[1]=1
renderer=mujoco.Renderer(model,height=448,width=448)
try:
 for camera in ('agentview','frontview'):
  folder=out/camera;folder.mkdir()
  renderer.update_scene(data,camera=camera,scene_option=option);rgb=renderer.render().copy()
  renderer.enable_segmentation_rendering();seg=renderer.render().copy();renderer.disable_segmentation_rendering()
  rendered=seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)
  roleids={'actor':[],'object':[],'goal':[]}
  for i in range(model.ngeom):
   name=model.geom(i).name or ''
   if name.startswith('gripper0_'):roleids['actor'].append(i)
   if name.startswith('black_book_1_'):roleids['object'].append(i)
   if name.startswith('desk_caddy_1_'):roleids['goal'].append(i)
  masks={role:rendered&np.isin(seg[:,:,0],ids) for role,ids in roleids.items()}
  cid=model.camera(camera).id;focal=224/np.tan(np.deg2rad(model.cam_fovy[cid])/2)
  calibration=dict(K=np.array([[focal,0,224],[0,focal,224],[0,0,1]]),camera_position=data.cam_xpos[cid].copy(),camera_rotation=data.cam_xmat[cid].reshape(3,3).copy())
  pixels=project(patch@T[:3,:3].T+T[:3,3],calibration)
  # A static-design polygon entirely on the same physical brown wall face.
  support=np.array([[-.05,.06,0],[.05,.06,0],[.05,-.06,0],[-.05,-.06,0]])
  polygon=project(support@T[:3,:3].T+T[:3,3],calibration)
  face=np.zeros((448,448),np.uint8);cv2.fillConvexPoly(face,np.rint(polygon).astype(np.int32),1)
  face=face.astype(bool)&masks['goal']
  Image.fromarray(rgb).save(folder/'reference_rgb.png')
  np.savez_compressed(folder/'reference_masks.npz',**masks,caddy=masks['goal'],front_wall=face)
  meta=dict(schema='prove.p51-opened-reference-render.v1',source_frame=str((base/'frame_0000.npz').relative_to(ROOT)),source_frame_sha256=hashlib.sha256((base/'frame_0000.npz').read_bytes()).hexdigest(),
            source_binding_sha256=hashlib.sha256((base/'BINDING.json').read_bytes()).hexdigest(),robot_actions=0,physics_steps=0,mj_forward_calls=1,camera=camera,resolution=448,image_convention='top_left',
            collision_group_rendered=False,mask_permission='GT visible role geom segmentation only',mujoco_version=mujoco.__version__,**{k:v.tolist() for k,v in calibration.items()})
  (folder/'RENDER.json').write_text(json.dumps(meta,indent=2)+'\n')
  record=dict(ref,reference_patch_pixels=pixels.tolist(),front_wall_polygon_pixels=polygon.tolist(),camera=meta,
              channel_scope=['agentview','frontview'],multiview_identifiability_claim='two distinct calibrated optical centers; benefit requires numeric qualification')
  (folder/'REFERENCE.json').write_text(json.dumps(record,indent=2)+'\n')
  print(json.dumps(dict(camera=camera,visible_role_pixels={k:int(v.sum()) for k,v in masks.items()},front_wall_pixels=int(face.sum()),reference_patch_pixels=pixels.tolist())),flush=True)
finally:
 renderer.close()
print('RENDERER_CLOSED; zero actions/physics steps',flush=True)
