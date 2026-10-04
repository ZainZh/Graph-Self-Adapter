#!/usr/bin/env python3
"""One opened P48 teacher-state render, zero actions/physics steps; offline only."""
from pathlib import Path
import hashlib,json,os,sys
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'evaluations/fast_adapter_p48_task_graph_v01/gt_replay_v01'
out=ROOT/'evaluations/fast_adapter_p51_libero_complete_v01/grounding_reference_v01'
out.mkdir(parents=True,exist_ok=False)
binding=json.loads((base/'BINDING.json').read_text())
frame=np.load(base/'frame_0000.npz')
model=mujoco.MjModel.from_xml_string(binding['relocated_xml'])
data=mujoco.MjData(model)
for field in ('qpos','qvel','act'):
 getattr(data,field)[:]=frame['supervisor_'+field]
mujoco.mj_forward(model,data)
option=mujoco.MjvOption();option.geomgroup[0]=0;option.geomgroup[1]=1
renderer=mujoco.Renderer(model,height=448,width=448)
try:
 renderer.update_scene(data,camera='agentview',scene_option=option)
 rgb=renderer.render().copy()
 renderer.enable_segmentation_rendering();seg=renderer.render().copy();renderer.disable_segmentation_rendering()
finally:
 renderer.close()
# Native mujoco.Renderer is top-left, unlike saved robosuite OpenGL arrays.
Image.fromarray(rgb).save(out/'reference_rgb.png')
ids={name:model.geom(name).id for name in ('desk_caddy_1_g0','black_book_1_g0') if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)>=0}
caddy=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(seg[:,:,0]==ids['desk_caddy_1_g0'])
np.savez_compressed(out/'reference_masks.npz',caddy=caddy)
cam=model.camera('agentview').id
K=np.array([[224/np.tan(np.deg2rad(model.cam_fovy[cam])/2),0,224],[0,224/np.tan(np.deg2rad(model.cam_fovy[cam])/2),224],[0,0,1]])
metadata=dict(schema='prove.p51-opened-reference-render.v1',source_frame=str((base/'frame_0000.npz').relative_to(ROOT)),
 source_frame_sha256=hashlib.sha256((base/'frame_0000.npz').read_bytes()).hexdigest(),source_binding_sha256=hashlib.sha256((base/'BINDING.json').read_bytes()).hexdigest(),
 robot_actions=0,physics_steps=0,mj_forward_calls=1,camera='agentview',resolution=448,image_convention='top_left',
 camera_position=data.cam_xpos[cam].tolist(),camera_rotation=data.cam_xmat[cam].reshape(3,3).tolist(),K=K.tolist(),
 collision_group_rendered=False,mask_permission='GT visible instance geom segmentation only',mujoco_version=mujoco.__version__)
(out/'RENDER.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(dict(output=str(out),mask_pixels=int(caddy.sum()),**metadata),indent=2))
