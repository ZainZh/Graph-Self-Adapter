"""P20 custom two-object MuJoCo insertion, no robot or kinematic teleporting.

Uses the physical engine version from the existing LIBERO environment. This is
not an official LIBERO benchmark task. Reset may set initial state; each control
step drives force-limited position actuators and runs contact dynamics.
"""
from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco

STEP=.008
HALF=np.array([.014,.0125,.030])
INNER=np.array([.0175,.016])
SLOT_TOP=.068
GOAL=np.array([0.,0.,.0385])
CAMERA=np.array([.24,-.29,.285])
LOOKAT=np.array([0.,0.,.080])

def scene_xml():
 z=CAMERA-LOOKAT;z=z/np.linalg.norm(z)
 x=np.cross([0.,0.,1.],z);x=x/np.linalg.norm(x);y=np.cross(z,x)
 axes=' '.join(map(str,np.r_[x,y]));pos=' '.join(map(str,CAMERA))
 return f'''<mujoco model="P20 two-object insertion">
 <compiler angle="radian"/>
 <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81" iterations="80" cone="elliptic"/>
 <visual><global offwidth="224" offheight="224"/><quality shadowsize="2048"/><headlight ambient=".3 .3 .3" diffuse=".6 .6 .6"/></visual>
 <default><geom friction=".7 .02 .001" solref=".006 1" solimp=".95 .99 .001" condim="4"/><joint damping=".15"/></default>
 <asset>
  <texture name="desktex" type="2d" builtin="checker" rgb1=".19 .23 .25" rgb2=".21 .25 .27" width="128" height="128"/>
  <material name="deskmat" texture="desktex" texrepeat="6 6" reflectance=".02"/>
  <material name="wood" rgba=".64 .40 .20 1" specular=".12" shininess=".1"/>
  <material name="pegwood" rgba=".88 .32 .09 1" specular=".18" shininess=".1"/>
 </asset>
 <worldbody>
  <light pos=".1 -.3 .8" dir="-.1 .2 -1" diffuse=".85 .85 .85" castshadow="true"/>
  <geom name="table" type="plane" size=".35 .35 .01" material="deskmat"/>
  <camera name="global" pos="{pos}" xyaxes="{axes}" fovy="36"/>
  <body name="socket">
   <geom name="socket_floor" type="box" pos="0 0 .004" size=".065 .06 .004" material="wood"/>
   <geom name="socket_left" type="box" pos="-.04125 0 .038" size=".02375 .06 .03" material="wood"/>
   <geom name="socket_right" type="box" pos=".04125 0 .038" size=".02375 .06 .03" material="wood"/>
   <geom name="socket_front" type="box" pos="0 -.038 .038" size=".0175 .022 .03" material="wood"/>
   <geom name="socket_back" type="box" pos="0 .038 .038" size=".0175 .022 .03" material="wood"/>
  </body>
  <body name="peg" gravcomp="1">
   <joint name="x" type="slide" axis="1 0 0"/>
   <joint name="y" type="slide" axis="0 1 0"/>
   <joint name="z" type="slide" axis="0 0 1"/>
   <geom name="peg_main" type="box" size=".014 .0125 .030" mass=".055" material="pegwood"/>
   <geom name="peg_marker" type="box" pos="0 0 .03005" size=".010 .0015 .00005" mass="0" rgba=".95 .82 .50 1" contype="0" conaffinity="0"/>
  </body>
 </worldbody>
 <actuator>
  <position name="px" joint="x" kp="450" kv="11" forcelimited="true" forcerange="-3 3"/>
  <position name="py" joint="y" kp="450" kv="11" forcelimited="true" forcerange="-3 3"/>
  <position name="pz" joint="z" kp="450" kv="11" forcelimited="true" forcerange="-5 5"/>
 </actuator>
 </mujoco>'''

class Insert3D:
 def __init__(self,render=True):
  self.model=mujoco.MjModel.from_xml_string(scene_xml());self.data=mujoco.MjData(self.model)
  self.renderer=mujoco.Renderer(self.model,height=224,width=224) if render else None
  self.peg_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ['peg_main','peg_marker']]
  self.socket_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ['socket_floor','socket_left','socket_right','socket_front','socket_back']]
  self.M=np.eye(3);self.bias=np.zeros(3);self.reset(np.array([.03,-.025,.145]))
 def set_domain(self,M,bias,friction=.7):
  self.M=np.array(M,dtype=float);self.bias=np.array(bias,dtype=float)
  self.model.geom_friction[:,0]=friction
 def reset(self,position):
  mujoco.mj_resetData(self.model,self.data);self.data.qpos[:]=position;self.data.ctrl[:]=position;mujoco.mj_forward(self.model,self.data)
  self.streak=0
 def state(self):return self.data.qpos.copy()
 def step(self,action):
  command=np.clip(np.asarray(action),-1,1)
  desired=self.state()+self.M@(STEP*command)+self.bias
  self.data.ctrl[:]=desired
  for _ in range(48):mujoco.mj_step(self.model,self.data)
  facts=self.facts();self.streak=self.streak+1 if facts['inserted'] else 0
  return facts
 def facts(self):
  pos=self.state();inside=bool(np.all(np.abs(pos[:2])+HALF[:2]<=INNER+.0002));bottom=pos[2]-HALF[2]
  pen=min([float(self.data.contact[i].dist) for i in range(self.data.ncon)]+[0.])
  inserted=inside and .0065<=bottom<=.013 and pen>=-.0015
  return {'position':pos.tolist(),'inside_xy':inside,'bottom_z':float(bottom),'inserted':bool(inserted),'contact_count':int(self.data.ncon),'max_penetration':-pen}
 def observe(self):
  self.renderer.disable_segmentation_rendering();self.renderer.update_scene(self.data,camera='global');rgb=self.renderer.render().copy()
  self.renderer.enable_segmentation_rendering();seg=self.renderer.render().copy();self.renderer.disable_segmentation_rendering()
  ids=seg[...,0];types=seg[...,1];boxes=[];counts=[]
  for selected in [self.peg_ids,self.socket_ids]:
   mask=np.isin(ids,selected)&(types==int(mujoco.mjtObj.mjOBJ_GEOM));yy,xx=np.nonzero(mask);counts.append(len(xx))
   if len(xx):boxes.append([int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)])
   else:boxes.append([0,0,1,1])
  return rgb,np.array(boxes,np.float32),counts
 def close(self):
  if self.renderer is not None:self.renderer.close()

def domain(seed,level='train'):
 rng=np.random.default_rng(seed);angle=rng.uniform(-.20,.20) if level!='ood' else rng.choice([-1,1])*rng.uniform(.20,.35)
 c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
 M=rot@np.diag(rng.uniform(.75,1.25,3));bias=rng.uniform(-.24,.24,3)*STEP
 return M,bias,float(rng.uniform(.45,.95))

def start_position(seed):
 rng=np.random.default_rng(seed);return np.r_[rng.uniform(-.045,.045,2),rng.uniform(.132,.16)]

def nominal_action(position,step):
 target=GOAL.copy()
 if step<14:target[2]=.112
 return np.clip(.5*(target-np.asarray(position))/STEP,-1,1)

def oracle_action(position,step,M,bias):
 base=nominal_action(position,step)
 return np.clip(np.linalg.solve(M,base-bias/STEP),-1,1)

PROBES=np.array([[.65,0,0],[0,.65,0],[-.65,0,0],[0,-.65,0],[0,0,.65]],dtype=np.float32)
