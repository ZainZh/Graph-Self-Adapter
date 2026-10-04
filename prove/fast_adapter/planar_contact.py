"""P39 engineering environment: planar near-contact pushing, mask-assisted RGB.

Object dynamics/coordinates are NOT observations. Segmentation masks, RGB,
camera calibration, and tool proprioception are explicitly allowed. A circular
object translates in two dimensions (no free rotation); forces are accurate.
"""
import numpy as np

DT = .04
STEPS = 40


def configuration(seed, split='train'):
    r = np.random.default_rng(seed)
    damping = r.uniform(2.5, 6., 2)
    friction = r.uniform(.025, .1, 2)
    if split == 'resistance':
        damping = r.uniform(6., 10., 2)
        friction = r.uniform(.1, .18, 2)
    origin = r.uniform(-.008, .008, 2)
    theta = r.uniform(-.65, .65)
    distance = r.uniform(.025, .037)
    size = r.uniform(.008, .012)
    if split == 'geometry':
        size = r.uniform(.013, .016)
        theta = r.choice([-1, 1])*r.uniform(.7, 1.1)
    return dict(seed=int(seed), split=split, damping=damping.tolist(),
                friction=friction.tolist(), mass=float(r.uniform(.045, .085)),
                origin=origin.tolist(), direction=[float(np.cos(theta)), float(np.sin(theta))],
                distance=float(distance), radius=float(size), tool_radius=.006,
                lateral=float(r.uniform(-.12, .12)), gap=float(r.uniform(.0001, .0004)),
                camera=r.uniform(-.006, .006, 2).tolist(),
                colors=r.uniform(.2, .95, (3, 3)).tolist(),
                background=r.uniform(.15, .45, 3).tolist())


class PlanarPush:
    def __init__(self, cfg, render=True, resolution=448):
        import mujoco
        self.mj=mujoco; self.cfg=dict(cfg); self.resolution=resolution
        origin=np.array(cfg['origin']); direction=np.array(cfg['direction'])
        self.goal=origin+direction*cfg['distance']
        tangent=np.array([-direction[1], direction[0]])
        tool=origin-direction*(cfg['radius']+cfg['tool_radius']+cfg['gap'])
        tool+=tangent*cfg['lateral']*(cfg['radius']+cfg['tool_radius'])
        def xyz(v,z): return f'{v[0]} {v[1]} {z}'
        def color(v): return ' '.join(map(str,v))+' 1'
        goal_parts=[]
        for axis in (0,1):
            for sign in (-1,1):
                p=self.goal.copy();p[axis]+=sign*.02
                size='.001 .021 .001' if axis==0 else '.021 .001 .001'
                goal_parts.append(f'<geom name="goal{axis}{sign}" type="box" pos="{xyz(p,.001)}" size="{size}" rgba="{color(cfg["colors"][2])}" contype="0" conaffinity="0"/>')
        self.xml=f'''<mujoco model="planar_near_push">
          <option timestep=".001" gravity="0 0 -9.81" integrator="implicitfast"/>
          <visual><global offwidth="448" offheight="448"/><quality shadowsize="1024" offsamples="0"/></visual>
          <default><geom friction="0 0 0" condim="3" solref=".002 1" solimp=".99 .99 .001"/></default>
          <worldbody><light pos="0 -.1 .4" dir="0 0 -1" diffuse=".8 .8 .8"/>
            <geom name="floor" type="plane" size=".2 .2 .01" rgba="{color(cfg['background'])}" contype="0" conaffinity="0"/>
            <camera name="top" pos="{xyz(cfg['camera'],.30)}" xyaxes="1 0 0 0 1 0" fovy="35"/>
            {''.join(goal_parts)}
            <body name="tool" pos="{xyz(tool,.006)}">
              <joint type="slide" axis="1 0 0" damping="2." frictionloss=".005"/>
              <joint type="slide" axis="0 1 0" damping="2." frictionloss=".005"/>
              <geom name="actor" type="cylinder" size="{cfg['tool_radius']} .006" mass=".03" rgba="{color(cfg['colors'][0])}"/>
            </body>
            <body name="block" pos="{xyz(origin,.006)}">
              <joint type="slide" axis="1 0 0" damping="{cfg['damping'][0]}" frictionloss="{cfg['friction'][0]}"/>
              <joint type="slide" axis="0 1 0" damping="{cfg['damping'][1]}" frictionloss="{cfg['friction'][1]}"/>
              <geom name="object" type="cylinder" size="{cfg['radius']} .006" mass="{cfg['mass']}" rgba="{color(cfg['colors'][1])}"/>
            </body>
          </worldbody>
        </mujoco>'''
        self.model=mujoco.MjModel.from_xml_string(self.xml); self.data=mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        self.tool_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'tool')
        self.obj_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'block')
        self.actor_geom=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'actor')
        self.object_geom=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'object')
        self.goal_geoms=[i for i in range(self.model.ngeom) if mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i).startswith('goal')]
        self.renderer=mujoco.Renderer(self.model,height=resolution,width=resolution) if render else None

    def render(self):
        self.renderer.update_scene(self.data,camera='top')
        self.renderer.disable_segmentation_rendering();rgb=self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering();seg=self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        # MuJoCo segmentation is (object id, object type). No depth/pose lookup.
        geom=seg[...,1]==int(self.mj.mjtObj.mjOBJ_GEOM)
        masks=np.stack([geom&(seg[...,0]==self.actor_geom),geom&(seg[...,0]==self.object_geom),
                        geom&np.isin(seg[...,0],self.goal_geoms)])
        return rgb,masks

    def public(self):
        rgb,masks=self.render();centers=[];radii=[];boxes=[]
        for i,mask in enumerate(masks):
            ys,xs=np.where(mask)
            if len(xs)<10: raise RuntimeError('Role not visible; no state fallback')
            # Bounding extent midpoint reduces partial occlusion bias for circles/ring.
            cx=(xs.min()+xs.max()+1)/2;cy=(ys.min()+ys.max()+1)/2
            z=.012 if i<2 else .002
            scale=2*(.30-z)*np.tan(np.deg2rad(17.5))/self.resolution
            centers.append([self.cfg['camera'][0]+(cx-self.resolution/2)*scale,
                            self.cfg['camera'][1]-(cy-self.resolution/2)*scale])
            radii.append(.25*(xs.max()-xs.min()+ys.max()-ys.min()+2)*scale)
            boxes.append([xs.min(),ys.min(),xs.max()+1,ys.max()+1])
        obs=dict(positions=np.array(centers,dtype=np.float32),radii=np.array(radii,dtype=np.float32),
                 boxes=np.array(boxes),proprio=np.r_[self.data.xpos[self.tool_id,:2],self.data.qvel[:2]].astype(np.float32))
        return obs,rgb,masks

    def step(self,action):
        self.data.xfrc_applied[self.tool_id,:2]=np.clip(action,-1.5,1.5)
        for _ in range(40):self.mj.mj_step(self.model,self.data)

    def state(self):
        return dict(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),time=float(self.data.time),
                    object=self.data.xpos[self.obj_id,:2].copy(),tool=self.data.xpos[self.tool_id,:2].copy())

    def close(self):
        if self.renderer is not None:self.renderer.close()


def controller(obs,response=None):
    """Shared analytic force control; response is effective gain/bias per axis."""
    pos=obs['positions'];d=pos[2]-pos[1];r=obs['radii']
    if response is None:response=np.array([[3.225806,-.241935],[3.225806,-.241935]])
    error=np.linalg.norm(d);v=d*min(4.,.09/max(error,1e-6))
    wanted=v*20
    desired=np.sign(wanted)*(np.abs(wanted)-response[:,1])/response[:,0]
    if error<.0018:return np.zeros(2)
    direction=desired/max(np.linalg.norm(desired),1e-6)
    normal=(pos[1]-pos[0]);normal/=max(np.linalg.norm(normal),1e-6)
    target=pos[1]-direction*(r[0]+r[1])
    side=target-pos[0];side-=normal*np.dot(side,normal)
    velocity=obs['proprio'][2:].copy();velocity-=normal*np.dot(velocity,normal)
    return np.clip(normal*np.linalg.norm(desired)+25*side-.4*velocity,-1.5,1.5)


def projected_force(obs,action):
    n=obs['positions'][1]-obs['positions'][0];distance=np.linalg.norm(n);n=n/max(distance,1e-8)
    gap=distance-obs['radii'][:2].sum()
    gate=np.clip(1-gap/.002,0,1)
    return n*max(0,float(np.dot(action,n)))*gate
