"""P38 short physical interaction: a force-driven tool pushes a damped slider.

Two MuJoCo rigid bodies make real unilateral contact. Object drag and dry
friction are persistent hidden conditions, not actuator calibration errors.
public() exposes RGB measurements and the tool's encoder only. Ground-truth
object coordinates and resistance are confined to engineering/oracle/evaluator.
"""
import numpy as np

DT = .04
NOMINAL_DAMPING = 4.2
NOMINAL_FRICTION = .07


def configuration(seed, split='train'):
    rng = np.random.default_rng(seed)
    if split == 'native':
        damping, friction = 4., .06
    elif split == 'ood':
        damping, friction = rng.uniform(6.5, 9.), rng.uniform(.13, .22)
    else:
        damping, friction = rng.uniform(2.5, 6.), rng.uniform(.02, .12)
    return {'seed': int(seed), 'split': split, 'damping': float(damping),
            'friction': float(friction), 'mass': float(rng.uniform(.04, .08)),
            'distance': float(rng.uniform(.026, .034)),
            'origin': float(rng.uniform(-.008, .008)),
            'halfwidth': float(rng.uniform(.006, .008))}


class NearPush:
    def __init__(self, cfg, render=True, resolution=448):
        import mujoco
        self.mj = mujoco; self.cfg = dict(cfg); self.resolution = resolution
        x, hw = cfg['origin'], cfg['halfwidth']
        self.goal = x+cfg['distance']
        self.xml = f'''<mujoco model="short_contact_push">
          <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
          <visual><global offwidth="448" offheight="448"/><quality shadowsize="1024"/></visual>
          <default><geom contype="1" conaffinity="1" condim="3" friction="0 0 0"
            solref=".002 1" solimp=".99 .99 .001"/></default>
          <worldbody>
            <light pos="0 -.1 .4" dir="0 0 -1" diffuse=".8 .8 .8"/>
            <geom type="plane" size=".2 .1 .01" rgba=".22 .24 .26 1" contype="0" conaffinity="0"/>
            <camera name="top" pos=".015 0 .25" xyaxes="1 0 0 0 1 0" fovy="30"/>
            <geom name="goal" type="box" pos="{self.goal} .023 .001"
              size=".001 .006 .001" rgba="1 .12 .12 1" contype="0" conaffinity="0"/>
            <body name="tool" pos="{x-hw-.006-.0002} 0 .006">
              <joint name="tool_x" type="slide" axis="1 0 0" damping=".2" frictionloss=".01"/>
              <geom name="tool_geom" type="box" size=".006 .005 .006" mass=".03" rgba=".1 .9 .18 1"/>
            </body>
            <body name="object" pos="{x} 0 .006">
              <joint name="object_x" type="slide" axis="1 0 0" damping="{cfg['damping']}" frictionloss="{cfg['friction']}"/>
              <geom name="object_geom" type="box" size="{hw} .008 .006" mass="{cfg['mass']}" rgba=".12 .3 1 1"/>
            </body>
          </worldbody>
          <actuator><motor name="force" joint="tool_x" gear="1" ctrllimited="true" ctrlrange="0 1"/></actuator>
        </mujoco>'''
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=resolution, width=resolution) if render else None
        self.tool_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'tool')
        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'object')
        self.history = []

    def rgb(self):
        self.renderer.update_scene(self.data, camera='top')
        return self.renderer.render().copy()

    def encoder(self):
        return np.array([self.data.xpos[self.tool_id, 0], self.data.qvel[0]], dtype=np.float32)

    def public(self):
        rgb = self.rgb()
        return rgb_measurements(rgb), self.encoder(), rgb

    def step(self, force):
        self.data.ctrl[0] = np.clip(float(force), 0., 1.)
        for _ in range(40):
            self.mj.mj_step(self.model, self.data)
        self.history.append(self.state())

    def state(self):
        return {'qpos': self.data.qpos.copy(), 'qvel': self.data.qvel.copy(),
                'time': float(self.data.time), 'object_x': float(self.data.xpos[self.object_id, 0]),
                'tool_x': float(self.data.xpos[self.tool_id, 0])}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def rgb_measurements(rgb):
    """Fixed color-key scene detector; no simulator segmentation/boxes/depth.

    Known camera calibration converts image displacement to table distance.
    This is not an arbitrary-object detector and its restrictions are reported.
    """
    v = rgb.astype(np.float32)/255.
    red, green, blue = v[..., 0], v[..., 1], v[..., 2]
    masks = [(green > red*1.5) & (green > blue*1.5) & (green > .25),
             (blue > red*1.5) & (blue > green*1.5) & (blue > .25),
             (red > green*1.5) & (red > blue*1.5) & (red > .25)]
    centers, boxes = [], []
    for mask in masks:
        ys, xs = np.where(mask)
        if len(xs) < 4:
            raise ValueError('RGB entity not visible; retain failure, do not use truth fallback')
        centers.append(float(xs.mean()+.5))
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)])
    # Top surfaces at z=.012 for actor/object and z=.002 for reference marker.
    z = np.array([.012, .012, .002])
    per_pixel = 2*(.25-z)*np.tan(np.deg2rad(15))/rgb.shape[1]
    world_x = .015+(np.array(centers)-rgb.shape[1]/2)*per_pixel
    return {'positions': world_x.astype(np.float32), 'boxes': np.array(boxes, dtype=np.int32),
            'pixel_centers': np.array(centers, dtype=np.float32)}


def nominal_force(error, damping=NOMINAL_DAMPING, friction=NOMINAL_FRICTION):
    desired_velocity = np.clip(4*error, 0., .08)
    return 0. if error < .0012 else float(np.clip(damping*desired_velocity+friction, 0, 1))
