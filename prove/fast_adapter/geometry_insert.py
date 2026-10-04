"""P22 geometry-varying insertion. Simulator truth is not a policy input."""
from __future__ import annotations
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from prove.fast_adapter.insert3d import Insert3D, scene_xml, domain, STEP, HALF, INNER

STRATA = ('id', 'position', 'scale')

def geometry(seed, stratum='id'):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-.020, .020, 2)
    scale = rng.uniform(.90, 1.10)
    if stratum == 'position':
        axis = int(rng.integers(2))
        xy[axis] = rng.choice([-1, 1]) * rng.uniform(.030, .045)
    elif stratum == 'scale':
        scale = rng.uniform(.75, .85) if rng.integers(2) == 0 else rng.uniform(1.15, 1.25)
    elif stratum != 'id':
        raise ValueError(stratum)
    return {'xy': xy.tolist(), 'scale': float(scale)}

def initial_position(seed):
    # Independent of slot position/size, so the reset does not encode its answer.
    rng = np.random.default_rng(seed)
    return np.r_[rng.uniform(-.05, .05, 2), rng.uniform(.17, .19)]

class GeometryInsert(Insert3D):
    def __init__(self, config, render=True):
        self.config = dict(config)
        self.xy = np.asarray(config['xy'], dtype=float)
        self.scale = float(config['scale'])
        root = ET.fromstring(scene_xml())
        root.find("./worldbody/camera[@name='global']").set('fovy', '42')
        socket = root.find("./worldbody/body[@name='socket']")
        socket.set('pos', ' '.join(map(str, [*self.xy, 0.])))
        for body_name in ('socket', 'peg'):
            for g in root.findall(f"./worldbody/body[@name='{body_name}']/geom"):
                for key in ('pos', 'size'):
                    if key in g.attrib:
                        g.set(key, ' '.join(map(str, np.fromstring(g.get(key), sep=' ') * self.scale)))
        # Mass, actuator gains, camera and friction distribution do not scale.
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=224, width=224) if render else None
        self.M = np.eye(3); self.bias = np.zeros(3)
        self.reset(initial_position(0))

    def rgb(self):
        # No segmentation, boxes, depth, goal coordinates or object IDs returned.
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera='global')
        return self.renderer.render().copy()

    def facts(self):
        pos = self.state(); s = self.scale
        inside = bool(np.all(np.abs(pos[:2] - self.xy) + HALF[:2]*s <= (INNER[:2]+.0002)*s))
        bottom = pos[2] - HALF[2]*s
        penetration = max([-float(self.data.contact[i].dist) for i in range(self.data.ncon)]+[0.])
        inserted = inside and .0065*s <= bottom <= .013*s and penetration <= .0015*s
        return {'inserted': bool(inserted), 'inside_xy': inside, 'bottom_z': float(bottom),
                'max_penetration': penetration, 'contact_count': int(self.data.ncon)}

    def expert_nominal(self):
        """Training label/reference only. Never called by a learned controller."""
        pos = self.state(); s = self.scale
        aligned = np.all(np.abs(pos[:2]-self.xy) <= .65*(INNER[:2]-HALF[:2])*s)
        goal = np.r_[self.xy, (.0385 if aligned else .110)*s]
        return np.clip(.65*(goal-pos)/STEP, -1, 1)

    def expert_action(self):
        return np.clip(np.linalg.solve(self.M, self.expert_nominal()-self.bias/STEP), -1, 1)

