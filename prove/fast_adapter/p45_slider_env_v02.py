"""P45 package 2 articulated-handle task: contact grasp, then open a passive slider.

The physical slide joint belongs to the object. It constrains a drawer-like
carriage to its rail; it neither attaches the object to the robot nor supplies
control feedback. Robot actuation, cameras and fingers use the P45 Panda
backend. The task shares the cup's visible rectangular-handle representation,
but requires translation along a constrained axis instead of free-body lift.

Only ``oracle`` and ``facts`` expose object state. Ordinary observations contain
rendered RGB/visible masks, camera calibration and robot proprioception only.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import numpy as np

from prove.fast_adapter.geometry3d_env import RES, configuration as cup_configuration, vec
from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
from prove.fast_adapter.p45_handle_surfaces import FRONT_FACE_PREFIX, SURFACE_PREFIX


SLIDE_RANGE_M = .095
OPEN_SUCCESS_M = .060


def configuration(task, seed, split='id'):
    """Evaluator-owned geometry, with exactly the cup functional-frame schema.

    Physics and opening objective are invariant across seeds and splits. The
    caller must not pass this record into a non-Oracle runtime controller.
    """
    if task != 'slider':
        raise ValueError('p45_slider_env accepts only the native slider task')
    if split not in ('id', 'geometry'):
        raise ValueError(f'Unknown geometry split: {split}')
    cfg = cup_configuration('cup', seed, split)
    cfg.update(task='slider', geometry_family='cup', slide_range=SLIDE_RANGE_M,
               slider_mass=.12, slider_damping=.3, slider_frictionloss=.15)
    return cfg


def slider_xml(panda_xml, cfg):
    """Replace only the scene in a constructed Panda cup model.

    This pure XML operation deliberately leaves robot meshes, inertias,
    actuators, finger exclusions, controller gains and cameras untouched.
    """
    root = ET.fromstring(panda_xml)
    root.set('model', 'p45_panda_slider')
    world = root.find('worldbody')
    old_object = world.find("body[@name='object']")
    if old_object is None:
        raise ValueError('Panda scene is missing its named object body')
    for geom in list(world.findall('geom')):
        if geom.get('name') == 'cup_support':
            world.remove(geom)
    carriage = ET.Element('body', name='object', pos=old_object.get('pos', '0 0 0'),
                          quat=old_object.get('quat', '1 0 0 0'))
    ET.SubElement(carriage, 'joint', name='slider_joint', type='slide',
                  axis='1 0 0', range=f'0 {SLIDE_RANGE_M}', limited='true',
                  damping='.3', frictionloss='.15', solreflimit='.004 1',
                  solimplimit='.99 .99 .001')
    ET.SubElement(carriage, 'inertial', pos='0 0 .035', mass='.12',
                  diaginertia='.00018 .00018 .00012')
    radius = cfg['radius']
    # The handle is precisely the same visible and colliding physical frame.
    for geom in old_object.findall('geom'):
        if (geom.get('name') or '').startswith('handle_'):
            carriage.append(copy.deepcopy(geom))
    if len(carriage.findall('geom')) != 4:
        raise ValueError('Expected four physical handle members')
    # The parent keeps these visual surfaces inside the four original collision
    # boxes. They are appearance-preserving semantic parts, not grasp aids.
    for geom in old_object.findall('geom'):
        if (geom.get('name') or '').startswith((FRONT_FACE_PREFIX, SURFACE_PREFIX)):
            carriage.append(copy.deepcopy(geom))
    ET.SubElement(carriage, 'geom', name='slider_front', type='box',
                  pos=vec([radius - .006, 0, .057]), size='.006 .030 .057',
                  rgba='.36 .55 .70 1')
    ET.SubElement(carriage, 'geom', name='slider_carriage', type='box',
                  pos=vec([radius - .035, 0, -.018]), size='.035 .028 .012',
                  rgba='.37 .43 .49 1')
    world.remove(old_object)
    world.insert(0, carriage)
    # Rails are visible collision geometry around the mechanical joint, with
    # clearance from the moving carriage. Their mass/drag are not randomized.
    fixture = ET.SubElement(world, 'body', name='slider_fixture',
                            pos=vec(cfg['center']), quat=old_object.get('quat', '1 0 0 0'))
    ET.SubElement(fixture, 'geom', name='rail_bed', type='box',
                  pos='-.005 0 -.078', size='.125 .065 .022',
                  rgba='.27 .31 .35 1')
    for sign in (-1, 1):
        ET.SubElement(fixture, 'geom', name=f'rail_{sign}', type='box',
                      pos=vec([.015, sign * .039, -.043]),
                      size='.125 .004 .013', rgba='.64 .67 .69 1')
    return ET.tostring(root, encoding='unicode')


class SliderGeometrySkill(RobotGeometrySkill):
    """Native slider environment with a cup-family public observation schema."""

    geometry_family = 'cup'

    def __init__(self, cfg, render=True):
        if cfg['task'] != 'slider':
            raise ValueError('Expected a slider configuration')
        # Build the shared robot in its cup grasp orientation. Then replace
        # only the scene and migrate robot state by joint name, because removing
        # the object's free joint changes all subsequent qpos/dof addresses.
        cup_cfg = dict(cfg, task='cup')
        super().__init__(cup_cfg, render=False)
        old_model, old_data = self.model, self.data
        self.xml = slider_xml(self.xml, cfg)
        self.model = self.mj.MjModel.from_xml_string(self.xml)
        self.data = self.mj.MjData(self.model)
        self.ik_data = self.mj.MjData(self.model)
        qwidth, dwidth = {0: 7, 1: 4, 2: 1, 3: 1}, {0: 6, 1: 3, 2: 1, 3: 1}
        for i in range(self.model.njnt):
            name = self.model.joint(i).name
            if name == 'slider_joint':
                continue
            old_id = old_model.joint(name).id
            kind = int(self.model.jnt_type[i])
            n, nd = qwidth[kind], dwidth[kind]
            q, oq = self.model.jnt_qposadr[i], old_model.jnt_qposadr[old_id]
            d, od = self.model.jnt_dofadr[i], old_model.jnt_dofadr[old_id]
            self.data.qpos[q:q+n] = old_data.qpos[oq:oq+n]
            self.data.qvel[d:d+nd] = old_data.qvel[od:od+nd]
        self.data.ctrl[:] = old_data.ctrl
        self.cfg = copy.deepcopy(cfg)
        self.task = 'slider'
        self.tool_id = self.model.body('tool').id
        self.obj_id = self.model.body('object').id
        self.slider_q = int(self.model.jnt_qposadr[self.model.joint('slider_joint').id])
        self.slider_d = int(self.model.jnt_dofadr[self.model.joint('slider_joint').id])
        self.arm_q = np.array([self.model.jnt_qposadr[self.model.joint(f'joint{i}').id] for i in range(1, 8)])
        self.arm_d = np.array([self.model.jnt_dofadr[self.model.joint(f'joint{i}').id] for i in range(1, 8)])
        self.arm_limits = np.array([self.model.jnt_range[self.model.joint(f'joint{i}').id] for i in range(1, 8)])
        self.finger_q = np.array([self.model.jnt_qposadr[self.model.joint(f'finger_joint{i}').id] for i in (1, 2)])
        actor_names = {'hand_visual', 'hand_collision', 'finger1_visual',
                       'finger2_visual', 'finger1_collision', 'finger2_collision',
                       'finger1_pad_collision', 'finger2_pad_collision'}
        self.goal_ids_prefix = FRONT_FACE_PREFIX
        self.goal_ids = [i for i in range(self.model.ngeom)
                         if (self.model.geom(i).name or '').startswith(FRONT_FACE_PREFIX)]
        if len(self.goal_ids) != 4:
            raise ValueError('Package 2 requires the four visible front-surface parts')
        self.actor_geom_ids = [i for i in range(self.model.ngeom)
                               if self.model.geom(i).name in actor_names]
        self.obj_ids = [i for i in range(self.model.ngeom)
                        if (self.model.geom(i).name or '').startswith('slider_')]
        self.mj.mj_forward(self.model, self.data)
        for _ in range(150):
            self._servo(False)
            self.mj.mj_step(self.model, self.data)
        self.t = 0
        self.start_pos, self.start_R = self.proprio()
        self.initial_object = self.data.xpos[self.obj_id].copy()
        self.initial_slide = float(self.data.qpos[self.slider_q])
        if render:
            self.renderer = self.mj.Renderer(self.model, RES, RES)

    def robot_tracking_diagnostics(self):
        return dict(super().robot_tracking_diagnostics(), native_task='slider',
                    geometry_family='cup', cup_support_riser=False,
                    slider_joint_is_object_mechanism=True,
                    slider_attachment_to_robot=False)

    def facts(self):
        """Evaluator only. This must be called after the trajectory is sealed."""
        target = self.model.geom('handle_right').id
        pairs = {frozenset((int(c.geom1), int(c.geom2)))
                 for c in self.data.contact if c.dist < .0005}
        sides = [('finger1_pad_collision', 'finger1_collision'),
                 ('finger2_pad_collision', 'finger2_collision')]
        bilateral = all(any(frozenset((self.model.geom(name).id, target)) in pairs
                            for name in names) for names in sides)
        displacement = float(self.data.qpos[self.slider_q] - self.initial_slide)
        velocity = float(self.data.qvel[self.slider_d])
        return dict(success=bool(bilateral and displacement > OPEN_SUCCESS_M
                                 and abs(velocity) < .01),
                    slider_displacement=displacement, slider_velocity=velocity,
                    bilateral=bool(bilateral), secure_grasp_assumed=False,
                    slider_attachment_to_robot=False,
                    slider_range_violation=max(0., -float(self.data.qpos[self.slider_q]),
                                               float(self.data.qpos[self.slider_q]) - SLIDE_RANGE_M))
