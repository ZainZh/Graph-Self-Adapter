"""P44 articulated Panda execution of the geometry tasks.

The robot is a fixed-base, torque-actuated seven-joint chain. Cartesian commands
are converted to joint targets using a separate kinematic scratch state; only
motor torques advance the live state. Cup grasping uses finger contacts only.
Insertion explicitly assumes a secure grasp: the peg is a rigid wrist child
from reset. Oracle geometry is reserved for engineering/evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from prove.fast_adapter.geometry3d_env import (
    GeometrySkill, RES, camera_axes, configuration, quat, vec,
)

ROOT = Path(__file__).resolve().parents[2]


def panda_assets():
    explicit = os.environ.get('PROVE_PANDA_ASSET_ROOT')
    candidates = ([Path(explicit)] if explicit else []) + [
        ROOT / 'data/runtime/libero-pi05-venv-clean/libero-pro-composite-assets',
        ROOT / 'data/runtime/libero-pi05-venv/libero-pro-composite-assets',
    ]
    for p in candidates:
        if (p / 'robots/panda/robot.xml').is_file() and (p / 'grippers/panda_gripper.xml').is_file():
            return p
    raise FileNotFoundError('Set PROVE_PANDA_ASSET_ROOT to the robosuite assets directory')


def source_assets():
    base = panda_assets()
    paths = [base / 'robots/panda/robot.xml', base / 'grippers/panda_gripper.xml']
    for path in paths[:]:
        tree = ET.parse(path)
        paths.extend(path.parent / x.attrib['file'] for x in tree.findall('.//mesh'))
    return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def robot_command(task, t, estimate, start_p, start_R, length=0., anchor=None):
    """Common fixed schedule; no privileged geometry or success feedback."""
    if t < 12:
        waypoints = np.array([[.018, 0, 0], [0, .018, 0], [-.018, 0, 0], [0, -.018, .01]])
        return start_p + start_R @ waypoints[t // 3], start_R, task == 'insert', 'shared observation motion'
    if estimate is None:
        return start_p, start_R, task == 'insert', 'target not observed: hold'
    p, R, size = estimate
    if task == 'insert':
        clearance = .04 if t < 60 else (.004 if t < 90 else -.04)
        return p + R[:, 2] * (length + clearance), R, True, ('align held peg' if t < 60 else 'approach slot' if t < 90 else 'insert along estimated axis')
    grip = p + R[:, 0] * (size[0] - .003) - R[:, 2] * .004
    if anchor is not None and t >= 105:
        grip, R = anchor
    if t < 60:
        return grip + R[:, 0] * .055, R, False, 'align outside cup handle'
    if t < 105:
        return grip, R, False, 'approach handle with open fingers'
    if t < 125:
        return grip, R, True, 'close physical Panda fingers'
    return grip + np.array([0., 0., .065]), R, True, 'lift with finger contacts'


class RobotGeometrySkill(GeometrySkill):
    """The P43 observation API with a physically actuated Panda backend."""

    def __init__(self, cfg, render=True):
        import mujoco
        self.mj = mujoco
        self.cfg = copy.deepcopy(cfg)
        self.task = cfg['task']
        self.t = 0
        self.renderer = None
        self.tool_calibration = {}
        # Obtain only the declared scene geometry from the unchanged P41 scene.
        # Its simulated floating tool is discarded before constructing P44.
        scene = GeometrySkill(cfg, render=False)
        self.truth_target = tuple(x.copy() for x in scene.truth_target)
        self.goal_ids_prefix = scene.goal_ids_prefix
        self.camera_local = tuple(x.copy() for x in scene.camera_local)
        self.start_pos = scene.start_pos.copy()
        self.start_R = scene.start_R.copy()
        if self.task == 'cup':
            # A safe side-grasp observation pose, independent of test geometry.
            self.start_pos = np.array([.13, -.10, .16])
            cp = np.array([.10, .12, .20])
            self.camera_local = cp, camera_axes(cp, np.array([-.07, -.04, 0.]), np.array([0., 1., 0.]))
        self.peg_length = cfg.get('length', 0.)
        with tempfile.NamedTemporaryFile(suffix='.xml') as f:
            mujoco.mj_saveLastXML(f.name, scene.model)
            scene_xml = ET.parse(f.name).getroot()
        scene.close()
        root = ET.Element('mujoco', model='p44_panda_' + self.task)
        ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
        ET.SubElement(root, 'option', timestep='.001', integrator='implicitfast', iterations='80', cone='elliptic')
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'global', offwidth='896', offheight='448')
        ET.SubElement(visual, 'quality', shadowsize='2048', offsamples='0')
        default = ET.SubElement(root, 'default')
        ET.SubElement(default, 'geom', condim='4', friction='1.5 .01 .001', solref='.005 1', solimp='.95 .99 .001')
        assets = ET.SubElement(root, 'asset')
        world = ET.SubElement(root, 'worldbody')
        ET.SubElement(world, 'light', pos='0 -.4 1.4', diffuse='.8 .8 .8')
        ET.SubElement(world, 'light', pos='-.6 .5 1.0', diffuse='.5 .5 .5')
        table_z = -.125 if self.task == 'cup' else -.025
        ET.SubElement(world, 'geom', name='table', type='box', pos=f'-.03 0 {table_z}', size='.44 .36 .025', rgba='.32 .38 .41 1')
        if self.task == 'cup':
            # A physical cup stand leaves room for the real horizontal hand.
            # Its top stays at canonical z=0: cup geometry/poses are unchanged.
            cx, cy, _ = cfg['center']
            ET.SubElement(world, 'geom', name='cup_support', type='cylinder', pos=f'{cx} {cy} -.05', size=f'{cfg["radius"] + .001} .05', rgba='.48 .52 .56 1')
        ET.SubElement(world, 'geom', name='ground', type='plane', pos='0 0 -.65', size='2 2 .01', rgba='.14 .17 .20 1')
        ET.SubElement(world, 'geom', name='robot_pedestal', type='box', pos='.50 .22 -.415', size='.10 .10 .235', rgba='.21 .25 .29 1')
        world.append(copy.deepcopy(scene_xml.find("worldbody/body[@name='object']")))
        self.global_camera = (np.array([.82, -.93, .72]), None)
        gp = self.global_camera[0]
        gr = camera_axes(gp, np.array([.18, .10, .20]))
        self.global_camera = gp, gr
        ET.SubElement(world, 'camera', name='global', pos=vec(gp), xyaxes=vec(gr[:, :2].T.flatten()), fovy='46')
        self.calibration_camera = (np.array([.36, -.48, .38]), None)
        cp = self.calibration_camera[0]
        cr = camera_axes(cp, np.array([0., 0., .09]))
        self.calibration_camera = cp, cr
        ET.SubElement(world, 'camera', name='calibration', pos=vec(cp), xyaxes=vec(cr[:, :2].T.flatten()), fovy='38')
        detail_pos = np.array([.29, -.37, .25])
        detail_R = camera_axes(detail_pos, np.array([0., 0., .09]))
        ET.SubElement(world, 'camera', name='detail', pos=vec(detail_pos), xyaxes=vec(detail_R[:, :2].T.flatten()), fovy='36')
        base = panda_assets()
        robot = ET.parse(base / 'robots/panda/robot.xml').getroot()
        gripper = ET.parse(base / 'grippers/panda_gripper.xml').getroot()
        for tree, folder in [(robot, base / 'robots/panda'), (gripper, base / 'grippers')]:
            for item in tree.find('asset'):
                item = copy.deepcopy(item)
                if 'file' in item.attrib:
                    item.set('file', str((folder / item.attrib['file']).resolve()))
                assets.append(item)
        arm = copy.deepcopy(robot.find('worldbody/body'))
        arm.set('pos', '.50 .22 -.18')
        arm.set('quat', '0 0 0 1')
        for parent in arm.iter():
            for cam in list(parent.findall('camera')):
                parent.remove(cam)
        for geom in arm.iter('geom'):
            if geom.attrib.get('group') == '0':
                geom.set('rgba', '.3 .3 .3 0')
        wrist = arm.find(".//body[@name='right_hand']")
        hand = copy.deepcopy(gripper.find('worldbody/body'))
        wrist.append(hand)
        for parent in hand.iter():
            for site in list(parent.findall('site')):
                parent.remove(site)
        for joint in hand.iter('joint'):
            joint.set('damping', '4')
            joint.set('armature', '.01')
            joint.set('frictionloss', '.1')
        for geom in hand.iter('geom'):
            if geom.attrib.get('group') == '0':
                geom.set('rgba', '.3 .3 .3 0')
        # Tool orientation maintains the old geometry interface. Cup tool x is
        # radial, y vertical, z the handle normal. Real fingers approach -x and
        # close along z. Insertion tool z points away from the peg tip.
        local_R = (Rotation.from_euler('z', 180, degrees=True).as_matrix() @ Rotation.from_euler('y', 90, degrees=True).as_matrix()) if self.task == 'cup' else Rotation.from_euler('x', 180, degrees=True).as_matrix()
        tool = ET.SubElement(hand, 'body', name='tool', pos='0 0 .0934', quat=vec(quat(local_R)))
        cp, cr = self.camera_local
        ET.SubElement(tool, 'camera', name='wrist', pos=vec(cp), xyaxes=vec(cr[:, :2].T.flatten()), fovy='65')
        if self.task == 'insert':
            L = cfg['length']; px, py = cfg['half']
            ET.SubElement(tool, 'geom', name='peg', type='box', pos=f'0 0 {-L/2}', size=f'{px} {py} {L/2}', mass='.06', rgba='.48 .25 .78 1')
            ET.SubElement(tool, 'geom', name='held_grasp_section', type='box', pos='0 0 .012', size=f'{px} {py} .012', mass='.015', rgba='.48 .25 .78 1', contype='0', conaffinity='0')
        world.append(arm)
        contact = ET.SubElement(root, 'contact')
        # The supplied padded finger collision approximations overlap at zero
        # opening. Exclude only the opposing fingers, as in a coupled gripper;
        # every finger/object/table contact remains active.
        ET.SubElement(contact, 'exclude', body1='leftfinger', body2='rightfinger')
        actuator = ET.SubElement(root, 'actuator')
        for a in robot.find('actuator'):
            actuator.append(copy.deepcopy(a))
        for a in gripper.find('actuator'):
            actuator.append(copy.deepcopy(a))
        self.xml = ET.tostring(root, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.tool_id = self.model.body('tool').id
        self.obj_id = self.model.body('object').id
        self.arm_q = np.array([self.model.jnt_qposadr[self.model.joint(f'joint{i}').id] for i in range(1, 8)])
        self.arm_d = np.array([self.model.jnt_dofadr[self.model.joint(f'joint{i}').id] for i in range(1, 8)])
        self.arm_limits = np.array([self.model.jnt_range[self.model.joint(f'joint{i}').id] for i in range(1, 8)])
        self.finger_q = np.array([self.model.jnt_qposadr[self.model.joint(f'finger_joint{i}').id] for i in (1, 2)])
        self.home_q = np.array([0., -.7, 0., -2., 0., 1.5, -1.5]) if self.task == 'cup' else np.array([0., -.45, 0., -2.15, 0., 1.7, .8])
        self.data.qpos[self.arm_q] = self.home_q
        self.data.qpos[self.finger_q] = [.035, -.035]
        mujoco.mj_forward(self.model, self.data)
        self.target_q, residual = self._reset_inverse_kinematics(self.start_pos, self.start_R)
        self.data.qpos[self.arm_q] = self.target_q
        if self.task == 'insert':
            width = cfg['half'][0]
            self.data.qpos[self.finger_q] = [width, -width]
        mujoco.mj_forward(self.model, self.data)
        self.command_pos = self.start_pos.copy()
        self.command_R = self.start_R.copy()
        self._reset_ik_residual = residual
        self._last_tracking = {}
        for _ in range(150):
            self._servo(self.task == 'insert')
            mujoco.mj_step(self.model, self.data)
        self.start_pos, self.start_R = self.proprio()
        self.initial_object = self.data.xpos[self.obj_id].copy()
        self.goal_ids = [i for i in range(self.model.ngeom) if (self.model.geom(i).name or '').startswith(self.goal_ids_prefix)]
        actor_names = {'hand_visual', 'hand_collision', 'finger1_visual', 'finger2_visual', 'finger1_collision', 'finger2_collision', 'finger1_pad_collision', 'finger2_pad_collision'}
        self.actor_geom_ids = [i for i in range(self.model.ngeom) if self.model.geom(i).name in actor_names]
        self.obj_ids = [i for i in range(self.model.ngeom) if (self.model.geom(i).name or '').startswith('cup_') or self.model.geom(i).name in ('peg', 'held_grasp_section')]
        if render:
            self.renderer = mujoco.Renderer(self.model, RES, RES)

    def _reset_inverse_kinematics(self, p, R):
        from scipy.optimize import least_squares
        def error(q):
            self.ik_data.qpos[:] = self.data.qpos
            self.ik_data.qpos[self.arm_q] = q
            self.mj.mj_forward(self.model, self.ik_data)
            current_R = self.ik_data.xmat[self.tool_id].reshape(3, 3)
            return np.r_[p - self.ik_data.xpos[self.tool_id], .2 * Rotation.from_matrix(R @ current_R.T).as_rotvec()]
        fit = least_squares(error, self.home_q, bounds=(self.arm_limits[:, 0] + .02, self.arm_limits[:, 1] - .02), max_nfev=200)
        e = error(fit.x)
        residual = dict(position=float(np.linalg.norm(e[:3])), angle=float(np.linalg.norm(e[3:]) / .2))
        if residual['position'] > .002 or residual['angle'] > .035:
            raise RuntimeError(f'Nominal robot reset is not reachable: {residual}')
        return fit.x, residual

    def _inverse_kinematics(self, p, R, iterations=12):
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        self.mj.mj_forward(self.model, d)
        jp = np.zeros((3, self.model.nv)); jr = jp.copy()
        for _ in range(iterations):
            current_R = d.xmat[self.tool_id].reshape(3, 3)
            e = np.r_[np.asarray(p) - d.xpos[self.tool_id], Rotation.from_matrix(R @ current_R.T).as_rotvec()]
            if np.linalg.norm(e[:3]) < .00015 and np.linalg.norm(e[3:]) < .002:
                break
            self.mj.mj_jacBody(self.model, d, jp, jr, self.tool_id)
            J = np.vstack([jp[:, self.arm_d], jr[:, self.arm_d]])
            damping = .002
            dq = J.T @ np.linalg.solve(J @ J.T + np.eye(6) * damping ** 2, e)
            dq *= min(1., .15 / max(np.max(np.abs(dq)), 1e-12))
            d.qpos[self.arm_q] = np.clip(d.qpos[self.arm_q] + dq, self.arm_limits[:, 0] + .015, self.arm_limits[:, 1] - .015)
            self.mj.mj_forward(self.model, d)
        return d.qpos[self.arm_q].copy(), dict(position=float(np.linalg.norm(e[:3])), angle=float(np.linalg.norm(e[3:])))

    def _servo(self, closed):
        mass = np.zeros((self.model.nv, self.model.nv))
        self.mj.mj_fullM(self.model, mass, self.data.qM)
        acc = 1600. * (self.target_q - self.data.qpos[self.arm_q]) - 80. * self.data.qvel[self.arm_d]
        torques = mass[np.ix_(self.arm_d, self.arm_d)] @ acc + self.data.qfrc_bias[self.arm_d]
        self.data.ctrl[:7] = np.clip(torques, self.model.actuator_ctrlrange[:7, 0], self.model.actuator_ctrlrange[:7, 1])
        opening = 0. if closed else .035
        self.data.ctrl[7:9] = [opening, -opening]

    def step(self, p, R, closed=False):
        old, old_R = self.proprio()
        delta = np.asarray(p) - old
        target_p = old + delta * min(1., .004 / max(np.linalg.norm(delta), 1e-12))
        dr = Rotation.from_matrix(np.asarray(R) @ old_R.T).as_rotvec()
        target_R = Rotation.from_rotvec(dr * min(1., .045 / max(np.linalg.norm(dr), 1e-12))).as_matrix() @ old_R
        self.command_pos = np.asarray(p).copy(); self.command_R = np.asarray(R).copy()
        self.target_q, residual = self._inverse_kinematics(target_p, target_R)
        for _ in range(60):
            self._servo(closed)
            self.mj.mj_step(self.model, self.data)
        self.t += 1
        actual_p, actual_R = self.proprio()
        self._last_tracking = dict(ik_position_error=residual['position'], ik_angle_error=residual['angle'], position_error=float(np.linalg.norm(target_p - actual_p)), angle_error=float(np.linalg.norm(Rotation.from_matrix(target_R @ actual_R.T).as_rotvec())), joint_velocity_max=float(np.max(np.abs(self.data.qvel[self.arm_d]))), joint_target_error_max=float(np.max(np.abs(self.target_q - self.data.qpos[self.arm_q]))))

    def robot_tracking_diagnostics(self):
        q = self.data.qpos[self.arm_q]
        violation = np.maximum(self.arm_limits[:, 0] - q, q - self.arm_limits[:, 1])
        finger_q = self.data.qpos[self.finger_q]
        finger_violation = np.maximum(np.array([0., -.04]) - finger_q, finger_q - np.array([.04, 0.]))
        return dict(self._last_tracking, robot='Franka Panda 7DOF', actuated_joint_count=7, live_qpos_reset_only=True, mocap_bodies=int(self.model.nmocap), reset_ik=self._reset_ik_residual, insertion_secure_grasp_assumed=self.task == 'insert', arm_joint_limit_violation_rad=float(max(0., np.max(violation))), finger_joint_limit_violation_m=float(max(0., np.max(finger_violation))), cup_support_riser=self.task == 'cup')

    def render_detail(self):
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera='detail')
        return self.renderer.render().copy()

    def snapshot(self):
        return dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(), act=self.data.act.copy(), ctrl=self.data.ctrl.copy(), qacc_warmstart=self.data.qacc_warmstart.copy(), time=float(self.data.time), t=self.t, target_q=self.target_q.copy(), command_pos=self.command_pos.copy(), command_R=self.command_R.copy())

    def restore(self, state):
        for key in ('qpos', 'qvel', 'act', 'ctrl', 'qacc_warmstart'):
            if key in state:
                getattr(self.data, key)[:] = state[key]
        self.data.time = state.get('time', self.data.time)
        self.t = state.get('t', self.t)
        for key in ('target_q', 'command_pos', 'command_R'):
            if key in state:
                setattr(self, key, np.asarray(state[key]).copy())
        self.mj.mj_forward(self.model, self.data)

    def calibrate_held_geometry(self):
        # Reuse the declared P43 visible-mask calibration with its same camera.
        old_global = self.global_camera
        camera_id = self.model.camera('global').id
        old_pos = self.model.cam_pos[camera_id].copy(); old_quat = self.model.cam_quat[camera_id].copy(); old_fovy = self.model.cam_fovy[camera_id]
        p, R = self.calibration_camera
        self.global_camera = p, R
        self.model.cam_pos[camera_id] = p; self.model.cam_quat[camera_id] = quat(R); self.model.cam_fovy[camera_id] = 38
        self.mj.mj_forward(self.model, self.data)
        try:
            return super().calibrate_held_geometry()
        finally:
            self.model.cam_pos[camera_id] = old_pos; self.model.cam_quat[camera_id] = old_quat; self.model.cam_fovy[camera_id] = old_fovy
            self.global_camera = old_global
            self.mj.mj_forward(self.model, self.data)

    def facts(self):
        if self.task == 'insert':
            return dict(super().facts(), secure_grasp_assumed=True)
        target_ids = {self.model.geom('handle_right').id}
        side_names = [('finger1_pad_collision', 'finger1_collision'), ('finger2_pad_collision', 'finger2_collision')]
        pairs = {frozenset((int(c.geom1), int(c.geom2))) for c in self.data.contact if c.dist < .0005}
        bilateral = all(any(frozenset((self.model.geom(n).id, target)) in pairs for n in names for target in target_ids) for names in side_names)
        lift = float(self.data.xpos[self.obj_id, 2] - self.initial_object[2])
        return dict(success=bool(bilateral and lift > .04), lift=lift, bilateral=bool(bilateral), secure_grasp_assumed=False)
