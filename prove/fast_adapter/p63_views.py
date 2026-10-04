"""P63 render-only public multiview observations from sealed P62 PRE states.

No policy, controller, target geometry or integration state crosses the public
observation boundary. Camera rotations use MuJoCo/OpenGL (+x right,+y up,-z
forward); raster origin is top left. Global means the existing camera, unchanged.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np

TIMES = (0, 4, 8, 12, 16, 20, 24, 29, 40, 60, 90, 120, 180, 239)
VIEWS = ('wrist', 'global')
RESOLUTION = 448
PUBLIC_ARRAY_KEYS = frozenset(('rgb', 'visible_masks', 'K', 'T_world_camera',
    'T_world_eef', 't', 'simulation_time', 'p62_geometry8', 'T_world_anchor'))


def pose_matrix(position, rotation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def camera_calibration(env, name):
    """Read actual camera world pose after mj_forward, never wrist substitutes."""
    camera_id = env.model.camera(name).id
    focal = (RESOLUTION / 2) / np.tan(np.deg2rad(env.model.cam_fovy[camera_id] / 2))
    K = np.array([[focal, 0, RESOLUTION/2], [0, focal, RESOLUTION/2], [0, 0, 1.]])
    return K, pose_matrix(env.data.cam_xpos[camera_id], env.data.cam_xmat[camera_id].reshape(3, 3))


def project_world(points, K, world_camera):
    cam = (np.asarray(points) - world_camera[:3, 3]) @ world_camera[:3, :3]
    if np.any(cam[:, 2] >= 0):
        raise ValueError('point not in front of camera')
    return np.stack((K[0, 2] + K[0, 0]*cam[:, 0]/-cam[:, 2],
                     K[1, 2] - K[1, 1]*cam[:, 1]/-cam[:, 2]), -1)


@contextmanager
def render_only_environment(config):
    """Suppress constructor settling AND servo; forbid integration thereafter.

    The old constructor is reused solely to compile the exact existing model.
    Its reset IK is initialization, discarded by the sealed integration restore.
    Single worker only: patches are process-global and are restored on exit.
    """
    import mujoco
    from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill
    suppressed = []
    def forbid(*args, **kwargs):
        raise RuntimeError('P63 forbids physics integration and controller execution')
    env = None
    with patch.object(mujoco, 'mj_step', side_effect=lambda *a, **kw: suppressed.append(1)), \
         patch.object(RobotGeometrySkill, '_servo', return_value=None):
        env = RobotGeometrySkill(config, render=False)
    try:
        with patch.object(mujoco, 'mj_step', side_effect=forbid), \
             patch.object(mujoco, 'mj_step1', side_effect=forbid), \
             patch.object(mujoco, 'mj_step2', side_effect=forbid), \
             patch.object(RobotGeometrySkill, '_servo', side_effect=forbid), \
             patch.object(RobotGeometrySkill, 'step', side_effect=forbid):
            env.renderer = mujoco.Renderer(env.model, height=RESOLUTION, width=RESOLUTION)
            yield env, len(suppressed)
    finally:
        if env is not None:
            env.close()


def restore_pre_action(env, row):
    """P47 integration-state coverage; mj_forward is not a dynamics step."""
    state = row['state']
    if int(row['t']) != int(state['t']):
        raise ValueError('PRE-action time mismatch')
    spec = env.mj.mjtState.mjSTATE_INTEGRATION
    saved = np.asarray(state['integration'])
    if saved.shape != (env.mj.mj_stateSize(env.model, spec),):
        raise ValueError('integration state shape mismatch')
    env.mj.mj_setState(env.model, env.data, saved, spec)
    env.t = int(row['t'])
    env.mj.mj_forward(env.model, env.data)
    for key in ('qpos', 'qvel', 'act', 'ctrl'):
        np.testing.assert_array_equal(getattr(env.data, key), state[key])
    if env.data.time != state['time']:
        raise ValueError('simulation time changed')



def restore_sensor_time(env, row):
    """Reconstruct P62's cached sensor kinematics, not its post-step qpos.

    implicitfast integrates these scalar hinge/slide joints by qpos += dt * qvel. Original
    P62 observe() used mj_step's pre-integration xpos/cameras. At t0 the public
    calibration called mj_forward, so current integration qpos is correct.
    This algebraic kinematic reconstruction executes no integration routine.
    """
    import hashlib
    restore_pre_action(env, row)
    integration_time = float(env.data.time)
    dt = float(env.model.opt.timestep)
    if env.model.nq != env.model.nv or not np.isin(env.model.jnt_type, [int(env.mj.mjtJoint.mjJNT_HINGE), int(env.mj.mjtJoint.mjJNT_SLIDE)]).all():
        raise ValueError('sensor reconstruction qualified only for scalar hinge/slide Panda insertion')
    if dt != .001:
        raise ValueError('frozen integration timestep changed')
    if int(row['t']) > 0:
        env.data.qpos[:] = np.asarray(row['state']['qpos']) - dt*np.asarray(row['state']['qvel'])
        env.data.time = integration_time - dt
        env.mj.mj_forward(env.model, env.data)
    return dict(integration_time=integration_time, sensor_time=float(env.data.time),
                sensor_qpos_sha256=hashlib.sha256(env.data.qpos.tobytes()).hexdigest(),
                original_integration_sha256=hashlib.sha256(np.asarray(row['state']['integration']).tobytes()).hexdigest(),
                reconstruction='identity after calibration' if int(row['t']) == 0 else 'qpos - dt*qvel; mj_forward only')


def render_public(env):
    spec = env.mj.mjtState.mjSTATE_INTEGRATION
    integration_before = np.empty(env.mj.mj_stateSize(env.model, spec))
    env.mj.mj_getState(env.model, env.data, integration_before, spec)
    before = (env.data.qpos.copy(), env.data.qvel.copy(), float(env.data.time))
    images, masks, Ks, cameras = [], [], [], []
    for name in VIEWS:
        env.renderer.disable_segmentation_rendering()
        env.renderer.update_scene(env.data, camera=name)
        images.append(env.renderer.render().copy())
        env.renderer.enable_segmentation_rendering()
        segmentation = env.renderer.render().copy()
        env.renderer.disable_segmentation_rendering()
        visible = segmentation[..., 1] == int(env.mj.mjtObj.mjOBJ_GEOM)
        masks.append(np.stack([visible & np.isin(segmentation[..., 0], ids)
                              for ids in (env.actor_geom_ids, env.obj_ids, env.goal_ids)]))
        K, camera = camera_calibration(env, name)
        Ks.append(K); cameras.append(camera)
    position, rotation = env.proprio()
    integration_after = np.empty_like(integration_before)
    env.mj.mj_getState(env.model, env.data, integration_after, spec)
    np.testing.assert_array_equal(integration_before, integration_after)
    np.testing.assert_array_equal(env.data.qpos, before[0])
    np.testing.assert_array_equal(env.data.qvel, before[1])
    if env.data.time != before[2]:
        raise RuntimeError('render advanced time')
    return dict(rgb=np.stack(images), visible_masks=np.stack(masks), K=np.stack(Ks),
                T_world_camera=np.stack(cameras), T_world_eef=pose_matrix(position, rotation))


def visibility_record(masks, t):
    from prove.fast_adapter.p60_contour import outer_hull_boundary
    rows = []
    for view, role_masks in zip(VIEWS, masks):
        boundary, guard = outer_hull_boundary(role_masks[2])
        rows.append(dict(t=int(t), view=view, role_pixels=role_masks.reshape(3, -1).sum(-1).tolist(),
                         closed_opening_admitted=boundary is not None, guard=guard))
    return rows
