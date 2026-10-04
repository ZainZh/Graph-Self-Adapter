"""Small shared P45 factories and visible-only observation serialization."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('evaluations/fast_adapter_p45_robot_mechanism_v01')
TASK_BASE = {'cup': 455000, 'insert': 456000, 'slider': 457000}
SOURCES = ['prove/fast_adapter/geometry3d_env.py',
           'prove/fast_adapter/geometry3d_adapter.py',
           'prove/fast_adapter/robot_geometry_env_p45.py',
           'prove/fast_adapter/p45_slider_env.py',
           'prove/fast_adapter/neural_geometry_visual.py',
           'prove/fast_adapter/lila_frontend.py',
           'prove/fast_adapter/visual_alignment.py',
           'tools/fast_adapter/p45_common.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def js(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def make_env(task, seed, split, render=True, backend=None):
    if backend is not None:
        from importlib import import_module
        module = import_module(backend['module'])
        cls = getattr(module, backend['class'])
        configuration = import_module(backend.get('configuration_module', backend['module'])).configuration
    elif task == 'slider':
        from prove.fast_adapter.p45_slider_env import SliderGeometrySkill, configuration
        cls = SliderGeometrySkill
    else:
        from prove.fast_adapter.robot_geometry_env_p45 import RobotGeometrySkill, configuration
        cls = RobotGeometrySkill
    return cls(configuration(task, seed, split), render=render)


def make_controller(env, length=0., backend=None):
    from importlib import import_module
    module = (backend or {}).get('controller_module', 'prove.fast_adapter.robot_geometry_env_p45')
    cls = getattr(import_module(module), (backend or {}).get('controller_class', 'GeometryTaskController'))
    return cls(env.task, env.start_pos, env.start_R, length=length)


def observe_roles(env):
    """Only rendered masks, calibrated view and robot proprioception."""
    from prove.fast_adapter.neural_geometry_visual import mask_geometry
    obs, rgb, goal_mask = env.observe()
    env.renderer.enable_segmentation_rendering()
    try:
        seg = env.renderer.render().copy()
    finally:
        env.renderer.disable_segmentation_rendering()
    rendered = seg[:, :, 1] == int(env.mj.mjtObj.mjOBJ_GEOM)
    masks = np.stack([np.isin(seg[:, :, 0], env.actor_geom_ids) & rendered,
                      np.isin(seg[:, :, 0], env.obj_ids) & rendered,
                      goal_mask.astype(bool)])
    boxes, visible = mask_geometry(masks)
    return dict(obs, role_bbox=boxes, role_visible=visible), rgb, masks.astype(np.uint8)


def public_row(obs, t, action):
    return dict(corners=np.full((4, 2), np.nan) if obs['corners'] is None else obs['corners'],
                corners_valid=obs['corners'] is not None,
                **{k: obs[k] for k in ('camera_position', 'camera_rotation', 'K', 'position',
                                      'rotation', 'role_bbox', 'role_visible')},
                t=int(t), previous_action=np.asarray(action).copy())


def privileged_training_label(env):
    """Separate training label, never added to a public runtime observation."""
    p, R, size = env.oracle()
    outer = size + (.006 if env.task == 'insert' else 0.)
    a, b = outer
    local = np.array([[-a, b, 0], [a, b, 0], [a, -b, 0], [-a, -b, 0]])
    visible_p = p + R[:, 2] * .0006 if env.task == 'insert' else p
    return dict(corners=local @ R.T + visible_p, center=p, rotation=R, size=size)
