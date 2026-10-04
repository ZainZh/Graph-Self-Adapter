"""Isolated P54 book-size intervention; no historical asset is edited."""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import numpy as np

BOOK_BODY = 'black_book_1_main'
BOOK_VISUAL = 'black_book_1_g0'
BOOK_COLLISION = 'black_book_1_g1'
BOOK_MESH = 'black_book_1_black_book_vis'
BOOK_JOINT = 'black_book_1_joint0'
SCALES = (1.0, 0.9)


def _floats(value):
    return np.fromstring(value, sep=' ', dtype=float)


def _format(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def book_elements(root):
    body = next((x for x in root.iter('body') if x.get('name') == BOOK_BODY), None)
    if body is None:
        raise ValueError('Target book body missing')
    geoms = {x.get('name'): x for x in body.findall('geom')}
    visual, collision = geoms.get(BOOK_VISUAL), geoms.get(BOOK_COLLISION)
    mesh = next((x for x in root.iter('mesh') if x.get('name') == BOOK_MESH), None)
    if visual is None or collision is None or mesh is None:
        raise ValueError('Target book geometry missing')
    if visual.get('mesh') != BOOK_MESH or visual.get('contype') != '0' or visual.get('conaffinity') != '0':
        raise ValueError('Unexpected visual book geometry')
    if collision.get('type') != 'box' or collision.get('mesh') is not None:
        raise ValueError('Unexpected book collision geometry')
    if len(body.findall('joint')) != 1 or body.find('joint').get('name') != BOOK_JOINT or body.find('joint').get('type') != 'free':
        raise ValueError('Unexpected book joint')
    if body.find('inertial') is not None:
        raise ValueError('Original book already has explicit inertial')
    return body, visual, collision, mesh


def original_properties(xml):
    """Read source geometry and actual compiled inertia, including the visual mesh."""
    import mujoco
    root = ET.fromstring(xml)
    _, visual, collision, mesh = book_elements(root)
    center, half, mesh_scale = (_floats(collision.get('pos')), _floats(collision.get('size')), _floats(mesh.get('scale')))
    if any(a.shape != (3,) or not np.isfinite(a).all() for a in (center, half, mesh_scale)) or np.any(half <= 0):
        raise ValueError('Invalid book geometry')
    if visual.get('pos') not in (None, '0 0 0'):
        raise ValueError('Visual mesh already translated')
    model = mujoco.MjModel.from_xml_string(xml)
    body_id = model.body(BOOK_BODY).id
    return dict(center=center.tolist(), halfsize=half.tolist(), mesh_scale=mesh_scale.tolist(),
                mass=float(model.body_mass[body_id]), inertial_pos=model.body_ipos[body_id].tolist(),
                inertial_quat=model.body_iquat[body_id].tolist(),
                diagonal_inertia=model.body_inertia[body_id].tolist(), collision_friction=collision.get('friction'),
                collision_density=collision.get('density'))


def scaled_xml(xml, scale, properties=None):
    if scale not in SCALES:
        raise ValueError('Only predeclared P54 scales allowed')
    if scale == 1.0:
        return xml
    props = original_properties(xml) if properties is None else properties
    root = ET.fromstring(xml)
    body, visual, collision, mesh = book_elements(root)
    center = np.asarray(props['center'])
    collision.set('size', _format(scale * np.asarray(props['halfsize'])))
    mesh.set('scale', _format(scale * np.asarray(props['mesh_scale'])))
    visual.set('pos', _format((1.0 - scale) * center))
    ET.SubElement(body, 'inertial', pos=_format(props['inertial_pos']), quat=_format(props['inertial_quat']),
                  mass=format(props['mass'], '.17g'), diaginertia=_format(props['diagonal_inertia']))
    return ET.tostring(root, encoding='unicode')


def verify_scaled_xml(original, candidate, scale, properties=None):
    expected = scaled_xml(original, scale, properties)
    if candidate != expected:
        raise ValueError('P54 scene differs from the one-book intervention')
    return original_properties(original) if properties is None else properties


def initial_bottom_shift(world_collision_rotation, old_halfsize, scale):
    """Z translation that preserves the old collision-box bottom at initial pose."""
    if scale not in SCALES:
        raise ValueError('Unsupported scale')
    rotation = np.asarray(world_collision_rotation, dtype=float).reshape(3, 3)
    half = np.asarray(old_halfsize, dtype=float)
    if half.shape != (3,) or np.any(half <= 0) or not np.isfinite(rotation).all():
        raise ValueError('Invalid initial geometry')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError('Invalid initial rotation')
    return -(1.0 - scale) * float(np.abs(rotation[2]) @ half)


def scaled_supervisor_calibration(original, scale):
    result = copy.deepcopy(original)
    result['book_collision_halfsize'] = (scale * np.asarray(original['book_collision_halfsize'], dtype=float)).tolist()
    return result
