"""P56 book geometry intervention, isolated from frozen P54/P55 contracts."""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import numpy as np

from tools.task_graph.p54_book_geometry import book_elements, original_properties


def validate_scale(axes, kind):
    scale = np.asarray(axes, dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Book scale must be three positive finite axes')
    if kind == 'isotropic':
        if not np.all(scale == scale[0]):
            raise ValueError('Isotropic geometry must scale all axes equally')
    elif kind == 'thickness':
        if not np.array_equal(scale[1:], np.ones(2)):
            raise ValueError('Thickness geometry may change only book local X')
    elif kind == 'mixed':
        if np.array_equal(scale, np.full(3, scale[0])):
            raise ValueError('Mixed geometry must change axes differently')
    else:
        raise ValueError('Unknown P56 geometry kind')
    return scale


def _fmt(values):
    return ' '.join(format(float(value), '.17g') for value in values)


def changed_xml(xml, axes, kind, properties=None):
    """Scale visible/collision geometry around the original collision center.

    The original compiled inertial is explicit in the new scene, so the visual
    mesh's contribution to mass is retained rather than silently recomputed.
    """
    scale = validate_scale(axes, kind)
    if np.array_equal(scale, np.ones(3)):
        return xml
    props = original_properties(xml) if properties is None else properties
    root = ET.fromstring(xml)
    body, visual, collision, mesh = book_elements(root)
    center = np.asarray(props['center'], dtype=float)
    half = np.asarray(props['halfsize'], dtype=float)
    mesh_scale = np.asarray(props['mesh_scale'], dtype=float)
    for value in (center, half, mesh_scale):
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Invalid original book geometry')
    collision.set('size', _fmt(scale * half))
    mesh.set('scale', _fmt(scale * mesh_scale))
    visual.set('pos', _fmt((1.0 - scale) * center))
    ET.SubElement(body, 'inertial', pos=_fmt(props['inertial_pos']),
                  quat=_fmt(props['inertial_quat']), mass=format(props['mass'], '.17g'),
                  diaginertia=_fmt(props['diagonal_inertia']))
    return ET.tostring(root, encoding='unicode')


def bottom_shift(rotation, old_halfsize, axes):
    """Preserve the original collision-box bottom at the initial book pose."""
    rotation = np.asarray(rotation, dtype=float)
    half = np.asarray(old_halfsize, dtype=float)
    scale = np.asarray(axes, dtype=float)
    if (rotation.shape != (3, 3) or half.shape != (3,) or scale.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(half).all()
            or np.any(half <= 0) or not np.isfinite(scale).all() or np.any(scale <= 0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)):
        raise ValueError('Invalid initial book geometry')
    return float(np.abs(rotation[2]) @ (scale * half - half))


def supervisor_calibration(original, axes):
    result = copy.deepcopy(original)
    result['book_collision_halfsize'] = (
        np.asarray(axes, dtype=float) * np.asarray(original['book_collision_halfsize'], dtype=float)).tolist()
    return result
