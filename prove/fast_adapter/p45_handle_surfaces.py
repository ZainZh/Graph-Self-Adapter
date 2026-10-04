"""Optional P45 visible-part semantics for the six surfaces of each handle bar.

This is a scene-construction transform, not a runtime geometry observer. It
preserves the original collision box and its body inertia. Six noncolliding,
zero-density visual slabs occupy the *inside* of its six existing surfaces.
Their exterior positions and material match the box exactly; no marker, new
color or protruding geometry is introduced. Rendered front-part identities
can then be exposed as an explicitly declared simulator visible-part mask.

The renderer still decides visibility and occlusion. Consumers get only the
front-face pixels that were actually rendered, never target projections or
unoccluded geometry. This finer part-mask prior must be disclosed separately
from RGB-only observation.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import numpy as np


FRONT_FACE_PREFIX = 'frame_front_'
SURFACE_PREFIX = 'frame_surface_'


def _vec(values):
    return ' '.join(f'{value:.12g}' for value in values)


def split_handle_visible_faces(xml):
    """Give existing handle surfaces separate render IDs; physics is unchanged.

    Only the four known axis-aligned box members of the declared rectangular
    handle asset are supported. Object yaw belongs to the containing body.
    This function does not project anything and is never called by a policy.
    """
    root = ET.fromstring(xml)
    body = root.find("worldbody/body[@name='object']")
    if body is None:
        raise ValueError('Expected an object body with four physical handle bars')
    if any((g.get('name') or '').startswith(FRONT_FACE_PREFIX) for g in body.findall('geom')):
        raise ValueError('Handle surfaces have already been split')
    bars = [body.find(f"geom[@name='handle_{side}']") for side in ('left', 'right', 'top', 'bottom')]
    if any(bar is None for bar in bars):
        raise ValueError('Expected exactly the four declared handle members')
    for bar in bars:
        if bar.get('type', 'sphere') != 'box':
            raise ValueError('Visible surface partition currently requires box members')
        if any(key in bar.attrib for key in ('quat', 'euler', 'axisangle', 'xyaxes', 'zaxis')):
            raise ValueError('Handle bar rotation must belong to the object body')
        center = np.fromstring(bar.get('pos', '0 0 0'), sep=' ')
        half = np.fromstring(bar.get('size', ''), sep=' ')
        color = np.fromstring(bar.get('rgba', '.5 .5 .5 1'), sep=' ')
        if center.shape != (3,) or half.shape != (3,) or color.shape != (4,):
            raise ValueError('Malformed handle box geometry')
        visual_source = copy.deepcopy(bar)
        # Exactly the old box remains the only contact/inertial geom. The
        # visual replacement has the same occupied exterior and same material.
        bar.set('rgba', _vec(np.r_[color[:3], 0.]))
        label = bar.get('name')[len('handle_'):]
        for axis in range(3):
            thickness = min(1e-6, half[axis] * .001)
            for sign in (-1, 1):
                surface = copy.deepcopy(visual_source)
                is_front = axis == 1 and sign == -1
                name = FRONT_FACE_PREFIX + label if is_front else f'{SURFACE_PREFIX}{label}_{axis}_{"neg" if sign < 0 else "pos"}'
                pos = center.copy()
                pos[axis] += sign * (half[axis] - thickness)
                size = half.copy()
                size[axis] = thickness
                surface.set('name', name)
                surface.set('pos', _vec(pos))
                surface.set('size', _vec(size))
                surface.set('contype', '0')
                surface.set('conaffinity', '0')
                surface.set('density', '0')
                surface.set('group', '2')
                surface.attrib.pop('mass', None)
                body.append(surface)
    return ET.tostring(root, encoding='unicode')
