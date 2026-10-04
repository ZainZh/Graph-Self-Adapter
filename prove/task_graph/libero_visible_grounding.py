"""Public-pixel input engineering for a NEW frozen-head LIBERO transfer candidate.

This module does not certify a physical rectangle from a silhouette. Caddy
instance segmentation merges noncoplanar mesh faces, so even a four-sided
contour is only diagnostic. No object pose, site projection or evaluator data
is accepted by the public packet converter. No actuator command is produced.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import cv2
import numpy as np

from prove.task_graph.fast_adapter_bridge import public_observation

ROLES = ('actor', 'object', 'goal')
PUBLIC_KEYS = {'rgb', 'role_masks', 'K', 'camera_position', 'camera_rotation',
               'position', 'rotation', 'camera_convention'}
CAMERA_CONVENTION = 'opengl_camera_to_world_top_left_image'


def _rotation(value, name):
    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (3, 3) or not np.isfinite(result).all()
            or not np.allclose(result.T @ result, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(result), 1., atol=1e-5)):
        raise ValueError(f'{name} must be a proper camera/world rotation')
    return result


def visible_role_boxes(masks):
    """Actor/object/reference xyxy pixel extrema, never physical corner points."""
    if set(masks) != set(ROLES):
        raise ValueError('Exactly actor, object and goal visible masks are required')
    boxes, visible = np.zeros((3, 4), np.float32), np.zeros(3, np.float32)
    for index, name in enumerate(ROLES):
        mask = np.asarray(masks[name])
        if mask.shape != (448, 448) or mask.dtype != np.bool_:
            raise ValueError('Visible masks must be boolean and aligned to native 448 RGB')
        y, x = np.nonzero(mask)
        if len(x):
            boxes[index] = [x.min(), y.min(), x.max(), y.max()]
            visible[index] = 1
    return boxes, visible


def public_geometry_packet(packet):
    """Validate available channels; withhold unqualified caddy corners.

    Native calibrated 448 rendering avoids inventing an image flip/resize
    convention. RGB/masks share top-left pixel coordinates. Camera rotation
    maps OpenGL (+x right,+y up,-z forward) to world, as P45 rays require.
    The explicit robot rotation is measured own state, never opaque state[3:6].
    A None corner packet is diagnostic and cannot initialize the adapter.
    """
    if set(packet) != PUBLIC_KEYS:
        raise ValueError('Public packet fields mismatch; truth and caller corners are forbidden')
    rgb = np.asarray(packet['rgb'])
    if rgb.shape != (448, 448, 3) or rgb.dtype != np.uint8:
        raise ValueError('Native calibrated 448x448 uint8 RGB required')
    if packet['camera_convention'] != CAMERA_CONVENTION:
        raise ValueError('Unqualified camera/image convention')
    K = np.asarray(packet['K'], dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])
            or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8
            or not (0 <= K[0, 2] < 448 and 0 <= K[1, 2] < 448)):
        raise ValueError('P45 requires finite, zero-skew calibrated image intrinsics')
    boxes, visible = visible_role_boxes(packet['role_masks'])
    observation = {key: packet[key] for key in ('K', 'camera_position', 'position')}
    for name in ('camera_rotation', 'rotation'):
        observation[name] = _rotation(packet[name], name)
    observation.update(corners=None, role_bbox=boxes, role_visible=visible)
    return public_observation(observation, 'geometry')


def silhouette_diagnostic(mask):
    """Image evidence only; explicitly does not promote bbox/hull to corners."""
    mask = np.asarray(mask)
    if mask.shape != (448, 448) or mask.dtype != np.bool_:
        raise ValueError('Expected native boolean visible mask')
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    clipped = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    largest = max(contours, key=cv2.contourArea) if contours else None
    vertices = 0 if largest is None else len(cv2.approxPolyDP(
        largest, max(1., .01 * cv2.arcLength(largest, True)), True))
    return dict(visible_pixels=int(mask.sum()), touches_image_border=clipped,
                external_contours=len(contours), largest_contour_vertices=vertices,
                physical_coplanarity_verified=False, adapter_corners=None,
                reason='Instance silhouette cannot establish one physical planar face')


def caddy_source_audit(xml):
    """Inspect development/design XML only, excluding all world placements.

    Collision boxes locate possible engineering references, but their corners
    are never projected or sent to the estimator. Mesh faces need pixel-based
    association; an invisible goal site is not a rendered target surface.
    """
    root = ET.fromstring(xml)
    body = root.find(".//body[@name='desk_caddy_1_main']")
    if body is None:
        raise ValueError('Selected development source lacks the caddy body')
    geoms = list(body.findall('geom'))
    meshes = [g.get('name') for g in geoms if g.get('type') == 'mesh']
    site = body.find("site[@name='desk_caddy_1_back_contain_region']")
    rear_wall = body.find("geom[@name='desk_caddy_1_g4']")
    return dict(goal_site_present=site is not None,
                goal_is_physical_geom=False,
                visual_mesh_geoms=meshes,
                collision_box_count=sum(g.get('type') == 'box' for g in geoms),
                candidate_reference='rear wall or back-compartment rim',
                rear_wall_collision_proxy_present=rear_wall is not None,
                face_specific_visible_mask_available=False,
                note='Single caddy mesh geom does not segment individual planar faces; collision proxy is not visible evidence')


def capability_report(source=None, packet=None):
    """No profile boolean, successful replay or task alias grants execution."""
    channels, diagnostic = 'not supplied', None
    if packet is not None:
        public_geometry_packet(packet)
        channels = 'shape/calibration convention checked; no physical corner producer'
        diagnostic = silhouette_diagnostic(packet['role_masks']['goal'])
    return dict(schema='prove.libero-adapter-input-qualification.v1',
                candidate='new OOD book/caddy transfer of frozen P45 insert geometry head',
                legacy_core_changed=False, old_validated_domain=False,
                input_qualified=False, execution_qualified=False,
                public_channels=channels, source_audit=source, pixel_diagnostic=diagnostic,
                missing=[
                    'Four visible coplanar physical points with RGB/visible-mask face association; whole-object bbox is insufficient',
                    'Point ordering and normal-sign qualification against frozen insert PnP cyclic/identity prior',
                    'Metric initialization qualification: frozen insert nominal rectangle is 0.038 x 0.032 m',
                    'Frozen object-local reference-to-goal transform from demonstration or declared design prior',
                    'Qualified absolute SE(3)/gripper to LIBERO delta OSC controller binding',
                ],
                next_gate='Collect scoped visible-mask/calibration development evidence; freeze new transfer candidate only after input qualification',
                adaptation_or_rollout_started=False)
