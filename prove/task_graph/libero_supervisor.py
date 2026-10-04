"""Privileged, frozen one-demo book/caddy supervisor. Never an action input.

This is an engineering relation judge, not a learned semantic observer. Unknown
measurements remain None. Only facts and scalar progress leave this boundary.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools
import math
from typing import Mapping

import numpy as np

SCHEMA = 'prove.libero-book-supervisor-calibration.v1'
SIGNALS = ('open_gripper', 'above_grasp_approach', 'grasp_ready', 'attached',
           'clear_of_table', 'above_destination_vicinity',
           'aligned_with_back_compartment', 'inside_back_compartment',
           'object_lost', 'wrong_target', 'local_error')
GOAL = ['in', 'black_book_1', 'desk_caddy_1_back_contain_region']


def rotation_xyzw(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError('invalid EEF quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def angle(a, b):
    return float(np.arccos(np.clip((np.trace(np.asarray(a).T @ b)-1)/2, -1, 1)))


def corners(center, halfsize):
    return np.asarray(center) + np.asarray(list(itertools.product((-1, 1), repeat=3))) * halfsize


@dataclass(frozen=True)
class Measurement:
    """Supervisor-private snapshot; never pass this into a backend payload."""
    timestamp: float
    instance_id: str
    book_position: np.ndarray
    book_rotation: np.ndarray
    eef_position: np.ndarray
    eef_rotation: np.ndarray
    back_position: np.ndarray
    back_rotation: np.ndarray
    gripper_width: float
    contacts: tuple[bool, bool] | None
    inside: bool | None
    wrong_target: bool | None

    def valid(self):
        vectors = (self.book_position, self.eef_position, self.back_position)
        matrices = (self.book_rotation, self.eef_rotation, self.back_rotation)
        return (math.isfinite(self.timestamp) and math.isfinite(self.gripper_width)
                and all(np.asarray(v).shape == (3,) and np.isfinite(v).all() for v in vectors)
                and all(np.asarray(m).shape == (3, 3) and np.isfinite(m).all()
                        and np.allclose(np.asarray(m).T @ m, np.eye(3), atol=1e-5)
                        and np.linalg.det(m) > 0 for m in matrices))


@dataclass(frozen=True)
class SupervisorReport:
    facts: Mapping[str, bool | None]
    progress: bool
    progress_value: float | None
    diagnostics: Mapping[str, float | str | None]


class RelationSupervisor:
    """Instance-bound stateful judge; history advances only on a new timestamp."""
    def __init__(self, calibration: Mapping, *, instance_id: str):
        if calibration.get('schema') != SCHEMA or not instance_id:
            raise ValueError('invalid supervisor calibration or instance')
        if calibration.get('authority') != 'supervisor_demo_only':
            raise ValueError('privileged calibration authority required')
        # Copy through JSON to sever the caller's mutable calibration references.
        import json
        self.calibration = json.loads(json.dumps(calibration, allow_nan=False))
        self.instance_id = instance_id
        self.history = deque(maxlen=3)
        self.held_reference = None
        self.last_timestamp = None
        self.cached = None
        self.stage = None
        self.best = None

    def observe_measurement(self, measurement: Measurement | None, stage_id: str):
        if stage_id not in {f's{i}' for i in range(1, 8)}:
            raise ValueError('unknown stage')
        if measurement is None or not measurement.valid():
            self.history.clear()
            self.cached = None
            return SupervisorReport(dict.fromkeys(SIGNALS), False, None, {'status': 'measurement_unavailable'})
        if measurement.instance_id != self.instance_id:
            raise ValueError('cross-instance supervisor measurement')
        t = measurement.timestamp
        if self.last_timestamp is not None and t < self.last_timestamp:
            raise ValueError('nonmonotonic measurement timestamp; create a new instance supervisor')
        if t != self.last_timestamp or self.cached is None:
            self.cached = self._measure(measurement)
            self.last_timestamp = t
        facts, residuals, diagnostics = self.cached
        value = residuals[stage_id]
        if self.stage != stage_id:
            self.stage, self.best = stage_id, value
            progress = False
        else:
            progress = value is not None and self.best is not None and value > self.best + .01
            if value is not None and (self.best is None or progress):
                self.best = value
        result_facts = dict(facts)
        local_goal = facts['aligned_with_back_compartment' if stage_id == 's6' else 'inside_back_compartment']
        result_facts['local_error'] = (False if stage_id not in ('s6', 's7') else
                                       None if local_goal is None else not local_goal)
        return SupervisorReport(result_facts, progress, value, dict(diagnostics))

    def _measure(self, m):
        c = self.calibration
        tol = c['tolerances']
        refs = c['references']
        eef_book = m.book_rotation.T @ (m.eef_position-m.book_position)
        eef_book_rot = m.book_rotation.T @ m.eef_rotation
        book_back = m.back_rotation.T @ (m.book_position-m.back_position)
        book_back_rot = m.back_rotation.T @ m.book_rotation
        held = (m.eef_rotation.T @ (m.book_position-m.eef_position),
                m.eef_rotation.T @ m.book_rotation)
        self.history.append((held, m.contacts))
        bilateral = None if m.contacts is None else all(m.contacts)
        if bilateral is False:
            attached = None if any(m.contacts) else False
        elif bilateral is None or len(self.history) < 3 or any(x[1] is None for x in self.history):
            attached = None
        elif not all(all(x[1]) for x in self.history):
            attached = None  # initial bilateral-contact warmup
        else:
            attached = bool(all(np.linalg.norm(held[0]-x[0][0]) <= tol['attachment_position']
                                and angle(held[1], x[0][1]) <= tol['angle_rad'] for x in self.history))
        if attached and self.held_reference is None:
            self.held_reference = (held[0].copy(), held[1].copy())
        if self.held_reference is None or m.inside is True or attached is True:
            lost = False
        elif bilateral is None or m.inside is None:
            lost = None
        else:
            drift = (np.linalg.norm(held[0]-self.held_reference[0]) > tol['attachment_position']
                     or angle(held[1], self.held_reference[1]) > tol['angle_rad'])
            lost = bool(drift) if bilateral is False and not any(m.contacts) else None
        book_points = corners(c['book_collision_center'], np.asarray(c['book_collision_halfsize']))
        world_points = book_points @ m.book_rotation.T + m.book_position
        bottom = float(world_points[:, 2].min())
        back_points = corners(np.zeros(3), np.asarray(c['back_halfsize'])) @ m.back_rotation.T + m.back_position
        back_bottom, back_top = float(back_points[:, 2].min()), float(back_points[:, 2].max())
        clearance = bottom-c['table_top_reference']
        pose_errors = {}
        for name, pos, rot in (('approach', eef_book, eef_book_rot), ('grasp', eef_book, eef_book_rot),
                               ('vicinity', book_back, book_back_rot), ('alignment', book_back, book_back_rot)):
            ref = refs[name]
            pose_errors[name] = (float(np.linalg.norm(pos-ref['position'])), angle(rot, ref['rotation']))
        approach, grasp, vicinity, alignment = [pose_errors[k] for k in ('approach', 'grasp', 'vicinity', 'alignment')]
        align_delta = np.abs(book_back[:2]-np.asarray(refs['alignment']['position'])[:2])
        align_pos = bool(np.all(align_delta <= np.asarray(c['back_halfsize'])[:2]*.5))
        align = bool(align_pos and alignment[1] <= tol['angle_rad'] and bottom > back_bottom)
        facts = dict(open_gripper=bool(m.gripper_width >= c['open_width_threshold']),
                     above_grasp_approach=bool(approach[0] <= tol['position'] and approach[1] <= tol['angle_rad']),
                     grasp_ready=bool(grasp[0] <= tol['position'] and grasp[1] <= tol['angle_rad']),
                     attached=attached,
                     clear_of_table=bool(clearance >= c['lift_clearance_reference']-tol['position']),
                     above_destination_vicinity=bool(vicinity[0] <= tol['vicinity_position'] and bottom >= back_top),
                     aligned_with_back_compartment=align,
                     inside_back_compartment=m.inside, object_lost=lost,
                     wrong_target=m.wrong_target, local_error=None)
        # Position/angle residuals, physical aperture/contact and clearance only.
        contact_score = None if m.contacts is None else sum(m.contacts)/2
        grasp_value = None if contact_score is None else (contact_score + .5*float(attached is True)
                         + .25*np.clip(1-m.gripper_width/c['open_width_threshold'], 0, 1))
        residuals = dict(s1=-(approach[0]/tol['position'] + approach[1]/tol['angle_rad']),
                         s2=-(grasp[0]/tol['position'] + grasp[1]/tol['angle_rad']), s3=grasp_value,
                         s4=clearance/max(c['lift_clearance_reference'], tol['position']),
                         s5=-vicinity[0]/tol['vicinity_position'],
                         s6=-(float(np.linalg.norm(align_delta/(np.asarray(c['back_halfsize'])[:2]*.5)))
                              + alignment[1]/tol['angle_rad']),
                         s7=(1.0 if m.inside else -float(np.linalg.norm(book_back/np.asarray(c['back_halfsize'])))))
        diagnostics = dict(status='supervisor_only', bottom_clearance=float(clearance),
                           contact_count=None if m.contacts is None else float(sum(m.contacts)),
                           approach_residual=approach[0], grasp_residual=grasp[0],
                           vicinity_residual=vicinity[0], alignment_angle=alignment[1])
        return facts, {k: None if v is None else float(v) for k, v in residuals.items()}, diagnostics


def contact_facts(names, contacts):
    """Only bilateral contact with another named movable object is wrong-target.

    Table/caddy contact during grasp/insertion is not a foreign-object grasp.
    """
    fingers = [{i for i, n in enumerate(names['geom']) if n and
                n.startswith(f'gripper0_finger{finger}_') and 'collision' in n}
               for finger in (1, 2)]
    book = {i for i, n in enumerate(names['geom']) if n and n.startswith('black_book_1_g')}
    if not book or any(not f for f in fingers) or contacts is None:
        return None, None
    def touching(geoms, finger):
        return any((r['geom1'] in geoms and r['geom2'] in finger) or
                   (r['geom2'] in geoms and r['geom1'] in finger) for r in contacts)
    bilateral = tuple(touching(book, f) for f in fingers)
    foreign = []
    for body in names.get('body', []):
        if body and body.endswith('_main') and body not in ('black_book_1_main', 'desk_caddy_1_main'):
            prefix = body[:-5] + '_g'
            geoms = {i for i, n in enumerate(names['geom']) if n and n.startswith(prefix)}
            if geoms:
                foreign.append(geoms)
    return bilateral, any(all(touching(g, f) for f in fingers) for g in foreign)


class LiberoSupervisor(RelationSupervisor):
    def __init__(self, env, calibration: Mapping, *, instance_id='libero_graph_instance'):
        super().__init__(calibration, instance_id=instance_id)
        self.env = env
        model = env.sim.model
        self.names = {kind: [getattr(model, kind+'_id2name')(i) for i in range(getattr(model, count))]
                      for kind, count in (('body', 'nbody'), ('geom', 'ngeom'))}
        self.book_id = model.body_name2id('black_book_1_main')
        self.back_id = model.site_name2id('desk_caddy_1_back_contain_region')
        if not np.allclose(model.site_size[self.back_id], calibration['back_halfsize']):
            raise ValueError('live compartment geometry differs from frozen calibration')

    def observe(self, obs, stage_id):
        data = self.env.sim.data
        try:
            contacts = [dict(geom1=int(data.contact[i].geom1), geom2=int(data.contact[i].geom2))
                        for i in range(data.ncon)]
            fingers, wrong = contact_facts(self.names, contacts)
            try:
                inside = bool(self.env._eval_predicate(GOAL))
            except (AttributeError, KeyError, ValueError):
                inside = None
            measurement = Measurement(float(data.time), self.instance_id,
                np.array(data.body_xpos[self.book_id]), np.array(data.body_xmat[self.book_id]).reshape(3, 3),
                np.asarray(obs['robot0_eef_pos']), rotation_xyzw(obs['robot0_eef_quat']),
                np.array(data.site_xpos[self.back_id]), np.array(data.site_xmat[self.back_id]).reshape(3, 3),
                float(np.abs(obs['robot0_gripper_qpos']).sum()), fingers, inside, wrong)
        except (AttributeError, KeyError, ValueError, IndexError):
            measurement = None
        return self.observe_measurement(measurement, stage_id)


LiveSupervisor = LiberoSupervisor
