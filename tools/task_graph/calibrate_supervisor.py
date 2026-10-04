#!/usr/bin/env python3
"""One-time offline calibration from the authorized P48 opened teacher replay.

No environment creation or rollout. Writes new files exclusively; existing
calibrations and historical artifacts cannot be overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prove.task_graph.libero_supervisor import (SCHEMA, GOAL, Measurement, RelationSupervisor,
                                               contact_facts, corners, rotation_xyzw)

BOUNDARIES = [0, 30, 57, 68, 85, 130, 200, 234]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_new(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def load_capture(capture):
    capture = Path(capture)
    binding = json.loads((capture/'BINDING.json').read_text())
    if binding.get('authority') != 'supervisor_demo_only':
        raise ValueError('Only explicitly authorized supervisor demo capture is accepted')
    if not binding.get('task', '').endswith('pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy'):
        raise ValueError('Only P48 selected book/caddy task is supported')
    rows = [json.loads(s) for s in (capture/'FRAMES.jsonl').read_text().splitlines()]
    if [r['frame'] for r in rows] != list(range(235)):
        raise ValueError('Expected complete P48 teacher replay with 235 frames')
    names = json.loads((capture/'SUPERVISOR_NAMES.json').read_text())
    b = names['body'].index('black_book_1_main')
    s = names['site'].index('desk_caddy_1_back_contain_region')
    samples = []
    for row in rows:
        path = capture/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('teacher frame hash mismatch')
        with np.load(path, allow_pickle=False) as z:
            contacts, wrong = contact_facts(names, row['supervisor'].get('contacts'))
            inside = row['supervisor']['goal_predicates'].get(json.dumps(GOAL))
            samples.append(Measurement(float(z['supervisor_sim_state'][0]), 'p48_teacher_offline',
                z['supervisor_body_xpos'][b], z['supervisor_body_xmat'][b].reshape(3, 3),
                z['observation_robot0_eef_pos'], rotation_xyzw(z['observation_robot0_eef_quat']),
                z['supervisor_site_xpos'][s], z['supervisor_site_xmat'][s].reshape(3, 3),
                float(np.abs(z['observation_robot0_gripper_qpos']).sum()), contacts, inside, wrong))
    return binding, samples


def calibrate(capture):
    capture = Path(capture)
    binding, samples = load_capture(capture)
    xml = ET.fromstring(binding['relocated_xml'])
    book = xml.find(".//geom[@name='black_book_1_g1']")
    site = xml.find(".//site[@name='desk_caddy_1_back_contain_region']")
    if book is None or site is None or book.get('type') != 'box' or site.get('type') != 'box':
        raise ValueError('Expected known book collision box and back compartment bounds')
    half = np.fromstring(book.get('size'), sep=' ')
    center = np.fromstring(book.get('pos'), sep=' ')
    # This box has a 180-degree y rotation, which preserves its symmetric corners.
    if book.get('quat') != '0 0 1 0':
        raise ValueError('Unexpected book collision rotation')
    back_half = np.fromstring(site.get('size'), sep=' ')
    refs = {}
    for name, frame in [('approach', 30), ('grasp', 57), ('vicinity', 130), ('alignment', 200)]:
        m = samples[frame]
        if name in ('approach', 'grasp'):
            pos, rot = m.book_rotation.T@(m.eef_position-m.book_position), m.book_rotation.T@m.eef_rotation
        else:
            pos, rot = m.back_rotation.T@(m.book_position-m.back_position), m.back_rotation.T@m.book_rotation
        refs[name] = dict(source_frame=frame, position=pos.tolist(), rotation=rot.tolist())
    def bottom(m):
        return float((corners(center, half)@m.book_rotation.T+m.book_position)[:, 2].min())
    table = bottom(samples[0])
    return dict(schema=SCHEMA, authority='supervisor_demo_only',
        qualification='one-shot supervised engineering relations; not learned semantic proof',
        source_status='OPENED_DEVELOPMENT', source_task=binding['task'],
        source_binding_sha256=sha(capture/'BINDING.json'),
        source_manifest_sha256=sha(capture/'FRAMES.jsonl'),
        source_names_sha256=sha(capture/'SUPERVISOR_NAMES.json'),
        source_demo_sha256=binding['source_sha256'], source_capture=str(capture),
        boundaries=BOUNDARIES, references=refs, book_collision_center=center.tolist(),
        book_collision_halfsize=half.tolist(), back_halfsize=back_half.tolist(),
        table_top_reference=table, lift_clearance_reference=bottom(samples[85])-table,
        open_width_threshold=(samples[57].gripper_width+samples[68].gripper_width)/2,
        tolerances=dict(position=float(min(half)), attachment_position=float(min(half)/2),
                        vicinity_position=float(2*max(back_half)), angle_rad=float(np.deg2rad(10))),
        rules=dict(attachment='three unique timestamps, bilateral book contact and stable EEF-frame book transform',
            unknown='missing or ambiguous evidence is None', progress='stage relation residual improvement > 0.01 normalized units',
            approach_grasp='within minimum book halfwidth and 10 degrees of teacher relative EEF pose',
            vicinity='within twice maximum back halfsize of teacher book pose, above back world top',
            alignment='half back bounds horizontal tolerance, 10 degrees, book bottom above region bottom',
            loss='previously certified attachment, no finger contact, transform drift, official In not true',
            wrong_target='bilateral same foreign named object contact; exclude table and caddy',
            terminal='official LIBERO In predicate only; no clock or waypoint completion'))


def timeline(capture, calibration):
    _, samples = load_capture(capture)
    judge = RelationSupervisor(calibration, instance_id='p48_teacher_offline')
    rows = []
    for frame, m in enumerate(samples):
        # Annotation selects the residual to report, never sets any predicate.
        stage = 's'+str(min(7, next((i for i, end in enumerate(BOUNDARIES[1:], 1) if frame < end), 7)))
        report = judge.observe_measurement(m, stage)
        rows.append(dict(frame=frame, stage_annotation=stage, facts=report.facts,
                         progress=report.progress, progress_value=report.progress_value,
                         diagnostics=report.diagnostics))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeline', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.timeline.exists():
        raise FileExistsError('Calibration and timeline outputs must both be new')
    calibration = calibrate(args.capture)
    rows = timeline(args.capture, calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_new(args.output, calibration)
    with args.timeline.open('x') as f:
        f.writelines(json.dumps(row, allow_nan=False)+'\n' for row in rows)
    print(json.dumps(dict(calibration=str(args.output), sha256=sha(args.output),
                         frames=len(rows), final_facts=rows[-1]['facts']), indent=2))


if __name__ == '__main__':
    main()
