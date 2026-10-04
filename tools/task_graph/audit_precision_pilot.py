#!/usr/bin/env python3
"""Read-only P50 sealed-command audit. No simulator or latent recurrence replay.

Declared before audit: cross-runtime command tolerances 1e-6 m / 1e-5 rad.
The local/server Torch versions differ; numerical agreement is not bitwise
agreement. Every completed failure stays in the 32-cell denominator. Missing
or aborted cells fail completeness rather than silently reducing that count.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from prove.fast_adapter.geometry3d_adapter import pose
from prove.fast_adapter.p45_neural import load_model
from prove.fast_adapter.robot_geometry_env_p45 import GeometryTaskController
from prove.task_graph.precision_action_policy import PrecisionActionController, decode_action, encode_state, load_frozen_head
from tools.fast_adapter.p47_evaluate import PUBLIC_KEYS, parameters_sha
from tools.task_graph.p50_precision_pilot import cells, domain_excursions
from tools.task_graph.present_precision_pilot import load_complete
from tools.task_graph.train_precision_action_policy import dump_new, sha
POSITION_TOLERANCE = 1e-6
ROTATION_TOLERANCE = 1e-5
DOMAIN_METADATA_TOLERANCE = 1e-12
SHADOW_CELLS = {0, 1, 16, 17}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command_error(command, p, R, closed, stage):
    dp = float(np.linalg.norm(np.asarray(command[0]) - p))
    dr = float(np.linalg.norm(Rotation.from_matrix(np.asarray(command[1]) @ np.asarray(R).T).as_rotvec()))
    require(dp <= POSITION_TOLERANCE and dr <= ROTATION_TOLERANCE, f'Command mismatch: {dp} m, {dr} rad')
    require(bool(command[2]) == bool(closed) and command[3] == stage, 'Gripper/stage mismatch')
    return dp, dr


def domain_error(actual, recorded):
    require(set(actual) == set(recorded), 'Head-domain field names mismatch')
    require(actual['outside_training_domain'] == recorded['outside_training_domain']
            and actual['violated_fields'] == recorded['violated_fields'], 'Head-domain violation flags mismatch')
    maximum = 0.
    for key in ('relative_position', 'rotation_angle', 'estimated_half_size', 'calibrated_held_length'):
        a, b = np.asarray(actual[key]), np.asarray(recorded[key])
        require(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=DOMAIN_METADATA_TOLERANCE),
                f'Head-domain numeric metadata mismatch: {key}')
        maximum = max(maximum, float(np.max(np.abs(a - b))))
    return maximum


def audit(freeze_path):
    freeze_path = Path(freeze_path)
    frozen, records = load_complete(freeze_path)
    require(frozen['cells'] == cells() and frozen['steps_per_episode'] == 240, 'Scope changed')
    for name, digest in frozen['sources'].items():
        require(sha(freeze_path.parent / 'source_snapshot' / name) == digest, f'Snapshot source changed: {name}')
        require(sha(ROOT / name) == digest, f'Local replay source differs: {name}')
    for name, item in frozen['files'].items():
        require(sha(ROOT / item['path']) == item['sha256'], f'Frozen artifact changed: {name}')
    torch.set_num_threads(1)
    head = load_frozen_head(ROOT / frozen['files']['action_head']['path'])
    adapter, _ = load_model(ROOT / frozen['files']['adapter']['path'], 'cpu')
    model_hashes = dict(adapter=parameters_sha(adapter), action_head=parameters_sha(head))
    prefixes, reports = {}, []
    for index, record in enumerate(records):
        directory, cell, result = Path(record['directory']), record['cell'], record['result']
        seal = json.loads((directory / 'SEAL.json').read_text())
        pre = json.loads((directory / 'PRE_RUN.json').read_text())
        require(seal['neural_parameters_before'] == model_hashes, 'Checkpoint/recorded parameter hash mismatch')
        require(pre['cell'] == cell and pre['cell_index'] == index and pre['freeze']['sha256'] == sha(freeze_path), 'Pre-run binding mismatch')
        traces = np.load(directory / 'TRACE.npy', allow_pickle=True).tolist()
        observations = np.load(directory / 'OBS.npy', allow_pickle=True).tolist()
        require(len(traces) == len(observations) == result['steps'] == 240, 'Incomplete action/observation trace')
        length = result['tool_calibration']['length']
        start = traces[0]['controller_before']
        controller = (PrecisionActionController(head, start['start_pos'], start['start_R'], length=length)
                      if cell['controller'] == 'neural' else GeometryTaskController('insert', start['start_pos'], start['start_R'], length=length))
        prefix = [(row['command_p'], row['command_R'], row['closed']) for row in traces[:12]]
        key = (cell['split'], cell['condition_seed'])
        if key in prefixes:
            require(all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
                        for a, b in zip(prefixes[key], prefix)), 'Four-arm shared prefix differs')
        prefixes[key] = prefix
        valid = events = gradient = neural = excursions = shadows = 0
        max_position = max_rotation = max_normalized = max_metadata = 0.
        initial_z = None
        for t, (row, obs) in enumerate(zip(traces, observations)):
            require(row['t'] == obs['t'] == t, 'Nonmonotonic frame index')
            require(set(obs) == set(PUBLIC_KEYS) | {'local', 't', 'previous_action', 'adaptation_frozen'}, 'Nonpublic observation key')
            require(np.array_equal(obs['local'], np.zeros(48)), 'Geometry arm reads appearance')
            require(np.array_equal(obs['position'], row['proprio'][0]) and np.array_equal(obs['rotation'], row['proprio'][1]), 'Proprioception mismatch')
            if t:
                previous = traces[t - 1]
                require(np.array_equal(obs['previous_action'], np.r_[previous['command_p'], previous['command_R'].ravel(), float(previous['closed'])]), 'Previous action mismatch')
            update = row['adaptation']
            available = obs['corners'] is not None and not obs['adaptation_frozen']
            require((update is not None) == available, 'Missing or unexpected adaptation record')
            if update is not None:
                valid += 1
                expected_u = 3 if cell['arm'] == 'continuous' and valid >= 3 else 0
                events += int(expected_u > 0)
                gradient += expected_u
                require(update['t'] == t and update['U'] == expected_u and len(update['inner']) == expected_u
                        and update['update_events'] == events and update['S'] == min(valid, 32), 'Adaptation budget/support count mismatch')
                if initial_z is None:
                    initial_z = np.asarray(update['initial_z'])
                require(np.array_equal(update['initial_z'], initial_z), 'Initial latent changed')
                if cell['arm'] == 'fixed':
                    require(np.array_equal(update['z'], initial_z) and update['latent_change'] == 0., 'Fixed latent changed')
            before = row['controller_before']
            require('model' not in before and 'records' not in before and before['length'] == length, 'Invalid compact controller state')
            controller.__dict__.update(copy.deepcopy(before))
            if cell['controller'] == 'neural':
                controller.records = []
            estimate, proprio = row['estimate'], row['proprio']
            phase = 'align' if before['phase'] == 'observe' else before['phase']
            features = encode_state(estimate, proprio, length, phase) if t >= 12 and estimate is not None else None
            if features is not None:
                domain = domain_excursions(features)
                max_metadata = max(max_metadata, domain_error(domain, row['head_domain']))
                excursions += int(domain['outside_training_domain'])
            else:
                require(row['head_domain'] is None, 'Unexpected domain record')
            issued = controller.command(t, estimate, proprio)
            dp, dr = command_error(issued, row['command_p'], row['command_R'], row['closed'], row['stage'])
            max_position, max_rotation = max(max_position, dp), max(max_rotation, dr)
            require(controller.phase == row['controller_after']['phase'] and controller.stable_steps == row['controller_after']['stable_steps'], 'Phase judge replay mismatch')
            action = row['neural_action']
            if cell['controller'] == 'neural' and features is not None:
                require(action is not None and action['phase'] == phase and np.array_equal(features, np.asarray(action['features'], np.float32)), 'Neural feature/phase mismatch')
                require(action['command_source'] == 'frozen_precision_action_mlp', 'Wrong precision command source')
                decoded = decode_action(action['normalized_action'], estimate[1], proprio)
                command_error((*decoded, row['closed'], row['stage']), row['command_p'], row['command_R'], row['closed'], row['stage'])
                max_normalized = max(max_normalized, float(np.max(np.abs(np.asarray(controller.records[-1]['normalized_action']) - action['normalized_action']))))
                neural += 1
            else:
                require(action is None, 'Unexpected neural action record')
            if index in SHADOW_CELLS and update is not None:
                controller.__dict__.update(copy.deepcopy(before))
                shadow = controller.command(t, pose(np.asarray(update['pre_update_points'])), proprio)
                command_error(shadow, update['shadow_command_position'], update['shadow_command_rotation'], shadow[2], shadow[3])
                require(abs(np.linalg.norm(row['command_p'] - update['shadow_command_position']) - update['same_state_action_position_delta']) <= POSITION_TOLERANCE, 'Shadow delta mismatch')
                rotation_delta = np.linalg.norm(Rotation.from_matrix(row['command_R'] @ np.asarray(update['shadow_command_rotation']).T).as_rotvec())
                require(abs(rotation_delta - update['same_state_action_rotation_delta']) <= ROTATION_TOLERANCE, 'Shadow rotation delta mismatch')
                require(update['same_state_closed_changed'] == (bool(row['closed']) != bool(shadow[2]))
                        and update['same_state_stage_changed'] == (row['stage'] != shadow[3]), 'Shadow gripper/stage delta mismatch')
                shadows += 1
        require(gradient == result['gradient_steps'] and events == result['adaptation_events'] and valid == result['valid_corner_frames'], 'Result adaptation counts mismatch')
        require(neural == result['neural_precision_commands'] and excursions == result['head_domain_excursion_steps'], 'Result neural/domain counts mismatch')
        reports.append(dict(cell_index=index, cell=cell, validated_success=result['validated_success'],
                            gradient_steps=gradient, neural_commands=neural, domain_excursions=excursions, shadow_commands_checked=shadows,
                            max_command_position_error_m=max_position, max_command_rotation_error_rad=max_rotation,
                            max_normalized_head_output_difference=max_normalized, max_domain_metadata_error=max_metadata))
    require(dict(adapter=parameters_sha(adapter), action_head=parameters_sha(head)) == model_hashes, 'Audit mutated model weights')
    return dict(schema='prove.p50-offline-command-audit.v1', passed=True, freeze_sha256=sha(freeze_path), cells_checked=32,
                task_success_denominator=32, validated_successes=sum(bool(r['validated_success']) for r in reports),
                tolerances=dict(position_m=POSITION_TOLERANCE, rotation_rad=ROTATION_TOLERANCE, bitwise_required=False,
                                domain_metadata_atol=DOMAIN_METADATA_TOLERANCE, domain_metadata_rtol=0),
                audit_source_sha256=sha(__file__), shadow_cell_indices=sorted(SHADOW_CELLS), records=reports,
                runtime=dict(audit_python=sys.version.split()[0], audit_torch=str(torch.__version__), pilot=frozen['packages']),
                max_domain_metadata_error=max(r['max_domain_metadata_error'] for r in reports),
                metadata_review='Only numeric domain metadata uses 1e-12 absolute tolerance; violation flags/field names and neural features remain exact',
                limits=['no physical reintegration or independent success-fact recomputation', 'no full latent recurrence/autograd replay',
                        'same-state shadow replay only cells 0/1/16/17', 'pixel content not re-rendered; sealed file hashes checked',
                        'cross-runtime command comparison is numerical, not bitwise'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Audit output must be a new file')
    try:
        report = audit(args.freeze)
    except Exception as exc:
        report = dict(schema='prove.p50-offline-command-audit.v1', passed=False, error=type(exc).__name__, reason=str(exc),
                      tolerances=dict(position_m=POSITION_TOLERANCE, rotation_rad=ROTATION_TOLERANCE), audit_source_sha256=sha(__file__))
    dump_new(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
