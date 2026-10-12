"""Offline sensor shadow replay: mj_forward and RGB-D only, never mj_step/evaluator."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from tools.task_graph.twin_sensor_rsi import REPO, digest, load_frozen_runner
from prove.task_graph.twin_sensor_adapter import packet_from_visible, sensor_holding_contract
from prove.task_graph.twin_sensor_judge import GapCalibration, SensorJudge


def sensor_predicate_predictions(stage, values, judge, references, confirmation_since):
    predictions = {}
    for group in ('completion_predicates', 'invariant_predicates'):
        for predicate in stage.get(group, []):
            kind = predicate.get('predicate')
            if kind in ('holding', 'holding_confirmed'):
                value = sensor_holding_contract(stage, values)
                if kind == 'holding_confirmed':
                    key = stage['stage_id']
                    if value is True:
                        start = confirmation_since.setdefault(key, judge.packet.timestamp_sec)
                        value = judge.packet.timestamp_sec - start + 1e-9 >= float(predicate.get('minimum_sec', .2))
                    else:
                        confirmation_since.pop(key, None)
            elif kind in ('released', 'loaded_released', 'stabilized', 'object_stabilized',
                          'supported_contact', 'object_support_contact', 'near', 'near_object', 'near_target'):
                value = values.get(kind)
            elif kind == 'stage_goal':
                relation = predicate.get('goal_relation', stage['goal_relation'])
                if relation == 'lifted':
                    value = values['lifted']
                elif relation == 'supported':
                    value = values['supported_contact']
                elif relation == 'accessible_to_right_arm':
                    anchor = references.get('handover_anchors', {}).get(stage.get('cycle_id'), {}).get('centroid_world_m')
                    if anchor is None:
                        anchor = references.get('stage_relations', {}).get(stage['stage_id'], {}).get('target_end')
                    value = judge.handover(stage['object'], stage['arm'], anchor)
                else:
                    value = None
            else:
                continue  # reference progress/actor command/simulator safety are not sensor judges
            predictions[predicate['id']] = {'kind':kind, 'value':value}
    return predictions


def shadow_job(job, *, config, provenance, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for key in ('report', 'visual_trace'):
        if digest(job[key]['path']) != job[key]['sha256']:
            raise ValueError('shadow input binding mismatch: ' + key)
    runner, snapshot, packages = load_frozen_runner(job['task'])
    solution = packages / ('twin_' + job['task']) / 'solution'
    runner.verify_freeze(solution)
    report = json.loads(Path(job['report']['path']).read_text())
    plan = json.loads((REPO / 'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json').read_text())
    task = runner.task_from_plan(plan, job['task'])
    scene = Path(task['sources']['scene.xml']['path'])
    physics = runner._physics()
    renderer_path = REPO / 'tools/evidence/render_twin_graph_video.py'
    spec = importlib.util.spec_from_file_location('_sensor_shadow_render_inputs', renderer_path)
    render = importlib.util.module_from_spec(spec); sys.modules[spec.name] = render; spec.loader.exec_module(render)
    _, trace = render.load_inputs(Path(job['report']['path']), Path(job['visual_trace']['path']), scene)
    world, model, data, initialization = render.frozen_initial_state(report, Path(job['report']['path']),
        scene, snapshot / 'prove/task_graph/twin_physics.py', solution / 'CANDIDATE_FREEZE.json')
    world._initialized = True  # offline reconstruction, never initialize() or actuator stepping
    world.observer = physics.VisibleRGBD(world, width=640, height=480, camera='overview', min_visible_pixels=12)
    references = json.loads((solution / 'refs/visible_centroids.json').read_text())
    graph = json.loads((solution / 'graph/task_graph.json').read_text())
    stages = {row['stage_id']: row for row in graph['stages']}
    motor = np.load(task['sources']['motor_targets.npz']['path'], allow_pickle=False)
    travel = float(model.numeric('gripper_closing_travel').data[0])
    cal = GapCalibration(2 * travel, (1., -1.), 2 * travel, 0., 'sim_encoder_equivalent_not_physical_jaw_width')
    judge, predictions, packets = SensorJudge(config), [], {}
    stage_start, previous_stage, previous_qpos, previous_time = None, None, None, None
    confirmation_since = {}
    try:
        for index, (t, qpos, stage_id) in enumerate(zip(trace['timestamps'], trace['qpos'], trace['stage_id'])):
            stage = stages[str(stage_id)]
            if previous_stage != stage_id:
                stage_start = float(t) - .1
                previous_stage = stage_id
                confirmation_since.clear()
            data.qpos[:] = qpos
            if previous_qpos is None:
                data.qvel[:] = 0.
            else:
                dt = float(t) - previous_time
                if dt <= 0:
                    raise ValueError('shadow timestamps must increase strictly')
                physics.mujoco.mj_differentiatePos(model, data.qvel, dt, previous_qpos, qpos)
            data.time = float(t)
            physics.mujoco.mj_forward(model, data)
            frame = world.observe_visible(roles=sorted(world.role_geom_ids))
            # Commanded setpoint is reconstructed from the frozen reference clock,
            # never taken from action slots as measured finger feedback.
            record = report['trace'][index]
            clock = record.get('controller', {}).get('reference_clock', {})
            ref_time = clock.get('command_endpoint_reference_time_sec')
            if ref_time is None:
                start_time = float(motor['timestamps'][stage['reference']['start_index']])
                end_time = float(motor['timestamps'][stage['reference']['end_index']])
                ref_time = min(end_time, start_time + float(t) - stage_start)
            commands = np.asarray([np.interp(ref_time, motor['timestamps'], motor['gripper'][:, arm]) for arm in range(2)])
            packet = packet_from_visible(frame, world.proprioception(), commands, {'left': cal, 'right': cal})
            judge.ingest(packet, float(t))
            reference = references.get('stage_relations', {}).get(stage_id, {})
            values = judge.facts(stage['object'], stage['arm'], target=stage['target'],
                relation=stage['goal_relation'], reference_delta=reference.get('object_target_end'))
            if stage['goal_relation'] == 'accessible_to_right_arm':
                anchor = references.get('handover_anchors', {}).get(stage.get('cycle_id'), {}).get('centroid_world_m', reference.get('target_end'))
                values['object_table_support_contact'] = judge.handover(stage['object'], stage['arm'], anchor)
            predicate_values = sensor_predicate_predictions(stage, values, judge, references, confirmation_since)
            predictions.append({'timestamp_sec': float(t), 'stage_id': stage_id,
                'facts': values, 'predicate_values': predicate_values, 'diagnostics': judge.diagnostics,
                'command_source': 'reconstructed_frozen_visible_reference_clock_not_measured_feedback'})
            for role, points in packet.role_points_m.items():
                packets[f'packet{index:05d}_role_{role}'] = points
            for arm, sensors in packet.arms.items():
                for key in ('q', 'qdot', 'tcp_position_m', 'tcp_rotation', 'finger_encoders',
                            'actual_aperture_m', 'commanded_aperture_m', 'timestamp_sec'):
                    packets[f'packet{index:05d}_{arm}_{key}'] = np.asarray(getattr(sensors, key))
            previous_qpos, previous_time = qpos.copy(), float(t)
    finally:
        world.close()
    output.mkdir(parents=True)
    archive = output / 'SENSOR_PACKETS.npz'; np.savez_compressed(archive, **packets)
    seal = output / 'PREDICTIONS_SEALED.json'
    seal.write_text(json.dumps({'schema': 'graph-self-adapter.sensor-shadow-predictions.v1',
        'offline_only': True, 'mj_step_controller_evaluator_executed': False,
        'actual_runtime_provenance': provenance, 'initialization': initialization,
        'source_report': job['report'], 'source_visual_trace': job['visual_trace'],
        'packet_archive': {'path': str(archive), 'sha256': digest(archive)},
        'predictions': predictions}, indent=2, sort_keys=True) + '\n')
    # Only after sealing: compare selected predicate labels from original report.
    known = unknown = agree = 0
    comparison = []
    by_kind = {}
    for pred, original in zip(predictions, report['trace']):
        labels = original.get('boolean_facts', {})
        for key, prediction in pred['predicate_values'].items():
            value, kind = prediction['value'], prediction['kind']
            label = labels.get(key)
            counts = by_kind.setdefault(kind, {'known':0,'unknown':0,'sensor_unknown':0,'stored_missing':0,'agreements':0,'disagreements':0})
            counts['sensor_unknown'] += value is None
            counts['stored_missing'] += label is None
            if value is None or label is None:
                unknown += 1; counts["unknown"] += 1
            else:
                known += 1; agree += value == label
                counts["known"] += 1; counts["agreements"] += value == label; counts["disagreements"] += value != label
            comparison.append({'time_sec': pred['timestamp_sec'], 'predicate': key,
                               'kind':kind, 'sensor': value, 'stored_gt': label})
    result = {'schema': 'graph-self-adapter.sensor-shadow-report.v1', 'task': job['task'],
        'trial': job['trial'], 'prediction_seal': {'path': str(seal), 'sha256': digest(seal)},
        'known_predicates': known, 'unknown_predicates': unknown, 'known_agreements': agree,
        'comparison': comparison, 'by_predicate_kind':by_kind, 'opened_regression_not_fresh_evidence': True}
    (output / 'SHADOW_REPORT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result
