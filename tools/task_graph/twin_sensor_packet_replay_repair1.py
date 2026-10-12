"""Repair predictions from ALREADY SEALED public packets; no simulator imports."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from prove.task_graph.twin_sensor_judge import ArmSensors, SensorPacket, GapCalibration
from prove.task_graph.twin_sensor_judge_repair1 import RepairJudge
from tools.task_graph.twin_sensor_rsi import REPO, digest, load_frozen_runner
from tools.task_graph.twin_sensor_shadow import sensor_predicate_predictions


def robot_encoder_calibration(scene_path):
    node=ET.parse(scene_path).find(".//numeric[@name='gripper_closing_travel']")
    if node is None:
        raise ValueError('explicit robot closing-travel calibration missing')
    travel=float(node.attrib['data'])
    if not np.isfinite(travel) or travel <= 0:
        raise ValueError('invalid robot closing-travel calibration')
    return GapCalibration(2*travel,(1.,-1.),2*travel,0.,
        'explicit_robot_numeric_closing_travel_encoder_equivalent_not_physical_inner_jaw_gap')


def packet_from_archive(archive,index,cal):
    prefix=f'packet{index:05d}_'
    points={key[len(prefix+'role_'):]:archive[key] for key in archive.files if key.startswith(prefix+'role_')}
    arms={}
    for arm in ('left','right'):
        def get(key):return archive[prefix+arm+'_'+key]
        fingers=get('finger_encoders')
        timestamp=float(get('timestamp_sec'))
        arms[arm]=ArmSensors(get('q'),get('qdot'),get('tcp_position_m'),get('tcp_rotation'),fingers,
            cal.aperture(fingers),float(get('commanded_aperture_m')),cal.provenance,cal.open_gap_m,cal.closed_gap_m,
            timestamp,'simulator_actual_joint_encoder')
    timestamp=arms['left'].timestamp_sec
    return SensorPacket(timestamp,timestamp,points,arms,{
        'mask':'sealed_simulator_GT_semantic_mask_proxy','geometry':'sealed_public_RGBD_pointcloud',
        'robot':'sealed_actual_simulator_finger_encoders_and_FK',
        'command':'separately_archived_commanded_aperture_not_used_as_measurement',
        'calibration':cal.provenance})


def replay_job(job, *,config,provenance,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    seal_binding=job['initial_prediction_seal']
    if digest(seal_binding['path'])!=seal_binding['sha256']:raise ValueError('initial shadow seal mismatch')
    initial=json.loads(Path(seal_binding['path']).read_text())
    packet_binding=initial['packet_archive']
    if digest(packet_binding['path'])!=packet_binding['sha256']:raise ValueError('sealed public packet hash mismatch')
    runner,snapshot,packages=load_frozen_runner(job['task'])
    solution=packages/('twin_'+job['task'])/'solution';runner.verify_freeze(solution)
    graph=json.loads((solution/'graph/task_graph.json').read_text())
    refs=json.loads((solution/'refs/visible_centroids.json').read_text())
    stages={stage['stage_id']:stage for stage in graph['stages']}
    plan=json.loads((REPO/'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json').read_text())
    task=runner.task_from_plan(plan,job['task']);scene=Path(task['sources']['scene.xml']['path'])
    cal=robot_encoder_calibration(scene)
    judge=RepairJudge(config);predictions=[];confirmation={};last_stage=None
    with np.load(packet_binding['path'],allow_pickle=False) as archive:
        for index,old_prediction in enumerate(initial['predictions']):
            packet=packet_from_archive(archive,index,cal)
            judge.ingest(packet,packet.received_sec);judge.expire(packet.received_sec)
            stage=stages[old_prediction['stage_id']]
            if stage['stage_id']!=last_stage:confirmation.clear();last_stage=stage['stage_id']
            ref=refs['stage_relations'].get(stage['stage_id'],{})
            facts=judge.facts(stage['object'],stage['arm'],target=stage['target'],relation=stage['goal_relation'],
                              reference_delta=ref.get('object_target_end'))
            if stage['goal_relation']=='accessible_to_right_arm':
                anchor=refs.get('handover_anchors',{}).get(stage.get('cycle_id'),{}).get('centroid_world_m',ref.get('target_end'))
                facts['object_table_support_contact']=judge.handover(stage['object'],stage['arm'],anchor)
            values=sensor_predicate_predictions(stage,facts,judge,refs,confirmation)
            predictions.append({'timestamp_sec':packet.timestamp_sec,'stage_id':stage['stage_id'],
                                'predicate_values':values,'facts':facts,'diagnostics':judge.diagnostics})
    output.mkdir(parents=True)
    sealed=output/'REPAIR_PREDICTIONS_SEALED.json'
    sealed.write_text(json.dumps({'schema':'graph-self-adapter.sensor-repair-packet-predictions.v1',
        'execution_variant':'sensor_only_repair1','no_simulation_or_collection':True,
        'initial_prediction_seal':seal_binding,'initial_public_packets':packet_binding,
        'calibration':{'offset_m':cal.offset_m,'encoder_gains':cal.encoder_gains,
                       'open_gap_m':cal.open_gap_m,'closed_gap_m':cal.closed_gap_m,'provenance':cal.provenance,
                       'robot_config':{'path':str(scene),'sha256':digest(scene)}},
        'actual_runtime_provenance':provenance,'predictions':predictions},indent=2,sort_keys=True)+'\n')
    # Stored privileged labels are read only AFTER repair prediction sealing.
    source=initial['source_report']
    if digest(source['path'])!=source['sha256']:raise ValueError('original sealed label report mismatch')
    report=json.loads(Path(source['path']).read_text());by_kind={};comparison=[]
    for prediction,original in zip(predictions,report['trace']):
        for key,predicate in prediction['predicate_values'].items():
            kind,value=predicate['kind'],predicate['value'];label=original['boolean_facts'].get(key)
            counts=by_kind.setdefault(kind,{'denominator':0,'known':0,'sensor_unknown':0,'stored_missing':0,
                'true_positive':0,'true_negative':0,'false_positive':0,'false_negative':0})
            counts['denominator']+=1;counts['sensor_unknown']+=value is None;counts['stored_missing']+=label is None
            if value is not None and label is not None:
                counts['known']+=1
                counts[('true_' if value==label else 'false_')+('positive' if value else 'negative')]+=1
            comparison.append({'time_sec':prediction['timestamp_sec'],'predicate':key,'kind':kind,'sensor':value,'stored_gt':label})
    result={'schema':'graph-self-adapter.sensor-repair-packet-report.v1','task':job['task'],'trial':job['trial'],
        'prediction_seal':{'path':str(sealed),'sha256':digest(sealed)},'by_predicate_kind':by_kind,
        'comparison':comparison,'opened_public_packet_regression_only':True,'new_physical_steps':0}
    (output/'PACKET_REPLAY_REPORT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result
