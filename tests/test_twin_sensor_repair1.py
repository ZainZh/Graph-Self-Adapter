from dataclasses import replace
import numpy as np
from prove.task_graph.twin_sensor_judge import ArmSensors, SensorPacket
from prove.task_graph.twin_sensor_judge_repair1 import RepairJudge


def packet(t, actual=.015, command=0., obj=(0.,0.,0.), tcp=(0.,0.,0.)):
    arm=ArmSensors(np.zeros(6),np.zeros(6),np.array(tcp),np.eye(3),np.zeros(2),
        actual,command,'test_calibration',.08,0.,t,'measured_encoder')
    return SensorPacket(t,t,{'obj':np.tile(obj,(10,1)),'target':np.tile([0.,0.,-.03],(10,1))},{'left':arm},{})


def read(j,p,**kwargs):
    j.ingest(p,p.timestamp_sec);j.expire(p.timestamp_sec)
    return j.facts('obj','left',**kwargs)


def test_persistent_obstruction_retains_compliant_grasp_and_caches_repeat_reads():
    j=RepairJudge()
    for index,gap in enumerate((.017,.0155,.0126,.011,.010)):
        result=read(j,packet(index*.1,actual=gap))
    assert result['holding'] is True
    assert j.diagnostics['holding']['physical_contact_verified'] is False
    assert j.diagnostics['holding']['aperture_change_is_hard_detachment_veto'] is False
    assert j.facts('obj','left')['holding'] is True


def test_centroid_motion_diagnostic_cannot_veto_obstructed_visible_grasp():
    j=RepairJudge()
    read(j,packet(0));read(j,packet(.1))
    assert read(j,packet(.2,tcp=(.015,0,0)))['holding'] is True
    assert 'provisional' in j.diagnostics['holding']['evidence_strength']
    assert j.diagnostics['holding']['centroid_motion_is_hard_detachment_veto'] is False
    assert read(j,packet(.3,tcp=(.1,0,0)))['holding'] is False
    assert read(j,packet(.4,actual=0.,command=0.))['holding'] is False


def test_relative_actual_opening_release_requires_acquisition_and_settled_relation():
    j=RepairJudge();kwargs={'target':'target','relation':'supported','reference_delta':[0,0,.03]}
    for t in (0.,.1,.2):read(j,packet(t),**kwargs)
    # .020 is well below .8 full opening (.064), but exceeds last blocked .015 + .003.
    for t in (.3,.4,.5,.6,.7):result=read(j,packet(t,actual=.020,command=.030),**kwargs)
    assert result['released'] is True and result['supported_contact'] is True
    empty=RepairJudge()
    for t in (0.,.1,.2,.3):result=read(empty,packet(t,actual=.020,command=.030),**kwargs)
    assert result['released'] is False and result['supported_contact'] is False


def test_missing_placeholder_or_stopped_stream_remains_unknown():
    j=RepairJudge();read(j,packet(0));read(j,packet(.2))
    j.expire(.5)
    assert j.facts('obj','left')['holding'] is None
    p=packet(.6);p=replace(p,arms={'left':replace(p.arms['left'],measurement_source='command_placeholder')})
    assert read(j,p)['holding'] is None
    p=packet(.7);p=replace(p,role_points_m={})
    assert read(j,p)['holding'] is None


def test_inside_nested_hanging_stay_unknown_after_repair():
    j=RepairJudge()
    for t in (0.,.1,.2):read(j,packet(t))
    for t in (.3,.4,.5):read(j,packet(t,actual=.020,command=.03))
    for relation in ('inside','nested_with_support','hanging'):
        assert j.facts('obj','left',target='target',relation=relation,reference_delta=[0,0,.03])['supported_contact'] is None


def test_relative_opening_with_jaw_still_blocked_cannot_claim_release():
    j=RepairJudge()
    for t in (0.,.1,.2):read(j,packet(t))
    for t in (.3,.4,.5,.6):result=read(j,packet(t,actual=.028,command=.020),target='target',relation='supported',reference_delta=[0,0,.03])
    assert result['released'] is False and result['supported_contact'] is False


def test_out_of_calibrated_range_or_missing_actual_aperture_is_unknown():
    j=RepairJudge()
    result=read(j,packet(0.,actual=.20,command=0.))
    assert result['holding'] is None
    assert j.diagnostics['holding']['reason']=='calibrated_aperture_range_or_missing_nonfinite'
    result=read(j,packet(.1,actual=.015,command=.20))
    assert result['released'] is None
    p=packet(.2);p=replace(p,arms={'left':replace(p.arms['left'],actual_aperture_m=None)})
    assert read(j,p)['holding'] is None


def test_repair_evaluator_variant_sealed_before_private_evidence(tmp_path):
    import json
    from types import SimpleNamespace
    from prove.task_graph.twin_sensor_adapter_repair1 import repair_world_class
    from prove.task_graph.twin_sensor_judge_repair1 import RepairConfig
    path=tmp_path/'seal.json'
    class Base:
        def __init__(self):
            self.model=SimpleNamespace(numeric=lambda _:SimpleNamespace(data=[.037]))
        def contact_evidence(self):
            return {'roles':{'original':True}}
        def finalize_evaluator(self,outcome):
            assert json.loads(path.read_text())['execution_variant']=='sensor_only_repair1'
            assert self.contact_evidence()['roles']['original']
            return {'original_evaluator':True}
    physics=SimpleNamespace(TwinWorld=Base)
    cls=repair_world_class(physics,config=RepairConfig(),references={},judge_mode='sensor_only',seal_path=path)
    world=cls()
    assert world.contact_evidence()['roles']=={}
    assert world.finalize_evaluator({})['original_evaluator']
    assert world.contact_evidence()['roles']=={}


def test_packet_replay_recomputes_actual_from_finger_measurement_not_command(tmp_path):
    from tools.task_graph.twin_sensor_packet_replay_repair1 import robot_encoder_calibration,packet_from_archive
    xml=tmp_path/'robot.xml';xml.write_text('<mujoco><custom><numeric name="gripper_closing_travel" data=".037"/></custom></mujoco>')
    cal=robot_encoder_calibration(xml)
    values={'packet00000_role_obj':np.zeros((10,3))}
    for arm in ('left','right'):
        for name,value in {'q':np.zeros(6),'qdot':np.zeros(6),'tcp_position_m':np.zeros(3),'tcp_rotation':np.eye(3),
                           'finger_encoders':np.array([-.01,.01]),'actual_aperture_m':999.,
                           'commanded_aperture_m':.01,'timestamp_sec':0.}.items():
            values['packet00000_'+arm+'_'+name]=value
    path=tmp_path/'public.npz';np.savez_compressed(path,**values)
    with np.load(path,allow_pickle=False) as archive:
        result=packet_from_archive(archive,0,cal)
    assert abs(result.arms['left'].actual_aperture_m-.054)<1e-9
    assert result.arms['left'].commanded_aperture_m==.01
    assert 'not_physical_inner_jaw_gap' in result.arms['left'].calibration_source


def test_repair_post_binding_check_retains_mismatch_and_missing_without_losing_batch(tmp_path):
    from tools.task_graph.twin_sensor_validate_repair1 import post_bindings
    path=tmp_path/'source';path.write_text('changed')
    rows=post_bindings([{'path':str(path),'sha256':'old'},{'path':str(tmp_path/'missing'),'sha256':'old'}])
    assert len(rows)==2 and all(row['matches'] is False for row in rows)
    assert rows[1]['error'].startswith('FileNotFoundError:')
