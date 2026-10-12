from dataclasses import replace
import numpy as np
from prove.task_graph.twin_sensor_judge import ArmSensors, SensorPacket, SensorJudge, rgbd_points


def packet(t, *, obj=(0, 0, 0), tcp=(0, 0, 0), actual=.04, command=.02, target=(0, 0, -.03)):
    arm = ArmSensors(np.zeros(6), np.zeros(6), np.asarray(tcp, float), np.eye(3),
                     np.zeros(2), actual, command, 'measured_test_calibration', .08, 0., t, 'measured_encoder')
    return SensorPacket(t, t, {'object': np.tile(obj, (10, 1)), 'target': np.tile(target, (10, 1))},
                        {'left': arm}, {'mask': 'external_segmentation', 'geometry': 'RGB-D', 'proprio': 'encoder_FK'})


def read(judge, p, **kwargs):
    judge.ingest(p, p.received_sec)
    return judge.facts('object', 'left', **kwargs)


def test_holding_gap_debounce_zero_motion_is_weaker_not_comotion():
    j = SensorJudge()
    assert read(j, packet(0))['holding'] is False
    p = packet(.1)
    assert read(j, p)['holding'] is False
    assert read(j, p)['holding'] is False  # duplicate cannot advance clock
    assert read(j, packet(.2))['holding'] is True
    assert j.diagnostics['holding']['evidence_strength'] == 'gap_obstruction_visible_vicinity_only'
    assert read(j, packet(.3, obj=(.01, 0, 0), tcp=(.01, 0, 0)))['holding'] is True
    assert j.diagnostics['holding']['evidence_strength'] == 'motion_verified'


def test_far_and_noncomoving_static_object_cannot_establish_holding():
    j = SensorJudge()
    for t in (0, .1, .2):
        assert read(j, packet(t, obj=(.3, 0, 0)))['holding'] is False
    j = SensorJudge()
    read(j, packet(0)); read(j, packet(.1))
    assert read(j, packet(.2, tcp=(.01, 0, 0)))['holding'] is False


def test_stale_out_of_order_missing_unknown_and_reset():
    j = SensorJudge()
    read(j, packet(0)); assert read(j, packet(.2))['holding'] is True
    assert j.ingest(packet(.1), .2) is False
    assert j.facts('object', 'left')['holding'] is None
    p = packet(.3)
    assert j.ingest(p, .7) is False
    assert j.facts('object', 'left')['holding'] is None
    read(j, packet(.8))
    assert j.facts('absent', 'left')['holding'] is None
    assert read(j, packet(.9))['holding'] is False


def test_release_settling_visible_support_proxy_and_unobservable_exact_relations():
    j = SensorJudge()
    kwargs = dict(target='target', relation='supported', reference_delta=[0, 0, .03])
    read(j, packet(-.3)); read(j, packet(-.1))
    assert read(j, packet(0, actual=.08, command=.08), **kwargs)['supported_contact'] is False
    read(j, packet(.1, actual=.08, command=.08), **kwargs)
    read(j, packet(.3, actual=.08, command=.08), **kwargs)
    read(j, packet(.5, actual=.08, command=.08), **kwargs)
    assert read(j, packet(.7, actual=.08, command=.08), **kwargs)['supported_contact'] is True
    for relation in ('inside', 'nested_with_support', 'hanging'):
        assert j.facts('object', 'left', target='target', relation=relation,
                       reference_delta=[0, 0, .03])['supported_contact'] is None
    assert read(j, packet(.8, actual=.04, command=.02), **kwargs)['supported_contact'] is False


def test_depth_mask_unprojection_and_invalid_depth_missing():
    points = rgbd_points(np.ones((3, 3)), {'obj': np.ones((3, 3), bool)}, np.eye(3), np.eye(4))
    assert np.allclose(points['obj'][4], [1, 1, 1])
    assert rgbd_points(np.full((3, 3), np.nan), {'obj': np.ones((3, 3), bool)}, np.eye(3), np.eye(4)) == {}


def test_multiframe_small_motion_rotation_and_cumulative_settling_drift():
    j = SensorJudge()
    for index in range(4):
        pos = (.003 * index, 0, 0)
        result = read(j, packet(index * .1, obj=pos, tcp=pos))
    assert result['holding'] is True
    assert j.diagnostics['holding']['evidence_strength'] == 'motion_verified'
    j = SensorJudge()
    read(j, packet(0, obj=(.03, 0, 0)))
    theta = .3
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    p = packet(.2, obj=(.03 * np.cos(theta), .03 * np.sin(theta), 0))
    p = replace(p, arms={'left': replace(p.arms['left'], tcp_rotation=rotation)})
    assert read(j, p)['holding'] is True
    assert j.diagnostics['holding']['evidence_strength'] == 'motion_verified'
    j = SensorJudge()
    for index in range(8):
        result = read(j, packet(index * .1, obj=(index * .004, 0, 0), actual=.08, command=.08),
                      target='target', relation='supported', reference_delta=[0, 0, .03])
        assert result['stabilized'] is False
        assert result['supported_contact'] is False


def test_missing_role_resets_only_its_history_and_stale_gap_does_not_compare():
    j = SensorJudge()
    read(j, packet(0)); read(j, packet(.1))
    p = packet(.2)
    p = replace(p, role_points_m={'object': p.role_points_m['object']})
    j.ingest(p, .2)
    assert j.facts('target', 'left')['holding'] is None
    assert j.facts('object', 'left')['holding'] is True
    assert read(j, packet(1., obj=(.01, 0, 0), tcp=(.01, 0, 0)))['holding'] is False
    assert j.diagnostics['holding']['evidence_strength'] != 'motion_verified'


def test_malformed_and_stale_robot_or_command_placeholder_feedback_unknown():
    j = SensorJudge()
    p = packet(0)
    malformed = replace(p, arms={'left': replace(p.arms['left'], tcp_position_m=np.zeros(2))})
    assert read(j, malformed)['holding'] is None
    assert j.diagnostics['holding']['reason'] == 'malformed_robot_or_calibration'
    p = packet(.1)
    stale = replace(p, arms={'left': replace(p.arms['left'], timestamp_sec=0.)})
    assert read(j, stale)['holding'] is None
    assert j.diagnostics['holding']['reason'] == 'stale_or_unsynchronized_robot_sample'
    p = packet(.2)
    placeholder = replace(p, arms={'left': replace(p.arms['left'], measurement_source='command_placeholder')})
    result = read(j, placeholder)
    assert result['near'] is True and result['holding'] is None and result['released'] is None
    assert j.diagnostics['holding']['reason'] == 'actual_gripper_feedback_missing_or_command_placeholder'
    for index in range(3, 7):
        assert read(j, packet(index * .1, actual=.02, command=.02))['holding'] is False


def test_sensor_judge_uses_visible_jaw_points_not_elongated_object_centroid():
    j = SensorJudge()
    for t in (0, .1, .2):
        p = packet(t, obj=(.2, 0, 0))
        points = p.role_points_m['object'].copy()
        points[0] = [0, 0, .02]  # observed functional part close to jaw; center remains far
        p = replace(p, role_points_m={**p.role_points_m, 'object': points})
        result = read(j, p)
    assert result['near'] is True and result['holding'] is True
    assert j.diagnostics['bilateral_contact']['legacy_contact_name_is_sensor_alias_only']


def test_window_rollover_repeated_reads_keep_identical_basic_motion_evidence():
    j = SensorJudge()
    for index in range(7):
        p = packet(index * .1, obj=(index * .003, 0, 0), tcp=(index * .003, 0, 0))
        result = read(j, p)
    first_strength = j.diagnostics['holding']['evidence_strength']
    for relation in ('supported', 'inside', 'lifted', None):
        again = j.facts('object', 'left', target='target', relation=relation, reference_delta=[0,0,.03])
        assert again['holding'] == result['holding']
        assert again['stabilized'] == result['stabilized']
        assert j.diagnostics['holding']['evidence_strength'] == first_strength


def test_stopped_sensor_stream_expires_and_duplicate_cannot_refresh_timestamp():
    j=SensorJudge()
    read(j,packet(0.)); last=packet(.2)
    assert read(j,last)['holding'] is True
    assert not j.expire(.5)
    assert j.facts('object','left')['holding'] is None
    assert not j.ingest(last,.5)
    assert j.facts('object','left')['holding'] is None
    assert read(j,packet(.6))['holding'] is False
