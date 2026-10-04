"""Synthetic offline fixtures; no simulator or formal evaluator artifacts."""
import json

import h5py
import numpy as np
import pytest

from tools.task_graph.inspect_demo import inspect_demo, sha256_file
from tools.task_graph.export_demo_evidence import action_events, export_demo


def fixture(path):
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        data.attrs['env_args'] = '{"controller":"OSC_POSE"}'
        g = data.create_group('demo_0')
        g.create_dataset('actions', data=[[0.2, 0, 0, -1], [0.2, 0, 0, 1], [-0.2, 0, 0, 1]])
        g.create_dataset('states', data=np.zeros((3, 25)))
        g.create_dataset('obs/ee_pos', data=np.ones((3, 3)))
        g.create_dataset('obs/agentview_rgb', data=np.zeros((3, 8, 8, 3), dtype=np.uint8))


def test_event_indices_and_zero_motion():
    actions = np.array([[.2, 0, 0, -1], [.2, 0, 0, 1], [-.2, 0, 0, 1], [0, 0, 0, 1]])
    assert action_events(actions) == [
        {'frame': 1, 'reasons': ['gripper_command_sign_switch']},
        {'frame': 2, 'reasons': ['translation_direction_change']}]
    with pytest.raises(ValueError):
        action_events(np.array([[float('nan'), 0, 0, 1]]))


def test_inventory_and_export(tmp_path):
    source = tmp_path / 'demo.hdf5'
    fixture(source)
    inv = inspect_demo(source)
    assert inv['datasets']['data/demo_0/states']['shape'] == [3, 25]
    args = dict(group='data/demo_0', source_kind='human_teleoperation', provenance='synthetic fixture',
                coordinate_conventions='synthetic arbitrary units', frames=[2, 1, 1], start=1)
    out = tmp_path / 'evidence'
    manifest = export_demo(source, out, **args)
    assert manifest['source']['sha256'] == sha256_file(source)
    assert manifest['image_evidence']['frames'] == [1, 2]
    assert manifest['heuristic_events'][0]['frame'] == 2
    payload = json.loads((out / 'ACTION_PROPRIO.json').read_text())
    assert payload['frame_indices'] == [1, 2]
    assert 'states' not in payload
    assert payload['proprioception']['ee_pos'] == [[1., 1., 1.], [1., 1., 1.]]
    assert (out / 'CONTACT_SHEET.png').is_file()
    with pytest.raises(FileExistsError):
        export_demo(source, out, **args)
    with pytest.raises(ValueError, match='outside'):
        export_demo(source, tmp_path / 'invalid', **dict(args, frames=[0]))
    second = tmp_path / 'second'
    assert export_demo(source, second, **args) == manifest


def test_explicit_annotation_boundaries_and_outcome():
    from tools.task_graph.render_demo_graph import validate_annotations, frame_label
    annotations = {'stages': [{'start': 0, 'stop': 2, 'stage': 'Move', 'roles': 'book'},
                              {'start': 2, 'stop': 3, 'stage': 'Insert'}]}
    stages = validate_annotations(annotations, 3)
    assert 'Move' in frame_label(1, 3, stages)[2]
    assert 'Insert' in frame_label(2, 3, stages)[2]
    assert not any('outcome' in line for line in frame_label(1, 3, stages))
    assert 'unverified' in frame_label(2, 3, stages)[-1]
    with pytest.raises(ValueError, match='contiguous'):
        validate_annotations({'stages': [{'start': 1, 'stop': 3, 'stage': 'Move'}]}, 3)


def test_chinese_annotation_outcome_only_at_end():
    from tools.task_graph.render_demo_graph import frame_label
    stages = [{'start': 0, 'stop': 3, 'stage': 'Insert'}]
    before = frame_label(1, 3, stages, 'zh')
    final = frame_label(2, 3, stages, 'zh')
    assert before[0] == '官方单条演示；未调用Fast Adapter'
    assert not any('源终点标记' in line for line in before)
    assert final[-1] == '尚未验证物理成功；源终点标记不能作成功依据'
    assert '非销孔插入技能' in final[2]


def test_review_page_omits_missing_links(tmp_path):
    from tools.task_graph.build_review_page import build_page
    root = tmp_path / 'experiment'
    output = root / 'presentation/index.html'
    build_page(root, output)
    page = output.read_text()
    assert '根报告尚未生成' in page
    assert 'href="../ROOT_REPORT.json"' not in page
    assert '自动链路尚未验证' in page
    with pytest.raises(FileExistsError):
        build_page(root, output)
