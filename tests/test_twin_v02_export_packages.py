import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v02_export_test',ROOT/'tools/task_graph/twin_v02_export_packages.py')
exporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(exporter)
SELECTION=ROOT/'evaluations/twin_rsi_v02/FINAL_SELECTION.json'


def test_export_preserves_selection_and_defaults_without_rollout(tmp_path,monkeypatch):
    root=tmp_path/'selected';result=exporter.export(SELECTION,root)
    assert result['tasks']==3
    assert (root/'provenance/FINAL_SELECTION.json').read_bytes()==SELECTION.read_bytes()
    manifest=exporter.load(root/'DERIVED_RUNTIME_MANIFEST.json')
    assert manifest['derived_paths_only'] and manifest['default_execution_mode']=='fixed_z'
    assert {r['task']:r['candidate_id'] for r in manifest['tasks']}==exporter.TASK_CANDIDATES
    sorting=next(r for r in manifest['tasks'] if r['task']=='sorting_cube_bimanual')
    assert 'research reuse only' in sorting['qualification'] and 'not qualified as an improvement' in sorting['qualification']
    def forbidden(*a,**kw):raise AssertionError('dry-run must never start runner')
    monkeypatch.setattr(exporter.subprocess,'run',forbidden)
    for task in exporter.TASK_CANDIDATES:
        args=SimpleNamespace(export_root=root,task=task,trial=0,output=tmp_path/task,python=None,online_z=False,dry_run=True)
        report=exporter.launch(args)
        assert report['controller']=='fixed_z' and report['dry_run']
        assert '--online-z' not in report['command']
        assert report['command'][report['command'].index('--condition-seed')+1]=='51005'
        assert report['command'][report['command'].index('--condition-set')+1]=='development_v02'
    args.online_z=True
    assert exporter.launch(args)['controller']=='online_z_research_ablation'
    args.output.mkdir()
    with pytest.raises(FileExistsError):exporter.launch(args)


def test_export_detects_frozen_model_tampering(tmp_path):
    root=tmp_path/'selected';exporter.export(SELECTION,root)
    model=root/'tasks/twin_put_knife_board_new/solution/models/local_skill.npz'
    model.write_bytes(model.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='export file mismatch'):exporter.verify(root)
