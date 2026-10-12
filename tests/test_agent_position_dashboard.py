"""Synthetic static display fixtures only; never read new episode results."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_position_dashboard_static',ROOT/'tools/evidence/build_project_dashboard.py')
dashboard=importlib.util.module_from_spec(spec);spec.loader.exec_module(dashboard)


def binding(root,path,value):
    file=root/path;file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(value))
    return {'path':path,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}


def fixture(tmp_path,monkeypatch,*,final=False):
    monkeypatch.setattr(dashboard,'REPO_ROOT',tmp_path)
    status={'schema':'graph-self-adapter.agent-position-status.v1','complete_task_episodes':10 if final else 0,'planned_task_episodes':10}
    bindings={'agent_position_status':binding(tmp_path,'status.json',status)}
    report=None
    if final:
        episodes=[{'trial':i,'strict_success':i<7,'graph_completed':i<8,'root_request_count':i+1,'unknown_count':int(i%2==0),
                   'termination_file':binding(tmp_path,f'runs/trial_{i:02d}/terminal.json',{'explicit_synthetic_fixture':True})} for i in range(10)]
        report={'schema':'graph-self-adapter.agent-position-final-report.v1','task':'put_mug_on_box','denominator':10,
                'strict_successes':7,'graph_completed_count':8,'root_request_count':55,'unknown_count':5,'episodes':episodes}
        bindings['agent_position_final_report']=binding(tmp_path,'final.json',report)
    return bindings,report


def test_pending_progress_never_displays_zero_percent_or_final_score(tmp_path,monkeypatch):
    bindings,_=fixture(tmp_path,monkeypatch)
    html=dashboard.render_agent_position(bindings)
    assert '完成 0/10' in html and '单任务' in html and '已打开位置变化回归' in html
    assert '0%' not in html and '严格成功' not in html and '最终结果尚未封存' in html
    assert '历史 8/10' in html and '不是盲评或因果归因' in html
    assert dashboard.render_agent_position({})==''


def test_final_statistics_are_recomputed_from_episodes(tmp_path,monkeypatch):
    bindings,_=fixture(tmp_path,monkeypatch,final=True)
    html=dashboard.render_agent_position(bindings)
    assert '<td>7/10</td>' in html and '<td>8/10</td>' in html and '<td>55</td>' in html and '<td>5</td>' in html
    assert '0%' not in html


@pytest.mark.parametrize('case',['successes','graph','requests','unknown','trial','task','denominator','boolean','source_hash','status_hash'])
def test_invalid_bound_statistics_and_sources_are_rejected(tmp_path,monkeypatch,case):
    bindings,report=fixture(tmp_path,monkeypatch,final=True)
    if case=='successes':report['strict_successes']=8
    if case=='graph':report['graph_completed_count']=9
    if case=='requests':report['root_request_count']=54
    if case=='unknown':report['unknown_count']=6
    if case=='trial':report['episodes'][1]['trial']=0
    if case=='task':report['task']='put_knife_board_new'
    if case=='denominator':report['denominator']=9
    if case=='boolean':report['episodes'][0]['strict_success']=1
    if case=='source_hash':report['episodes'][0]['termination_file']['sha256']='changed'
    if case=='status_hash':bindings['agent_position_status']['sha256']='changed'
    bindings['agent_position_final_report']=binding(tmp_path,'final.json',report)
    with pytest.raises(ValueError):dashboard.render_agent_position(bindings)
