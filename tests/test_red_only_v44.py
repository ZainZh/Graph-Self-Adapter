import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from api_v44_red_entry import route,extended_code_route
from prove.xtrainer.runtime import load_bundle

def test_single_red_contract_never_executes_white():
    g=route(load_bundle('put_mug_on_box')['graph'])
    assert g['logical_task']=='put_red_mug_on_box';assert len(g['stages'])==17
    assert all(s['object']=='red_mug' for s in g['stages'])
    assert not any(s['stage_id'].startswith('white') for s in g['stages'])
    assert g['stages'][-1]['phase']=='Complete';assert g['stages'][-1]['next'] is None
    report=extended_code_route(g);assert report['graph_nodes']==17
    assert len(report['code_trajectory_stage_ids'])==11
    assert len(report['precision_closed_loop_stage_ids'])==4
    assert len(report['semantic_boundary_stage_ids'])==2
    assert report['weights_modified'] is False
