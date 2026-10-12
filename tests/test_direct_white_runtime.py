import sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
from direct_white_geometry import upright_support_prior,direct_targets,tcp_at_handle
from direct_white_route import direct_route_graph
from direct_white_runtime import DirectExecutor


def test_upright_relationship_and_horizontal_transport_not_airborne_attachment():
    cloud=np.c_[np.linspace(0,.08,100),np.zeros(100),np.linspace(.03,.11,100)];table=cloud.copy();table[:,2]=.03
    prior=upright_support_prior(cloud,np.array([-.03,0,.09]),table);box=cloud.copy();box[:,2]=.15
    targets=direct_targets(prior,box,[.1,-.5,.15],[.16,-.48,.2],[.1,-.5,.15])
    assert np.allclose(targets['white_visible_goal_xy_m'],[.16,-.48])
    assert np.isclose(targets['handle_support_goal_m'][2],.15+.06)
    assert targets['transport_handle_height_m']>targets['observed_box_top_z_m']+prior['visible_extent_radius_m']
    point=np.array(targets['handle_support_goal_m']);pose=tcp_at_handle(point,np.eye(3),[0,0,-.03]);assert np.allclose(pose[0]+pose[1]@[0,0,-.03],point)


def test_new_route_red_actions_unchanged_white_no_regrasp_and_complete_after_retreat():
    original=json.loads((ROOT/'evaluations/xtrainer_laptop_v01/portable/cup/graph/task_graph.json').read_text());route=direct_route_graph(original)
    red=[s['stage_id'] for s in route['stages'] if s['cycle_id'] in ('c01','c02') and s['phase']!='Complete']
    assert red==[s['stage_id'] for s in original['stages'] if s['cycle_id'] in ('c01','c02')]
    white=[s for s in route['stages'] if s['object']=='white_mug'];assert [s['phase'] for s in white]==['Approach','Grasp','Lift','Raise','Move','Lower','Release','Retreat','Complete']
    assert all(s['arm']=='left' and s['target']=='box' for s in white) and len(route['stages'])==25
    assert sum(s['phase']=='Grasp' for s in route['stages'])==3 and sum(s['phase']=='Complete' for s in route['stages'])==3


def test_unknown_real_aperture_allows_only_command_profile_progress_not_measured_release(tmp_path):
    executor=DirectExecutor.__new__(DirectExecutor);stage=next(s for s in direct_route_graph(json.loads((ROOT/'evaluations/xtrainer_laptop_v01/portable/cup/graph/task_graph.json').read_text()))['stages'] if s['cycle_id']=='white_direct' and s['phase']=='Release')
    executor.stage=stage;executor._last_cartesian_goal=(np.zeros(3),np.eye(3));executor._shared_pose=lambda *a:(np.zeros(3),np.eye(3));executor.command_g=np.ones(2);executor.initial_state_diagnostics={'actual_aperture_m':[None,None]};executor.stage_started=0.;executor.clock=lambda:5.
    samples=[SimpleNamespace(q=np.zeros(6),aperture=None)]*2;facts=executor.predicates(stage,1.,np.zeros((2,6)),samples,None)
    assert facts['own_progress'] and executor.release_progress_record['sensor_measured'] is False
    assert 'unknown' in executor.release_progress_record['criterion'] and 'released' not in facts
