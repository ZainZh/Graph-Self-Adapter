from types import SimpleNamespace
import numpy as np
from prove.task_graph.p70_stage_judge import independent_facts


def fixture(attached=True):
    report=SimpleNamespace(facts=dict(attached=attached,above_grasp_approach=True,grasp_ready=False,clear_of_table=True,object_lost=False),diagnostics={})
    B=np.eye(4);B[2,3]=1.
    private=dict(world_book=B,world_caddy=np.eye(4),finger_contacts=(True,True),wrong_target=False,official_success=False)
    calibration=dict(book_collision_center=[0,0,0],book_collision_halfsize=[.01,.02,.03],back_halfsize=[.1,.1,.1],tolerances=dict(angle_rad=.1,vicinity_position=.03))
    return report,private,calibration,np.eye(4),dict(near_ready=False),.5


def test_independent_stage_facts_and_no_contact_alias():
    args=fixture();facts,diag=independent_facts(*args)
    assert facts['grasp_ready'] and not facts['at_grasp_pose']
    assert facts['attached'] and facts['lift_clear'] and facts['orient_clear']
    assert facts['above_destination_vicinity'] and not facts['near_entry']
    assert all(type(v) is bool for v in facts.values())
    args[1]['world_book'][0,3]=.2
    facts,_=independent_facts(*args)
    assert facts['orient_clear'] and not facts['above_destination_vicinity']
    args[1]['world_book'][:3,:3]=np.diag([-1.,-1.,1.])
    facts,_=independent_facts(*args)
    assert not facts['orient_clear']


def test_unknown_attachment_cannot_pass_held_or_near_gates():
    args=fixture(None);args[-2]['near_ready']=True
    facts,_=independent_facts(*args)
    assert facts['bilateral_contact'] and not facts['attached']
    assert not any(facts[k] for k in ('lift_clear','orient_clear','above_destination_vicinity','near_entry'))


def test_old_clearance_cannot_skip_frozen_safe_rotation_height():
    args=fixture();args[1]['world_book'][2,3]=.4
    facts,diag=independent_facts(*args)
    assert args[0].facts['clear_of_table'] is True and facts['attached']
    assert not diag['safe_clearance']
    assert not any(facts[k] for k in ('lift_clear','orient_clear','above_destination_vicinity'))
