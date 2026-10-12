import json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
from prove.xtrainer.multitask_bundle import load_bundle
from prove.xtrainer.io import SafetyError
from multitask_ordinary_route import ordinary_route,role_contract
import multitask_ordinary_runtime as runtime

@pytest.mark.parametrize('task,nodes',[('collect_apple',8),('put_knife_board_new',8),('sort_two_spoon',16),('collect_cubes',40),('sorting_cube_bimanual',32),('put_mug_on_box',32)])
def test_actual_frozen_per_task_bundles_and_routes(task,nodes):
    b=load_bundle(task);g=ordinary_route(b['graph'])
    assert b['task']==task and len(g['stages'])==nodes and b['skill'].model.input_dim==29
    assert len(b['initial_q'])==2 and b['qcurve'](b['times'][0]).shape==(2,6)
    for complete in (s for s in g['stages'] if s['phase']=='Complete'):
        index=g['stages'].index(complete);assert g['stages'][index-1]['phase']=='Wait'
        original=next(s for s in b['graph']['stages'] if s['stage_id']==g['stages'][index-1]['stage_id'])
        assert complete['goal_relation']==original['goal_relation'] and complete['target']==original['target']
    for stage in g['stages']:
        if stage['phase'] in ('Lift','Move','Align'):assert 'tentative_holding' in stage['invariants'] and 'placement_complete' not in stage['done']
    roles=role_contract(g,b['refs']);assert roles['SAM_roles']

@pytest.mark.parametrize('task,phase',[('hanging_mug','Insert'),('open_drawer_and_place_object','Pull'),('stack_cups','Insert'),('sweep_objects','Push')])
def test_specialist_assets_load_but_missing_handler_never_false_completes(task,phase):
    b=load_bundle(task)
    with pytest.raises(SafetyError,match=phase):ordinary_route(b['graph'])


def test_actor_gripper_lock_and_empty_retreat_no_early_open_or_reclose(monkeypatch):
    x=runtime.OrdinaryExecutor.__new__(runtime.OrdinaryExecutor);original=np.array([[.5,.7],[.8,.9]])
    x.b={'gripper':original};x.command_g=np.array([.12,1.]);x.accepted_grasp={'cycle':'c1','index':0,'g_command':.12};seen=[]
    def command(self,*a,**k):seen.append(self.b['gripper'].copy());return None,0,{}
    monkeypatch.setattr(runtime.ObjectCenteredExecutor,'command',command)
    for phase in ('Lift','Move','Align','Wait'):x.command({'phase':phase,'cycle_id':'c1','arm':'left'})
    assert all(np.all(row[:,0]==.12) for row in seen) and all(np.array_equal(row[:,1],original[:,1]) for row in seen)
    assert x.b['gripper'] is original
    x.command({'phase':'Release','cycle_id':'c1','arm':'left'});assert np.array_equal(seen[-1],original)


def test_release_unknown_aperture_uses_command_minimum_time_but_not_released_truth():
    x=runtime.OrdinaryExecutor.__new__(runtime.OrdinaryExecutor);x._last_cartesian_goal=(np.zeros(3),np.eye(3));x._shared_pose=lambda *a:x._last_cartesian_goal
    x.current_holding=False;x.accepted_grasp=None;x.command_g=np.ones(2);x.initial_aperture=[None,None];x.stage_started=0.;x.clock=lambda:1.
    stage={'phase':'Release','arm':'left','holding_contract':'none','reference':{'start_time_sec':0.,'end_time_sec':2.}}
    samples=[SimpleNamespace(q=np.zeros(6),aperture=None),SimpleNamespace(q=np.zeros(6),aperture=None)]
    assert not x.predicates(stage,1.,np.zeros((2,6)),samples,None)['opening_progress']
    x.clock=lambda:3.;facts=x.predicates(stage,1.,np.zeros((2,6)),samples,None)
    assert facts['opening_progress'] and 'released' not in facts and x.opening_record['measured'] is False

def test_multitask_complete_context_preserves_inside_and_supported_relations(tmp_path):
    from PIL import Image
    from prove.xtrainer.agent import digest
    from compact_multitask_judge import compact_context
    frames=[]
    for i in range(3):
        p=tmp_path/f'RGB_{i}.png';Image.new('RGB',(640,360)).save(p)
        frames.append({'frame_id':f'f{i}','path':p.name,'sha256':digest(p),'observed_monotonic_sec':10.+i*.2,'public_robot':[]})
    path=tmp_path/'REQUEST.json'
    value={'stage_id':'c1.complete','stage_contract':{'phase':'Complete','actor':'left_gripper','arm':'left','object':'apple','target':'bowl','goal_relation':'inside'},'requested_facts':['released','placement_complete'],'frames':frames,'public_robot':[{'actual_aperture_m':None},{'actual_aperture_m':None}]}
    path.write_text(json.dumps(value));inside=compact_context(path).context
    assert inside['goal_relation']=='inside' and inside['query_goal_relation']=='released_inside_declared_target'
    value['stage_contract']['goal_relation']='supported';value['stage_contract']['target']='board';path.write_text(json.dumps(value))
    assert compact_context(path).context['query_goal_relation']=='released_supported_by_declared_target'

def test_lift_sam_attachment_registration_preserves_next_move_start_continuity():
    x=runtime.OrdinaryExecutor.__new__(runtime.OrdinaryExecutor);x.command_g=np.array([0.,1.]);x.coarse_source=np.zeros(3)
    actual=np.array([.2,.1,.3]);point=actual+np.array([.01,0,0]);demo=np.array([.05,0,.1])
    x._shared_pose=lambda q,g,arm:(actual.copy(),np.eye(3)) if np.all(q==1) else (demo.copy(),np.eye(3))
    x.b={'qcurve':lambda t:np.zeros((2,6))};x.attachment={'a_actual':np.zeros(3),'a_ref':np.array([.02,0,0]),'R_demo_end':np.eye(3),'R_actual_grasp':np.eye(3)}
    samples=[SimpleNamespace(q=np.ones(6)),SimpleNamespace(q=np.ones(6))];stage={'object':'apple','arm':'left','stage_id':'c1.lift','reference':{'end_time_sec':1.}}
    x.refresh_lift_attachment(stage,samples,SimpleNamespace(roles={'apple':point}))
    goal,objgoal=x.attached_goal((demo,np.eye(3)),x.coarse_source)
    assert np.allclose(goal[0],actual) and np.allclose(objgoal,point)
    assert 'not measured contact' in x.attachment_updates[0]['source']
