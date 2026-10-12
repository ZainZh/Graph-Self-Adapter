import sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
from prove.xtrainer.multitask_bundle import load_bundle
from multitask_special_route import special_route
from multitask_special_runtime import SpecialExecutor
from prove.xtrainer.io import SafetyError

@pytest.mark.parametrize('task,count',[('hanging_mug',8),('stack_cups',16),('open_drawer_and_place_object',14),('sweep_objects',11)])
def test_actual_special_routes_preserve_phase_goal_and_held_terminal(task,count):
    b=load_bundle(task);g=special_route(b['graph']);assert len(g['stages'])==count
    assert [s['phase'] for s in g['stages'] if s['phase']!='Complete']==[s['phase'] for s in b['graph']['stages']]
    completes=[s for s in g['stages'] if s['phase']=='Complete']
    if task=='sweep_objects':
        assert all(s['hold_at_end'] and s['semantic_facts']==['holding','placement_complete'] for s in completes)
        assert len(completes[0]['declared_held_tools'])==1 and len(completes[1]['declared_held_tools'])==2
        assert not any(s['phase']=='Release' for s in g['stages'])
    elif task=='open_drawer_and_place_object':assert [s['goal_relation'] for s in completes]==['drawer_extended','inside']

@pytest.mark.parametrize('phase',['Insert','Pull','Push'])
def test_frozen_neural_phase_unchanged_and_delta_applied_to_reference(tmp_path,phase):
    x=SpecialExecutor.__new__(SpecialExecutor);seen=[];actual=np.zeros((2,6));actual[0,0]=.8
    class Skill:
        def command(self,features,**kwargs):kwargs['features']=features;seen.append(kwargs);return SimpleNamespace(action=np.array([.005,0,0,0,0,0]),record={'phase':kwargs['phase']})
    x.b={'qcurve':lambda t:np.array([[.2,0,0,0,0,0],[0,0,0,0,0,0]]),'times':np.array([0.,1.]),'gripper':np.array([[0.,1.],[0.,1.]]),'skill':Skill()}
    x._cycle='c1';x._phase=None;x._phase_elapsed=0.;x._last_cartesian_goal=None;x.coarse_source=np.array([.01,0,0]);x.coarse_target=np.array([.02,0,0]);x.static_source_env=np.zeros(3);x.static_target_env=np.zeros(3);x.code_static_source_handled=np.zeros(3);x.code_static_target_handled=np.zeros(3);x.attachment=None;x.current_holding=True;x.command_g=np.array([0.,1.]);x.T=np.eye(4);x.cfg={'hardware_reference_speed_scale':1.,'safety':{'max_age_sec':.25}}
    x.nearest_ref=lambda t:{'time_sec':t,'surface_centroids_world_m':{'tool':np.zeros(3)},'tcp_position':np.zeros((2,3)),'tcp_rotation':np.repeat(np.eye(3)[None],2,axis=0)}
    x._shared_pose=lambda q,g,arm:(q[0 if arm=='left' else 1,:3].copy(),np.eye(3))
    x._step_ik=lambda q,g,arm,goal:(q.copy(),None,np.r_[goal[0]-q[0,:3],np.zeros(3)])
    samples=[SimpleNamespace(q=q,qdot=np.zeros(6)) for q in actual];x.capture=lambda:(samples,SimpleNamespace(stamp=10.));x.clock=lambda:10.;x.sleep=lambda dt:None
    x.safety=SimpleNamespace(validate=lambda *a:None,samples=lambda *a:None,last_target=None);x.robot=SimpleNamespace(samples=lambda:samples,command=lambda *a:None);x.output=tmp_path;x._hardware_attempts=0;x._hardware_acknowledged=0;x.command_index=0
    stage={'stage_id':'c1.'+phase,'cycle_id':'c1','arm':'left','phase':phase,'controller':'neural_short','object':'tool','reference':{'start_time_sec':0.,'end_time_sec':1.}}
    _,_,record=x._reference_command(stage,.1,samples,SimpleNamespace(roles={'tool':np.zeros(3)}),.1)
    assert seen[0]['phase']==phase and np.array_equal(seen[0]['features'][:6],actual[0])
    assert seen[0]['features'][21:26].tolist()==np.eye(5)[('Grasp','Align','Insert','Push','Pull').index(phase)].tolist()
    assert record['goal_TCP_shared_m'][0]==pytest.approx(.225) # reference .2 + NN .005 + coarse target .02
    assert record['goal_TCP_shared_m'][0]!=pytest.approx(.825) # never actual .8 + NN


def test_two_tool_certificates_passive_unknown_stops_before_command():
    x=SpecialExecutor.__new__(SpecialExecutor);x.stage={'arm':'right','object':'broom','cycle_id':'c2','phase':'Push'};x.accepted_grasp=None
    x.certificates={('left','dustpan'):{'index':0,'aperture_m':None,'object_TCP_distance_m':.05,'g_command':0.},('right','broom'):{'index':1,'aperture_m':None,'object_TCP_distance_m':.05,'g_command':0.}}
    x.cfg={'grasp_region_margin_m':.05};x.command_g=np.array([0.,0.]);x.T=np.eye(4)
    from multitask_ordinary_runtime import OrdinaryExecutor
    original=OrdinaryExecutor.capture
    OrdinaryExecutor.capture=lambda owner:([SimpleNamespace(tcp=np.zeros(3),aperture=None),SimpleNamespace(tcp=np.zeros(3),aperture=None)],SimpleNamespace(roles={'dustpan':None,'broom':np.zeros(3)}))
    try:
        with pytest.raises(SafetyError,match='passive tool'):x.capture()
    finally:OrdinaryExecutor.capture=original


def test_held_terminal_prompt_and_context_do_not_require_release(tmp_path):
    import compact_special_judge as judge
    from PIL import Image
    frames=[]
    for i in range(3):
        p=tmp_path/f'RGB_{i}.png';Image.new('RGB',(640,360)).save(p)
        frames.append({'frame_id':f'f{i}','path':p.name,'sha256':judge.digest(p),'observed_monotonic_sec':10.+i*.2,'public_robot':[]})
    path=tmp_path/'REQUEST.json';path.write_text(json.dumps({'stage_id':'c2.complete','stage_contract':{'phase':'Complete','actor':'right_gripper','arm':'right','object':'broom','target':'yellow_cube','goal_relation':'block_inside_dustpan_with_tools_held','hold_at_end':True,'declared_held_tools':[{'arm':'left','object':'dustpan'},{'arm':'right','object':'broom'}]},'requested_facts':['holding','placement_complete'],'frames':frames,'public_robot':[{'actual_aperture_m':None},{'actual_aperture_m':None}]}))
    compact=judge.compact_context(path)
    assert compact.context['hold_at_end'] is True and 'released' not in compact.context['query_goal_relation']
    assert len(compact.context['declared_held_tools'])==2 and compact.context['actual_jaw_m_by_arm']=={'left':None,'right':None}
    assert 'without release' in judge.SYSTEM_PROMPT and len(judge.SYSTEM_PROMPT.split())<350
    reply=judge.parse_reply({'decision':'unknown','facts':{'holding':True},'evidence_frame_ids':['f0','f2'],'reason':'occluded'},compact)
    assert reply['facts']=={'holding':None,'placement_complete':None}

def test_special_carry_gripper_locks_both_tool_certificates_and_restores_profile():
    x=SpecialExecutor.__new__(SpecialExecutor);original=np.array([[.6,.8],[.7,.9]])
    x.b={'gripper':original};x.command_g=np.array([0.,.12]);x.current_holding=True;x.certificates={('left','dustpan'):{'index':0,'state':True,'g_command':0.},('right','broom'):{'index':1,'state':True,'g_command':.12}}
    seen=[];x._reference_command=lambda *a:(seen.append(x.b['gripper'].copy()) or (None,0,{}))
    stage={'phase':'Push','arm':'right','object':'broom','holding_contract':'bilateral_contact'}
    x.command(stage,0.,[],None,.1)
    assert np.all(seen[0][:,1]==.12) and np.array_equal(seen[0][:,0],original[:,0])
    assert x.b['gripper'] is original and x.command_g[0]==0. # passive is preserved by real command's g[1-index] rule

def test_held_terminal_visual_true_cannot_override_actual_empty_tool_invariant(monkeypatch):
    from prove.xtrainer.runtime import Executor
    x=SpecialExecutor.__new__(SpecialExecutor);x.visual_facts={};x.semantic_log=[];x.certificates={('left','dustpan'):{'state':False}}
    def semantics(owner,stage,facts,before):owner.visual_facts.update({f:True for f in facts});return before
    monkeypatch.setattr(Executor,'semantics',semantics)
    stage={'hold_at_end':True,'declared_held_tools':[{'arm':'left','object':'dustpan'}],'stage_id':'c1.complete','goal_relation':'receptacle_ready_and_held'}
    with pytest.raises(SafetyError,match='actual tool invariant'):x.semantics(stage,['holding','placement_complete'],None)
    assert not x.semantic_log
