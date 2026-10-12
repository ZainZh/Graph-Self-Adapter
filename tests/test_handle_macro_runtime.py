import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/deployment'))
from handle_macro_route import macro_route_graph
from handle_macro_runtime import MacroExecutor,MacroVision
from prove.xtrainer.io import SafetyError


def test_macro_route_preserves_action_order_and_does_not_fake_internal_object_facts():
    original=json.loads((ROOT/'evaluations/xtrainer_laptop_v01/portable/cup/graph/task_graph.json').read_text());route=macro_route_graph(original)
    assert len(route['stages'])==28 and [s['stage_id'] for s in route['stages']]==[s['stage_id'] for s in original['stages']]
    for stage in route['stages']:
        if stage['phase'] in ('Lift','Move','Align'):assert all(p['predicate']!='stage_goal' for p in stage['completion_predicates'])
        if stage['phase']=='Release':assert any(p['predicate']=='stage_goal' for p in stage['completion_predicates'])
    assert route['macro_contract']['Root_boundaries']==['Grasp','Release'] and original.get('macro_contract') is None


def test_memo_reuses_only_same_update_fresh_stamp_and_invalidates_on_next_update(monkeypatch):
    import handle_macro_runtime as runtime
    vision=MacroVision.__new__(MacroVision);clock=[1.];vision.clock=lambda:clock[0];vision.config={'max_age_sec':.25};vision._memo=None;vision.memo_enabled=False;vision.memo_reuse_count=0;calls=[]
    def actual(self):calls.append(1);return SimpleNamespace(stamp=clock[0],roles={'source':'actual_new_capture'})
    monkeypatch.setattr(runtime.HandleFullVision,'capture',actual)
    vision.begin_control_update(1);first=vision.capture();clock[0]=1.1;assert vision.capture() is first and first.stamp==1.
    clock[0]=1.3;assert vision.capture() is not first and len(calls)==2
    vision.begin_control_update(2);vision.capture();assert len(calls)==3


def test_macro_run_requests_root_only_at_four_grasp_and_four_release_boundaries(tmp_path):
    graph=macro_route_graph(json.loads((ROOT/'evaluations/xtrainer_laptop_v01/portable/cup/graph/task_graph.json').read_text()));ex=MacroExecutor.__new__(MacroExecutor)
    ex.b={'graph':graph,'skill':SimpleNamespace(reset_phase=lambda:None)};ex.cfg={'maximum_node_control_updates':1,'command_dt_sec':.1,'mode':'sim','hardware_reference_speed_scale':1.}
    ex.output=tmp_path;ex.stage_trace=[];ex.trace=[];ex.stops=[];ex._hardware_attempts=0;ex._hardware_acknowledged=0;ex.command_index=0;ex.command_g=np.zeros(2);ex.handle_routes=[];ex.attachment_refresh_log=[]
    ex.vision=SimpleNamespace(begin_control_update=lambda token:None,invalidate=lambda:None,memo_reuse_count=0);ex.judge=SimpleNamespace(records=[]);ex.robot=SimpleNamespace(commands=0);ex.clock=lambda:0.;ex.start=lambda:None;ex.guard_limits=lambda:None
    samples=[SimpleNamespace(q=np.zeros(6),aperture=.01)]*2;obs=SimpleNamespace(roles={'red_mug':[0,0,0],'white_mug':[0,0,0]});ex.capture=lambda:(samples,obs)
    ex.holding=lambda samples,stage:None if stage['phase']=='Grasp' else True
    def command(*args):ex.robot.commands+=1;return np.zeros((2,6)),1.,{'test_fixture':True}
    ex.command=command
    def semantics(stage,facts,before):ex.judge.records.append({'stage':stage['phase'],'facts':facts,'test_fixture_only':True});return before
    ex.semantics=semantics;ex.predicates=lambda stage,*a:{key:True for key in stage['done']+stage['invariants']};ex.refresh_lift_attachment=lambda *a:None
    result=ex.run();assert result['graph_completed'] and result['unsupported_Agent_waits']==0
    assert [r['stage'] for r in ex.judge.records]==['Grasp','Release']*4 and len(result['stage_trace'])==28
    assert not any('lifted' in r['facts'] or 'align_ready' in r['facts'] for r in ex.judge.records)


def test_formal_scope_budget_is_not_reset_by_new_output(tmp_path):
    from macro_one_command import reserve
    from prove.xtrainer.agent import write
    scope=tmp_path/'SCOPE.json';write(scope,{'execution_authorized':True,'hardware_motion_jobs':0,'maximum_full_simulator_episodes':2})
    ledger=tmp_path/'LEDGER.json';reserve(scope,ledger,tmp_path/'a',tmp_path/'ca');reserve(scope,ledger,tmp_path/'b',tmp_path/'cb')
    with pytest.raises(SafetyError,match='budget exhausted'):reserve(scope,ledger,tmp_path/'c',tmp_path/'cc')
    assert len(json.loads(ledger.read_text())['runs'])==2


def test_carry_locks_certified_g_for_both_arms_restores_original_profile_and_release(monkeypatch):
    import handle_macro_runtime as runtime
    ex=MacroExecutor.__new__(MacroExecutor);original=np.array([[.2,.3],[.4,.5]]);ex.b={'gripper':original}
    observed=[]
    def command(self,stage,*args,**kwargs):observed.append(self.b['gripper'].copy());return np.zeros((2,6)),.5,{}
    monkeypatch.setattr(runtime.HandleFullExecutor,'command',command)
    for arm,index,g in [('left',0,0.),('right',1,.12)]:
        ex.macro_grasp={'role':'mug','cycle':'cycle','g_command':g};stage={'phase':'Align','arm':arm,'object':'mug','cycle_id':'cycle'}
        ex.command(stage);assert np.all(observed[-1][:,index]==g) and np.array_equal(observed[-1][:,1-index],original[:,1-index])
        assert ex.b['gripper'] is original
        ex.command({**stage,'phase':'Release'});assert np.array_equal(observed[-1],original)
