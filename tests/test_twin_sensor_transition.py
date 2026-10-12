from dataclasses import replace
import numpy as np
import pytest
from prove.task_graph.twin_sensor_judge import ArmSensors,SensorPacket
from prove.task_graph.twin_sensor_judge_repair1 import RepairJudge
from prove.task_graph.twin_sensor_transition import support_transition_candidate,transition_holding_contract
from prove.task_graph.twin_sensor_adapter import sensor_holding_contract

STAGE={'phase':'Align','holding_contract':'held_or_supported','object':'obj','arm':'left','target':'target','goal_relation':'supported'}
REF={'authority':'offline_visible_rgbd_reference_only','object_target_end':[0,0,.03]}

def feed(j,t,*,obj=(0,0,0),actual=.015,source='measured_encoder',target=True):
    arm=ArmSensors(np.zeros(6),np.zeros(6),np.zeros(3),np.eye(3),np.zeros(2),actual,0.,'test_calibration',.08,0.,t,source)
    points={'obj':np.tile(obj,(10,1))}
    if target:points['target']=np.tile([0,0,-.03],(10,1))
    j.ingest(SensorPacket(t,t,points,{'left':arm},{}),t)
    return j.facts('obj','left',target='target',relation='supported',reference_delta=[0,0,.03])

def candidate(j,f,stage=STAGE,**kwargs):
    return support_transition_candidate(j,stage,f,REF,observed_role='obj',observed_target=stage['target'],**kwargs)

def acquired():
    j=RepairJudge()
    for t in (0,.1,.2):f=feed(j,t)
    assert f['holding'] is True
    return j,f

def test_permission_never_becomes_release_or_strict_support_completion():
    j,f=acquired();value,diag=candidate(j,f)
    assert value is True and diag['placement_completed'] is False
    assert f['released'] is False and f['supported_contact'] is False
    facts={**f,'bilateral_contact':False,'object_support_contact':False,'support_transition_candidate':value}
    assert transition_holding_contract(STAGE,facts,sensor_holding_contract) is True
    for phase in ('Release','Lift','Move'):
        assert transition_holding_contract({**STAGE,'phase':phase},facts,sensor_holding_contract) is False
    assert transition_holding_contract({**STAGE,'holding_contract':'none'},facts,sensor_holding_contract) is True


def test_settling_wait_is_unknown_then_permission_true_strict_release_stays_false():
    j,f=acquired()
    # Small visible shift resets cumulative anchor without losing prior acquisition.
    f=feed(j,.3,obj=(.006,0,0));assert candidate(j,f)[0] is None
    f=feed(j,.4,obj=(.006,0,0));assert candidate(j,f)[0] is None
    f=feed(j,.5,obj=(.006,0,0));assert candidate(j,f)[0] is True
    assert f['released'] is False and f['supported_contact'] is False


def test_virtual_handover_support_none_normalizes_only_declared_virtual_target():
    j,f=acquired();stage={**STAGE,'target':'table_handover_region','goal_relation':'accessible_to_right_arm'}
    value,diag=support_transition_candidate(j,stage,f,REF,observed_role='obj',observed_target=None,handover_anchor=[0,0,0])
    assert value is True and diag['prior_is_fresh_measurement'] is False
    assert support_transition_candidate(j,STAGE,f,REF,observed_role='obj',observed_target=None)[0] is None
    assert support_transition_candidate(j,stage,f,REF,observed_role='obj',observed_target='wrong',handover_anchor=[0,0,0])[0] is False


@pytest.mark.parametrize('phase,contract',[('Lift','held_or_supported'),('Move','held_or_supported'),('Release','held_or_supported'),('Align','none')])
def test_permission_unavailable_outside_explicit_contract(phase,contract):
    j,f=acquired();assert candidate(j,f,{**STAGE,'phase':phase,'holding_contract':contract})[0] is None


def test_far_wrong_missing_expired_and_placeholder_sensors():
    j,f=acquired()
    assert support_transition_candidate(j,STAGE,f,REF,observed_role='wrong',observed_target='target')[0] is False
    f=feed(j,.3,obj=(.05,0,0));assert candidate(j,f)[0] is False
    f=feed(j,.4,target=False);assert candidate(j,f)[0] is None
    j.expire(.8);assert candidate(j,j.facts('obj','left'))[0] is None
    f=feed(j,.9,source='command_placeholder');assert candidate(j,f)[0] is None


def test_v04_prepare_preserves_history_and_exact_scoped_jobs(tmp_path):
    from pathlib import Path
    from tools.task_graph.twin_sensor_transition_validate import prepare,REPO
    freeze=prepare(REPO/'evaluations/twin_sensor_v04/SCOPE.json',REPO/'evaluations/twin_sensor_v04/CONFIG_PROPOSED.json',
        REPO/'evaluations/twin_sensor_v04/CONTRACT_PROPOSED.json',tmp_path/'new')
    assert len(freeze['jobs'])==10 and all(j['judge_mode']=='sensor_only' for j in freeze['jobs'])
    assert all(j['task']!='sorting_cube_bimanual' for j in freeze['jobs'])
    import json
    history=json.loads((REPO/'evaluations/twin_sensor_v03/repair1/EXECUTION_FREEZE.json').read_text())
    bindings={r['path']:r['sha256'] for r in freeze['bindings']}
    assert all(bindings[r['path']]==r['sha256'] for r in history['bindings'])


def test_world_support_none_handover_and_evaluator_seal_keep_strict_hooks(monkeypatch,tmp_path):
    import json
    from types import SimpleNamespace
    from prove.task_graph import twin_sensor_adapter as old
    from prove.task_graph.twin_sensor_transition_adapter import transition_world_class
    from prove.task_graph.twin_sensor_judge_repair1 import RepairConfig
    path=tmp_path/'seal.json'
    class Physics:
        def __init__(self):
            self.model=SimpleNamespace(numeric=lambda _:SimpleNamespace(data=[.037]))
            self.data=SimpleNamespace(time=.2)
        def contact_evidence(self):return {'roles':{'original':True}}
        def finalize_evaluator(self,outcome):
            seal=json.loads(path.read_text())
            assert seal['execution_variant']=='sensor_transition_v04'
            assert seal['transition_permission_is_strict_placement'] is False
            assert seal['transition_permissions'][0]['value'] is True
            assert self.contact_evidence()['roles']=={'original':True}
            return {'private_evaluator_after_seal':True}
    refs={'stage_relations':{'align':REF},'handover_anchors':{'cycle':{'centroid_world_m':[0,0,0]}}}
    # Isolate process-local router installation from other tests.
    monkeypatch.setattr(old,'sensor_holding_contract',sensor_holding_contract)
    cls=transition_world_class(SimpleNamespace(TwinWorld=Physics),config=RepairConfig(),references=refs,
        judge_mode='sensor_only',seal_path=path)
    world=cls();j,f=acquired();world.sensor_judge=j
    world.sensor_stage={**STAGE,'stage_id':'align','cycle_id':'cycle','target':'table_handover_region','goal_relation':'accessible_to_right_arm'}
    strict={**f,'object_support_contact':False,'object_table_support_contact':False,'stage_goal_complete':False}
    # Override only inherited public fact collection, leaving the new facade method real.
    sensor_base=next(base for base in cls.__mro__ if base.__name__=='SensorWorld')
    monkeypatch.setattr(sensor_base,'_sensor_facts',lambda self,*args,**kwargs:dict(strict))
    values=world._sensor_facts('obj','left',support=None,relation='accessible_to_right_arm')
    assert values['support_transition_candidate'] is True
    assert values['object_table_support_contact'] is False and values['stage_goal_complete'] is False
    assert world.contact_evidence()['roles']=={}
    assert world.finalize_evaluator({})['private_evaluator_after_seal']
    assert world.contact_evidence()['roles']=={}
