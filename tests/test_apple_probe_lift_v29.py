import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from api_v29_apple_entry import route
from prove.xtrainer.multitask_bundle import load_bundle
from apple_probe_lift_v29_runtime import ObservedLiftExecutor,Executor
from prove.xtrainer.io import SafetyError

def test_closure_probe_cannot_enable_transport_gate():
    graph=route(load_bundle('collect_apple')['graph'])
    stages={s['phase']:s for s in graph['stages']}
    assert 'tentative_holding' not in stages['Grasp']['done']
    assert 'tentative_holding' not in stages['Lift']['invariants']
    assert stages['Lift']['holding_contract']=='none'
    assert 'tentative_holding' in stages['Move']['invariants']
    assert stages['Complete']['semantic_facts']==['released','placement_complete']
    assert stages['Complete']['goal_relation']=='inside'

def test_unknown_after_probe_stops_before_attachment(monkeypatch):
    obj=SimpleNamespace(visual_facts={'holding':None},accepted_grasp=None)
    monkeypatch.setattr(Executor,'semantics',lambda *a,**k:([],None))
    with pytest.raises(SafetyError,match='transport forbidden'):
        ObservedLiftExecutor.refresh_lift_attachment(obj,{},[],None)
    assert obj.accepted_grasp is None
