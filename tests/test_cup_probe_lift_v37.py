import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from api_v37_cup_entry import route
from prove.xtrainer.runtime import load_bundle,Executor
from cup_probe_lift_v37_runtime import ObservedCupLiftExecutor
from prove.xtrainer.io import SafetyError

def test_probe_graph_keeps_carry_gates_and_complete_nodes():
    g=route(load_bundle('put_mug_on_box')['graph']);assert len(g['stages'])==25
    for s in g['stages']:
        if s['phase'] in ('Grasp','Lift'):
            assert s['holding_contract']=='none'
            assert all(p['predicate'] not in ('holding','holding_confirmed') for p in s['completion_predicates']+s['invariant_predicates'])
        elif s['phase']=='Move':assert s['holding_contract']!='none'
    assert sum(s['phase']=='Complete' for s in g['stages'])==3

def test_unknown_lift_cannot_set_transport_certificate(monkeypatch):
    vision=SimpleNamespace(invalidate=lambda:None,memo_enabled=True)
    obj=SimpleNamespace(vision=vision,visual_facts={'holding':None},macro_grasp=None)
    monkeypatch.setattr(Executor,'semantics',lambda *a,**k:([],None))
    with pytest.raises(SafetyError,match='no transport'):
        ObservedCupLiftExecutor.confirm_lift(obj,{},[],None)
    assert obj.macro_grasp is None;assert obj.vision.memo_enabled
