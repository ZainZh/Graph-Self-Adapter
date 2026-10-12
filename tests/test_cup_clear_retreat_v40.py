import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_clear_retreat_v40_runtime import empty_clearance_goals

def test_empty_fingers_clear_observed_cup_and_move_outward():
    rng=np.random.default_rng(3);cup=rng.uniform([0,0,.2],[.06,.06,.27],(400,3));pads=rng.uniform([-.02,-.05,-.08],[.02,.05,0],(300,3));p=np.array([.07,.03,.28]);R=np.eye(3)
    goals,record=empty_clearance_goals((p,R),cup,pads)
    assert goals[0][0][2]+pads[:,2].min()>cup[:,2].max()
    assert np.linalg.norm(goals[-1][0][:2]-np.median(cup,axis=0)[:2])>np.linalg.norm(p[:2]-np.median(cup,axis=0)[:2])
    assert record['not_contact_or_release_certificate']
