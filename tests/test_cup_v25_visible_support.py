import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_v25_visible_support_geometry import dominant_visible_surface,support_geometry
from prove.xtrainer.io import SafetyError

def test_disconnected_fringe_does_not_become_cup_bottom():
    x,y=np.meshgrid(np.linspace(0,.04,20),np.linspace(0,.04,20))
    cup=np.c_[x.ravel(),y.ravel(),np.full(x.size,.2)]
    fringe=np.c_[x.ravel()[:20]+1,y.ravel()[:20],np.full(20,.01)]
    box=np.c_[x.ravel(),y.ravel(),np.full(x.size,.14)]
    report=support_geometry(np.r_[cup,fringe],box,[0,0,.25],np.eye(3))
    assert report['lowest_observed_cup_surface_m']==pytest.approx(.2)
    assert report['target_visible_surface_clearance_m']==.01
    assert report['proposed_lower_delta_z_m']==pytest.approx(-.05)
    assert report['contact_observed'] is False and report['hidden_bottom_inferred'] is False
    assert report['dominant_points']==400

def test_insufficient_visible_geometry_stays_unknown():
    with pytest.raises(SafetyError):dominant_visible_surface(np.zeros((20,3)))
