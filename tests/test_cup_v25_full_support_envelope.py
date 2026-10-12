import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_v25_full_support_envelope import plan_support

def cloud(cx,cy,width,height,z,n=30):
    x,y=np.meshgrid(np.linspace(cx-width/2,cx+width/2,n),np.linspace(cy-height/2,cy+height/2,n))
    return np.c_[x.ravel(),y.ravel(),np.full(x.size,z)]

def test_whole_contour_fits_original_margin_and_red_separating_axis():
    cup=cloud(0,0,.04,.06,.2);box=cloud(0,0,.20,.20,.14);red=cloud(.04,-.05,.03,.03,.18)
    target,report=plan_support(cup,box,red,np.array([.095,.095]))
    assert report['feasible'] and target is not None
    placed=np.asarray(report['projected_cup_contour_xy_m'])
    lo=np.asarray(report['box_top_lower_xy_m']);hi=np.asarray(report['box_top_upper_xy_m'])
    assert (placed.min(axis=0)>=lo+.010-1e-9).all()
    assert (placed.max(axis=0)<=hi-.010+1e-9).all()
    normal=np.asarray(report['separating_axis']);bound=report['separating_bound_m']
    assert normal@target<=bound+1e-8
    assert report['hidden_full_shape_or_COM_verified'] is False

def test_infeasible_visible_envelope_stays_infeasible():
    target,report=plan_support(cloud(0,0,.05,.07,.2),cloud(0,0,.02,.02,.14),cloud(0,0,.02,.02,.18),np.zeros(2))
    assert target is None and report['feasible'] is False
