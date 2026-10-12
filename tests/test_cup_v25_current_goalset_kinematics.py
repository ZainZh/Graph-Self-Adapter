from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_v25_current_goalset_solver import solve_goalset

def cloud(cx,cy,width,height,z,n=22):
    x,y=np.meshgrid(np.linspace(cx-width/2,cx+width/2,n),np.linspace(cy-height/2,cy+height/2,n))
    return np.c_[x.ravel(),y.ravel(),np.full(x.size,z)]

def fk(q,g,arm):return q[0,:3].copy(),Rotation.from_rotvec(q[0,3:6]).as_matrix()

def test_goalset_recomputes_current_observed_height_and_verifies_contours():
    q=np.zeros((2,6));q[0,:3]=[-.07,0,.25];limits=SimpleNamespace(joint_min=np.full(6,-1.),joint_max=np.ones(6))
    box=cloud(0,0,.25,.25,.14);red=cloud(.06,-.065,.04,.04,.18)
    goals=[]
    for observed_bottom in [.22,.20]:
        goal,report=solve_goalset(fk,q,np.ones(2),cloud(-.07,0,.035,.045,observed_bottom),box,red,np.array([.03,.06]),limits)
        assert report['feasible'] and goal is not None
        assert np.allclose(goal[0][2],.25+.14+.020-observed_bottom,atol=1e-5)
        for values in report['verified_box_clearances_m'].values():assert min(values)>=.010-1e-6
        assert report['NN_updated'] is False and report['frame_anchor_consistent'] is True
        footprint=np.asarray(report['projected_cup_contour_xy_m']);occupied=np.asarray(report['red_exclusion_contour_xy_m']);axis=np.asarray(report['separating_axis'])
        assert (footprint@axis).min()>=(occupied@axis).max()-1e-6
        goals.append(goal[0][2])
    assert not np.isclose(goals[0],goals[1])
