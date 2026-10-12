import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_clear_goalset_v41 import solve_clearance

def test_free_wrist_finds_clear_relation_under_low_height_limit():
    rng=np.random.default_rng(1);cup=rng.uniform([-.025,-.025,.20],[.025,.025,.27],(300,3));pads=rng.uniform([-.01,-.03,-.07],[.01,.03,-.03],(300,3))
    q=np.zeros((2,6));q[0,:3]=[.08,0,.24]
    def fk(q,g,arm):
        x=q[0 if arm=='left' else 1];return x[:3],Rotation.from_rotvec(x[3:]).as_matrix()
    limits=SimpleNamespace(joint_min=[-.2,-.2,.1,-2.,-2.,-2.],joint_max=[.2,.2,.30,2.,2.,2.])
    goals,report=solve_clearance(fk,q,np.ones(2),'left',cup,pads,limits,np.eye(4))
    assert report['feasible'];assert len(goals)==4;assert report['verified_min_margin']>=-1e-5
    p,R=goals[-1];assert p[2]<=.30+1e-6
    assert (pads@R.T+p)[:,2].min()>=cup[:,2].max()-.001
    assert np.linalg.norm(report['goal_q_actor'][3:])>.1
