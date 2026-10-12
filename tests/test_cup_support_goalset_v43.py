import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from cup_support_goalset_v43 import solve_goalset

def cylinder(center,r,z0,z1):
    t,z=np.meshgrid(np.linspace(0,2*np.pi,50,endpoint=False),np.linspace(z0,z1,35));return np.c_[center[0]+r*np.cos(t).ravel(),center[1]+r*np.sin(t).ravel(),z.ravel()]

def test_joint_support_solution_keeps_full_footprint_and_clearance():
    cup=cylinder([.10,.10],.035,.42,.48);red=cylinder([.245,.245],.025,.30,.36)
    x,y=np.meshgrid(np.linspace(0,.30,35),np.linspace(0,.30,35));box=np.c_[x.ravel(),y.ravel(),np.full(x.size,.30)]
    q=np.zeros((2,6));q[0,:3]=[.10,.10,.50]
    def fk(q,g,arm):return q[0,:3],Rotation.from_euler('xyz',q[0,3:]).as_matrix()
    limits=SimpleNamespace(joint_min=[-.3,-.3,0,-3.,-3.,-3.],joint_max=[.4,.4,.8,3.,3.,3.])
    goal,report=solve_goalset(fk,q,np.ones(2),cup,box,red,np.array([.10,.10]),limits,maximum_iterations=120)
    assert report['feasible'];assert goal is not None
    projected=np.asarray(report['projected_cup_contour_xy_m']);lo=np.asarray(report['box_lower_xy_m']);hi=np.asarray(report['box_upper_xy_m'])
    assert np.all(projected.min(0)>=lo+.010-1e-6);assert np.all(projected.max(0)<=hi-.010+1e-6)
    assert report['release_clearance_m']==.020
