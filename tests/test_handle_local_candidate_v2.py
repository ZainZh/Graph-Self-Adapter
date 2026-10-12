import importlib.util
from pathlib import Path
import numpy as np
import pytest
from prove.xtrainer.io import SafetyError
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_handle_v2',ROOT/'tools/deployment/handle_local_candidate_v2.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


def test_root_seed_depth_component_excludes_hole_background_without_changing_sam_mask():
    depth=np.full((8,8),.55);depth[:,4:]=.62;mask=np.ones((8,8),bool);before=mask.copy();K=np.diag([400.,400.,1.])
    selected=candidate.seed_depth_component(depth,mask,K,[0,0,8,8],[2,3])
    assert selected[:,:4].all() and not selected[:,4:].any() and np.array_equal(mask,before)
    back=candidate.seed_depth_component(depth,mask,K,[0,0,8,8],[6,3]);assert back[:,4:].all() and not back[:,:4].any()


def test_seed_must_have_exact_foreground_support_no_nearest_pixel_search():
    depth=np.full((8,8),.55);mask=np.ones((8,8),bool);mask[3,2]=False
    with pytest.raises(SafetyError,match='support'):candidate.seed_depth_component(depth,mask,np.diag([400.,400.,1.]),[0,0,8,8],[2,3])


def test_handle_plane_normal_goal_is_right_handed_and_preserves_observed_pad_center():
    rng=np.random.default_rng(3);cloud=np.c_[rng.normal(.1,.0005,80),rng.normal(.2,.0005,80),np.linspace(.04,.08,80)];cup=cloud+[-.04,0,0]
    relation=candidate.handle_plane_relation(cloud,[.1,-1,.5],cup);radial=relation['center']-np.median(cup,axis=0)
    assert abs(np.dot(relation['closing_axis'],radial))<1e-10 and abs(np.dot(relation['closing_axis'],relation['tangent_down']))<1e-10
    pads={'closing_axis_TCP_local':np.array([0.,-1.,0]),'pad_midpoint_TCP_local_m':np.array([0,0,-.006])}
    p,R=candidate.candidate_goal(relation,pads);assert np.allclose(R.T@R,np.eye(3)) and np.linalg.det(R)>.999
    assert np.allclose(p+R@pads['pad_midpoint_TCP_local_m'],relation['center'])


def test_original_tube_gate_remains_strict_after_component_filtering():
    rng=np.random.default_rng(6);cloud=rng.normal(size=(100,3))*.03
    with pytest.raises(SafetyError,match='tube hypothesis'):candidate.handle_plane_relation(cloud,[0,-1,.5],cloud+[.1,0,0])
