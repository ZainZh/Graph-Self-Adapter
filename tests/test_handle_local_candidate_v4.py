import importlib.util
from pathlib import Path
import numpy as np
import pytest
from prove.xtrainer.io import SafetyError
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_handle_v4',ROOT/'tools/deployment/handle_local_candidate_v4.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


def rectangle(y,z0,z1,x0,x1,inward_sign):
    vertices=np.array([[x0,y,z0],[x1,y,z0],[x1,y,z1],[x0,y,z1]])
    triangles=vertices[[[0,1,2],[0,2,3]]]
    normal=np.cross(triangles[0,1]-triangles[0,0],triangles[0,2]-triangles[0,0])
    if np.sign(normal[1])!=inward_sign:triangles=triangles[:,[0,2,1]]
    return triangles


def test_connected_planes_overlap_defines_common_robot_pad_not_tip_or_unmatched_area():
    first=rectangle(.03,-.05,-.01,-.015,.015,-1);second=rectangle(-.03,-.08,-.005,-.02,.02,1)
    pads=candidate.paired_flat_overlaps(first,second,[0,-1,0]);best=pads[0]
    assert np.isclose(best['usable_overlap_area_m2'],.03*.04)
    assert np.allclose(best['pad_midpoint_TCP_local_m'],[0,0,-.03])
    assert np.allclose(best['inner_pad_points_TCP_local_m'][:,2],-.03)


def test_disconnected_flat_regions_do_not_fill_gap_with_hull():
    first=np.concatenate([rectangle(.03,-.05,-.04,-.01,.01,-1),rectangle(.03,-.02,-.01,-.01,.01,-1)])
    second=rectangle(-.03,-.06,0,-.02,.02,1)
    pads=candidate.paired_flat_overlaps(first,second,[0,-1,0]);assert len(pads)==2
    assert all(np.isclose(p['usable_overlap_area_m2'],.02*.01) for p in pads)
    assert all(not -.04<p['pad_midpoint_TCP_local_m'][2]<-.02 for p in pads)


def test_nonopposing_or_missing_flat_faces_fail_closed():
    first=rectangle(.03,-.05,-.01,-.015,.015,-1)
    with pytest.raises(SafetyError,match='overlap unavailable'):candidate.paired_flat_overlaps(first,first,[0,-1,0])


def test_real_robot_overlap_is_upstream_flat_area_without_any_object_geometry():
    import mujoco
    model=mujoco.MjModel.from_xml_path('/home/zheng/github/Mujoco-RSI/runs/physics_v2/put_mug_on_box/scene.xml');data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    for arm in ('left','right'):
        pads=candidate.pad_relation_from_robot_mesh(model,data,arm)
        assert pads['usable_overlap_area_m2']>700e-6 and pads['pad_midpoint_TCP_local_m'][2]<-.02
        assert len(pads['overlap_candidates'])>=1 and 'connected' in pads['source']
