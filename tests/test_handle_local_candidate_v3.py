import importlib.util
from pathlib import Path
import numpy as np
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_handle_v3',ROOT/'tools/deployment/handle_local_candidate_v3.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)


def test_down_approach_is_independent_of_bent_tube_tangent_and_right_handed():
    rng=np.random.default_rng(5);line=np.linspace(-.02,.02,80);cloud=np.c_[.1+line*.7,rng.normal(.2,.0003,80),.06+line*.7];cup=cloud+[-.04,0,0]
    relation=candidate.handle_plane_relation(cloud,[.1,-1,.5],cup)
    assert relation['approach_down'][2]<-.99 and abs(np.dot(relation['approach_down'],relation['closing_axis']))<1e-10
    assert abs(np.dot(relation['tangent_down'],[0,0,-1]))<.8
    pads={'closing_axis_TCP_local':np.array([0.,-1.,0]),'pad_midpoint_TCP_local_m':np.array([0,0,-.002])}
    p,R=candidate.candidate_goal(relation,pads);assert np.allclose(R.T@R,np.eye(3)) and np.linalg.det(R)>.999
    assert np.allclose(R[:,2],relation['approach_down']) and np.allclose(p+R@pads['pad_midpoint_TCP_local_m'],relation['center'])


def test_distal_pad_center_uses_robot_shape_end_not_shank_average():
    import mujoco
    model=mujoco.MjModel.from_xml_path('/home/zheng/github/Mujoco-RSI/runs/physics_v2/put_mug_on_box/scene.xml');data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    pads=candidate.pad_relation_from_robot_mesh(model,data)
    faces=pads['inner_pad_points_TCP_local_m']
    # Each selected face lies in distal 10% of its own named robot mesh extent.
    for row,name in zip(faces,('left_J1_7_03_00_collision','left_J1_8_03_00_collision')):
        geom=model.geom(name);mesh=int(geom.dataid[0]);i=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh]);tcp=data.site('left_tcp');world=model.mesh_vert[i:i+n]@data.geom_xmat[geom.id].reshape(3,3).T+data.geom_xpos[geom.id];cloud=(world-tcp.xpos)@tcp.xmat.reshape(3,3)
        assert row[2]>=cloud[:,2].max()-.1*np.ptp(cloud[:,2])-1e-9
    assert pads['pad_midpoint_TCP_local_m'][2]>-.004 and 'distal' in pads['source']
