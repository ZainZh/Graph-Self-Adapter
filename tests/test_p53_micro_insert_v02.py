"""V02 changes arrival tolerance only; privileged physical gate stays strict."""
import numpy as np
from prove.task_graph.p53_micro_insert import PublicPosePath,near_supervisor


def test_public_arrival_tolerance_does_not_qualify_physical_entry():
    half=[.01434,.05519,.0672];position_tol=min(half)/4
    rotation_tol=float(np.arctan((min(half)/16)/max(half)))
    references=[dict(position=[0,0,0],quaternion_xyzw=[0,0,0,1])]
    pose=[.0008,-.0002,.0011]
    _,old_reached,_=PublicPosePath(references,min(half)/16,rotation_tol).command(pose,[0,0,0,1],0,0)
    _,new_reached,_=PublicPosePath(references,position_tol,rotation_tol).command(pose,[0,0,0,1],0,0)
    assert not old_reached and new_reached
    # Actual book remains 5mm above rim, outside the unchanged 3.585mm near band.
    book=np.eye(4);book[2,3]=-(.1+half[2]+.005)
    private=dict(world_caddy=np.eye(4),world_book=book,rim_height=.1,finger_contacts=(True,True),official_success=False,wrong_target=False)
    relation=dict(book_collision_center=[0,0,0],book_collision_halfsize=half,back_halfsize=[.03,.065,.06],mouth_clearance_margin=.0007)
    assert not near_supervisor(relation,private)['near_ready']

