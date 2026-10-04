import numpy as np
from prove.task_graph.p52_training_teacher import transform,box_vertices
from prove.task_graph.p52_training_teacher_v02 import centered_relation,TrainingOnlyBookTeacher


def relation():
    target=np.eye(4);target[:3,3]=[-.015,.009,.04]
    return dict(final_caddy_book=target.tolist(),book_collision_center=[.001,-.012,.069],
                book_collision_halfsize=[.014,.055,.067],back_halfsize=[.028,.062,.060],demonstrated_clearance_bottom=.12)


def test_centering_preserves_demo_orientation_z_and_uses_geometry_margin():
    r=relation();new=centered_relation(r);T=np.asarray(new['final_caddy_book']);old=np.asarray(r['final_caddy_book'])
    points=box_vertices(r['book_collision_center'],r['book_collision_halfsize'])@T[:3,:3].T+T[:3,3]
    np.testing.assert_allclose((points.min(0)+points.max(0))[:2]/2,0,atol=1e-12)
    np.testing.assert_array_equal(T[:3,:3],old[:3,:3]);assert T[2,3]==old[2,3]
    assert abs(new['mouth_clearance_margin']-.2*(.062-.055))<1e-12


def test_bad_footprint_interrupts_insert_with_upward_clearance_motion():
    teacher=TrainingOnlyBookTeacher(relation());teacher.phase='insert'
    B=np.asarray(teacher.relation['final_caddy_book']);B[0,3]+=.04
    E=B@transform([0,0,-.1],np.eye(3))
    action,label=teacher.command(dict(world_caddy=np.eye(4),world_book=B,world_eef=E,rim_height=.1,finger_contacts=(True,True),official_success=False))
    assert label['teacher_phase']=='lift_clear'
    assert label['recovery_event']=='descent_stopped_footprint_outside_mouth'
    assert action[2]<0  # mouth local insertion +z in this identity fixture; clearance is -z
    assert label['stop_reason'] is None and not label['footprint_qualified']
    assert action[6]==1
