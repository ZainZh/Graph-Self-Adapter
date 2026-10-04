import numpy as np
from prove.task_graph.p53_micro_insert_v03 import MicroTeacher,near_supervisor
from prove.task_graph.p53_micro_insert import near_supervisor as original_gate


def fixture(gap=.005,x=.015):
    target=np.eye(4);target[2,3]=-.13
    r=dict(final_caddy_book=target.tolist(),book_collision_center=[0,0,0],book_collision_halfsize=[.014,.05,.06],back_halfsize=[.028,.06,.06],mouth_clearance_margin=.001)
    book=np.eye(4);book[:3,3]=[x,0,-(.1+.06+gap)]
    p=dict(world_caddy=np.eye(4),world_book=book,world_eef=book.copy(),rim_height=.1,finger_contacts=(True,True),official_success=False,wrong_target=False)
    return r,p


def test_new_near_scope_accepts_small_alignment_responsibility_not_success():
    r,p=fixture();assert near_supervisor(r,p)['near_ready'];assert not original_gate(r,p)['near_ready']
    p['world_book'][0,3]=.020;assert not near_supervisor(r,p)['near_ready']
    r,p=fixture();p['official_success']=True;assert not near_supervisor(r,p)['near_ready']


def test_local_align_preserves_entry_clearance_then_inserts():
    r,p=fixture();teacher=MicroTeacher(r);action,label=teacher.command(p)
    assert label['teacher_phase']=='local_align' and label['stop_reason'] is None
    assert action[0]<0 and abs(action[2])<1e-12
    target=np.asarray(label['desired_caddy_book']);assert abs(-target[2,3]-.06-.1-.005)<1e-12
    p['world_book'][0,3]=0.;_,label=teacher.command(p);assert label['teacher_phase']=='insert'


def test_fit_loss_below_rim_stops_instead_of_large_recovery():
    r,p=fixture(gap=-.002);teacher=MicroTeacher(r);teacher.phase='insert';teacher.entry_gap=.005
    _,label=teacher.command(p);assert label['stop_reason']=='micro_unsafe_realign_below_rim'
    assert label['teacher_phase']=='insert'
    r,p=fixture(gap=.002);teacher=MicroTeacher(r);teacher.phase='insert';teacher.entry_gap=.005
    _,label=teacher.command(p);assert label['teacher_phase']=='local_align' and label['local_realign_event']
