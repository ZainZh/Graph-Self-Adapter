"""P52 training-teacher algebra and data contract; no environment construction."""
import numpy as np
from scipy.spatial.transform import Rotation
from prove.task_graph.p52_training_teacher import TrainingOnlyBookTeacher,held_compensated_eef,transform


def relation():
    final=np.eye(4);final[:3,3]=[0,0,.08]
    return dict(final_caddy_book=final.tolist(),demonstrated_clearance_bottom=.12,
                book_collision_center=[0,0,0],book_collision_halfsize=[.014,.05,.06],back_halfsize=[.028,.06,.06])


def test_current_held_offset_is_compensated_exactly():
    WC=transform([.2,.3,.9],Rotation.from_euler('xyz',[.2,.3,.1]).as_matrix())
    WE=transform([.1,-.2,1.2],Rotation.from_euler('xyz',[.1,-.2,.4]).as_matrix())
    EB=transform([.012,-.01,.1],Rotation.from_euler('xyz',[.1,0,.1]).as_matrix())
    target=transform([.01,.02,-.04],Rotation.from_euler('xyz',[0,.2,0]).as_matrix())
    out=held_compensated_eef(WC,target,WE,WE@EB)
    np.testing.assert_allclose(out@EB,WC@target,atol=1e-12)
    shifted=EB.copy();shifted[0,3]+=.01
    different=held_compensated_eef(WC,target,WE,WE@shifted)
    assert np.linalg.norm(different[:3,3]-out[:3,3])>.009


def test_teacher_fixed_grip_bounded_pose_and_native_success_required():
    teacher=TrainingOnlyBookTeacher(relation())
    observation=dict(world_caddy=np.eye(4),world_book=transform([0,0,-.1],np.eye(3)),
                     world_eef=transform([0,0,-.2],np.eye(3)),rim_height=.1,finger_contacts=(True,True),official_success=False)
    for _ in range(10):
        action,label=teacher.command(observation)
        assert np.linalg.norm(action[:3]*.05)<=.0100000001
        assert np.linalg.norm(action[3:6]*.5)<=.0500000001
        assert action[6]==1
        assert label['stop_reason'] is None
        # No clock-triggered stage transition while actual book never reaches goal.
        assert teacher.phase=='lift_clear'
    observation['official_success']=True
    for i in range(3):
        action,label=teacher.command(observation)
        np.testing.assert_array_equal(action,[0,0,0,0,0,0,1])
        assert (label['stop_reason']=='qualified_goal_hold')==(i==2)


def test_clearance_uses_entire_rotated_book_box():
    teacher=TrainingOnlyBookTeacher(relation())
    pose=transform([0,0,.1],Rotation.from_euler('xyz',[.4,.3,.2]).as_matrix())
    lifted=teacher.raised(pose,.15)
    assert teacher.bottom(lifted)>=.15-1e-12
    np.testing.assert_array_equal(lifted[:3,:3],pose[:3,:3])
    np.testing.assert_allclose(lifted[:2,3],pose[:2,3])
