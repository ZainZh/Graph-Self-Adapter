"""Read-only first red-cup grasp-entry planning from the actual joint pose.

No command, robot connection, camera access, model invocation or demo-q0 gate.
Caller owns measured input freshness, cell/arm collision checks and any later
motion authorization. Frozen policy residuals are not reinterpreted here.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def finite(value,shape,label):
    out=np.asarray(value,dtype=float)
    if out.shape!=shape or not np.isfinite(out).all():raise ValueError('invalid '+label)
    return out.copy()


def first_red_grasp_reference(bundle):
    stage=next(s for s in bundle['graph']['stages'] if s['phase']=='Grasp')
    if stage['object']!='red_mug' or stage['arm']!='left':raise ValueError('scope is only the first left-arm red-cup grasp entry')
    time=float(stage['reference']['start_time_sec'])
    index=int(np.argmin(abs(bundle['frame_times']-time)));frame=bundle['frames'][index]
    obj=frame['surface_centroids_world_m'].get('red_mug')
    if obj is None:raise ValueError('declared grasp-entry visible reference unavailable')
    if abs(float(frame['time_sec'])-time)>.15:raise ValueError('grasp-entry visible reference is not time-matched')
    return {'stage_id':stage['stage_id'],'reference_time_sec':time,'visible_reference_time_sec':float(frame['time_sec']),
            'reference_q':finite(bundle['qcurve'](time),(2,6),'reference q'),
            'reference_object_shared_m':finite(obj,(3,),'reference red-cup surface')}


def plan_first_red_grasp_entry(*,actual_q,current_object_shared_m,reference_q,reference_object_shared_m,
                               native_fk,gripper_command,T_left_from_right,
                               joint_min_rad=None,joint_max_rad=None,workspace_min_m=None,workspace_max_m=None,
                               max_iterations=400,max_joint_step_rad=.01,damping=.015,
                               position_tolerance_m=.001,rotation_tolerance_rad=.005,observed_timestamp_sec=None):
    q=finite(actual_q,(2,6),'actual q');reference_q=finite(reference_q,(2,6),'reference q')
    current_object=finite(current_object_shared_m,(3,),'current visible red-cup surface')
    ref_object=finite(reference_object_shared_m,(3,),'reference visible red-cup surface')
    g=finite(gripper_command,(2,),'declared gripper command')
    transform=finite(T_left_from_right,(4,4),'left-from-right transform')
    if not np.allclose(transform[3],[0,0,0,1]) or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5):
        raise ValueError('invalid shared-world rigid transform')
    if not 0<max_joint_step_rad<=.01 or not 1<=max_iterations<=1000 or damping<=0:
        raise ValueError('candidate planning budget/step exceeds reviewed bounds')
    lower=np.full((2,6),-np.inf) if joint_min_rad is None else np.broadcast_to(finite(joint_min_rad,np.asarray(joint_min_rad).shape,'joint minimum'),(2,6)).copy()
    upper=np.full((2,6),np.inf) if joint_max_rad is None else np.broadcast_to(finite(joint_max_rad,np.asarray(joint_max_rad).shape,'joint maximum'),(2,6)).copy()
    if np.any(lower>=upper) or np.any(q<lower) or np.any(q>upper):raise ValueError('actual pose outside declared joint bounds')
    wmin=np.full(3,-np.inf) if workspace_min_m is None else finite(workspace_min_m,(3,),'workspace minimum')
    wmax=np.full(3,np.inf) if workspace_max_m is None else finite(workspace_max_m,(3,),'workspace maximum')
    if np.any(wmin>=wmax):raise ValueError('invalid workspace bounds')
    fk_calls=0
    def pose(value):
        nonlocal fk_calls
        # NativeRPC.fk-compatible signature; first arm is the shared left base.
        xyz,rotation=native_fk(np.asarray(value).copy(),g.copy(),'left');fk_calls+=1
        xyz=finite(xyz,(3,),'native FK position');rotation=finite(rotation,(3,3),'native FK rotation')
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<=0:
            raise ValueError('native FK returned a non-rotation')
        return xyz,rotation
    ref_tcp,goal_rotation=pose(reference_q)
    object_offset=ref_tcp-ref_object;goal=current_object+object_offset
    path=[q.copy()];poses=[];errors=[];reason='iteration_budget_exhausted';converged=False
    for iteration in range(max_iterations+1):
        xyz,rotation=pose(q)
        position_error=goal-xyz;rotation_error=Rotation.from_matrix(goal_rotation@rotation.T).as_rotvec()
        poses.append({'TCP_shared_m':xyz.tolist(),'rotation_shared':rotation.tolist()})
        errors.append({'position_error_m':float(np.linalg.norm(position_error)),
                       'rotation_error_rad':float(np.linalg.norm(rotation_error))})
        if np.any(xyz<wmin) or np.any(xyz>wmax):reason='candidate_TCP_outside_workspace';break
        if errors[-1]['position_error_m']<=position_tolerance_m and errors[-1]['rotation_error_rad']<=rotation_tolerance_rad:
            converged=True;reason='candidate_goal_reached';break
        if np.any(goal<wmin) or np.any(goal>wmax):reason='desired_goal_outside_workspace';break
        if iteration==max_iterations:break
        jacobian=np.empty((6,6));eps=1e-4
        for joint in range(6):
            signed_eps=eps if q[0,joint]+eps<=upper[0,joint] else -eps
            if not lower[0,joint]<=q[0,joint]+signed_eps<=upper[0,joint]:
                jacobian[:,joint]=0.;continue
            probe=q.copy();probe[0,joint]+=signed_eps;probe_xyz,probe_rotation=pose(probe)
            jacobian[:3,joint]=(probe_xyz-xyz)/signed_eps
            jacobian[3:,joint]=Rotation.from_matrix(probe_rotation@rotation.T).as_rotvec()/signed_eps
        error=np.r_[position_error,rotation_error]
        delta=jacobian.T@np.linalg.solve(jacobian@jacobian.T+damping*damping*np.eye(6),error)
        delta*=min(1.,max_joint_step_rad/max(float(np.max(abs(delta))),1e-12))
        accepted=False
        for backtrack in range(12):
            candidate=q.copy();candidate[0]+=delta*(.5**backtrack)
            if np.any(candidate<lower) or np.any(candidate>upper):continue
            candidate_xyz,candidate_rotation=pose(candidate)
            if np.any(candidate_xyz<wmin) or np.any(candidate_xyz>wmax):continue
            candidate_error=np.r_[goal-candidate_xyz,Rotation.from_matrix(goal_rotation@candidate_rotation.T).as_rotvec()]
            if np.linalg.norm(candidate_error)<np.linalg.norm(error)-1e-10:
                q=candidate;path.append(q.copy());accepted=True;break
        if not accepted:reason='no_safe_descending_IK_step';break
    return {'schema':'graph-self-adapter.object-relative-entry-plan.v1','task':'put_mug_on_box','object':'red_mug','arm':'left',
            'read_only':True,'commands_sent':0,'converged':converged,'termination':reason,
            'goal_TCP_shared_m':goal.tolist(),'goal_rotation_shared':goal_rotation.tolist(),
            'declared_reference_object_TCP_offset_m':object_offset.tolist(),'current_object_shared_m':current_object.tolist(),
            'actual_start_q':path[0].tolist(),'q_path':[v.tolist() for v in path],'poses':poses,'errors':errors,
            'FK_calls':fk_calls,'max_candidate_joint_step_rad':max_joint_step_rad,'observed_timestamp_sec':observed_timestamp_sec,
            'freshness_and_collision_verified':False,'joint_limits_supplied':joint_min_rad is not None and joint_max_rad is not None,
            'workspace_limits_supplied':workspace_min_m is not None and workspace_max_m is not None,
            'limitations':['candidate only; caller must validate fresh actual feedback and geometry before any motion',
                           'no whole-arm collision/obstacle planner','does not reinterpret frozen delta-q-reference neural policy']}
