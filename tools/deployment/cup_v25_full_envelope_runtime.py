"""Full observed cup envelope inside support with red exclusion and bounded yaw."""
import numpy as np
from scipy.spatial.transform import Rotation
from cup_v25_own_history_retreat_runtime import OwnHistoryRetreatExecutor, SharedSAMVision
from cup_v25_full_support_envelope import plan_support
from cup_v25_visible_support_geometry import dominant_visible_surface
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError

RELATIVE_SUPPORT_YAW_DEG=-15.0

class FullEnvelopeExecutor(OwnHistoryRetreatExecutor):
    def segment_goals(self,stage,pose):
        goals=super().segment_goals(stage,pose)
        if stage['cycle_id']!='white_direct' or stage['phase']!='Lower':return goals
        cup=self.vision.cloud('white_mug');box=self.vision.cloud('box');red=self.vision.cloud('red_mug')
        surface,_=dominant_visible_surface(cup);center=np.median(surface,axis=0)
        Q=Rotation.from_euler('z',RELATIVE_SUPPORT_YAW_DEG,degrees=True).as_matrix()
        projected=(cup-center)@Q.T+center
        nominal=np.asarray(self.white_object_yaw_plan['target']['cup_center_m'])[:2]
        target,envelope=plan_support(projected,box,red,nominal)
        write(self.output/'FULL_VISIBLE_SUPPORT_ENVELOPE.json',envelope)
        if target is None:raise SafetyError('full current visible cup envelope/red exclusion infeasible')
        delta=np.asarray(pose[0])-center;goal=np.r_[target,center[2]]+Q@delta
        goal[2]=self.visible_support_goal[0][2]  # same current surface20mm release clearance
        rotation=Q@np.asarray(pose[1])
        if np.linalg.norm(goal[:2]-np.asarray(pose[0])[:2])>.10:raise SafetyError('full envelope goal outside declared bounded adjustment')
        self.visible_support_goal=(goal,rotation)
        self.visible_support_plan.update(footprint=envelope,goal_TCP_m=goal.tolist(),goal_rotation=rotation.tolist(),
            relative_support_yaw_deg=RELATIVE_SUPPORT_YAW_DEG,
            public_projection_proxy_center_m=center.tolist(),current_red_exclusion=True,
            whole_observed2_to98percent_envelope_used=True,
            rigid_held_contour_yaw_is_declared_hypothesis_not_contact_proof=True,
            task_goal='both cups released stably supported by box; exact demonstration XY/yaw not required',
            continuous_geometric_free_region_not_cm_search=True,
            original_final_goal_and_all_servo_endpoint_caps_unchanged=True)
        np.savez_compressed(self.output/'FULL_SUPPORT_PUBLIC_GEOMETRY.npz',cup_cloud=cup,box_cloud=box,red_cloud=red,current_TCP_m=np.asarray(pose[0]),current_rotation=np.asarray(pose[1]))
        write(self.output/'WHITE_LOWER_VISIBLE_SUPPORT_PLAN.json',self.visible_support_plan)
        return [(goal,rotation)]

SharedCupExecutor=FullEnvelopeExecutor
