"""After release, reverse Lower to this episode's measured pre-Lower free pose."""
import numpy as np
from cup_v25_release_clearance_runtime import ReleaseClearanceExecutor
from cup_v25_root_grounding_vision_v3 import RootFallbackGroundingVision
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError

class OwnHistoryRetreatExecutor(ReleaseClearanceExecutor):
    def segment_goals(self,stage,pose):
        if stage['cycle_id']=='white_direct' and stage['phase']=='Retreat':
            if self.visible_support_plan is None:raise SafetyError('current-episode pre-Lower own pose unavailable')
            point=np.asarray(self.visible_support_plan['current_TCP_m']);rotation=np.asarray(self.visible_support_plan['current_rotation'])
            write(self.output/'WHITE_EMPTY_RETREAT_PLAN.json',{
                'source':'same native episode actually attained pre-Lower own measured TCP/rotation; reverse executed direct Lower path',
                'goal_TCP_m':point.tolist(),'goal_rotation':rotation.tolist(),'current_TCP_m':np.asarray(pose[0]).tolist(),
                'gripper_command':'explicit open; actual release remains for final actual Root visual verification',
                'demonstration_prefix_or_teacher_used':False,'same_position_tolerance_m':.001,'same_rotation_tolerance_rad':.02,
                'full_collision_planning_claim':False,'original_final_goal_unchanged':True})
            return [(point,rotation)]
        return super().segment_goals(stage,pose)

SharedCupExecutor=OwnHistoryRetreatExecutor
SharedSAMVision=RootFallbackGroundingVision
