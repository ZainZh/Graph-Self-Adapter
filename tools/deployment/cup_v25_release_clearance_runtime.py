"""V25 release from20mm visible surface clearance; all acceptance caps unchanged."""
import numpy as np
from cup_v25_combined_support_runtime import CombinedSupportExecutor, SharedSAMVision
from prove.xtrainer.agent import write

RELEASE_CLEARANCE_M=.020

class ReleaseClearanceExecutor(CombinedSupportExecutor):
    def segment_goals(self,stage,pose):
        goals=super().segment_goals(stage,pose)
        if stage['cycle_id']=='white_direct' and stage['phase']=='Lower':
            goal,R=goals[-1];goal=np.asarray(goal).copy()
            previous=self.visible_support_plan['target_visible_surface_clearance_m']
            difference=RELEASE_CLEARANCE_M-previous;goal[2]+=difference
            self.visible_support_goal=(goal,R)
            self.visible_support_plan.update(target_visible_surface_clearance_m=RELEASE_CLEARANCE_M,
                proposed_lower_delta_z_m=self.visible_support_plan['proposed_lower_delta_z_m']+difference,
                goal_TCP_m=goal.tolist(),release_clearance_development_constant_not_learned=True,
                previous10mm_failure='actual endpoint remained6.5mm above goal; original3mm own pose gate refused',
                original_endpoint_and_step_caps_unchanged=True)
            write(self.output/'WHITE_LOWER_VISIBLE_SUPPORT_PLAN.json',self.visible_support_plan)
            return [(goal,R)]
        return goals

SharedCupExecutor=ReleaseClearanceExecutor
