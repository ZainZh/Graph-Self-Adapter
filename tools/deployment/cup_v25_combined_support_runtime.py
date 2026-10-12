"""Same visible cup support goal, one direct combined TCP segment."""
from cup_v25_visible_support_runtime import VisibleSupportExecutor, SharedSAMVision
from prove.xtrainer.agent import write

class CombinedSupportExecutor(VisibleSupportExecutor):
    def segment_goals(self,stage,pose):
        goals=super().segment_goals(stage,pose)
        if stage['cycle_id']=='white_direct' and stage['phase']=='Lower':
            self.visible_support_plan.update(path_strategy='direct combined horizontal/descent segment to same current visible support goal',
                unreachable_transport_height_horizontal_intermediate_removed=True,
                planned_goal_count=1,same_public_support_goal_orientation_clearance_and_caps=True)
            write(self.output/'WHITE_LOWER_VISIBLE_SUPPORT_PLAN.json',self.visible_support_plan)
            return goals[-1:]
        return goals

SharedCupExecutor=CombinedSupportExecutor
