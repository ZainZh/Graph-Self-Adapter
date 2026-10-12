"""V24 box-support runtime with offline white Move IK continuation vias."""
import numpy as np

from autonomous_stage_runtime_box_support import BoxSupportExecutor
from autonomous_stage_runtime import AutonomousVision
from direct_white_geometry import tcp_at_handle
from prove.xtrainer.agent import write


VARIANT = "cup_autonomous_stage_v24_box_support_white_move_vias"


def white_move_vias(current_handle, target_handle, rotation, pad):
    current = np.asarray(current_handle, float); target = np.asarray(target_handle, float)
    points = [(1. - alpha) * current + alpha * target for alpha in (1 / 3, 2 / 3, 1.)]
    return [tcp_at_handle(point, rotation, pad) for point in points]


class WhiteMoveContinuationExecutor(BoxSupportExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.white_move_via_plan = None

    def segment_goals(self, stage, pose):
        goals = super().segment_goals(stage, pose)
        if stage["cycle_id"] != "white_direct" or stage["phase"] != "Move": return goals
        plan = self.direct_plan; rotation = plan["entry"][1]
        current = pose[0] + pose[1] @ plan["pad"]
        target = np.asarray(plan["targets"]["handle_support_goal_m"], float).copy()
        target[2] = plan["targets"]["transport_handle_height_m"]
        goals = white_move_vias(current, target, rotation, plan["pad"])
        self.white_move_via_plan = {"fractions": [1 / 3, 2 / 3, 1.], "current_handle_m": current.tolist(),
            "target_handle_m": target.tolist(), "via_TCP_m": [goal[0].tolist() for goal in goals],
            "same_transport_height": True, "final_target_unchanged": True,
            "contract": "offline sequential IK continuation; preceding solution seeds next waypoint; one continuous streamed trajectory"}
        write(self.output / "WHITE_MOVE_VIA_PLAN.json", self.white_move_via_plan)
        return goals

    def run(self):
        result = super().run()
        result.update(execution_variant=VARIANT, white_move_via_plan=self.white_move_via_plan)
        return result


SharedCupExecutor = WhiteMoveContinuationExecutor
SharedSAMVision = AutonomousVision
