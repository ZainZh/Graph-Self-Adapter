"""V24 fixed object-centered white yaw avoids the diagnosed wrist singularity."""
import numpy as np
from scipy.spatial.transform import Rotation

from autonomous_stage_runtime_initial_part import InitialPartContractExecutor, InitialPartContractVision
from direct_white_geometry import tcp_at_handle
from prove.xtrainer.agent import write

VARIANT = "cup_autonomous_stage_v24_white_object_yaw"
WHITE_YAW_DEG = -30.0  # bounded CAD-planning development constant; not learned


def object_centered_yaw(handle, handle_minus_cup, rotation, yaw_deg=WHITE_YAW_DEG):
    handle = np.asarray(handle, float); delta = np.asarray(handle_minus_cup, float); rotation = np.asarray(rotation, float)
    gravity_yaw = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    cup_center = handle - delta
    return cup_center + gravity_yaw @ delta, gravity_yaw @ rotation, cup_center


class ObjectYawExecutor(InitialPartContractExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.white_object_yaw_plan = None

    def _ensure_yaw_plan(self, pose):
        if self.white_object_yaw_plan is not None: return self.white_object_yaw_plan
        plan = self.direct_plan; base_rotation = np.asarray(plan["entry"][1], float)
        delta = np.asarray(plan["prior"]["handle_minus_cup_visible_m"], float)
        pad = np.asarray(plan["pad"], float); height = float(plan["targets"]["transport_handle_height_m"])
        current_handle = np.asarray(pose[0]) + np.asarray(pose[1]) @ pad; current_handle[2] = height
        target_handle = np.asarray(plan["targets"]["handle_support_goal_m"], float).copy(); target_handle[2] = height
        support_handle = np.asarray(plan["targets"]["handle_support_goal_m"], float).copy()
        source_yaw, yaw_rotation, source_cup = object_centered_yaw(current_handle, delta, base_rotation)
        target_yaw, target_rotation, target_cup = object_centered_yaw(target_handle, delta, base_rotation)
        support_yaw, support_rotation, support_cup = object_centered_yaw(support_handle, delta, base_rotation)
        self.white_object_yaw_plan = {
            "yaw_deg": WHITE_YAW_DEG, "development_constant_not_learned": True,
            "base_transport_handle_height_m": height,
            "source": {"original_handle_m": current_handle.tolist(), "yaw_handle_m": source_yaw.tolist(), "cup_center_m": source_cup.tolist()},
            "target": {"original_handle_m": target_handle.tolist(), "yaw_handle_m": target_yaw.tolist(), "cup_center_m": target_cup.tolist()},
            "support": {"original_handle_m": support_handle.tolist(), "yaw_handle_m": support_yaw.tolist(), "cup_center_m": support_cup.tolist()},
            "cup_center_invariant_under_yaw": True, "upright_preserved_by_gravity_axis_yaw": True,
            "final_support_height_unchanged": True, "original_task_goal_unchanged": True,
            "grasp_transform": "tool rotation and observed handle-minus-cup vector rotated together about shared-base gravity Z",
            "yaw_rotation": yaw_rotation.tolist(), "pad_TCP_local_m": pad.tolist(),
        }
        write(self.output / "WHITE_OBJECT_YAW_PLAN.json", self.white_object_yaw_plan)
        return self.white_object_yaw_plan

    def segment_goals(self, stage, pose):
        if stage["cycle_id"] != "white_direct" or stage["phase"] not in ("Raise", "Move", "Lower", "Retreat"):
            return super().segment_goals(stage, pose)
        if stage["phase"] == "Retreat":
            point = np.asarray(pose[0]).copy(); point[2] += self.direct_plan["prior"]["observed_visible_height_m"]
            return [(point, np.asarray(pose[1]).copy())]
        yaw = self._ensure_yaw_plan(pose); rotation = np.asarray(yaw["yaw_rotation"], float); pad = np.asarray(yaw["pad_TCP_local_m"], float)
        if stage["phase"] == "Raise":
            base_rotation = np.asarray(self.direct_plan["entry"][1], float)
            original = tcp_at_handle(np.asarray(yaw["source"]["original_handle_m"]), base_rotation, pad)
            turned = tcp_at_handle(np.asarray(yaw["source"]["yaw_handle_m"]), rotation, pad)
            return [original, turned]
        key = "target" if stage["phase"] == "Move" else "support"
        return [tcp_at_handle(np.asarray(yaw[key]["yaw_handle_m"]), rotation, pad)]

    def run(self):
        result = super().run()
        result.update(execution_variant=VARIANT, white_object_yaw_plan=self.white_object_yaw_plan,
                      white_yaw_development_constant_deg=WHITE_YAW_DEG)
        return result

SharedCupExecutor = ObjectYawExecutor
SharedSAMVision = InitialPartContractVision
