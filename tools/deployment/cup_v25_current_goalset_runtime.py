"""Current observed goal-set joint planning; support yaw is solved, never fixed."""
import numpy as np
from cup_v25_own_history_retreat_runtime import OwnHistoryRetreatExecutor,SharedSAMVision
from cup_v25_current_goalset_solver import solve_goalset
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError

class CurrentGoalSetExecutor(OwnHistoryRetreatExecutor):
    def segment_goals(self,stage,pose):
        goals=super().segment_goals(stage,pose)
        if stage['cycle_id']!='white_direct' or stage['phase']!='Lower':return goals
        cup=self.vision.cloud('white_mug');box=self.vision.cloud('box');red=self.vision.cloud('red_mug')
        current=self.robot.samples();q=np.stack([row.q for row in current]);g=self.command_g.copy()
        nominal=np.asarray(self.white_object_yaw_plan['target']['cup_center_m'])[:2]
        solution,report=solve_goalset(self.robot.fk,q,g,cup,box,red,nominal,self.trajectory_limits)
        write(self.output/'CURRENT_ENTRY_GOALSET_PLAN.json',report)
        np.savez_compressed(self.output/'CURRENT_ENTRY_PUBLIC_GEOMETRY.npz',cup_cloud=cup,box_cloud=box,red_cloud=red,current_q_rad=q,current_TCP_m=np.asarray(pose[0]),current_rotation=np.asarray(pose[1]))
        if solution is None:raise SafetyError('current-entry support/own-joint/yaw goalset infeasible')
        self.visible_support_goal=solution
        self.visible_support_plan.update(footprint=report,goal_TCP_m=solution[0].tolist(),goal_rotation=solution[1].tolist(),
            release_clearance_m=.020,current_entry_goalset_solved=True,relative_support_yaw_deg=report['relative_yaw_deg'],
            static_old_entry_yaw_not_reused=True,frame_anchor_consistent=True,
            task_goal='both cups released stably supported by box; exact demonstration XY/yaw not required',
            NN_weights_and_latent_unchanged=True,original_final_goal_and_servo_caps_unchanged=True)
        write(self.output/'WHITE_LOWER_VISIBLE_SUPPORT_PLAN.json',self.visible_support_plan)
        return [solution]

SharedCupExecutor=CurrentGoalSetExecutor
