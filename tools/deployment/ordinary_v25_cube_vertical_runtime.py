"""Observed-height vertical Lift from actual accepted Grasp; no lateral sweep or tool rotation."""
import numpy as np
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from ordinary_v25_cube_pad_runtime import CubePadExecutor,CubePadVision
class VerticalLiftExecutor(CubePadExecutor):
    def _code_goals_and_gripper(self,stage,pose,obs):
        if stage['phase']!='Lift':return super()._code_goals_and_gripper(stage,pose,obs)
        anchor=self.accepted_grasp_pose.get(stage['cycle_id']);cert=self.accepted_grasp
        if anchor is None or cert is None or cert['cycle']!=stage['cycle_id']:raise SafetyError('accepted provisional own Grasp anchor missing')
        height=float(self.geometry['cube_measurements'][stage['object']]['observed_height_m']);clearance=float(self.cfg['vertical_lift_clearance_m'])
        if clearance!=.010:raise SafetyError('exact declared vertical Lift clearance must be10mm')
        endpoint=anchor[0]+np.array([0.,0.,height+clearance]);g=self.command_g.copy();g[cert['index']]=cert['g_command']
        write(self.output/(stage['cycle_id']+'_VERTICAL_LIFT_PLAN.json'),{'source':'actual accepted Grasp own TCP/R and declared observed cube height; current gravity +worldZ; no object truth','start_actual_TCP_m':anchor[0].tolist(),'end_TCP_m':endpoint.tolist(),'observed_object_height_m':height,'additional_clearance_m':clearance,'XY_preserved':True,'rotation_preserved':True,'holding_provisional_until_real_fresh_Lift_observation':True,'reference_lateral_motion_not_used':True})
        return [(anchor[0].copy(),anchor[1].copy()),(endpoint,anchor[1].copy())],g
SharedCupExecutor=VerticalLiftExecutor
SharedSAMVision=CubePadVision
