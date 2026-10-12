"""Uncertain held cube targets current observed bowl interior rather than reference endpoint."""
import numpy as np
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from cube_single_v45_runtime import SingleCubeExecutor,SharedSAMVision
class BowlCenterExecutor(SingleCubeExecutor):
    def _code_goals_and_gripper(self,stage,pose,obs):
        if stage['phase']!='Move':return super()._code_goals_and_gripper(stage,pose,obs)
        if not self.lift_verified or not self.attachment:raise SafetyError('verified current holding required before bowl transport')
        height=self.geometry['cube_measurements'][stage['object']]['observed_height_m'];radius=height/np.sqrt(2)+self.attachment['uncertainty_xy_m']+.003
        margin=self.bowl_region['inner_radius_m']-radius
        if margin<0:raise SafetyError('current bowl too small for cube uncertainty envelope')
        stamp,rgb,depth,K=self.vision.camera.latest_public_capture;mask=self.vision.public_masks['bowl']&np.isfinite(depth)&(depth>0);y,x=np.where(mask);z=depth[y,x];pts=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]@self.vision.T[:3,:3].T+self.vision.T[:3,3]
        rim_z=float(np.quantile(pts[:,2],.95));center=np.asarray(self.bowl_region['center_xy_m']);goal=np.r_[center,rim_z+height/2+self.attachment['uncertainty_z_m']]
        align=next(s for s in self.b['graph']['stages'] if s['phase']=='Align');native=self._shared_pose(np.asarray(self.b['qcurve'](align['reference']['end_time_sec'])),self.command_g,stage['arm'])
        nominal=native[0]+native[1]@self.attachment['a_ref'];self.coarse_target=goal-nominal
        current_body=pose[0]+pose[1]@self.attachment['a_actual'];safe_z=max(current_body[2],goal[2]+height)
        raised=current_body.copy();raised[2]=safe_z;transport=np.r_[center,safe_z]
        goals=[(p-pose[1]@self.attachment['a_actual'],pose[1].copy()) for p in (raised,transport)]
        g=self.command_g.copy();g[self.accepted_grasp['index']]=self.accepted_grasp['g_command']
        write(self.output/'CURRENT_BOWL_CENTER_TARGET.json',{'source':'current public SAM bowl rim + declared uncertain initial-cube attachment','desired_body_goal_m':goal.tolist(),'transport_body_goal_m':transport.tolist(),'body_envelope_radius_m':radius,'containment_margin_m':margin,'old_reference_endpoint_not_target':True,'hidden_object_pose_read':False})
        return goals,g
SharedCupExecutor=BowlCenterExecutor
