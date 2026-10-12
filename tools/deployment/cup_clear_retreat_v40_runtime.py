"""Object-relative empty-gripper clearance makes release observable before Complete."""
import numpy as np
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError
from cup_visible_surfaces_v39 import SharedCupExecutor as BaseExecutor,SharedSAMVision
EXTRA='c02.07b.clear_retreat'

def empty_clearance_goals(pose,cup_cloud,pad_vertices):
    p,R=pose;cup=np.asarray(cup_cloud);verts=np.asarray(pad_vertices)@R.T
    if len(cup)<200 or not np.isfinite(cup).all():raise SafetyError('delivered cup current geometry unavailable')
    center=np.median(cup,axis=0);height=float(np.ptp(cup[:,2]));span=float(np.linalg.norm(np.ptp(cup[:,:2],axis=0)))
    if height<=0 or span<=0:raise SafetyError('delivered cup extent unknown')
    upper=np.asarray(p).copy();upper[2]=max(upper[2],float(cup[:,2].max()-verts[:,2].min()+.25*height))
    direction=upper[:2]-center[:2]
    if np.linalg.norm(direction)<.001:direction=np.array([1.,0.]) # robot-base +X only for coincident horizontal positions
    direction/=np.linalg.norm(direction)
    end=upper.copy();end[:2]+=direction*(span+float(np.max(np.linalg.norm(verts[:,:2],axis=1))))
    return [(upper,R.copy()),(end,R.copy())],{'source':'current delivered-cup SAM depth extent + own CAD pad envelope','observed_height_m':height,'observed_horizontal_diagonal_m':span,'cup_center_m':center.tolist(),'raised_TCP_m':upper.tolist(),'cleared_TCP_m':end.tolist(),'not_contact_or_release_certificate':True}

class ClearRetreatExecutor(BaseExecutor):
    def _code_goals_and_gripper(self,stage,pose,obs):
        if stage['stage_id']!=EXTRA:return super()._code_goals_and_gripper(stage,pose,obs)
        cup=self.vision.cloud(stage['object']);verts=self.pad_library['arms'][stage['arm']]['pad_vertices_TCP_local_m']
        goals,record=empty_clearance_goals(pose,cup,verts);write(self.output/'EMPTY_GRIPPER_CLEARANCE_PLAN.json',record)
        g=self.command_g.copy();g[0 if stage['arm']=='left' else 1]=1.
        return goals,g

SharedCupExecutor=ClearRetreatExecutor
