"""V25 Lower follows fresh held-cup surface geometry; release settles naturally."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from autonomous_stage_runtime_object_yaw import ObjectYawExecutor, InitialPartContractVision
from autonomous_stage_runtime_box_support import supported_box_goal
from cup_v25_visible_support_geometry import support_geometry, dominant_visible_surface
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError

class VisibleSupportExecutor(ObjectYawExecutor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.visible_support_plan=None;self.visible_support_goal=None
    def segment_goals(self,stage,pose):
        if stage['cycle_id']!='white_direct' or stage['phase']!='Lower':return super().segment_goals(stage,pose)
        cup=self.vision.cloud('white_mug');box=self.vision.cloud('box')
        surface,_=dominant_visible_surface(cup)
        report=support_geometry(cup,box,pose[0],pose[1])
        nominal=np.asarray(self.white_object_yaw_plan['target']['cup_center_m'])[:2]
        safe,footprint=supported_box_goal(box,surface,nominal)
        center=np.median(surface,axis=0);delta_xy=safe-center[:2]
        goal=np.asarray(pose[0]).copy();goal[:2]+=delta_xy;goal[2]+=report['proposed_lower_delta_z_m']
        if not -.10<=report['proposed_lower_delta_z_m']<=.005 or np.linalg.norm(delta_xy)>.10:
            raise SafetyError('observed support adjustment outside bounded development envelope')
        aligned=np.asarray(pose[0]).copy();aligned[:2]+=delta_xy
        rotation=np.asarray(pose[1]).copy()
        report.update(footprint=footprint,visible_cup_center_m=center.tolist(),goal_TCP_m=goal.tolist(),
                      goal_rotation=rotation.tolist(),alignment_TCP_m=aligned.tolist(),
                      original_initial_pad_support_goal_replaced=True,
                      orientation_contract='keep currently measured wrist orientation; tilted cup not declared upright',
                      completion_contract='own TCP within original3mm/.02rad; later opened/retreated actual Root Complete proves stable original support goal',
                      clearance_not_contact_proof=True)
        self.visible_support_plan=report;self.visible_support_goal=(goal,rotation)
        Image.fromarray(self.vision._memo.rgb).save(self.output/'WHITE_LOWER_PUBLIC_RGB.png') if self.vision._memo is not None else None
        np.savez_compressed(self.output/'WHITE_LOWER_PUBLIC_GEOMETRY.npz',cup_cloud=cup,box_cloud=box,
                            current_TCP_m=np.asarray(pose[0]),current_rotation=rotation)
        write(self.output/'WHITE_LOWER_VISIBLE_SUPPORT_PLAN.json',report)
        return [(aligned,rotation),(goal,rotation)]
    def command(self,stage,*args,**kwargs):
        try:return super().command(stage,*args,**kwargs)
        finally:
            if stage['cycle_id']=='white_direct' and stage['phase']=='Lower' and self.visible_support_goal is not None:
                rows=self.robot.samples();q=np.stack([row.q for row in rows]);pose=self._shared_pose(q,self.command_g,'left')
                goal,R=self.visible_support_goal
                write(self.output/'WHITE_LOWER_ACTUAL_ENDPOINT.json',{
                    'stamp':rows[0].stamp,'actual_q_rad':q.tolist(),'actual_TCP_m':pose[0].tolist(),
                    'actual_rotation':pose[1].tolist(),'goal_TCP_m':goal.tolist(),'goal_rotation':R.tolist(),
                    'position_error_m':float(np.linalg.norm(pose[0]-goal)),
                    'rotation_error_rad':float(Rotation.from_matrix(R@pose[1].T).magnitude()),
                    'source':'actual own robot feedback; no contact/object truth',
                    'position_tolerance_m':.003,'rotation_tolerance_rad':.02})
    def run(self):
        result=super().run();result.update(visible_support_plan=self.visible_support_plan,
                    execution_variant='cup_v25_current_visible_support_with_real_capabilities')
        return result

SharedCupExecutor=VisibleSupportExecutor
SharedSAMVision=InitialPartContractVision
