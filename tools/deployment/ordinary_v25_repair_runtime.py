"""Separate cube repair: fresh elevated holding judgement replaces brittle 6mm SAM median equality."""
import copy,io,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from ordinary_v25_runtime import OrdinaryV25Executor
from shared_cup_vision import SharedSAMVision as BaseVision
from prove.xtrainer.runtime import Executor
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from multitask_ordinary_runtime import OrdinaryExecutor
from three_task_v25_public_io import PublicRobot
class EvidenceVision(BaseVision):
    def capture(self):
        obs=super().capture();root=self.evidence_root/'BOUNDARIES';root.mkdir(parents=True,exist_ok=True)
        buffer=io.BytesIO();Image.fromarray(obs.rgb).save(buffer,format='PNG');raw=buffer.getvalue();sha=hashlib.sha256(raw).hexdigest();dest=root/('RGB_'+sha+'.png')
        if not dest.exists():
            tmp=dest.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(dest)
        if hasattr(self.camera,'latest_public_capture'):
            stamp,rgb,depth,K=self.camera.latest_public_capture
            np.savez_compressed(root/('RGBD_'+sha+'.npz'),depth_m=depth,K=K,T=self.T,stamp=np.asarray(stamp))
        write(root/'LATEST.json',{'path':str(dest),'sha256':sha,'observed_stamp':obs.stamp,'actual_RGB':True,'stage':self.stage_context()})
        return obs
class RepairExecutor(OrdinaryV25Executor):
    def __init__(self,bundle,robot,*args,**kwargs):super().__init__(bundle,PublicRobot(robot,source='sim_real_public_contract'),*args,**kwargs)
    def refresh_lift_attachment(self,stage,samples,obs):
        anchor=self.accepted_grasp_pose.get(stage['cycle_id']);current=self._actual_pose(samples,stage['arm'])
        if anchor is None or np.linalg.norm(current[0]-anchor[0])<.003:raise SafetyError('actual proprio nonzero Lift missing')
        if obs.roles.get(stage['object']) is None:raise SafetyError('fresh lifted object role missing')
        after=Executor.semantics(self,stage,['holding'],(samples,obs))
        if self.visual_facts.get('holding') is not True:raise SafetyError('fresh elevated semantic holding unknown/false')
        OrdinaryExecutor.refresh_lift_attachment(self,stage,*after)
        actual=self._actual_pose(after[0],stage['arm']);ref=self._shared_pose(np.asarray(self.b['qcurve'](stage['reference']['end_time_sec'])),self.command_g,stage['arm'])
        self.attachment['R_demo_end']=ref[1].copy();self.attachment['R_actual_grasp']=actual[1].copy()
        self.attachment_updates[-1].update(reason='fresh elevated RGB multi-frame holding plus own nonzero Lift; surface median equality is not rigid contact truth',semantic_record=self.judge.records[-1],actual_Lift_distance_m=float(np.linalg.norm(actual[0]-anchor[0])))
    def run(self):
        result=super().run();result.update(execution_variant='ordinary_public_semantic_lift_repair_v25',semantic_lift_boundary=True);write(self.output/'RESULT.json',result);return result
SharedCupExecutor=RepairExecutor
SharedSAMVision=EvidenceVision
