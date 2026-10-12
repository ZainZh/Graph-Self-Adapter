"""Opened engineering contact tolerance revision; all V17 motion logic unchanged."""
from dataclasses import replace
import numpy as np
from trajectory_white_runtime_v17 import TrajectoryWhiteExecutor as V17Executor,SharedSAMVision
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write

VARIANT='white_stage_trajectory_v18_lower_contact'

class TrajectoryWhiteExecutor(V17Executor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lower_contact_tolerance=float(self.cfg['trajectory']['lower_contact_position_tolerance_m'])
        if not 0<self.lower_contact_tolerance<=.003:raise SafetyError('Lower contact tolerance must be positive and <=3mm')
    def _is_lower(self,stage):return bool(stage and stage['cycle_id']=='white_direct' and stage['phase']=='Lower')
    def _reached(self,goal,pose):
        if self._is_lower(self.stage):
            error=self._error(goal,pose)
            return bool(np.linalg.norm(error[:3])<=self.lower_contact_tolerance and np.linalg.norm(error[3:])<=.02)
        return super()._reached(goal,pose)
    def command(self,stage,*args,**kwargs):
        original=self.trajectory_limits
        if self._is_lower(stage):self.trajectory_limits=replace(original,position_tolerance=self.lower_contact_tolerance)
        try:return super().command(stage,*args,**kwargs)
        finally:self.trajectory_limits=original
    def run(self):
        result=super().run();result.update(execution_variant=VARIANT,white_Lower_contact_position_tolerance_m=self.lower_contact_tolerance,threshold_revision_claim='opened engineering; not blind evaluation')
        write(self.output/'RESULT.json',result);return result

SharedCupExecutor=TrajectoryWhiteExecutor
