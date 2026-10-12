"""One bounded Lift command; stationary actual-visual wait has separate existing judge budget."""
from prove.xtrainer.io import SafetyError
from ordinary_v25_cube_baseline_vertical_runtime import BaselineVerticalExecutor,CubePadVision
from multitask_ordinary_runtime import OrdinaryExecutor
class StationaryJudgeExecutor(BaselineVerticalExecutor):
    def __init__(self,*args,**kwargs):
        self.lift_active=False;self.lift_active_started=None
        super().__init__(*args,**kwargs)
    def capture(self):
        samples,obs=OrdinaryExecutor.capture(self)
        if self.provisional and not self.lift_verified and self.stage and self.stage['phase'] in ('Grasp','Lift'):
            cert=self.accepted_grasp
            if cert and self.command_g[cert['index']]<=cert['g_command']+.1:
                self.current_holding=True;self.holding_source='provisional_one_Lift_permission; stationary judge wait is not holding evidence'
        return samples,obs
    def guard_limits(self):
        super().guard_limits()
        if self.lift_active and self.clock()-self.lift_active_started>60:raise SafetyError('one Lift active plan/stream budget exhausted')
    def command(self,stage,*args,**kwargs):
        if stage['phase']!='Lift':return super().command(stage,*args,**kwargs)
        self.lift_active=True;self.lift_active_started=self.clock()
        try:return super().command(stage,*args,**kwargs)
        finally:self.lift_active=False
SharedCupExecutor=StationaryJudgeExecutor
SharedSAMVision=CubePadVision
