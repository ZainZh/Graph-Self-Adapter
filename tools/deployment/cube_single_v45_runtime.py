"""Single yellow cube, bounded unconfirmed closure/Lift then fresh API + colour-witness guard."""
import numpy as np
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError
from multitask_ordinary_runtime import CARRY,VARIANT
from ordinary_v25_cube_state_guard_runtime import StatePoseExecutor,SharedSAMVision
class SingleCubeExecutor(StatePoseExecutor):
    def start(self):
        self.capture()
        roi=self.vision.config.get('background_table_roi')
        if roi:
            cx=(roi[0]+roi[2])//2;cy=(roi[1]+roi[3])//2;self.cfg['cube_table_ROI']=[cx-8,cy-8,cx+8,cy+8]
        super().start()
    def remember_attempt(self,stage,samples,obs):
        i=0 if stage['arm']=='left' else 1
        self.accepted_grasp={'object':stage['object'],'cycle':stage['cycle_id'],'index':i,'g_command':float(self.command_g[i]),'aperture_m':None,'object_TCP_distance_m':.10,'confirmed':False,'source':'unconfirmed closure parameters for one bounded diagnostic Lift, not holding proof'}
        self.accepted_grasp_pose[stage['cycle_id']]=self._actual_pose(samples,stage['arm']);self.provisional=False;self.lift_verified=False;self.current_holding=None;self.attachment=None
        self.pre_lift_frame=obs.history[-1]
    def capture(self):
        cert=getattr(self,'accepted_grasp',None)
        pending=cert and cert.get('confirmed') is False and not self.lift_verified
        if pending:self.accepted_grasp=None
        try:samples,obs=super().capture()
        finally:
            if pending:self.accepted_grasp=cert;self.current_holding=None
        obs.roles={k:v for k,v in obs.roles.items() if k!='table_patch'}
        return samples,obs
    def refresh_lift_attachment(self,stage,samples,obs):
        super().refresh_lift_attachment(stage,samples,obs)
        self.accepted_grasp['confirmed']=True
    def predicates(self,stage,progress,target,samples,obs):
        previous=self.attachment;source=None if self.coarse_source is None else self.coarse_source.copy()
        facts=super().predicates(stage,progress,target,samples,obs)
        if stage['phase'] in ('Grasp','Lift') and not self.lift_verified:
            self.attachment=previous;self.coarse_source=source;facts['tentative_holding']=None
        return facts

    def run(self):
        complete=False;failure=None;stopped=[]
        try:
            self.start()
            for stage in self.b['graph']['stages']:
                self.stage=stage;self.stage_started=self.clock();self.visual_facts={};self.b['skill'].reset_phase();done=False;elapsed=0.
                write(self.output/'NODE_STATUS.json',{'task':self.b['task'],'stage_id':stage['stage_id'],'phase':stage['phase'],'supported_semantic_boundary':stage['phase'] in ('Grasp','Complete')})
                if stage['phase']=='Complete':
                    self.semantics(stage,stage['semantic_facts'],self.capture());self.stage_trace.append({'stage_id':stage['stage_id'],'completed':True,'commands':0});continue
                for update in range(self.cfg['maximum_node_control_updates']):
                    self.guard_limits();samples,obs=self.capture()
                    if stage['phase'] in CARRY and stage['phase']!='Lift' and self.holding(samples,stage) is not True:raise SafetyError('fresh task carrying invariant unknown/false')
                    dt=self.cfg['command_dt_sec'];elapsed+=dt;target,progress,record=self.command(stage,elapsed,samples,obs,dt);samples,obs=self.capture()
                    if False:samples,obs=self.semantics(stage,['holding'],(samples,obs))
                    facts=self.predicates(stage,progress,target,samples,obs)
                    self.trace.append({'stage_id':stage['stage_id'],'command_index':self.command_index,'facts':facts,'controller':record,'issued_q':target.tolist(),'actual_q':[s.q.tolist() for s in samples],'actual_aperture_m':[s.aperture for s in samples],'issued_g':self.command_g.tolist(),'holding_state_source':getattr(self,'holding_source','unknown')})
                    if any(facts.get(k) is not True for k in stage['invariants']):raise SafetyError('ordinary stage invariant unknown/false')
                    if all(facts.get(k) is True for k in stage['done']):done=True;break
                self.stage_trace.append({'stage_id':stage['stage_id'],'completed':done,'commands':update+1})
                if not done:raise SafetyError('task stage own-pose/progress budget exhausted')
                if stage['phase']=='Grasp':self.remember_attempt(stage,*self.capture())
                if stage['phase']=='Lift':self.refresh_lift_attachment(stage,*self.capture())
            complete=True
        except Exception as error:
            failure=str(error)
            for hook in self.stops:
                try:hook.stop();stopped.append({'stopped':True})
                except Exception as stop_error:stopped.append({'error':str(stop_error)})
        self.b['skill'].reset_phase()
        result={'schema':'graph-self-adapter.multitask-ordinary-run.v1','task':self.b['task'],'execution_variant':VARIANT,'graph_completed':complete,'failure':failure,'graph_stages':len(self.b['graph']['stages']),'stage_trace':self.stage_trace,'trace':self.trace,'judge_records':self.judge.records,'semantic_log':self.semantic_log,'attachment_updates':getattr(self,'attachment_updates',[]),'command_attempts':self._hardware_attempts,'command_acknowledgments':self._hardware_acknowledged,'hardware_command_attempts':self._hardware_attempts,'hardware_command_acknowledgments':self._hardware_acknowledged,'stop_results':stopped,'fixed_z':True,'latent_updates':0,'final_original_goal_requires_postseal_private_evaluation':True,'model_action_semantics':'q_reference plus frozen NN residual → FK → object-frame Code IK from actual q','hardware_qualified':False,'unsupported_Agent_waits':0}
        write(self.output/'RESULT.json',result);return result

SharedCupExecutor=SingleCubeExecutor
