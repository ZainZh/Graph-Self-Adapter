"""Unconfirmed closure then bounded original Lift; transport needs fresh API confirmation."""
import copy,numpy as np
from pathlib import Path
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError
from prove.xtrainer.runtime import Executor
from multitask_ordinary_route import CARRY
from multitask_ordinary_runtime import VARIANT
from ordinary_v25_apple_pad_runtime import ApplePadExecutor,SharedSAMVision
from multitask_ordinary_runtime_v21 import relative_lift_goal

class ObservedLiftExecutor(ApplePadExecutor):
    def remember_attempt(self,stage,samples,obs):
        from PIL import Image
        import json,hashlib
        frame=self.output/'PRE_LIFT_PUBLIC_RGB.png';Image.fromarray(obs.rgb).save(frame)
        write(self.output/'PRE_LIFT_PUBLIC_OBSERVATION.json',{'RGB':str(frame.resolve()),'RGB_sha256':hashlib.sha256(frame.read_bytes()).hexdigest(),'stamp':obs.stamp,'public_robot':[{'TCP_m':v.tcp.tolist(),'q':v.q.tolist(),'stamp':v.stamp} for v in samples],'object_role_point':None if obs.roles.get(stage['object']) is None else obs.roles[stage['object']].tolist(),'source':'current actual public RGB and robot feedback before original Lift'})
        self.attempt_pose=self._actual_pose(samples,stage['arm']);self.attempt_g=self.command_g.copy()
        self.accepted_grasp=None;self.attachment=None;self.current_holding=None
        self.holding_source='unconfirmed closure; bounded diagnostic Lift only, no carry permission'
        write(self.output/'GRASP_ATTEMPT.json',{'holding':None,'pose':self.attempt_pose[0].tolist(),'gripper_command':self.attempt_g.tolist(),'next':'bounded original Lift for visible outcome; no transfer yet'})
    def _code_goals_and_gripper(self,stage,pose,obs):
        if stage['phase']!='Lift':return super()._code_goals_and_gripper(stage,pose,obs)
        self._ground_red_cycle(stage,obs);times=self._reference_times(stage)
        start=self._shared_pose(np.asarray(self.b['qcurve'](times[0])),self.command_g,stage['arm'])
        goals=[relative_lift_goal(self.attempt_pose,start,self._shared_pose(np.asarray(self.b['qcurve'](t)),self.command_g,stage['arm'])) for t in times]
        return goals,self.attempt_g.copy()
    def predicates(self,stage,progress,target,samples,obs):
        previous=self.attachment;coarse=None if self.coarse_source is None else self.coarse_source.copy()
        facts=super().predicates(stage,progress,target,samples,obs)
        if stage['phase'] in ('Grasp','Lift') and self.accepted_grasp is None:
            self.attachment=previous;self.coarse_source=coarse;facts['tentative_holding']=None
        return facts
    def refresh_lift_attachment(self,stage,samples,obs):
        after=Executor.semantics(self,stage,['holding'],(samples,obs))
        if self.visual_facts.get('holding') is not True:raise SafetyError('API did not confirm actual lifted apple; transport forbidden')
        i=0 if stage['arm']=='left' else 1;point=after[1].roles.get(stage['object'])
        if point is None:raise SafetyError('lifted object current public pose unavailable')
        pose=self._actual_pose(after[0],stage['arm']);own_delta=pose[0]-self.attempt_pose[0]
        visible_delta=np.asarray(point)-np.asarray(self.initial_roles[stage['object']])
        if own_delta[2]<.003 or visible_delta[2]<.003:raise SafetyError('public object and actor elevation evidence insufficient')
        self.accepted_grasp={'object':stage['object'],'cycle':stage['cycle_id'],'index':i,'g_command':float(self.command_g[i]),'aperture_m':after[0][i].aperture,'object_TCP_distance_m':float(np.linalg.norm(np.asarray(point)-pose[0])),'source':'fresh API confirmed elevated visible object, not contact truth'}
        self.current_holding=True;self._manipulated.add(stage['object']);self.accepted_grasp_pose[stage['cycle_id']]=self.attempt_pose
        self.register_attachment(stage,*after)
        self.attachment_updates=[{'stage_id':stage['stage_id'],'API_record':self.judge.records[-1],'own_delta_m':own_delta.tolist(),'visible_delta_m':visible_delta.tolist(),'source':'current SAM RGBD and own feedback; original goal unchanged'}]

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

SharedCupExecutor=ObservedLiftExecutor
