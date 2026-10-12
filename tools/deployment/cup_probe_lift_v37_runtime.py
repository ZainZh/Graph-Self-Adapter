"""Cup closure-Lift observation macro; carrying requires fresh API-confirmed elevation."""
import numpy as np,hashlib
from pathlib import Path
from PIL import Image
from prove.xtrainer.runtime import Executor
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError
from direct_white_runtime import CARRY,VARIANT
from cup_visible_plane_v36 import SharedCupExecutor as BaseExecutor,SharedSAMVision
from multitask_ordinary_runtime_v21 import relative_lift_goal

class ObservedCupLiftExecutor(BaseExecutor):
    def remember_attempt(self,stage,samples,obs):
        self.attempt_pose=self._shared_pose(np.stack([s.q for s in samples]),self.command_g,stage['arm']);self.attempt_g=self.command_g.copy()
        self.attempt_object=np.asarray(obs.roles[stage['object']]).copy()
        self.current_holding=None;self.root_node_state=None;self.macro_grasp=None;self.attachment=None
        self.holding_source='unconfirmed closed command; only original bounded diagnostic Lift permitted'
        frame=self.output/'PRE_LIFT_PUBLIC_RGB.png';Image.fromarray(obs.rgb).save(frame)
        write(self.output/'PRE_LIFT_PUBLIC_OBSERVATION.json',{'RGB':str(frame.resolve()),'RGB_sha256':hashlib.sha256(frame.read_bytes()).hexdigest(),'stamp':obs.stamp,'public_robot':[{'TCP_m':s.tcp.tolist(),'q':s.q.tolist(),'stamp':s.stamp} for s in samples],'object_role_point':self.attempt_object.tolist(),'source':'actual same-episode pre-Lift RGB and own feedback','cycle':stage['cycle_id']})
    def _red_goal(self,stage,t,pose):
        if stage['phase']!='Lift':return super()._red_goal(stage,t,pose)
        start=stage['reference']['start_time_sec']
        return relative_lift_goal(self.attempt_pose,self._shared_pose(np.asarray(self.b['qcurve'](start)),self.attempt_g,stage['arm']),self._shared_pose(np.asarray(self.b['qcurve'](t)),self.attempt_g,stage['arm']))
    def _code_goals_and_gripper(self,stage,pose,obs):
        if stage['phase']!='Lift':return super()._code_goals_and_gripper(stage,pose,obs)
        previous=self.macro_grasp
        self.macro_grasp={'role':stage['object'],'cycle':stage['cycle_id'],'g_command':float(self.attempt_g[0 if stage['arm']=='left' else 1]),'confirmed':False,'source':'command profile only; not holding belief'}
        try:return super()._code_goals_and_gripper(stage,pose,obs)
        finally:self.macro_grasp=previous
    def predicates(self,stage,progress,target,samples,obs):
        prior=self.attachment;coarse=None if self.coarse_source is None else self.coarse_source.copy()
        facts=super().predicates(stage,progress,target,samples,obs)
        if stage['phase'] in ('Grasp','Lift') and self.macro_grasp is None:
            self.attachment=prior;self.coarse_source=coarse;facts['unconfirmed_holding']=None
        return facts
    def confirm_lift(self,stage,samples,obs):
        self.vision.invalidate();self.vision.memo_enabled=False
        try:after=Executor.semantics(self,stage,['holding'],(samples,obs))
        finally:self.vision.invalidate();self.vision.memo_enabled=True
        if self.visual_facts.get('holding') is not True:raise SafetyError('current elevated cup holding UNKNOWN/false; no transport')
        index=0 if stage['arm']=='left' else 1;point=after[1].roles.get(stage['object'])
        if point is None:raise SafetyError('current lifted cup public position missing')
        pose=self._shared_pose(np.stack([s.q for s in after[0]]),self.command_g,stage['arm'])
        if pose[0][2]-self.attempt_pose[0][2]<.003:raise SafetyError('own actual Lift displacement missing')
        self.macro_grasp={'role':stage['object'],'cycle':stage['cycle_id'],'aperture_m':after[0][index].aperture,'g_command':float(self.command_g[index]),'Root_certified_at':self.clock(),'source':'API confirmed real pre/post-Lift RGB; not measured contact'}
        self.root_node_state={'role':stage['object'],'holding':True,'support_permission':None,'certified_at':self.clock(),'certified_node':stage['stage_id'],'certified_aperture_m':after[0][index].aperture,'initial_object_TCP_delta':np.asarray(point)-pose[0],'state_source':'fresh actual API visual holding'}
        self.current_holding=True;self._manipulated.add(stage['object']);self.node_state_log.append({**self.root_node_state,'initial_object_TCP_delta':self.root_node_state['initial_object_TCP_delta'].tolist()})
        if stage['cycle_id']!='white_direct':self.register_attachment(stage,*after)
        write(self.output/('LIFT_CONFIRMED_'+stage['cycle_id']+'.json'),{'API':self.judge.records[-1],'own_delta_m':(pose[0]-self.attempt_pose[0]).tolist(),'actual_public_object':np.asarray(point).tolist(),'holding':True,'transport_now_permitted':True})

    def run(self):
        failure=None;complete=False;updates=0;stops=[]
        try:
            self.start()
            for stage in self.b['graph']['stages']:
                self.stage=stage;self.stage_started=self.clock();self.visual_facts={};self.fact_stamp=-float('inf');self.b['skill'].reset_phase();done=False;elapsed=0.
                write(self.output/'NODE_STATUS.json',{'stage_id':stage['stage_id'],'phase':stage['phase'],'direct_box_route':True,'Root_boundary':stage['phase'] in ('Grasp','Complete')})
                if stage['phase']=='Complete':self.semantics(stage,['released','placement_complete'],self.capture());self.stage_trace.append({'stage_id':stage['stage_id'],'completed':True,'commands':0});continue
                for count in range(self.cfg['maximum_node_control_updates']):
                    updates+=1;self.vision.begin_control_update(updates);self.guard_limits();samples,obs=self.capture()
                    if stage['phase'] in CARRY and stage['phase']!='Lift' and self.holding(samples,stage) is not True:raise SafetyError('direct transfer carrying anomaly')
                    elapsed+=self.cfg['command_dt_sec'];target,progress,record=self.command(stage,elapsed,samples,obs,self.cfg['command_dt_sec']);samples,obs=self.capture()
                    if False:samples,obs=self.semantics(stage,['holding'],(samples,obs))
                    facts=self.predicates(stage,progress,target,samples,obs);self.trace.append({'stage_id':stage['stage_id'],'command_index':self.command_index,'facts':facts,'issued_q':target.tolist(),'issued_g':self.command_g.tolist(),'actual_q':[s.q.tolist() for s in samples],'actual_aperture_m':[s.aperture for s in samples],'controller':record,'holding_state_source':getattr(self,'holding_source','unknown'),'release_open_progress':getattr(self,'release_progress_record',None) if stage['phase']=='Release' else None})
                    if any(facts.get(k) is not True for k in stage['invariants']):raise SafetyError('direct route invariant unknown/false')
                    if all(facts.get(k) is True for k in stage['done']):done=True;break
                self.stage_trace.append({'stage_id':stage['stage_id'],'completed':done,'commands':count+1})
                if not done:raise SafetyError('direct route node budget exhausted')
                if stage['phase']=='Grasp':self.remember_attempt(stage,*self.capture())
                if stage['phase']=='Lift':self.confirm_lift(stage,*self.capture())
            complete=True
        except Exception as exc:
            failure=str(exc)
            for stop in self.stops:
                try:stop.stop();stops.append({'stopped':True})
                except Exception as err:stops.append({'error':str(err)})
        result={'schema':'graph-self-adapter.xtrainer-run.v1','execution_variant':VARIANT,'graph_completed':complete,'failure':failure,'graph_stages':len(self.b['graph']['stages']),'stage_trace':self.stage_trace,'trace':self.trace,'Root_decisions':self.judge.records,'root_request_count':len(self.judge.records),'fixed_z':True,'latent_updates':0,'mode':self.cfg['mode'],'hardware_motion_commands':self.robot.commands,'hardware_command_attempts':self._hardware_attempts,'hardware_command_acknowledgments':self._hardware_acknowledged,'stop_results':stops,'rgbd_discard_log':getattr(self,'rgbd_discard_log',[]),'SAM_same_update_reuse_count':self.vision.memo_reuse_count,'unsupported_Agent_waits':0,'Root_boundaries':['Grasp','Complete after open and empty retreat'],'original_final_goal_unchanged':True,'white_upright_prior_no_airborne_target':True,'policy_claim':'red frozen reference+residual; explicit observed white direct-place Code; not learned handle policy'}
        write(self.output/'RESULT.json',result);return result

SharedCupExecutor=ObservedCupLiftExecutor
