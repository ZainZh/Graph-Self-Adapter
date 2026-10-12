"""Generic ordinary-stage Code+frozen residual runtime; not a cup geometry clone.
New route changes only semantic boundary timing. No deployment success is inferred.
"""
import numpy as np
from prove.xtrainer.runtime import Executor
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from object_centered_runtime import ObjectCenteredExecutor
from multitask_ordinary_route import ORDINARY,CARRY,ordinary_route,role_contract
from rgbd_capture_retry import bounded_rgbd_capture
from shared_cup_vision import SharedSAMVision

VARIANT='multitask_ordinary_object_centered_v20'

class OrdinaryExecutor(ObjectCenteredExecutor):
    def __init__(self,bundle,*args,**kwargs):
        if bundle['graph'].get('schema')!='graph-self-adapter.multitask-ordinary-route.v1':raise SafetyError('explicit ordinary semantic route required')
        super().__init__(bundle,*args,**kwargs);self.accepted_grasp=None;self._registered_cycles=set();self.semantic_log=[]
        self.roles=role_contract(bundle['graph'],bundle['refs'])
    def start(self):
        samples,obs=self.capture()
        if any(s.running or s.mode!=5 or np.max(abs(s.qdot))>.01 for s in samples):raise SafetyError('actual current robot not stationary')
        if any(obs.roles.get(role) is None for role in self.roles['SAM_roles']):raise SafetyError('fresh task-specific SAM grounding missing')
        if np.any(self.command_g<0) or np.any(self.command_g>1):raise SafetyError('planned current gripper command invalid')
        self.initial_roles={k:None if p is None else np.asarray(p).copy() for k,p in obs.roles.items()}
        self.initial_aperture=[s.aperture for s in samples];self.previous_cycle=None
    def capture(self):
        samples,obs=bounded_rgbd_capture(self,lambda:Executor.capture(self))
        cert=self.accepted_grasp
        if cert and self.stage and self.stage['cycle_id']==cert['cycle'] and self.stage['phase'] not in ('Release','Wait','Complete'):
            i=cert['index'];point=obs.roles.get(cert['object']);tcp=samples[i].tcp if i==0 else (self.T@np.r_[samples[i].tcp,1])[:3]
            if point is not None and self.visual_anchor is not None:
                object_delta=np.asarray(point)-self.visual_anchor[4];tcp_delta=tcp-self.visual_anchor[3]
                self.comotion_record={'role':cert['object'],'object_delta_m':object_delta.tolist(),'own_TCP_delta_m':tcp_delta.tolist(),'disagreement_m':float(np.linalg.norm(object_delta-tcp_delta)),'nonzero_evidence':bool(np.linalg.norm(tcp_delta)>=.003 and np.linalg.norm(object_delta)>=.003),'source':'actual SAM visible surface+own FK; not rigid/contact truth'}
            empty=samples[i].aperture is not None and cert['aperture_m'] is not None and cert['aperture_m']>.003 and samples[i].aperture<.001
            far=point is not None and np.linalg.norm(np.asarray(point)-tcp)>cert['object_TCP_distance_m']+self.cfg.get('grasp_region_margin_m',.05)
            opened=self.command_g[i]>cert['g_command']+.1
            if point is None:self.current_holding=None;self.holding_source='actual task role unknown'
            elif empty or far or opened:self.current_holding=False;self.holding_source='actual jaw/visible-region/command anomaly'
            else:
                positive=self.current_holding is True and self.holding_source=='nonzero_visible_object_TCP_comotion'
                self.current_holding=True;self.holding_source='positive actual SAM/TCP comotion' if positive else 'supported-boundary tentative grasp within bounded continuous macro; not contact measurement'
        return samples,obs
    def holding(self,samples,stage):
        if stage['holding_contract']=='none':return True
        if not self.accepted_grasp or self.accepted_grasp['object']!=stage['object']:return None
        return self.current_holding
    def target_point(self,obs,stage):
        role=stage['target']
        if role not in self._manipulated and getattr(self,'initial_roles',{}).get(role) is not None:return np.asarray(self.initial_roles[role]).copy()
        return super().target_point(obs,stage)
    def semantics(self,stage,facts,before):
        if stage['phase'] not in ('Grasp','Complete'):raise SafetyError('unsupported transfer anomaly stops; no suspended semantic wait')
        after=before
        while True:
            after=Executor.semantics(self,stage,facts,after)
            if all(self.visual_facts.get(fact) is True for fact in facts):break
            if any(self.visual_facts.get(fact) is False for fact in facts):raise SafetyError('task semantic boundary explicitly false')
            # Executor.semantics owns the original two UNKNOWN budget; no None advancement.
            self.sleep(.1);after=self.capture()
        if stage['phase']=='Grasp':
            i=0 if stage['arm']=='left' else 1;point=after[1].roles.get(stage['object'])
            if point is None:raise SafetyError('accepted task grasp role not visible')
            tcp=after[0][i].tcp if i==0 else (self.T@np.r_[after[0][i].tcp,1])[:3]
            self.accepted_grasp={'object':stage['object'],'cycle':stage['cycle_id'],'index':i,'g_command':float(self.command_g[i]),'aperture_m':after[0][i].aperture,'object_TCP_distance_m':float(np.linalg.norm(np.asarray(point)-tcp)),'source':'supported semantic tentative permission, not measured attachment'}
            self.current_holding=True;self._manipulated.add(stage['object']);self.register_attachment(stage,*after)
        else:self.accepted_grasp=None;self.current_holding=False
        self.semantic_log.append({'stage_id':stage['stage_id'],'facts':{f:self.visual_facts[f] for f in facts},'declared_goal_relation':stage['goal_relation']})
        return after
    def refresh_lift_attachment(self,stage,samples,obs):
        if self.attachment is None or obs.roles.get(stage['object']) is None:raise SafetyError('Lift attachment update unknown')
        index=0 if stage['arm']=='left' else 1;point=np.asarray(obs.roles[stage['object']]);pose=self._shared_pose(np.stack([s.q for s in samples]),self.command_g,stage['arm'])
        previous=self.attachment['a_actual'].copy();self.attachment['a_actual']=pose[1].T@(point-pose[0])
        demo=self._shared_pose(np.asarray(self.b['qcurve'](stage['reference']['end_time_sec'])),self.command_g,stage['arm'])
        self.coarse_source=point-(demo[0]+demo[1]@self.attachment['a_ref'])
        if not hasattr(self,'attachment_updates'):self.attachment_updates=[]
        self.attachment_updates.append({'stage_id':stage['stage_id'],'source':'actual Lift-end SAM+own TCP estimated attachment; not measured contact','old_local_m':previous.tolist(),'new_local_m':self.attachment['a_actual'].tolist(),'comotion':getattr(self,'comotion_record',None),'next_Move_source_shift_m':self.coarse_source.tolist()})
    def command(self,stage,*args,**kwargs):
        if stage['phase'] not in ORDINARY:raise SafetyError('specialist phase handler missing: '+stage['phase'])
        original=self.b['gripper']
        if stage['phase']=='Wait':
            i=0 if stage['arm']=='left' else 1;locked=original.copy();locked[:,i]=self.command_g[i];self.b['gripper']=locked
        if stage['phase'] in CARRY:
            cert=self.accepted_grasp
            if not cert or cert['cycle']!=stage['cycle_id']:raise SafetyError('accepted task gripper command missing')
            locked=original.copy();locked[:,cert['index']]=cert['g_command'];self.b['gripper']=locked
        try:return super().command(stage,*args,**kwargs)
        finally:self.b['gripper']=original
    def predicates(self,stage,progress,target,samples,obs):
        pose=self._shared_pose(np.stack([s.q for s in samples]),self.command_g,stage['arm'])
        reached=self._last_cartesian_goal is not None and self._reached(self._last_cartesian_goal,pose)
        facts={'own_pose':reached,'own_progress':progress>=1.-1e-9,'finite':True,'tentative_holding':self.holding(samples,stage)}
        if stage['phase']=='Release':
            i=0 if stage['arm']=='left' else 1;gap=samples[i].aperture;initial=self.initial_aperture[i]
            measured=gap is not None and initial is not None;duration=stage['reference']['end_time_sec']-stage['reference']['start_time_sec']
            facts['opening_progress']=bool(gap>=.9*initial) if measured else bool(self.command_g[i]>=.9 and self.clock()-self.stage_started>=duration)
            self.opening_record={'measured':measured,'actual_aperture_m':gap,'commanded_open':float(self.command_g[i]),'visual_released_confirmation_deferred_to_Complete':True}
            if facts['own_pose'] and facts['own_progress'] and facts['opening_progress']:self._empty_hand_anchor=(pose[0].copy(),pose[1].copy())
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
                    if stage['phase'] in CARRY and self.holding(samples,stage) is not True:raise SafetyError('fresh task carrying invariant unknown/false')
                    dt=self.cfg['command_dt_sec'];elapsed+=dt;target,progress,record=self.command(stage,elapsed,samples,obs,dt);samples,obs=self.capture()
                    if stage['phase']=='Grasp' and progress>=1.-1e-9 and self.holding(samples,stage) is not True:samples,obs=self.semantics(stage,['holding'],(samples,obs))
                    facts=self.predicates(stage,progress,target,samples,obs)
                    self.trace.append({'stage_id':stage['stage_id'],'command_index':self.command_index,'facts':facts,'controller':record,'issued_q':target.tolist(),'actual_q':[s.q.tolist() for s in samples],'actual_aperture_m':[s.aperture for s in samples],'issued_g':self.command_g.tolist(),'holding_state_source':getattr(self,'holding_source','unknown')})
                    if any(facts.get(k) is not True for k in stage['invariants']):raise SafetyError('ordinary stage invariant unknown/false')
                    if all(facts.get(k) is True for k in stage['done']):done=True;break
                self.stage_trace.append({'stage_id':stage['stage_id'],'completed':done,'commands':update+1})
                if not done:raise SafetyError('task stage own-pose/progress budget exhausted')
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
