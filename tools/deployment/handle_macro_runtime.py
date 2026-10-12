"""Continuous transfer macro executor; same flatCode/NN/SAM, no suspended Agent waits."""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from handle_full_runtime import HandleFullExecutor,HandleFullVision
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write

VARIANT='handle_macro_continuous_transfer_v1'


class MacroVision(HandleFullVision):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.control_token=None;self._memo=None;self.memo_reuse_count=0;self.memo_enabled=False;self.initial_static_clouds={};self.use_initial_white_prior=False
    def cloud(self,role):
        if self.use_initial_white_prior and role in self.initial_static_clouds:return self.initial_static_clouds[role].copy()
        return super().cloud(role)
    def begin_control_update(self,token):self.control_token=token;self._memo=None;self.memo_enabled=True
    def invalidate(self):self._memo=None
    def capture(self):
        if self.memo_enabled and self._memo is not None and self.clock()-self._memo.stamp<=self.config['max_age_sec']:
            self.memo_reuse_count+=1;return self._memo # actual stamp unchanged; robot samples read independently by Executor
        self._memo=None;obs=super().capture();self._memo=obs if self.memo_enabled else None;return obs
    def reground(self,cycle):self.invalidate();return super().reground(cycle)


class MacroExecutor(HandleFullExecutor):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.macro_grasp=None;self.macro_events=[]
    def start(self):
        super().start()
        self.vision.initial_static_clouds={role:self.vision.cloud(role).copy() for role in ('white_mug','white_handle','table_patch')}
        self.initial_white_prior_stamp=self.vision.latest_observation_stamp
    def initialize_white_cycle(self,stage,samples,obs):
        self.vision.use_initial_white_prior=stage['cycle_id']=='c03'
        try:result=super().initialize_white_cycle(stage,samples,obs)
        finally:self.vision.use_initial_white_prior=False
        if stage['cycle_id']=='c03':
            self.handle_routes[-1].update(grounding='scene-initial actual SAM static white/handle/table prior; unmanipulated assumption, not fresh geometry',prior_observation_stamp=self.initial_white_prior_stamp)
        return result
    def holding(self,samples,stage):
        if stage['holding_contract']=='none':return True
        if stage['phase']=='Grasp':return super().holding(samples,stage)
        cert=self.macro_grasp;index=0 if stage['arm']=='left' else 1
        if not cert or cert['role']!=stage['object']:return None
        if self.command_g[index]>.5:return False
        gap=samples[index].aperture
        if gap is not None and cert['aperture_m'] is not None and cert['aperture_m']>.003 and gap<.001:
            self.holding_source='actual measured empty-jaw anomaly';return False
        if self.current_holding is not True:return None
        self.holding_source='tentative Root Grasp state within continuous transfer macro; not measured contact'
        return True
    def semantics(self,stage,facts,before):
        if stage['phase'] not in ('Grasp','Release'):raise SafetyError('Agent request at unsupported macro internal node forbidden')
        self.vision.invalidate();self.vision.memo_enabled=False
        try:after=super().semantics(stage,facts,before)
        finally:self.vision.invalidate();self.vision.memo_enabled=True
        if stage['phase']=='Grasp' and self.visual_facts.get('holding') is True:
            index=0 if stage['arm']=='left' else 1
            self.macro_grasp={'role':stage['object'],'cycle':stage['cycle_id'],'aperture_m':after[0][index].aperture,'g_command':float(self.command_g[index]),'Root_certified_at':self.clock()}
        if stage['phase']=='Release' and self.visual_facts.get('released') is True:self.macro_grasp=None
        return after
    def command(self,stage,*args,**kwargs):
        if stage['phase'] not in ('Lift','Move','Align'):return super().command(stage,*args,**kwargs)
        cert=self.macro_grasp
        if not cert or cert['role']!=stage['object'] or cert['cycle']!=stage['cycle_id']:raise SafetyError('confirmed macro gripper command missing')
        index=0 if stage['arm']=='left' else 1
        original=self.b['gripper'];locked=original.copy();locked[:,index]=cert['g_command']
        self.b['gripper']=locked # in-memory command-profile adapter; original graph/model/reference bytes unchanged
        try:target,progress,record=super().command(stage,*args,**kwargs)
        finally:self.b['gripper']=original
        record.update(gripper_contract='hold accepted Grasp command until explicit Release profile',confirmed_gripper_command=cert['g_command'])
        return target,progress,record
    def predicates(self,stage,progress,target,samples,obs):
        facts=super().predicates(stage,progress,target,samples,obs)
        facts.pop('semantic.align_ready',None) # no synthetic alignment fact inside macro
        return facts
    def run(self):
        completed=False;failure=None;stopped=[];node_budget=self.cfg['maximum_node_control_updates'];updates=0
        try:
            self.start()
            for stage in self.b['graph']['stages']:
                self.stage=stage;self.stage_started=self.clock();self.visual_facts={};self.fact_stamp=-float('inf');self.b['skill'].reset_phase();elapsed=0.;done=False
                write(self.output/'NODE_STATUS.json',{'stage_id':stage['stage_id'],'phase':stage['phase'],'continuous_macro':True,'Root_boundary':stage['phase'] in ('Grasp','Release')})
                for budget in range(node_budget):
                    updates+=1;self.vision.begin_control_update(updates);self.guard_limits();samples,obs=self.capture()
                    if budget==0 and stage['entry'].get('role_visible_or_held') and obs.roles.get(stage['object']) is None:raise SafetyError('macro entry visible role unknown')
                    if stage['phase'] in ('Lift','Move','Align') and self.holding(samples,stage) is not True:raise SafetyError('actual visible/gripper macro carrying anomaly; no unsupported Agent wait')
                    dt=self.cfg['command_dt_sec'];elapsed+=dt;target,progress,controller=self.command(stage,elapsed,samples,obs,dt)
                    samples,obs=self.capture()
                    if progress>=1.-1e-9:
                        if stage['phase']=='Grasp' and self.holding(samples,stage) is not True:samples,obs=self.semantics(stage,['holding'],(samples,obs))
                        if stage['phase']=='Release':
                            self.vision.invalidate();samples,obs=self.semantics(stage,['released','placement_complete'],(samples,obs))
                    facts=self.predicates(stage,progress,target,samples,obs)
                    self.trace.append({'stage_id':stage['stage_id'],'command_index':self.command_index,'facts':facts,'issued_q':target.tolist(),'issued_g':self.command_g.tolist(),
                        'actual_q':[s.q.tolist() for s in samples],'actual_aperture_m':[s.aperture for s in samples],'controller':controller,'holding_state_source':getattr(self,'holding_source','unknown')})
                    if any(facts.get(key) is not True for key in stage['invariants']):raise SafetyError('macro internal invariant unknown/false')
                    if all(facts.get(key) is True for key in stage['done']):done=True;break
                self.stage_trace.append({'stage_id':stage['stage_id'],'completed':done,'commands':budget+1,'control_update_budget':node_budget})
                if not done:raise SafetyError('macro internal control budget exhausted')
                if stage['phase']=='Lift':
                    # Sensor-only estimated re-registration at reached own pose; no lifted fact is asserted.
                    self.refresh_lift_attachment(stage,*self.capture())
                    if self.attachment_refresh_log:
                        self.attachment_refresh_log[-1].update(reason='actual Lift endpoint pose and tentative macro grasp; fresh SAM registration',source='SAM+own FK estimate; not Root lifted judgement or contact truth')
            completed=True
        except Exception as exc:
            failure=str(exc)
            for arm,stop in zip(('left','right'),self.stops):
                try:stop.stop();stopped.append({'arm':arm,'stop_acknowledged':True})
                except Exception as error:stopped.append({'arm':arm,'stop_acknowledged':False,'error':str(error)})
        self.b['skill'].reset_phase()
        result={'schema':'graph-self-adapter.xtrainer-run.v1','execution_variant':VARIANT,'mode':self.cfg['mode'],'graph_completed':completed,'failure':failure,
            'graph_stages':28,'stage_trace':self.stage_trace,'trace':self.trace,'Root_decisions':self.judge.records,'root_request_count':len(self.judge.records),
            'fixed_z':True,'latent_updates':0,'reference_clock_speed':self.cfg['hardware_reference_speed_scale'],'maximum_node_control_updates':node_budget,
            'hardware_command_attempts':self._hardware_attempts,'hardware_command_acknowledgments':self._hardware_acknowledged,'hardware_motion_commands':self.robot.commands,
            'stop_results':stopped,'rgbd_discard_log':getattr(self,'rgbd_discard_log',[]),'SAM_same_update_reuse_count':self.vision.memo_reuse_count,
            'Agent_boundaries':['supported Grasp','opened Release/placement'],'unsupported_Agent_waits':0,'local_60sec_gate_applied':False,
            'internal_completion_contract':'own pose/progress and tentative grasp, no fabricated lifted/align/support facts',
            'original_object_goal_required_after_seal':True,'unpromoted_sim_only':True,'white_handle_Code_route':self.handle_routes,'attachment_refresh_log':self.attachment_refresh_log,
            'policy_claim':'frozen reference+residual plus explicit Code primitives; no learned handle claim'}
        write(self.output/'RESULT.json',result);return result


SharedCupExecutor=MacroExecutor
SharedSAMVision=MacroVision
