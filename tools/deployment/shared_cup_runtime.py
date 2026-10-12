"""Shared sim/real node boundary layer over the same reviewed Code and frozen NN.

Only I/O differs. A terminal reference index does not request Root completion
until the full Cartesian target has been reached by actual robot feedback.
"""
import importlib.util
from pathlib import Path
import sys
import json
import hashlib
import time
import numpy as np
from prove.xtrainer.agent import RealRootJudge,write
from prove.xtrainer.io import SafetyError

path=Path(__file__).resolve().with_name('object_centered_runtime.py')
spec=importlib.util.spec_from_file_location('_shared_cup_base_runtime',path)
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)


class SharedCupExecutor(base.ObjectCenteredExecutor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.static_reference_points=None
        binding=self.cfg.get('static_reference_points_file')
        if binding is not None:
            prior=Path(binding)
            expected=self.cfg.get('static_reference_points_sha256')
            if expected is not None and hashlib.sha256(prior.read_bytes()).hexdigest()!=expected:raise SafetyError('paired static reference hash mismatch')
            payload=json.loads(prior.read_text())
            if payload.get('schema')!='graph-self-adapter.sam-static-reference-points.v1':raise SafetyError('paired static reference schema invalid')
            self.static_reference_points=payload['role_points_world_m']
            if set(self.static_reference_points)!=set(('red_mug','white_mug','box')) or any(np.asarray(v).shape!=(3,) or not np.isfinite(v).all() for v in self.static_reference_points.values()):raise SafetyError('paired static reference geometry invalid')
        if self.cfg.get('hardware_reference_speed_scale')!=1.0:raise SafetyError('common sim/real reference clock must be 1.0')
        self.cfg.setdefault('maximum_node_control_updates',1000)
        if self.cfg.get('maximum_hardware_commands') is None:self.cfg['maximum_hardware_commands']=5000
        if type(self.cfg['maximum_node_control_updates']) is not int or self.cfg['maximum_node_control_updates']<=0:raise SafetyError('node control budget must be positive integer')
        self.root_node_state=None;self._node_anomaly_requests=set();self.node_state_log=[];self._lift_attachment_refresh=set();self.attachment_refresh_log=[]
        if hasattr(self.vision,'stage_context'):self.vision.stage_context=lambda:self.stage
    def capture(self):
        samples,obs=super().capture();state=self.root_node_state
        if state and self.stage and state['role']==self.stage['object']:
            index=0 if self.stage['arm']=='left' else 1;point=obs.roles.get(state['role'])
            tcp=samples[index].tcp if index==0 else (self.T@np.r_[samples[index].tcp,1])[:3]
            radius=max(.12,float(np.linalg.norm(state['initial_object_TCP_delta']))+.08)
            missing=point is None;far=point is not None and np.linalg.norm(np.asarray(point)-tcp)>radius
            closed_command=self.command_g[index]<=.5
            actual_gap=samples[index].aperture;certified_gap=state.get('certified_aperture_m')
            jaw_opened=actual_gap is not None and certified_gap is not None and actual_gap>certified_gap+.01
            expired=self.clock()-state['certified_at']>self.cfg['max_stage_wall_sec']
            if state['holding'] is True and not expired and not missing and not far and closed_command and not jaw_opened:
                self.current_holding=True;self.holding_source='estimated_Root_node_grasp_state_not_measured_contact'
            elif state['holding'] is True:
                self.current_holding=None;self.holding_source='Root_node_grasp_state_real_anomaly_or_open_command'
            elif state['holding'] is False:self.current_holding=False;self.holding_source='Root_node_grasp_state_false'
            else:self.current_holding=None;self.holding_source='Root_node_grasp_state_unknown'
            state['latest_anomaly']={'missing':missing,'far_from_grip_region':far,'closed_command':closed_command,'actual_jaw_opened':jaw_opened,'expired':expired}
        return samples,obs
    def holding(self,samples,stage):
        if stage['holding_contract']=='none':return True
        if stage['holding_contract']=='held_or_supported' and self.root_node_state and self.root_node_state.get('support_permission') is True:return True
        return self.current_holding
    def semantics(self,stage,facts,before):
        holding_only=set(facts)<={'holding','support_permission'}
        if holding_only and self.holding(before[0],stage) is True:return before
        if holding_only and stage['phase']!='Grasp':
            key=stage['stage_id']
            if key in self._node_anomaly_requests:raise SafetyError('repeated real holding anomaly within node')
            self._node_anomaly_requests.add(key)
        after=before
        for retry in range(3):
            after=super().semantics(stage,facts,after)
            if all(self.visual_facts.get(fact) is not None for fact in facts):break
            if retry==2:raise SafetyError('same-node UNKNOWN observation retry budget exhausted')
            # No motion/target change: next Root request uses genuinely fresh after-observations.
        if 'holding' in facts:
            value=self.visual_facts.get('holding');point=after[1].roles.get(stage['object'])
            index=0 if stage['arm']=='left' else 1;tcp=after[0][index].tcp if index==0 else (self.T@np.r_[after[0][index].tcp,1])[:3]
            self.root_node_state={'role':stage['object'],'holding':value,'support_permission':self.visual_facts.get('support_permission'),
                'certified_at':self.clock(),'certified_node':stage['stage_id'],'certified_aperture_m':after[0][index].aperture,
                'initial_object_TCP_delta':np.zeros(3) if point is None else np.asarray(point)-tcp,
                'state_source':'actual Root node judgement; estimated semantic state, not measured contact'}
            self.node_state_log.append({**self.root_node_state,'initial_object_TCP_delta':self.root_node_state['initial_object_TCP_delta'].tolist()})
            self.current_holding=value;self.holding_source='estimated_Root_node_grasp_state_not_measured_contact' if value is True else 'Root_node_grasp_state_unknown_or_false'
        if self.visual_facts.get('released') is True:self.root_node_state=None;self.current_holding=False
        if stage['phase']=='Lift' and self.visual_facts.get('holding') is True and self.visual_facts.get('lifted') is True:
            self.refresh_lift_attachment(stage,*after)
        return after
    def refresh_lift_attachment(self,stage,samples,obs):
        if stage['stage_id'] in self._lift_attachment_refresh:return
        if self.attachment is None:raise SafetyError('Lift attachment reference missing')
        point=obs.roles.get(stage['object']);index=0 if stage['arm']=='left' else 1
        if point is None:raise SafetyError('Root-confirmed Lift fresh SAM attachment unknown')
        tcp=samples[index].tcp;rotation=samples[index].rotation
        if index:tcp=(self.T@np.r_[tcp,1])[:3];rotation=self.T[:3,:3]@rotation
        old=self.attachment['a_actual'].copy();self.attachment['a_actual']=rotation.T@(np.asarray(point)-tcp)
        end=stage['reference']['end_time_sec'];demo=self._shared_pose(np.asarray(self.b['qcurve'](end)),self.command_g,stage['arm'])
        self.coarse_source=np.asarray(point)-(demo[0]+demo[1]@self.attachment['a_ref'])
        self._lift_attachment_refresh.add(stage['stage_id'])
        self.attachment_refresh_log.append({'stage_id':stage['stage_id'],'reason':'fresh Root holding+lifted node confirmation after possible grasp slip',
            'source':'SAM visible object and measured robot TCP/R; estimated, not private contact/pose',
            'old_actual_local_offset_m':old.tolist(),'new_actual_local_offset_m':self.attachment['a_actual'].tolist(),
            'change_local_m':(self.attachment['a_actual']-old).tolist(),'new_source_role_shift_m':self.coarse_source.tolist(),
            'original_reference_offset_rotation_NN_static_cues_preserved':True})
    def start(self):
        samples,obs=self.capture()
        if any(s.running or s.mode!=5 or np.max(abs(s.qdot))>.01 for s in samples):raise SafetyError('actual fresh robot is not stationary')
        if not all(obs.roles.get(role) is not None for role in ('red_mug','white_mug','box')):raise SafetyError('initial actual cup/box grounding missing')
        if np.any(self.command_g<0) or np.any(self.command_g>1):raise SafetyError('invalid planned initial gripper command')
        self.initial_roles={k:None if p is None else np.asarray(p).copy() for k,p in obs.roles.items()};self.previous_cycle=None
        self.initial_state_diagnostics={'actual_q_rad':[s.q.tolist() for s in samples],'actual_aperture_m':[s.aperture for s in samples],
            'gripper_reference_command_not_measurement':self.command_g.tolist(),'physical_qualification_inferred':False}
    def run(self):
        completed=False;failure=None;stopped=[];node_budget=self.cfg['maximum_node_control_updates']
        try:
            self.start()
            for stage in self.b['graph']['stages']:
                self.stage=stage;self.stage_started=self.clock();self.visual_facts={};self.fact_stamp=-float('inf');self.b['skill'].reset_phase()
                elapsed=0.;done=False
                for budget in range(node_budget):
                    self.guard_limits();samples,obs=self.capture()
                    if budget==0 and stage['entry'].get('role_visible_or_held') and obs.roles.get(stage['object']) is None:
                        raise SafetyError('stage entry role visibility unknown')
                    carrying=stage['phase'] in ('Lift','Move','Align') and stage['holding_contract']!='none'
                    if carrying and self.holding(samples,stage) is not True:
                        need=['holding']+(['support_permission'] if stage['holding_contract']=='held_or_supported' else [])
                        samples,obs=self.semantics(stage,need,(samples,obs))
                        if self.holding(samples,stage) is not True:raise SafetyError('carrying invariant unknown/false')
                    dt=self.cfg['command_dt_sec'];elapsed+=dt
                    target,progress,controller=self.command(stage,elapsed,samples,obs,dt)
                    samples,obs=self.capture();need=[]
                    if progress>=1.-1e-9:
                        if stage['phase']=='Grasp' and self.holding(samples,stage) is not True:need=['holding']
                        if stage['phase']=='Lift':need=['holding','lifted']
                        if stage['phase']=='Align':need=['holding','support_permission','align_ready']
                        if stage['phase']=='Release':need=['released','placement_complete']
                    if need:samples,obs=self.semantics(stage,need,(samples,obs))
                    facts=self.predicates(stage,progress,target,samples,obs)
                    self.trace.append({'stage_id':stage['stage_id'],'command_index':self.command_index,'facts':facts,
                        'issued_q':target.tolist(),'issued_g':self.command_g.tolist(),'actual_q':[s.q.tolist() for s in samples],
                        'actual_aperture_m':[s.aperture for s in samples],'sensor_source':obs.source,'controller':controller,
                        'holding_evidence_source':getattr(self,'holding_source','unknown')})
                    if any(facts.get(k) is not True for k in stage['invariants']):raise SafetyError('stage invariant unknown/false')
                    done_ids=stage['done']+(['semantic.align_ready'] if stage['phase']=='Align' else [])
                    if all(facts.get(k) is True for k in done_ids):done=True;break
                self.stage_trace.append({'stage_id':stage['stage_id'],'completed':done,'commands':budget+1,'control_update_budget':node_budget})
                if not done:raise SafetyError('node control update budget exhausted with unknown/false completion')
            completed=True
        except Exception as exc:
            failure=str(exc)
            for arm,stop in zip(('left','right'),self.stops):
                try:stop.stop();stopped.append({'arm':arm,'stop_acknowledged':True})
                except Exception as stop_exc:stopped.append({'arm':arm,'stop_acknowledged':False,'error':str(stop_exc)})
        self.b['skill'].reset_phase()
        result={'schema':'graph-self-adapter.xtrainer-run.v1','execution_variant':'shared_object_centered_SAM_cup_v07_paired_static_prior',
            'mode':self.cfg['mode'],'graph_completed':completed,'failure':failure,'original_physical_goal_evaluated':False,
            'hardware_motion_commands':self.robot.commands,'graph_stages':len(self.b['graph']['stages']),
            'stage_trace':self.stage_trace,'trace':self.trace,'Root_decisions':self.judge.records,'stop_results':stopped,
            'bindings':self.b['bindings'],'fixed_z':True,'latent_updates':self.b['skill'].latent.updates,
            'root_request_count':self.root_request_count,'task_wall_sec':self.clock()-self.task_started,
            'maximum_node_control_updates':node_budget,'maximum_hardware_commands':self.cfg['maximum_hardware_commands'],
            'reference_clock_speed':self.cfg['hardware_reference_speed_scale'],'clock_does_not_define_command_budget':True,
            'hardware_command_attempts':self._hardware_attempts,'hardware_command_acknowledgments':self._hardware_acknowledged,
            'partial_motion_possible_without_ack':True,'initial_state_diagnostics':getattr(self,'initial_state_diagnostics',None),
            'source_coarse_reference_view':'Approach START before manipulation; current SAM minus declared Grasp END after successful grasp','source_relocked_at_END':False,'median_shift_scene_failure_gate':False,
            'not_a_historical_score':True,'policy_claim':'frozen reference plus residual policy; Code coarse grounding; not a new end-to-end or Fast-Adapter benefit',
            'NN_feature_contract':'original cycle-latched static environment cues minus Code-handled static shift','frozen_graph_nodes_unchanged':True,'grasp_state_source':'estimated Root-node semantic state; comotion positive-only; never measured contact',
            'Root_node_state_log':self.node_state_log,'attachment_refresh_log':self.attachment_refresh_log,'carried_attachment_geometry_registered_at_actual_Grasp_success':True,
            'Release_pose_policy':'hold actual Align-end pose while opening','Wait_pose_policy':'actual Release-end plus demonstrated relative empty-hand motion'}
        write(self.output/'RESULT.json',result);return result
    def command(self,*args,**kwargs):
        target,progress,record=super().command(*args,**kwargs)
        if progress>=1. and not self._goal_reached:
            record['node_request_deferred_until_full_actual_pose']=True
            progress=1.-1e-6
        return target,progress,record
    def predicates(self,*args,**kwargs):
        facts=super().predicates(*args,**kwargs)
        stage=args[0] if args else kwargs['stage']
        reached=all(facts.get(row['id']) is True for row in stage['completion_predicates'] if row['predicate']=='actor_command_reached')
        if reached and args[1]>=1.-1e-9:
            write(self.output/'NODE_STATUS.json',{'stage_id':stage['stage_id'],'reference_and_actual_pose_ready':True,
                'phase':stage['phase'],'facts':facts,'command_count':self.command_index})
        return facts
