"""Object-centered frozen A2 execution through real hardware-neutral interfaces."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import time
from collections import deque
import numpy as np
from scipy.interpolate import PchipInterpolator
from .agent import digest,write
from .io import SafetyError,SafetyLayer,dls_retarget

REPO=Path(__file__).resolve().parents[2]
SHORT_PHASES=('Grasp','Align','Insert','Push','Pull')


def load_bundle(task='put_mug_on_box'):
    solution=REPO/f'task_packages/twin_rsi_v02/A2/packages/twin_{task}/solution'
    frozen=REPO/'runs/twin_rsi_v02/A2/source_snapshot'
    freeze=json.loads((solution/'CANDIDATE_FREEZE.json').read_text())
    paths={'graph':solution/'graph/task_graph.json','model':solution/'models/local_skill.npz',
           'references':solution/'refs/visible_centroids.json','skill_source':frozen/'prove/fast_adapter/twin_local_skill.py'}
    for key in ('graph','model','references'):
        if digest(paths[key])!=freeze[key+'_sha256']:raise SafetyError('frozen package hash mismatch: '+key)
    if digest(paths['skill_source'])!=freeze['code']['prove/fast_adapter/twin_local_skill.py']:
        raise SafetyError('frozen skill source hash mismatch')
    spec=importlib.util.spec_from_file_location('_xtrainer_frozen_skill',paths['skill_source'])
    import sys
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    graph=json.loads(paths['graph'].read_text());refs=json.loads(paths['references'].read_text())
    plan=json.loads((REPO/'evaluations/twin_rsi_v02/ROOT_TASK_PLAN.json').read_text())
    taskrow=next(x for x in plan['tasks'] if x['task']==task)
    motorpath=Path(taskrow['sources']['motor_targets.npz']['path'])
    if digest(motorpath)!=taskrow['sources']['motor_targets.npz']['sha256']:raise SafetyError('motor prior hash mismatch')
    with np.load(motorpath,allow_pickle=False) as data:
        times=data['timestamps'].copy();q=data['joints'].copy();g=data['gripper'].copy()
    paths['motor_prior']=motorpath;paths['candidate_freeze']=solution/'CANDIDATE_FREEZE.json'
    skill=module.TwinLocalSkill(paths['model'],online_z=False,feedback_contract=graph['local_feedback_contract'])
    if skill.model.input_dim!=29:raise SafetyError('migration requires frozen 29D geometry model')
    frames=sorted(refs['frames'].values(),key=lambda f:f['time_sec'])
    return {'graph':graph,'refs':refs,'skill':skill,'paths':paths,'bindings':{str(p):digest(p) for p in paths.values()},
            'times':times,'qcurve':PchipInterpolator(times,q,axis=0),'gripper':g,'initial_q':q[0],'initial_g':g[0],
            'frame_times':np.array([x['time_sec'] for x in frames]),'frames':frames}


class Executor:
    def __init__(self,bundle,robot,vision,judge,stops,config,output,*,clock=time.monotonic,sleep=time.sleep):
        self.b=bundle;self.robot=robot;self.vision=vision;self.judge=judge;self.stops=stops;self.cfg=config
        self.output=Path(output);self.clock=clock;self.task_started=clock();self.stage_started=None;self.root_request_count=0;self.sleep=sleep;self.trace=[];self.stage_trace=[]
        self.safety=SafetyLayer(config['safety'],clock);self.T=np.asarray(config['T_left_from_right'])
        self.command_g=np.asarray(config['known_initial_gripper_commands'],dtype=float)
        self.holding_since={};self.visual_facts={};self.fact_stamp=-float('inf')
        self.rgb_history=deque(maxlen=3);self.visual_anchor=None;self.current_holding=None;self.unknown_counts={}
        self.stage=None;self.command_index=0;self.previous_cycle=None;self.source_delta=None;self.target_delta=None
    def guard_limits(self):
        if self.clock()-self.task_started>self.cfg['max_task_wall_sec']:raise SafetyError('task wall budget exhausted')
        if self.stage_started is not None and self.clock()-self.stage_started>self.cfg['max_stage_wall_sec']:raise SafetyError('stage wall budget exhausted')
    def capture(self):
        self.guard_limits()
        obs=self.vision.capture();samples=self.robot.samples()
        self.safety.samples(samples)
        if not self.rgb_history or self.rgb_history[-1][0]!=obs.stamp:
            self.rgb_history.append((obs.stamp,obs.rgb.copy(),[{'q':s.q.tolist(),'qdot':s.qdot.tolist(),'TCP_m':s.tcp.tolist(),'stamp':s.stamp} for s in samples]))
        obs.history=list(self.rgb_history)
        if not 0<=self.clock()-obs.stamp<=self.cfg['safety']['max_age_sec']:raise SafetyError('stale real RGB-D')
        if self.visual_anchor is not None and self.stage is not None:
            role,index,offset,start_tcp,start_object,root_stamp=self.visual_anchor;point=obs.roles.get(role)
            tcp=samples[index].tcp if index==0 else (self.T@np.r_[samples[index].tcp,1])[:3]
            if point is None:self.current_holding=None;self.holding_source='role_unknown'
            else:
                tcp_delta=tcp-start_tcp;object_delta=np.asarray(point)-start_object
                moving=np.linalg.norm(tcp_delta)>=.003 and np.linalg.norm(object_delta)>=.003
                if moving:
                    self.current_holding=bool(np.linalg.norm(object_delta-tcp_delta)<=.006)
                    self.holding_source='nonzero_visible_object_TCP_comotion'
                else:
                    self.current_holding=True if self.clock()-root_stamp<=.25 else None
                    self.holding_source='bounded_actual_Root_visual_judgment' if self.current_holding else 'static_overlap_is_unknown'
        return samples,obs
    def semantics(self,stage,facts,before):
        self.guard_limits()
        if self.root_request_count>=self.cfg['max_root_requests_per_task']:raise SafetyError('Root task request budget exhausted')
        self.root_request_count+=1
        settled_start=self.clock()
        while any(s.running or s.mode!=5 or np.max(abs(s.qdot))>.01 for s in before[0]) or len(before[1].history)<3 or before[1].history[-1][0]-before[1].history[0][0]<.19:
            if self.clock()-settled_start>2.:raise SafetyError('robot did not settle for fresh multi-frame Root review')
            self.sleep(.1);before=self.capture()
        baseline=[s.q.copy() for s in before[0]]
        def heartbeat():
            self.guard_limits()
            actual=self.robot.samples();self.safety.samples(actual)
            if max(np.max(abs(s.q-q)) for s,q in zip(actual,baseline))>.015:
                raise SafetyError('robot drift during real Root wait')
        judged,after=self.judge.ask(stage,facts,before=before,recapture=self.capture,heartbeat=heartbeat)
        self.visual_facts.update(judged);self.fact_stamp=self.clock()
        if judged.get('holding') is True:
            index=0 if stage['arm']=='left' else 1;point=after[1].roles.get(stage['object'])
            tcp=after[0][index].tcp if index==0 else (self.T@np.r_[after[0][index].tcp,1])[:3]
            if point is not None:
                if self.visual_anchor is not None and self.visual_anchor[:2]==(stage['object'],index):
                    self.visual_anchor=(*self.visual_anchor[:5],self.clock())
                else:self.visual_anchor=(stage['object'],index,np.asarray(point)-tcp,tcp.copy(),np.asarray(point).copy(),self.clock())
                self.current_holding=True;self.holding_source='actual_Root_visual_judgment'
        if judged.get('holding') is False:self.visual_anchor=None;self.current_holding=False
        if judged.get('released') is True:self.rgb_history=deque(maxlen=3);self.visual_anchor=None;self.current_holding=False
        if any(judged.get(f) is None for f in facts):
            key=(stage['stage_id'],tuple(facts));self.unknown_counts[key]=self.unknown_counts.get(key,0)+1
            if self.unknown_counts[key]>2:raise SafetyError('real semantic UNKNOWN retry budget exhausted')
        return after
    def holding(self,samples,stage):
        # No current aperture->contact shortcut: use current visual observation only.
        held=self.current_holding
        if stage['object']!=(self.visual_anchor[0] if self.visual_anchor else None):held=None
        if stage['holding_contract']=='held_or_supported':
            supported=self.visual_facts.get('support_permission') if self.clock()-self.fact_stamp<=.25 else None
            if held is True or supported is True:return True
            if held is False and supported is False:return False
            return None
        if stage['holding_contract']=='none':return True
        return held
    def start(self):
        samples,obs=self.capture();q=np.stack([s.q for s in samples])
        if any(s.running or s.mode!=5 for s in samples):raise SafetyError('initial robot is not stationary')
        if np.max(np.abs(q-self.b['initial_q']))>self.cfg['initial_q_tolerance_rad']:
            raise SafetyError('current pose is not the frozen native starting pose; no home/teleport permitted')
        if not np.allclose(self.command_g,self.b['initial_g'],atol=.01):raise SafetyError('known initial gripper command differs from native start')
        if self.cfg.get('initial_gripper_state_confirmed') is not True:raise SafetyError('actual initial gripper state must be separately confirmed')
        self.initial_roles={k:None if p is None else np.asarray(p).copy() for k,p in obs.roles.items()}
        self.previous_cycle=None
    def nearest_ref(self,t):return self.b['frames'][int(np.argmin(abs(self.b['frame_times']-t)))]
    def target_point(self,obs,stage):
        value=obs.roles.get(stage['target'])
        if value is None and stage['target']=='table_handover_region':
            value=self.b['refs']['handover_anchors'].get(stage['cycle_id'],{}).get('centroid_world_m')
        return None if value is None else np.asarray(value)
    def command(self,stage,elapsed,samples,obs,dt):
        refstart=stage['reference']['start_time_sec'];refend=stage['reference']['end_time_sec']
        speed=stage.get('reference_speed_scale',1.)*self.cfg['hardware_reference_speed_scale']
        t=min(refend,refstart+elapsed*speed);progress=(t-refstart)/max(refend-refstart,1e-9)
        ref=self.nearest_ref(t);index=0 if stage['arm']=='left' else 1
        if self.previous_cycle!=stage['cycle_id']:
            self.previous_cycle=stage['cycle_id'];self.source_delta=None;self.target_delta=None
        if self.source_delta is None:
            role=stage.get('source_anchor_role',stage['object'])
            first_ref=self.nearest_ref(refstart)['surface_centroids_world_m'].get(role)
            current=obs.roles.get(role)
            if current is None or first_ref is None:raise SafetyError('cycle source visible grounding unknown')
            self.source_delta=np.asarray(current)-np.asarray(first_ref)
        if self.target_delta is None:
            current=self.target_point(obs,stage)
            reference=ref['surface_centroids_world_m'].get(stage['target'])
            if reference is None and stage['target']=='table_handover_region':
                reference=self.b['refs']['handover_anchors'].get(stage['cycle_id'],{}).get('centroid_world_m')
            if current is None or reference is None:raise SafetyError('declared target grounding unknown')
            self.target_delta=current-np.asarray(reference)
        q=np.asarray(self.b['qcurve'](t));g=np.array([np.interp(t,self.b['times'],self.b['gripper'][:,i]) for i in range(2)])
        record={'controller':'Code_DLS_retarget','progress':progress,'source_delta_m':self.source_delta.tolist(),
                'target_delta_m':self.target_delta.tolist()}
        if stage['controller']=='neural_short':
            features=np.r_[samples[index].q,samples[index].qdot,self.source_delta,self.target_delta,progress,
                           np.eye(2)[index],np.eye(5)[SHORT_PHASES.index(stage['phase'])],np.zeros(3)]
            step=self.b['skill'].command(features,current_surface_xyz=obs.roles.get(stage['object']),
                reference_surface_xyz=ref['surface_centroids_world_m'].get(stage['object']),phase=stage['phase'],
                holding=self.current_holding,
                matched_reference_time_sec=ref['time_sec'],current_tcp_xyz=(samples[index].tcp if index==0 else (self.T@np.r_[samples[index].tcp,1])[:3]),
                current_tcp_rotation=(samples[index].rotation if index==0 else self.T[:3,:3]@samples[index].rotation),reference_tcp_xyz=np.asarray(ref['tcp_position'])[index],
                reference_tcp_rotation=np.asarray(ref['tcp_rotation'])[index])
            q[index]+=step.action;record=step.record
        else:
            delta=self.source_delta if stage['phase'] in ('Approach','Lift') else (1-progress)*self.source_delta+progress*self.target_delta
            q=dls_retarget(self.robot,q,g,stage['arm'],delta,self.T)
        # Preserve the original frozen synchronous two-arm prior; residual retargets only the actor.
        self.safety.validate(q,g,samples,dt,self.robot,self.T)
        # Finite-difference FK RPC can be slow: re-read independent actual feedback immediately before send.
        fresh=self.robot.samples();self.safety.samples(fresh)
        if not 0<=self.clock()-obs.stamp<=self.cfg['safety']['max_age_sec']:raise SafetyError('role geometry stale after FK RPC')
        actual=np.stack([s.q for s in fresh])
        if np.max(abs(q-actual))>self.cfg['safety']['max_step_rad']:raise SafetyError('pre-send actual pose changed during FK')
        self.robot.command(q,g,[True,True]);self.safety.last_target=q.copy();self.command_g=g.copy()
        self.sleep(dt);self.command_index+=1
        return q,progress,record
    def predicates(self,stage,progress,target,samples,obs):
        index=0 if stage['arm']=='left' else 1
        holding=self.holding(samples,stage);known=self.clock()-self.fact_stamp<=.25
        facts={};now=self.clock()
        if holding is True:
            self.holding_since.setdefault(stage['stage_id'],now)
        else:self.holding_since.pop(stage['stage_id'],None)
        for row in stage['completion_predicates']+stage['invariant_predicates']:
            kind=row['predicate'];value=None
            if kind=='reference_progress':value=progress>=1.-1e-9
            elif kind=='actor_command_reached':value=np.max(abs(samples[index].q-target[index]))<=row.get('tolerance_rad',.025)
            elif kind=='finite_simulation':value=True # Real alias: numerical finite values independently checked by SafetyLayer.
            elif kind=='holding':value=holding
            elif kind=='holding_confirmed':
                value=None if holding is None else bool(holding and now-self.holding_since.get(stage['stage_id'],now)>=row['minimum_sec'])
            elif kind=='released':
                # Actual jaws remain unknown unless measured; current Root visual release is separately declared.
                value=self.visual_facts.get('released') if known else None
            elif kind=='stage_goal':
                key='lifted' if row.get('goal_relation')=='lifted' or row.get('key')=='goal_lifted' else 'placement_complete'
                value=self.visual_facts.get(key) if known else None
            else:raise SafetyError('unimplemented graph predicate '+kind)
            facts[row['id']]=value
        if stage['phase']=='Align':facts['semantic.align_ready']=self.visual_facts.get('align_ready') if known else None
        return {key: bool(value) if isinstance(value,np.bool_) else value for key,value in facts.items()}
    def run(self):
        completed=False;failure=None;stopped=[]
        try:
            self.start()
            for stage in self.b['graph']['stages']:
                self.stage=stage;self.stage_started=self.clock();self.visual_facts={};self.fact_stamp=-float('inf');self.b['skill'].reset_phase()
                elapsed=0.;done=False
                hardware_budget=int(np.ceil(stage['max_budget_commands']/self.cfg['hardware_reference_speed_scale']))+2
                for budget in range(hardware_budget):
                    self.guard_limits()
                    if hasattr(self.vision,'set_context'):self.vision.set_context(stage,elapsed,self.b)
                    samples,obs=self.capture()
                    if budget==0 and stage['entry'].get('role_visible_or_held') and obs.roles.get(stage['object']) is None:
                        raise SafetyError('stage entry role visibility unknown')
                    carrying=stage['phase'] in ('Lift','Move','Align') and stage['holding_contract']!='none'
                    lease_expiring=(self.visual_anchor is not None and self.clock()+self.cfg['command_dt_sec']-self.visual_anchor[5]>.25 and getattr(self,'holding_source','')!='nonzero_visible_object_TCP_comotion')
                    if carrying and (self.holding(samples,stage) is not True or lease_expiring):
                        need=['holding']
                        if stage['holding_contract']=='held_or_supported':need.append('support_permission')
                        samples,obs=self.semantics(stage,need,(samples,obs))
                        if self.holding(samples,stage) is not True:raise SafetyError('carrying invariant unknown/false')
                    dt=self.cfg['command_dt_sec'];elapsed+=dt
                    target,progress,controller=self.command(stage,elapsed,samples,obs,dt)
                    if hasattr(self.vision,'set_context'):self.vision.set_context(stage,elapsed,self.b)
                    samples,obs=self.capture()
                    need=[]
                    if progress>=1.-1e-9:
                        if stage['phase']=='Grasp' and self.holding(samples,stage) is not True:need=['holding']
                        if stage['phase']=='Lift':need=['holding','lifted']
                        if stage['phase']=='Align':need=['holding','support_permission','align_ready']
                        if stage['phase']=='Release':need=['released','placement_complete']
                    if need:samples,obs=self.semantics(stage,need,(samples,obs))
                    facts=self.predicates(stage,progress,target,samples,obs)
                    self.trace.append({'stage_id':stage['stage_id'],'command_index':self.command_index,'facts':facts,
                        'issued_q':target.tolist(),'issued_g':self.command_g.tolist(),'actual_q':[s.q.tolist() for s in samples],
                        'actual_aperture_m':[s.aperture for s in samples],'sensor_source':obs.source,'holding_evidence_source':getattr(self,'holding_source','unknown'),'controller':controller})
                    if any(facts.get(k) is not True for k in stage['invariants']):raise SafetyError('stage invariant unknown/false')
                    done_ids=stage['done']+(['semantic.align_ready'] if stage['phase']=='Align' else [])
                    if all(facts.get(k) is True for k in done_ids):done=True;break
                self.stage_trace.append({'stage_id':stage['stage_id'],'completed':done,'commands':budget+1,'hardware_budget':hardware_budget})
                if not done:raise SafetyError('stage budget exhausted with unknown/false completion')
            completed=True
        except Exception as exc:
            failure=str(exc)
            for arm,stop in zip(('left','right'),self.stops):
                try:stop.stop();stopped.append({'arm':arm,'stop_acknowledged':True})
                except Exception as stop_exc:stopped.append({'arm':arm,'stop_acknowledged':False,'error':str(stop_exc)})
        self.b['skill'].reset_phase()
        result={'schema':'graph-self-adapter.xtrainer-run.v1','mode':self.cfg['mode'],'mock_only':self.cfg['mode']=='mock',
            'graph_completed':completed,'failure':failure,'original_physical_goal_evaluated':False,
            'hardware_motion_commands':0 if self.cfg['mode']=='mock' else self.robot.commands,
            'mock_command_count':self.robot.commands if self.cfg['mode']=='mock' else 0,
            'graph_stages':len(self.b['graph']['stages']),'stage_trace':self.stage_trace,'trace':self.trace,
            'Root_decisions':self.judge.records,'stop_results':stopped,'bindings':self.b['bindings'],
            'limitations':['mock completion is protocol integration evidence only; no hardware/SAM/physical task performance',
                           'real visual gripper judgments are not measured aperture/contact','no full-arm collision planner implemented'],
            'root_request_count':self.root_request_count,'task_wall_sec':self.clock()-self.task_started,
            'hardware_reference_speed_scale':self.cfg['hardware_reference_speed_scale'],'fixed_z':True,'latent_updates':self.b['skill'].latent.updates}
        self.output.mkdir(parents=True,exist_ok=True);write(self.output/'RESULT.json',result)
        return result
