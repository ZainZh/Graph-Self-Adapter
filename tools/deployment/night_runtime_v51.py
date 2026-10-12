"""Common skill-phase runtime: supported closure, observable consequence, then confirmed transport."""
import numpy as np,hashlib
from pathlib import Path
from PIL import Image
from prove.xtrainer.runtime import Executor
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError
from multitask_special_runtime import SpecialExecutor
from three_task_v26_public_io import PublicRobot
from all_code_stage_runtime import ExactFKCacheRobot
from night_vision_v49 import AutoTaskVision

class NightExecutor(SpecialExecutor):
    def __init__(self,b,robot,*args,**kw):
        super().__init__(b,ExactFKCacheRobot(PublicRobot(robot,source='sim_real_public_contract')),*args,**kw)
        self.roles['SAM_roles']=[r for r in self.roles['SAM_roles'] if r!='table_patch']
        self.pending=None;self.pre_motion=None;self.initial_public_points={}
    def start(self):
        super().start();self.initial_public_points={k:np.asarray(v).copy() for k,v in self.initial_roles.items() if v is not None}
        write(self.output/'INITIAL_PUBLIC_ROLE_POINTS.json',{'source':'actual SAM + metric RGBD initial visible medians','role_points':{k:v.tolist() for k,v in self.initial_public_points.items()}})
        self.paired_points={}
        if self.cfg.get('paired_SAM_reference'):
            import json
            f=Path(self.cfg['paired_SAM_reference'])
            if hashlib.sha256(f.read_bytes()).hexdigest()!=self.cfg['paired_SAM_reference_sha256']:raise SafetyError('public SAM reference binding changed')
            self.paired_points=json.loads(f.read_text())['role_points']
            self.static_reference_points={k:np.asarray(v) for k,v in self.paired_points.items()}
            offsets={role:np.asarray(point)-np.asarray(self.b['frames'][0]['surface_centroids_world_m'][role]) for role,point in self.paired_points.items() if role in self.b['frames'][0]['surface_centroids_world_m']}
            for ref in self.b['frames']:
                for role,delta in offsets.items():
                    if ref['surface_centroids_world_m'].get(role) is not None:ref['surface_centroids_world_m'][role]=(np.asarray(ref['surface_centroids_world_m'][role])+delta).tolist()

    def remember_closure(self,s,samples,obs):
        i=0 if s['arm']=='left' else 1;self.pending={'role':s['object'],'arm':s['arm'],'cycle':s['cycle_id'],'g':float(self.command_g[i]),'point':obs.roles.get(s['object'])};self.current_holding=None
        self.pre_motion=(obs.stamp,obs.rgb.copy(),[{'q':v.q.tolist(),'qdot':v.qdot.tolist(),'TCP_m':v.tcp.tolist(),'stamp':v.stamp} for v in samples])
        f=self.output/'PRE_LIFT_PUBLIC_RGB.png';Image.fromarray(obs.rgb).save(f);write(self.output/'PRE_LIFT_PUBLIC_OBSERVATION.json',{'RGB':str(f.resolve()),'RGB_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'stamp':obs.stamp,'public_robot':self.pre_motion[2]})
    def confirm_consequence(self,s,samples,obs):
        if s['phase']=='Pull':return # Drawer is fixed: final released/open relation, not carried-object attachment.
        if s['phase']!='Lift' and self.pre_motion:
            from dataclasses import replace
            obs=replace(obs,history=[self.pre_motion]+list(obs.history[-2:]))
        after=Executor.semantics(self,s,['holding'],(samples,obs))
        if self.visual_facts.get('holding') is not True:raise SafetyError('post-Lift holding UNKNOWN/false; no carry')
        i=0 if s['arm']=='left' else 1;point=after[1].roles.get(s['object']);tcp=after[0][i].tcp if i==0 else (self.T@np.r_[after[0][i].tcp,1])[:3]
        if point is None:raise SafetyError('current confirmed object role unavailable')
        cert={'object':s['object'],'cycle':s['cycle_id'],'index':i,'g_command':float(self.command_g[i]),'aperture_m':after[0][i].aperture,'object_TCP_distance_m':float(np.linalg.norm(point-tcp)),'state':True,'source':'current API visual motion confirmation'}
        self.certificates[(s['arm'],s['object'])]=cert;self.accepted_grasp=cert;self.current_holding=True;self._manipulated.add(s['object']);self.register_attachment(s,*after)
    def command(self,s,*args,**kw):
        index=0 if s['arm']=='left' else 1;original=self.b['gripper'];profile=original.copy()
        if s['phase'] in ('Lift','Move','Align','Insert','Pull','Push','Hold'):
            c=self.certificates.get((s['arm'],s['object']))
            grip=c['g_command'] if c else self.pending['g'] if self.pending and self.pending['role']==s['object'] else None
            if grip is not None:profile[:,index]=grip
        self.b['gripper']=profile
        try:
            diagnostic=s.get('diagnostic_consequence_only') is True or s['phase'] in ('Lift','Pull') and s['holding_contract']=='none'
            if diagnostic:
                if not self.pending or (self.pending['role'],self.pending['arm'],self.pending['cycle'])!=(s['object'],s['arm'],s['cycle_id']):raise SafetyError('diagnostic motion without matching closure attempt')
                # This bypasses only the premature certificate gate. No holding
                # belief is created; reference, IK, freshness and dispatch checks remain.
                return self._reference_command(s,*args,**kw)
            return super().command(s,*args,**kw)
        finally:self.b['gripper']=original
    def predicates(self,s,progress,target,samples,obs):
        previous=self.attachment;coarse=None if self.coarse_source is None else self.coarse_source.copy();f=super().predicates(s,progress,target,samples,obs)
        if s['phase']=='Grasp':self.attachment=previous;self.coarse_source=coarse
        return f
    def run(self):
        complete=False;failure=None;stopped=[]
        try:
            self.start()
            for s in self.b['graph']['stages']:
                self.stage=s;self.stage_started=self.clock();self.visual_facts={};self.b['skill'].reset_phase();elapsed=0.;done=False
                write(self.output/'NODE_STATUS.json',{'task':self.b['task'],'stage_id':s['stage_id'],'phase':s['phase']})
                if s['phase']=='Complete':
                    self.semantics(s,s['semantic_facts'],self.capture());self.stage_trace.append({'stage_id':s['stage_id'],'completed':True,'commands':0});continue
                for count in range(self.cfg['maximum_node_control_updates']):
                    self.guard_limits();samples,obs=self.capture()
                    if s['phase'] in ('Move','Align','Insert','Push','Hold') and s['holding_contract']!='none' and self.holding(samples,s) is not True:raise SafetyError('current carry invariant unknown/false')
                    dt=self.cfg['command_dt_sec'];elapsed+=dt;target,progress,record=self.command(s,elapsed,samples,obs,dt);samples,obs=self.capture();facts=self.predicates(s,progress,target,samples,obs)
                    self.trace.append({'stage_id':s['stage_id'],'command_index':self.command_index,'facts':facts,'controller':record,'actual_q':[v.q.tolist() for v in samples],'issued_g':self.command_g.tolist()})
                    if any(facts.get(k) is not True for k in s['invariants']):raise SafetyError('node invariant unknown/false')
                    if all(facts.get(k) is True for k in s['done']):done=True;break
                self.stage_trace.append({'stage_id':s['stage_id'],'completed':done,'commands':count+1})
                if not done:raise SafetyError('node pose/progress budget exhausted')
                if s['phase']=='Grasp':self.remember_closure(s,*self.capture())
                if s['phase']=='Lift' or s.get('diagnostic_consequence_only'):self.confirm_consequence(s,*self.capture())
            complete=True
        except Exception as e:
            failure=str(e)
            for stop in self.stops:
                try:stop.stop();stopped.append({'stopped':True})
                except Exception as e:stopped.append({'error':str(e)})
        r={'schema':'graph-self-adapter.night-task-run.v1','task':self.b['task'],'logical_task':self.b['graph']['logical_task'],'graph_completed':complete,'failure':failure,'graph_stages':len(self.b['graph']['stages']),'stage_trace':self.stage_trace,'trace':self.trace,'judge_records':self.judge.records,'fixed_z':True,'latent_updates':0,'command_attempts':self._hardware_attempts,'command_acknowledgments':self._hardware_acknowledged,'stop_results':stopped,'hardware_qualified':False}
        write(self.output/'RESULT.json',r);return r
SharedCupExecutor=NightExecutor
SharedSAMVision=AutoTaskVision
