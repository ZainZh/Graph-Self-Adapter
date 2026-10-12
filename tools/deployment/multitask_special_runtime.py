"""New specialist controller. NN sees original Insert/Pull/Push phase labels.
Single actor commands preserve passive tool pose/grip; no cup-specific geometry.
"""
import numpy as np
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from object_centered_runtime import ObjectCenteredExecutor
from multitask_ordinary_runtime import OrdinaryExecutor
from multitask_special_route import CARRY,role_contract

class SpecialExecutor(OrdinaryExecutor):
    def __init__(self,bundle,*args,**kwargs):
        if bundle['graph'].get('schema')!='graph-self-adapter.multitask-special-route.v1':raise SafetyError('explicit special route required')
        ObjectCenteredExecutor.__init__(self,bundle,*args,**kwargs)
        self.accepted_grasp=None;self.certificates={};self._registered_cycles=set();self.semantic_log=[];self.roles=role_contract(bundle['graph'],bundle['refs']);self._hold_pose=None
    def capture(self):
        if self.stage:self.accepted_grasp=self.certificates.get((self.stage['arm'],self.stage['object']))
        samples,obs=super().capture()
        for (arm,role),cert in self.certificates.items():
            i=cert['index'];point=obs.roles.get(role);tcp=samples[i].tcp if i==0 else (self.T@np.r_[samples[i].tcp,1])[:3]
            empty=samples[i].aperture is not None and cert['aperture_m'] is not None and cert['aperture_m']>.003 and samples[i].aperture<.001
            far=point is not None and np.linalg.norm(np.asarray(point)-tcp)>cert['object_TCP_distance_m']+self.cfg.get('grasp_region_margin_m',.05)
            cert['state']=None if point is None else False if empty or far or self.command_g[i]>cert['g_command']+.1 else True
            cert['state_source']='fresh SAM region+own jaw/closed command and bounded semantic certificate; tentative, not contact'
        if self.stage and any(c.get('state') is not True for (arm,role),c in self.certificates.items() if arm!=self.stage['arm']):raise SafetyError('passive tool holding unknown/false before dispatch')
        if self.accepted_grasp:self.current_holding=self.accepted_grasp['state']
        else:self.current_holding=None
        return samples,obs
    def holding(self,samples,stage):
        if stage['holding_contract']=='none':return True
        cert=self.certificates.get((stage['arm'],stage['object']))
        return None if cert is None else cert.get('state',self.current_holding)
    def semantics(self,stage,facts,before):
        if stage.get('hold_at_end'):
            after=before
            from prove.xtrainer.runtime import Executor
            while True:
                after=Executor.semantics(self,stage,facts,after)
                if all(self.visual_facts.get(f) is True for f in facts):break
                if any(self.visual_facts.get(f) is False for f in facts):raise SafetyError('held-terminal goal explicitly false')
                self.sleep(.1);after=self.capture()
            tools=stage.get('declared_held_tools',[])
            if not tools or any(self.certificates.get((tool['arm'],tool['object']),{}).get('state') is not True for tool in tools):raise SafetyError('held-terminal actual tool invariant unknown/false')
            self.semantic_log.append({'stage_id':stage['stage_id'],'hold_at_end':True,'facts':dict(self.visual_facts),'goal_relation':stage['goal_relation']});return after
        after=super().semantics(stage,facts,before)
        key=(stage['arm'],stage['object'])
        if stage['phase']=='Grasp':self.accepted_grasp['state']=True;self.certificates[key]=self.accepted_grasp
        elif stage['phase']=='Complete':self.certificates.pop(key,None)
        return after
    def command(self,stage,elapsed,samples,obs,dt):
        if stage['phase']=='Hold':return self._hold(stage,elapsed,samples,obs,dt)
        cert=self.certificates.get((stage['arm'],stage['object']));self.accepted_grasp=cert
        original=self.b['gripper']
        if stage['phase'] in CARRY:
            if cert is None or self.holding(samples,stage) is not True:raise SafetyError('special carrying certificate unknown/false')
            locked=original.copy();locked[:,cert['index']]=cert['g_command'];self.b['gripper']=locked
        elif stage['phase']=='Wait':
            locked=original.copy();locked[:,0 if stage['arm']=='left' else 1]=self.command_g[0 if stage['arm']=='left' else 1];self.b['gripper']=locked
        try:return self._reference_command(stage,elapsed,samples,obs,dt)
        finally:self.b['gripper']=original
    def _hold(self,stage,elapsed,samples,obs,dt):
        cert=self.certificates.get((stage['arm'],stage['object']))
        if cert is None or self.holding(samples,stage) is not True:raise SafetyError('held tool evidence unknown/false')
        actual=np.stack([s.q for s in samples]);pose=self._shared_pose(actual,self.command_g,stage['arm'])
        if self._phase!=stage['stage_id']:self._phase=stage['stage_id'];self._hold_pose=(pose[0].copy(),pose[1].copy())
        target,_,error=self._step_ik(actual,self.command_g,stage['arm'],self._hold_pose);fresh,frame=self.capture();self.safety.validate(target,self.command_g,fresh,dt,self.robot,self.T)
        latest=self.robot.samples();self.safety.samples(latest)
        if not 0<=self.clock()-frame.stamp<=self.cfg['safety']['max_age_sec'] or np.max(abs(target-np.stack([s.q for s in latest])))>.01:raise SafetyError('Hold fresh target changed/stale')
        i=cert['index'];self._hardware_attempts+=1;write(self.output/'COMMAND_INTENT.json',{'stage_id':stage['stage_id'],'attempt_count':self._hardware_attempts,'acknowledged_count':self._hardware_acknowledged,'q_target_rad':target.tolist(),'gripper_command':self.command_g.tolist(),'flags':[i==0,i==1],'partial_motion_possible_without_ack':True})
        self.robot.command(target,self.command_g,[i==0,i==1]);self._hardware_acknowledged+=1;self.command_index+=1;self.safety.last_target=target.copy();self.sleep(dt);self._last_cartesian_goal=self._hold_pose
        duration=stage['reference']['end_time_sec']-stage['reference']['start_time_sec']
        return target,float(elapsed>=duration and self._reached(self._hold_pose,pose)),{'controller':'actual tool pose/grip Hold; no opening profile','original_phase':'Hold','error':error.tolist()}
    def _reference_command(self,stage,elapsed,samples,obs,dt):
        index=0 if stage['arm']=='left' else 1;arm=stage['arm'];q_actual=np.stack([s.q for s in samples])
        if self._cycle!=stage['cycle_id']:
            if stage['phase']!='Approach':raise SafetyError('new manipulation cycle must begin with object Approach')
            self._cycle=stage['cycle_id'];self.coarse_source=None;self.coarse_target=None;self._coarse_source_finalized=False;self.attachment=None;self._empty_hand_anchor=None;self._release_pose_anchor=None
            self.static_source_env=None;self.static_target_env=None;self.code_static_source_handled=None;self.code_static_target_handled=None
        start=stage['reference']['start_time_sec'];end=stage['reference']['end_time_sec']
        first_query=self._phase!=stage['stage_id']
        if first_query:
            self._phase=stage['stage_id'];self._phase_elapsed=0.;self._last_cartesian_goal=None
        actual_pose=self._shared_pose(q_actual,self.command_g,arm)
        if stage['phase']=='Approach':
            t=end;progress=0.;ref=self.nearest_ref(end)
            if self.coarse_source is None:
                object_now=obs.roles.get(stage['object']);start_ref=self.nearest_ref(start);object_ref=start_ref['surface_centroids_world_m'].get(stage['object'])
                initial_ref=getattr(self,'static_reference_points',None) or self.b['frames'][0]['surface_centroids_world_m'];initial=getattr(self,'initial_roles',{})
                self.source_prior='fresh cycle Approach START paired static environment cue'
                if stage['object'] not in self._manipulated and initial.get(stage['object']) is not None and initial_ref.get(stage['object']) is not None:
                    object_now=initial[stage['object']];object_ref=initial_ref[stage['object']]
                    self.source_prior='scene initial visible paired prior; assumed unmanipulated static role, not fresh measurement'
                if object_now is None or object_ref is None:raise SafetyError('static source environment grounding unknown')
                self.source_anchor=np.asarray(object_now).copy();self.coarse_source=self.source_anchor-np.asarray(object_ref)
                target=self.target_point(obs,stage);target_ref=start_ref['surface_centroids_world_m'].get(stage['target'])
                self.target_prior='cycle static visible/fixed-handover target prior'
                if stage['target']=='box' and initial.get('box') is not None and initial_ref.get('box') is not None:
                    target=initial['box'];target_ref=initial_ref['box']
                    self.target_prior='scene initial SAM/reference paired box prior; static assumption, not fresh measurement'
                if target_ref is None and stage['target']=='table_handover_region':target_ref=self.b['refs']['handover_anchors'][stage['cycle_id']]['centroid_world_m']
                if target is None or target_ref is None:raise SafetyError('static target environment grounding unknown')
                self.target_anchor=np.asarray(target).copy();self.coarse_target=self.target_anchor-np.asarray(target_ref)
                self.static_source_env=self.coarse_source.copy();self.static_target_env=self.coarse_target.copy()
                self.code_static_source_handled=self.coarse_source.copy();self.code_static_target_handled=self.coarse_target.copy()
        else:
            if not first_query and (self._last_cartesian_goal is None or self._reached(self._last_cartesian_goal,actual_pose)):self._phase_elapsed+=dt
            t=min(end,start+self._phase_elapsed*stage.get('reference_speed_scale',1.)*self.cfg['hardware_reference_speed_scale'])
            progress=(t-start)/max(end-start,1e-9);ref=self.nearest_ref(t)
        q_ref=np.asarray(self.b['qcurve'](t));g=np.array([np.interp(t,self.b['times'],self.b['gripper'][:,i]) for i in range(2)])
        g[1-index]=self.command_g[1-index]
        if stage['phase']=='Approach':
            g[index]=max(float(self.command_g[index]),float(np.interp(start,self.b['times'],self.b['gripper'][:,index])))
        if stage['phase'] in ('Approach','Grasp','Lift'):coarse=self.coarse_source.copy()
        elif stage['phase']=='Move':coarse=(1-progress)*self.coarse_source+progress*self.coarse_target
        elif stage['phase'] in ('Align','Insert','Pull','Push','Release','Wait'):coarse=self.coarse_target.copy()
        else:raise SafetyError('undeclared object-centered engineering phase')
        # Frozen features12:18 are cycle-latched environment cues, NOT current
        # carried-object motion or visibility-dependent surface median errors.
        local_source_known=self.static_source_env is not None and self.static_target_env is not None
        if not local_source_known:raise SafetyError('frozen static environment feature contract unknown')
        local_source=self.static_source_env-self.code_static_source_handled
        local_target=self.static_target_env-self.code_static_target_handled
        record={'controller':'object_centered_Code_6D_current_pose_IK','reference_time_sec':t,
                'gripper_primitive_contract':'Approach actor remains declared open; closing only in Grasp profile' if stage['phase']=='Approach' else 'declared node gripper profile',
                'coarse_world_translation_m':coarse.tolist(),'locked_source_translation_m':self.coarse_source.tolist(),
                'locked_target_translation_m':self.coarse_target.tolist(),'local_source_residual_m':local_source.tolist(),
                'local_target_residual_m':local_target.tolist(),'local_source_known':local_source_known,
                'NN_feature_contract':'fixed cycle environment residual after subtracting Code-handled static shift; never object-motion/occlusion error',
                'static_source_environment_cue_m':self.static_source_env.tolist(),'static_target_environment_cue_m':self.static_target_env.tolist(),
                'Code_handled_static_source_m':self.code_static_source_handled.tolist(),'Code_handled_static_target_m':self.code_static_target_handled.tolist(),
                'source_environment_prior':getattr(self,'source_prior','cycle-latched'),'target_environment_prior':getattr(self,'target_prior','cycle-latched'),'coarse_reference_view':'actual Grasp-node attachment registration' if self.attachment is not None else 'Approach START; never re-locked at occluded Approach END','NN_action_semantics':'delta_q_reference_then_native_FK; never delta_q_current'}
        if stage['controller']=='neural_short':
            if not local_source_known:raise SafetyError('neural local object residual unknown')
            phases=('Grasp','Align','Insert','Push','Pull')
            features=np.r_[samples[index].q,samples[index].qdot,local_source,local_target,progress,np.eye(2)[index],np.eye(5)[phases.index(stage['phase'])],np.zeros(3)]
            step=self.b['skill'].command(features,current_surface_xyz=obs.roles.get(stage['object']),
                reference_surface_xyz=ref['surface_centroids_world_m'].get(stage['object']),phase=stage['phase'],holding=self.current_holding,
                matched_reference_time_sec=ref['time_sec'],current_tcp_xyz=actual_pose[0],current_tcp_rotation=actual_pose[1],
                reference_tcp_xyz=np.asarray(ref['tcp_position'])[index],reference_tcp_rotation=np.asarray(ref['tcp_rotation'])[index])
            q_ref=q_ref.copy();q_ref[index]+=step.action;record['frozen_neural_record']=step.record;record['frozen_delta_q_reference_rad']=step.action.tolist()
        native_goal=self._shared_pose(q_ref,g,arm)
        if stage['phase']=='Release':
            if self._release_pose_anchor is None:self._release_pose_anchor=(actual_pose[0].copy(),actual_pose[1].copy())
            goal=self._release_pose_anchor
            record['pose_anchor_source']='actual Align-end/Release-entry; hold while opening'
        elif stage['phase']=='Wait':
            if self._empty_hand_anchor is None:raise SafetyError('actual Release-end empty-hand anchor missing')
            ref_start=self._shared_pose(np.asarray(self.b['qcurve'](start)),g,arm)
            goal=(self._empty_hand_anchor[0]+native_goal[0]-ref_start[0],native_goal[1]@ref_start[1].T@self._empty_hand_anchor[1])
            record['pose_anchor_source']='actual Release-end plus demonstrated empty-hand relative motion'
        elif stage['phase'] in ('Lift','Move','Align','Insert','Push') and self.attachment is not None:
            goal,obj_goal=self.attached_goal(native_goal,coarse)
            record.update(reference_object_goal_shared_m=obj_goal.tolist(),actual_attached_offset_local_m=self.attachment['a_actual'].tolist(),
                          declared_attached_offset_local_m=self.attachment['a_ref'].tolist(),attachment_source='fresh SAM+actual robot at successful Grasp node; estimate not contact truth')
        else:goal=(native_goal[0]+coarse,native_goal[1])
        self._last_cartesian_goal=goal
        self._goal_reached=self._reached(goal,actual_pose)
        if stage['phase']=='Approach':progress=1. if self._goal_reached else 0.
        target,_,error=self._step_ik(q_actual,g,arm,goal)
        fresh_samples,fresh_obs=self.capture()
        # Visible surface medians change with EEF occlusion; do not treat them as rigid-body drift truth.
        self.safety.validate(target,g,fresh_samples,dt,self.robot,self.T)
        final_samples=self.robot.samples();self.safety.samples(final_samples)
        if not 0<=self.clock()-fresh_obs.stamp<=self.cfg['safety']['max_age_sec']:raise SafetyError('fresh post-IK roles stale before command')
        if np.max(abs(target-np.stack([s.q for s in final_samples])))>.01+1e-9:raise SafetyError('actual pose changed during IK beyond bounded step')
        self._hardware_attempts+=1
        intent={'attempt_count':self._hardware_attempts,'acknowledged_count':self._hardware_acknowledged,'stage_id':stage['stage_id'],
                'q_target_rad':target.tolist(),'gripper_command':g.tolist(),'flags':[index==0,index==1],'status':'attempted_not_acknowledged',
                'partial_motion_possible_without_ack':True,'monotonic_sec':self.clock()}
        write(self.output/'COMMAND_INTENT.json',intent)
        self.robot.command(target,g,[index==0,index==1]);self._hardware_acknowledged+=1
        intent.update(status='acknowledged',acknowledged_count=self._hardware_acknowledged);write(self.output/'COMMAND_INTENT.json',intent)
        self.safety.last_target=target.copy();self.command_g=g.copy();self.sleep(dt);self.command_index+=1
        record.update(goal_TCP_shared_m=goal[0].tolist(),goal_rotation_shared=goal[1].tolist(),Cartesian_error_before_command=error.tolist(),
                      candidate_joint_step_rad=(target-q_actual).tolist(),planned_joint_step_limit_rad=.009,dispatch_joint_step_limit_rad=.01,actor_only_flags=[index==0,index==1])
        return target,progress,record
    def predicates(self,stage,*args,**kwargs):
        facts=super().predicates(stage,*args,**kwargs)
        facts['passive_tools_held']=all(c.get('state') is True for key,c in self.certificates.items() if key!=(stage['arm'],stage['object']))
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
        result={'schema':'graph-self-adapter.multitask-special-run.v1','task':self.b['task'],'execution_variant':'multitask_special_object_centered_v20','graph_completed':complete,'failure':failure,'graph_stages':len(self.b['graph']['stages']),'stage_trace':self.stage_trace,'trace':self.trace,'judge_records':self.judge.records,'certificates':[{k:v for k,v in cert.items() if k!='state'} for cert in self.certificates.values()],'semantic_log':self.semantic_log,'attachment_updates':getattr(self,'attachment_updates',[]),'command_attempts':self._hardware_attempts,'command_acknowledgments':self._hardware_acknowledged,'hardware_command_attempts':self._hardware_attempts,'hardware_command_acknowledgments':self._hardware_acknowledged,'stop_results':stopped,'fixed_z':True,'latent_updates':0,'final_original_goal_requires_postseal_private_evaluation':True,'model_action_semantics':'q_reference plus frozen NN residual → FK → object-frame Code IK from actual q','hardware_qualified':False,'unsupported_Agent_waits':0}
        write(self.output/'RESULT.json',result);return result
