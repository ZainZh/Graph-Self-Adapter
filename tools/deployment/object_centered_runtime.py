"""Additive real engineering reference-frame layer, not frozen simulator results.

Frozen NN outputs remain delta-q-reference: q-reference + delta is evaluated by
native FK, translated by a locked coarse object frame, then reached from actual
q using bounded 6D Code IK. No direct actual-q + learned-delta interpretation.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from prove.xtrainer.runtime import Executor
from prove.xtrainer.io import SafetyError,array
from prove.xtrainer.agent import write

VARIANT='object_centered_real_reference_v01'
SCOPE_STOP='engineering_scope_stop_after_first_stage'


class ObjectCenteredExecutor(Executor):
    first_stage_only=False
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        budget=self.cfg.get('maximum_hardware_commands')
        if budget is not None and (type(budget) is not int or budget<=0):raise SafetyError('maximum_hardware_commands must be a positive integer')
        self.engineering_first_stage_only=self.first_stage_only;self._scope_complete=False
        self.coarse_source=None;self.coarse_target=None;self.source_anchor=None;self.target_anchor=None
        self._coarse_source_finalized=False;self._cycle=None;self._phase=None;self._phase_elapsed=0.;self._last_cartesian_goal=None;self._goal_reached=False
        self.attachment=None;self._empty_hand_anchor=None;self._release_pose_anchor=None
        self.static_source_env=None;self.static_target_env=None;self.code_static_source_handled=None;self.code_static_target_handled=None
        self._grasp_bootstrap=None;self._hardware_attempts=0;self._hardware_acknowledged=0
        self._manipulated=set();self._first_stage=self.b['graph']['stages'][0]['stage_id']
    def guard_limits(self):
        if self.engineering_first_stage_only and self._scope_complete:
            raise SafetyError(SCOPE_STOP)
        super().guard_limits()
        budget=self.cfg.get('maximum_hardware_commands')
        if budget is not None and self.command_index>=budget:raise SafetyError('maximum hardware command budget exhausted')
    def capture(self):
        if self.engineering_first_stage_only and self._scope_complete and self.stage and self.stage['stage_id']!=self._first_stage:
            raise SafetyError(SCOPE_STOP)
        samples,obs=super().capture()
        if self._grasp_bootstrap is not None and self.visual_anchor is not None:
            role,certified=self._grasp_bootstrap;anchor=self.visual_anchor
            if self.stage and self.stage['object']==role and anchor[0]==role and 0<=self.clock()-certified<=10.:
                index=anchor[1];point=obs.roles.get(role);tcp=samples[index].tcp if index==0 else (self.T@np.r_[samples[index].tcp,1])[:3]
                if point is not None and np.linalg.norm(tcp-anchor[3])<.003 and np.linalg.norm(np.asarray(point)-tcp-anchor[2])<=.006:
                    self.current_holding=True;self.holding_source='bounded_visual_grasp_bootstrap'
        return samples,obs
    def semantics(self,stage,facts,before):
        if facts==['holding'] and self.current_holding is True and getattr(self,'holding_source','') in ('bounded_visual_grasp_bootstrap','nonzero_visible_object_TCP_comotion'):
            return before
        after=super().semantics(stage,facts,before)
        if 'holding' in facts and self.visual_facts.get('holding') is not True:self._grasp_bootstrap=None
        if stage['phase']=='Grasp' and self.visual_facts.get('holding') is True:self._grasp_bootstrap=(stage['object'],self.clock())
        return after
    def start(self):
        samples,obs=self.capture()
        if any(s.running or s.mode!=5 or np.max(abs(s.qdot))>.01 for s in samples):raise SafetyError('current robot not stationary')
        if self.cfg.get('initial_gripper_state_confirmed') is not True:raise SafetyError('operator initial gripper declaration absent')
        if np.any(self.command_g<0) or np.any(self.command_g>1):raise SafetyError('invalid declared current gripper commands')
        if obs.roles.get('red_mug') is None or obs.roles.get('white_mug') is None or obs.roles.get('box') is None:
            raise SafetyError('fresh current cup/box role geometry missing')
        self.initial_roles={k:None if v is None else np.asarray(v).copy() for k,v in obs.roles.items()}
        self.previous_cycle=None
    def target_point(self,obs,stage):
        initial=getattr(self,'initial_roles',{})
        if stage['target']=='box' and 'box' not in self._manipulated and initial.get('box') is not None:return np.asarray(initial['box']).copy()
        return super().target_point(obs,stage)
    def _shared_pose(self,q,g,arm):
        xyz,rotation=self.robot.fk(q,g,arm)
        xyz=array(xyz,(3,));rotation=array(rotation,(3,3))
        if arm=='right':return (self.T@np.r_[xyz,1])[:3],self.T[:3,:3]@rotation
        return xyz,rotation
    @staticmethod
    def _error(goal,pose):
        return np.r_[np.asarray(goal[0])-pose[0],Rotation.from_matrix(np.asarray(goal[1])@pose[1].T).as_rotvec()]
    def _reached(self,goal,pose):
        error=self._error(goal,pose)
        return bool(np.linalg.norm(error[:3])<=.003 and np.linalg.norm(error[3:])<=.02)
    def _step_ik(self,actual_q,g,arm,goal):
        q=np.asarray(actual_q).copy();index=0 if arm=='left' else 1;pose=self._shared_pose(q,g,arm)
        error=self._error(goal,pose);jac=np.zeros((6,6));eps=1e-4
        for j in range(6):
            probe=q.copy();probe[index,j]+=eps;xyz,rotation=self._shared_pose(probe,g,arm)
            jac[:3,j]=(xyz-pose[0])/eps;jac[3:,j]=Rotation.from_matrix(rotation@pose[1].T).as_rotvec()/eps
        delta=jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),error)
        delta*=min(1.,.009/max(float(np.max(abs(delta))),1e-12));q[index]+=delta
        return q,pose,error
    def command(self,stage,elapsed,samples,obs,dt):
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
        elif stage['phase'] in ('Align','Release','Wait'):coarse=self.coarse_target.copy()
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
        elif stage['phase'] in ('Lift','Move','Align') and self.attachment is not None:
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
    def register_attachment(self,stage,samples,obs):
        index=0 if stage['arm']=='left' else 1;point=obs.roles.get(stage['object'])
        end=float(stage['reference']['end_time_sec']);ref=self.nearest_ref(end);ref_object=ref['surface_centroids_world_m'].get(stage['object'])
        if point is None or ref_object is None:raise SafetyError('successful Grasp attachment geometry unknown')
        tcp=samples[index].tcp;rotation=samples[index].rotation
        if index:tcp=(self.T@np.r_[tcp,1])[:3];rotation=self.T[:3,:3]@rotation
        nominal_end=self._shared_pose(np.asarray(self.b['qcurve'](end)),self.command_g,stage['arm'])
        self.attachment={'role':stage['object'],'a_actual':rotation.T@(np.asarray(point)-tcp),
            'a_ref':nominal_end[1].T@(np.asarray(ref_object)-nominal_end[0]),
            'R_demo_end':nominal_end[1].copy(),'R_actual_grasp':rotation.copy()}
        self.coarse_source=np.asarray(point)-np.asarray(ref_object)
    def attached_goal(self,native_demo_pose,coarse):
        a=self.attachment;object_goal=native_demo_pose[0]+native_demo_pose[1]@a['a_ref']+coarse
        rotation=native_demo_pose[1]@a['R_demo_end'].T@a['R_actual_grasp']
        return (object_goal-rotation@a['a_actual'],rotation),object_goal
    def predicates(self,stage,progress,target,samples,obs):
        facts=super().predicates(stage,progress,target,samples,obs)
        current_q=np.stack([s.q for s in samples]);pose=self._shared_pose(current_q,self.command_g,stage['arm'])
        reached=self._last_cartesian_goal is not None and self._reached(self._last_cartesian_goal,pose)
        if stage['phase']=='Approach':
            progress=1. if reached else 0.
        for row in stage['completion_predicates']:
            if row['predicate']=='reference_progress':facts[row['id']]=bool(progress>=1.-1e-9)
            if row['predicate']=='actor_command_reached':facts[row['id']]=bool(facts[row['id']] is True and reached)
        if stage['phase']=='Grasp' and self.current_holding is True:self._manipulated.add(stage['object'])
        if all(facts.get(k) is True for k in stage['done']):
            if stage['phase']=='Grasp' and self.attachment is None:self.register_attachment(stage,samples,obs)
            if stage['phase']=='Release':self._empty_hand_anchor=(pose[0].copy(),pose[1].copy())
        if self.engineering_first_stage_only and stage['stage_id']==self._first_stage and all(facts.get(k) is True for k in stage['done']):self._scope_complete=True
        return facts
    def run(self):
        result=super().run()
        result.update(execution_variant=VARIANT,engineering_reference_frame_contract=True,first_stage_only=self.engineering_first_stage_only,
            hardware_command_attempts=self._hardware_attempts,hardware_command_acknowledgments=self._hardware_acknowledged,
            partial_motion_possible_without_ack=True,holding_bootstrap_contract='actual Root grasp certificate <=10s; fresh relative offset <=6mm before TCP displacement 3mm',
            frozen_graph_stage_count=len(self.b['graph']['stages']),engineering_stage_subset=([self._first_stage] if self.engineering_first_stage_only else None),
            source_coarse_lock_at_Approach_START=True,median_shift_scene_failure_gate=False,not_a_historical_score=True,does_not_apply_NN_delta_to_actual_q=True)
        if self.engineering_first_stage_only and result.get('failure')==SCOPE_STOP:
            result.update(graph_completed=False,engineering_stage_completed=True,failure=None,scoped_stop_reason=SCOPE_STOP)
        write(self.output/'RESULT.json',result);return result
