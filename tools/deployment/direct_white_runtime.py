"""One complete task: red legacy two cycles, white left directly onto observed box."""
import copy
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from handle_macro_runtime import MacroExecutor,MacroVision
from shared_cup_runtime import SharedCupExecutor as CommonExecutor
from handle_local_candidate_v4 import handle_plane_relation,choose_clear_pad_overlap
from direct_white_geometry import upright_support_prior,direct_targets,tcp_at_handle
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError

VARIANT='red_legacy_white_direct_box_v15'
CARRY=('Lift','Raise','Move','Align','Lower')


class DirectExecutor(MacroExecutor):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.direct_plan=None;self.phase_goals={};self._direct_closed_at=None
    def _reached(self,goal,pose):
        if self.stage and self.stage['cycle_id']=='white_direct':
            error=self._error(goal,pose);return bool(np.linalg.norm(error[:3])<=.001 and np.linalg.norm(error[3:])<=.02)
        return super()._reached(goal,pose)
    def prepare_white(self,stage,samples,obs):
        self.vision.use_initial_white_prior=True
        try:
            handle=self.vision.cloud('white_handle');cup=self.vision.cloud('white_mug');table=self.vision.cloud('table_patch')
        finally:self.vision.use_initial_white_prior=False
        relation=handle_plane_relation(handle,self.vision.T[:3,3],cup)
        pads=copy.deepcopy(self.pad_library['arms']['left'])
        for key,value in list(pads.items()):
            if isinstance(value,list) and key!='overlap_candidates':pads[key]=np.asarray(value)
        pads,entry=choose_clear_pad_overlap(relation,pads,float(np.median(table[:,2])))
        prior=upright_support_prior(cup,relation['center'],table)
        final=self.b['frames'][-1]['surface_centroids_world_m'];initial=self.b['frames'][0]['surface_centroids_world_m']
        target=direct_targets(prior,self.vision.cloud('box'),obs.roles['box'],final['white_mug'],initial['box'])
        self.direct_plan={'entry':entry,'pre':(entry[0]-relation['approach_down']*relation['span_m'],entry[1]),'prior':prior,'targets':target,'pad':pads['pad_midpoint_TCP_local_m'],'pre_reached':False}
        write(self.output/'WHITE_DIRECT_PLAN.json',{'prior':prior,'targets':target,'entry_TCP_m':entry[0].tolist(),'initial_upright_relationship_R':entry[1].tolist(),'robot_pad_local_m':self.direct_plan['pad'].tolist(),
            'source':'actual SAM initial upright relation and current SAM box; declared finalwhite relative fixed referencebox XY; no airborne attachment used as desired support pose'})
    def dispatch_direct(self,goal,g,samples,dt):
        actual=np.stack([s.q for s in samples]);target,_,error=self._step_ik(actual,g,'left',goal);fresh,frame=self.capture()
        self.safety.validate(target,g,fresh,dt,self.robot,self.T);latest=self.robot.samples();self.safety.samples(latest)
        if self.clock()-frame.stamp>self.cfg['safety']['max_age_sec']:raise SafetyError('direct fresh observation expired')
        if np.max(abs(target-np.stack([s.q for s in latest])))>.01+1e-9:raise SafetyError('direct actual pose changed during bounded IK')
        self._hardware_attempts+=1;intent={'attempt_count':self._hardware_attempts,'acknowledged_count':self._hardware_acknowledged,'stage_id':self.stage['stage_id'],'q_target_rad':target.tolist(),'gripper_command':g.tolist(),'flags':[True,False],'partial_motion_possible_without_ack':True}
        write(self.output/'COMMAND_INTENT.json',intent);self.robot.command(target,g,[True,False]);self._hardware_acknowledged+=1
        self.safety.last_target=target.copy();self.command_g=g.copy();self.command_index+=1;self.sleep(dt)
        return target,error
    def command(self,stage,elapsed,samples,obs,dt):
        if stage['cycle_id']!='white_direct':return super().command(stage,elapsed,samples,obs,dt)
        if self.direct_plan is None:self.prepare_white(stage,samples,obs)
        plan=self.direct_plan;phase=stage['phase'];pose=self._shared_pose(np.stack([s.q for s in samples]),self.command_g,'left');g=self.command_g.copy();progress=0.
        if phase=='Approach':
            g[0]=max(g[0],self.b['initial_g'][0])
            if self._reached(plan['pre'],pose):plan['pre_reached']=True
            goal=plan['entry'] if plan['pre_reached'] else plan['pre'];progress=1. if plan['pre_reached'] and self._reached(goal,pose) else 0.
        elif phase=='Grasp':
            goal=plan['entry'];g[0]=0.
            if self._direct_closed_at is None:self._direct_closed_at=self.clock()
            duration=stage['reference']['end_time_sec']-stage['reference']['start_time_sec'];progress=1. if self.clock()-self._direct_closed_at>=duration and self._reached(goal,pose) else 0.
        else:
            if phase in CARRY:
                if self.macro_grasp is None:raise SafetyError('direct accepted grip command missing')
                g[0]=self.macro_grasp['g_command']
            else:g[0]=1. # explicit open; no nominal carry opening profile
            if phase not in self.phase_goals:
                point=pose[0]+pose[1]@plan['pad'];R=plan['entry'][1]
                if phase=='Lift':point[2]=max(point[2],plan['prior']['initial_table_z_m']+plan['prior']['visible_extent_radius_m']+.25*plan['prior']['observed_visible_height_m']);goal=tcp_at_handle(point,R,plan['pad'])
                elif phase=='Raise':point[2]=max(point[2],plan['targets']['transport_handle_height_m']);goal=tcp_at_handle(point,R,plan['pad'])
                elif phase=='Move':point[:2]=plan['targets']['handle_support_goal_m'][:2];point[2]=plan['targets']['transport_handle_height_m'];goal=tcp_at_handle(point,R,plan['pad'])
                elif phase=='Lower':goal=tcp_at_handle(plan['targets']['handle_support_goal_m'],R,plan['pad'])
                elif phase=='Release':goal=(pose[0].copy(),pose[1].copy())
                elif phase=='Retreat':p=pose[0].copy();p[2]+=plan['prior']['observed_visible_height_m'];goal=(p,pose[1].copy())
                else:raise SafetyError('unsupported direct white phase')
                self.phase_goals[phase]=goal
            goal=self.phase_goals[phase];progress=1. if self._reached(goal,pose) else 0.
        self._last_cartesian_goal=goal;self._goal_reached=self._reached(goal,pose);target,error=self.dispatch_direct(goal,g,samples,dt)
        return target,progress,{'controller':'observed_upright_white_direct_Code','phase':phase,'goal_TCP_m':goal[0].tolist(),'goal_R':goal[1].tolist(),'error_before_command':error.tolist(),'holdingG_locked_until_Release':phase in CARRY,'initial_upright_support_target':True}
    def predicates(self,stage,progress,target,samples,obs):
        if stage['cycle_id']!='white_direct':facts=super().predicates(stage,progress,target,samples,obs)
        else:
            reached=self._last_cartesian_goal is not None and self._reached(self._last_cartesian_goal,self._shared_pose(np.stack([s.q for s in samples]),self.command_g,'left'))
            facts={'own_pose':reached,'own_progress':progress>=1.-1e-9,'finite':True}
            if stage['phase']=='Approach':facts['own_progress']=bool(facts['own_progress'] and self.direct_plan['pre_reached'])
            if stage['phase']=='Grasp':facts['Root_holding']=self.holding(samples,stage)
        if stage['phase']=='Release':
            index=0 if stage['arm']=='left' else 1;gap=samples[index].aperture;initial=self.initial_state_diagnostics['actual_aperture_m'][index]
            measured=gap is not None and initial is not None
            duration=stage['reference']['end_time_sec']-stage['reference']['start_time_sec']
            opened=(gap>=.9*initial) if measured else (self.command_g[index]>=.9 and self.clock()-self.stage_started>=duration)
            self.release_progress_record={'sensor_measured':measured,'actual_aperture_m':gap,'issued_open_command':float(self.command_g[index]),'criterion':'observed >=90% initial aperture' if measured else 'commanded open plus original minimum profile time; measured opening unknown, Complete Root must confirm released'}
            key=next(p['id'] for p in stage['completion_predicates'] if p['predicate']=='reference_progress');facts[key]=bool(facts.get(key) and opened)
        return facts
    def semantics(self,stage,facts,before):
        if stage['phase']!='Complete':return super().semantics(stage,facts,before)
        self.vision.invalidate();self.vision.memo_enabled=False
        try:after=CommonExecutor.semantics(self,stage,facts,before)
        finally:self.vision.invalidate();self.vision.memo_enabled=True
        if any(self.visual_facts.get(f) is not True for f in facts):raise SafetyError('supported Root completion unknown/false')
        self.macro_grasp=None;return after
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
                    if stage['phase'] in CARRY and self.holding(samples,stage) is not True:raise SafetyError('direct transfer carrying anomaly')
                    elapsed+=self.cfg['command_dt_sec'];target,progress,record=self.command(stage,elapsed,samples,obs,self.cfg['command_dt_sec']);samples,obs=self.capture()
                    if stage['phase']=='Grasp' and progress>=1.-1e-9 and self.holding(samples,stage) is not True:samples,obs=self.semantics(stage,['holding'],(samples,obs))
                    facts=self.predicates(stage,progress,target,samples,obs);self.trace.append({'stage_id':stage['stage_id'],'command_index':self.command_index,'facts':facts,'issued_q':target.tolist(),'issued_g':self.command_g.tolist(),'actual_q':[s.q.tolist() for s in samples],'actual_aperture_m':[s.aperture for s in samples],'controller':record,'holding_state_source':getattr(self,'holding_source','unknown'),'release_open_progress':getattr(self,'release_progress_record',None) if stage['phase']=='Release' else None})
                    if any(facts.get(k) is not True for k in stage['invariants']):raise SafetyError('direct route invariant unknown/false')
                    if all(facts.get(k) is True for k in stage['done']):done=True;break
                self.stage_trace.append({'stage_id':stage['stage_id'],'completed':done,'commands':count+1})
                if not done:raise SafetyError('direct route node budget exhausted')
                if stage['phase']=='Lift' and stage['cycle_id']!='white_direct':self.refresh_lift_attachment(stage,*self.capture())
            complete=True
        except Exception as exc:
            failure=str(exc)
            for stop in self.stops:
                try:stop.stop();stops.append({'stopped':True})
                except Exception as err:stops.append({'error':str(err)})
        result={'schema':'graph-self-adapter.xtrainer-run.v1','execution_variant':VARIANT,'graph_completed':complete,'failure':failure,'graph_stages':len(self.b['graph']['stages']),'stage_trace':self.stage_trace,'trace':self.trace,'Root_decisions':self.judge.records,'root_request_count':len(self.judge.records),'fixed_z':True,'latent_updates':0,'mode':self.cfg['mode'],'hardware_motion_commands':self.robot.commands,'hardware_command_attempts':self._hardware_attempts,'hardware_command_acknowledgments':self._hardware_acknowledged,'stop_results':stops,'rgbd_discard_log':getattr(self,'rgbd_discard_log',[]),'SAM_same_update_reuse_count':self.vision.memo_reuse_count,'unsupported_Agent_waits':0,'Root_boundaries':['Grasp','Complete after open and empty retreat'],'original_final_goal_unchanged':True,'white_upright_prior_no_airborne_target':True,'policy_claim':'red frozen reference+residual; explicit observed white direct-place Code; not learned handle policy'}
        write(self.output/'RESULT.json',result);return result

SharedCupExecutor=DirectExecutor
SharedSAMVision=MacroVision
