"""Independent full-graph engineering route; active local candidates never import it.

Only white c03/c04 Approach/Grasp use functional-part Code. All other nodes
retain frozen reference+residual policy and reviewed actual attachment mapping.
"""
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from shared_cup_runtime import SharedCupExecutor as CommonExecutor
from handle_local_driver_v4r4 import HandleVision,lock_grasp_envelope,lift_from_locked_envelope
from rgbd_capture_retry import bounded_rgbd_capture
from handle_local_candidate_v4 import handle_plane_relation,candidate_goal,choose_clear_pad_overlap
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write,digest

VARIANT='handle_full_object_centered_Code_v1'


class HandleFullVision(HandleVision):
    def __init__(self,camera,*args,**kwargs):
        # Real ZED has no simulator physical clock; diagnostic metadata is None.
        if not hasattr(camera,'last_physical_time'):camera.last_physical_time=None
        super().__init__(camera,*args,**kwargs)
        self.stage_context=lambda:None;self.frame_records=[]
    def capture(self):
        previous=len(self.mask_records);obs=super().capture();self.latest_observation_stamp=obs.stamp;context=self.stage_context() or {}
        record={'frame_index':self.frame_count-1,'stage_id':context.get('stage_id'),'phase':context.get('phase'),
            'observed_monotonic_sec':obs.stamp,'recorded_monotonic_sec':self.clock(),
            'physical_time_sec':self.raw_camera.last_physical_time,
            'clock_source':'simulator camera snapshot physical clock' if self.raw_camera.last_physical_time is not None else 'real camera monotonic wall timestamp; physical clock unavailable',
            'mask_pixels':{k:int(v.sum()) for k,v in self.latest_masks.items()},'mask_source':'actual_SAM2','GT_fallback':False}
        if len(self.mask_records)>previous:record.update(files=self.mask_records[-1]['files'],role_points=self.mask_records[-1]['role_points'])
        self.frame_records.append(record)
        if self.evidence_root is not None and (len(self.mask_records)>previous or len(self.frame_records)%10==0):self.save_frame_evidence()
        return obs
    def save_frame_evidence(self):
        write(Path(self.evidence_root)/'MASK_EVIDENCE.json',{'frames':self.frame_records,'every_accepted_capture_logged':True,'RGB_mask_depth_files_sparse_every10':True,'actual_SAM':True,'private_object_pose_recorded':False})
    def close(self):
        if self.evidence_root is not None and self.frame_records:self.save_frame_evidence()
        return super().close()
    def connect(self):self.raw_camera.connect()
    def reground(self,cycle):
        # Retain one real current RGB for Root; LiveVision initializes that same
        # frame, then truly captures/tracks a new frame. Physics never pauses.
        if self.evidence_root is None:raise SafetyError('fresh cycle Grounding output absent')
        root=Path(self.evidence_root).parent/'GROUNDING'/cycle;root.mkdir(parents=True,exist_ok=True)
        self.config=copy.deepcopy(self.config)
        self.config.update(role_boxes={},prompt_frame_sha256=None,prompt_frame_output=str(root/'INITIAL_RGB.png'),prompt_reply_path=str(root/'INITIAL_BOXES.json'))
        self.tracker=None
        for retry in range(3):
            try:return self.capture()
            except SafetyError as exc:
                # New Grounding initializes a new tracker; stale first tracking
                # output is discarded, next capture really grabs a new image.
                if str(exc)!='RGBD/SAM processing too stale' or retry==2:raise



def static_cycle_cues(stage,initial,current,refs,initial_reference,handover):
    # Same original frozen static feature semantics; never live motion error.
    source=current.get(stage['object']);source_ref=refs.get(stage['object'])
    if stage['object'] not in stage.get('_already_manipulated',set()):
        source=initial.get(stage['object'],source);source_ref=initial_reference.get(stage['object'],source_ref)
    target=current.get(stage['target']);target_ref=refs.get(stage['target'])
    if stage['target']=='box':target=initial.get('box',target);target_ref=initial_reference.get('box',target_ref)
    if stage['target']=='table_handover_region':target=handover;target_ref=handover
    if any(v is None for v in (source,source_ref,target,target_ref)):raise SafetyError('white static cycle cue unknown')
    return np.asarray(source)-source_ref,np.asarray(target)-target_ref


class HandleFullExecutor(CommonExecutor):
    def capture(self):return bounded_rgbd_capture(self,lambda:super(HandleFullExecutor,self).capture())
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.handle_goals={};self.handle_routes=[];self._jaw_history=[]
        library=json.loads(Path(self.cfg['robot_pad_geometry_file']).read_text())
        if library.get('schema')!='graph-self-adapter.robot-pad-geometry.v1' or library.get('robot_only_no_object_geometry') is not True:raise SafetyError('robot-only pad geometry contract absent')
        self.pad_library=library
    def _white_route(self,stage):return stage['object']=='white_mug' and stage['cycle_id'] in ('c03','c04') and stage['phase'] in ('Approach','Grasp','Lift')
    def _reached(self,goal,pose):
        if self.stage and self._white_route(self.stage):
            error=self._error(goal,pose);tol=self.handle_goals.get(self.stage['cycle_id'],{}).get('position_tolerance_m',.001)
            return bool(np.linalg.norm(error[:3])<=tol and np.linalg.norm(error[3:])<=.02)
        return super()._reached(goal,pose)
    def initialize_white_cycle(self,stage,samples,obs):
        cycle=stage['cycle_id']
        if cycle=='c04':
            self.vision.reground(cycle);samples,obs=self.capture()
        self._cycle=cycle;self._phase=stage['stage_id'];self._phase_elapsed=0.;self._last_cartesian_goal=None;self._jaw_history=[]
        self.attachment=None;self._empty_hand_anchor=None;self._release_pose_anchor=None;self._coarse_source_finalized=False
        initial_ref=self.static_reference_points or self.b['frames'][0]['surface_centroids_world_m']
        ref=self.nearest_ref(stage['reference']['start_time_sec'])['surface_centroids_world_m']
        handover=self.b['refs']['handover_anchors'].get(cycle,{}).get('centroid_world_m')
        cues_stage={**stage,'_already_manipulated':self._manipulated}
        self.coarse_source,self.coarse_target=static_cycle_cues(cues_stage,self.initial_roles,obs.roles,ref,initial_ref,handover)
        self.static_source_env=self.coarse_source.copy();self.static_target_env=self.coarse_target.copy()
        self.code_static_source_handled=self.coarse_source.copy();self.code_static_target_handled=self.coarse_target.copy()
        self.source_prior='cycle static SAM/reference cue; independent functional handle Code goal'
        self.target_prior='scene-initial static box or fixed declared handover'
        self.source_anchor=np.asarray(obs.roles['white_mug']).copy();self.target_anchor=np.asarray(handover if stage['target']=='table_handover_region' else self.initial_roles['box']).copy()
        relation=handle_plane_relation(self.vision.cloud('white_handle'),self.vision.T[:3,3],self.vision.cloud('white_mug'))
        if relation['radial_extent_m']<.0002:raise SafetyError('observed handle radius unresolved')
        pads=copy.deepcopy(self.pad_library['arms'][stage['arm']])
        for key,value in list(pads.items()):
            if isinstance(value,list) and key!='overlap_candidates':pads[key]=np.asarray(value)
        # Tool-frame rebind is robot-only: real Tool1 is197mm, original sim TCP196.4mm.
        offset=float(self.cfg.get('robot_TCP_offset_m',self.pad_library['cad_TCP_offset_m']))-self.pad_library['cad_TCP_offset_m']
        for key in ('pad_midpoint_TCP_local_m','inner_pad_points_TCP_local_m','pad_vertices_TCP_local_m'):
            pads[key]=pads[key].copy();pads[key][...,2]-=offset
        for patch in pads['overlap_candidates']:
            for key in ('pad_midpoint_TCP_local_m','inner_pad_points_TCP_local_m'):
                points=np.asarray(patch[key]).copy();points[...,2]-=offset;patch[key]=points.tolist()
        table_z=float(np.median(self.vision.cloud('table_patch')[:,2]))
        pads,goal=choose_clear_pad_overlap(relation,pads,table_z)
        pre=(goal[0]-relation['approach_down']*relation['span_m'],goal[1])
        envelope=lock_grasp_envelope(self.vision,relation['center']);envelope['robot_pad_midpoint_TCP_local_m']=pads['pad_midpoint_TCP_local_m'].tolist()
        self.handle_goals[cycle]={'envelope_prior':envelope,'lift_goal':None,'goal':goal,'pre':pre,'pre_reached':False,'position_tolerance_m':min(.001,relation['radial_extent_m']/4.),'closed_at':None,'observed_radial_extent_m':relation['radial_extent_m']}
        record={'cycle':cycle,'arm':stage['arm'],'grounding':'fresh current RGB+SAM reinitialization' if cycle=='c04' else 'declared scene-initial SAM part association; current tracked cloud',
            'goal_TCP_m':goal[0].tolist(),'R_goal':goal[1].tolist(),'position_tolerance_m':self.handle_goals[cycle]['position_tolerance_m'],
            'static_source_cue_m':self.static_source_env.tolist(),'static_target_cue_m':self.static_target_env.tolist(),'NN_residual_cue_m':[0.]*6,
            'robot_tool_frame_offset_correction_m':offset,'standard_closed_gripper_command':0.,'not_neural_handle_learning':True}
        self.handle_routes.append(record);write(self.output/'HANDLE_ROUTE_PLANS.json',{'plans':self.handle_routes})
        return samples,obs
    def command(self,stage,elapsed,samples,obs,dt):
        if not self._white_route(stage):return super().command(stage,elapsed,samples,obs,dt)
        cycle=stage['cycle_id'];index=0 if stage['arm']=='left' else 1
        if cycle not in self.handle_goals:
            if stage['phase']!='Approach':raise SafetyError('white functional entry missing')
            samples,obs=self.initialize_white_cycle(stage,samples,obs)
        plan=self.handle_goals[cycle];q_actual=np.stack([s.q for s in samples]);pose=self._shared_pose(q_actual,self.command_g,stage['arm'])
        g=self.command_g.copy();progress=0.
        if stage['phase']=='Approach':
            g[index]=max(g[index],self.b['initial_g'][index])
            if self._reached(plan['pre'],pose):plan['pre_reached']=True
            goal=plan['goal'] if plan['pre_reached'] else plan['pre']
            progress=1. if plan['pre_reached'] and self._reached(goal,pose) else 0.
        elif stage['phase']=='Lift':
            g[index]=0.
            if plan['lift_goal'] is None:
                if self.attachment is None:raise SafetyError('actual successful handle attachment missing before white Lift')
                demo_end=self._shared_pose(np.asarray(self.b['qcurve'](stage['reference']['end_time_sec'])),g,stage['arm'])
                rotation=demo_end[1]@self.attachment['R_demo_end'].T@self.attachment['R_actual_grasp']
                pad=np.asarray(plan['envelope_prior']['robot_pad_midpoint_TCP_local_m'])
                grasp_point=pose[0]+pose[1]@pad
                anchored=(grasp_point-rotation@pad,rotation)
                plan['lift_goal'],measurement=lift_from_locked_envelope(anchored,plan['envelope_prior'])
                measurement['orientation_contract']='declared demonstrated relative Lift-END rotation from actual successful Grasp; grasp-pad XY anchored'
                self.handle_routes.append({'cycle':cycle,'stage':stage['stage_id'],'clearance_goal':measurement,'same_qualified_local_envelope_contract':True})
                write(self.output/'HANDLE_ROUTE_PLANS.json',{'plans':self.handle_routes})
            goal=plan['lift_goal'];progress=1. if self._reached(goal,pose) else 0.
        else:
            goal=plan['goal'];g[index]=0.
            if plan['closed_at'] is None:plan['closed_at']=self.clock()
            minimum_profile=stage['reference']['end_time_sec']-stage['reference']['start_time_sec']
            # Own pose plus declared shortest profile; Root judges grasp at supported boundary.
            # A partial observed tube radius is not an actual jaw-width limit.
            progress=1. if self.clock()-plan['closed_at']>=minimum_profile and self._reached(goal,pose) else 0.
        self._last_cartesian_goal=goal;self._goal_reached=self._reached(goal,pose)
        target,_,error=self._step_ik(q_actual,g,stage['arm'],goal)
        fresh,frame=self.capture();self.safety.validate(target,g,fresh,dt,self.robot,self.T)
        final=self.robot.samples();self.safety.samples(final)
        if self.clock()-frame.stamp>self.cfg['safety']['max_age_sec']:raise SafetyError('white fresh observation expired before dispatch')
        if np.max(abs(target-np.stack([s.q for s in final])))>.01+1e-9:raise SafetyError('white actual pose changed during bounded IK')
        self._hardware_attempts+=1;flags=[index==0,index==1]
        intent={'attempt_count':self._hardware_attempts,'acknowledged_count':self._hardware_acknowledged,'stage_id':stage['stage_id'],'q_target_rad':target.tolist(),'gripper_command':g.tolist(),'flags':flags,'partial_motion_possible_without_ack':True}
        write(self.output/'COMMAND_INTENT.json',intent);self.robot.command(target,g,flags);self._hardware_acknowledged+=1
        self.safety.last_target=target.copy();self.command_g=g.copy();self.sleep(dt);self.command_index+=1
        return target,progress,{'controller':'functional_handle_flat_pad_envelope_Code','goal_TCP_shared_m':goal[0].tolist(),'goal_rotation_shared':goal[1].tolist(),
            'Cartesian_error_before_command':error.tolist(),'actor_only_flags':flags,'planned_joint_step_limit_rad':.009,'dispatch_joint_step_limit_rad':.01,
            'gripper_measurement_available':samples[index].aperture is not None,'Grasp_completion_gate':'actual pose + minimum declared profile then supported-boundary Root; no partial-radius or sampling-window gate','completion_requires_actual_pose_and_Root_holding':True,'NN_action_used':False}
    def predicates(self,stage,progress,target,samples,obs):
        facts=super().predicates(stage,progress,target,samples,obs)
        if self._white_route(stage) and stage['phase']=='Approach':
            for row in stage['completion_predicates']:
                if row['predicate']=='reference_progress':facts[row['id']]=bool(progress>=1.-1e-9 and self.handle_goals[stage['cycle_id']]['pre_reached'] and self._goal_reached)
        return facts
    def run(self):
        result=super().run();result.update(execution_variant=VARIANT,white_handle_Code_route=self.handle_routes,rgbd_discard_log=getattr(self,'rgbd_discard_log',[]),
            original_nodes_order_preserved=True,white_Grasp_is_Code_not_NN=True,policy_claim='original reference+frozen residual on unchanged nodes; explicit white handle Code primitive')
        write(self.output/'RESULT.json',result);return result


# Thin entry imports this symbol for both simulator and real robot I/O.
SharedCupExecutor=HandleFullExecutor
SharedSAMVision=HandleFullVision
