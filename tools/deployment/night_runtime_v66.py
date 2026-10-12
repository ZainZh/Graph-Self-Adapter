"""Common skill-phase runtime: supported closure, observable consequence, then confirmed transport."""
import numpy as np,hashlib
from pathlib import Path
from PIL import Image
from prove.xtrainer.runtime import Executor
from prove.xtrainer.agent import write
from prove.xtrainer.io import SafetyError
from multitask_special_runtime import SpecialExecutor
from three_task_v26_public_io import PublicRobot
from all_code_stage_runtime import ExactFKCacheRobot,AllCodeStageExecutor
from all_code_stage_trajectory import Limits,plan_stage,stream_stage
from night_vision_v61 import AutoTaskVision

class NightExecutor(SpecialExecutor):
    def __init__(self,b,robot,*args,**kw):
        super().__init__(b,ExactFKCacheRobot(PublicRobot(robot,source='sim_real_public_contract')),*args,**kw)
        self.roles['SAM_roles']=[r for r in self.roles['SAM_roles'] if r!='table_patch']
        self.judge.zoom_Tcamera=np.asarray(self.vision.T);self.judge.zoom_Tright=self.T
        self.code_reports=[]
        self.tool_confirmations={}
        self.visible_witness=None;self.witness_anchor=None;self.witness_point=None;self.witness_qualified=False
        self.holding_source='unconfirmed';self.knife_bias=None
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

        self.grasp_rotations={}
        self.grasp_centers={};self.grasp_geometry={};self.geometry_cycles=set()
        if self.b['task'] in ('collect_cubes','sorting_cube_bimanual','put_knife_board_new'):
            import json
            self.pad_prior=json.loads(Path(self.cfg['robot_pad_prior']).read_text())
            self.capture();stamp,rgb,depth,K=self.vision.camera.latest_public_capture
            if not 0<=self.clock()-stamp<=self.cfg['safety']['max_age_sec']:raise SafetyError('initial functional geometry stale')
            if self.b['task'] in ('collect_cubes','sorting_cube_bimanual'):
                from public_grasp_geometry_v55 import cube_geometry
                if not self.vision.background_roi:raise SafetyError('public support plane ROI missing')
                self.grasp_centers,self.grasp_geometry=cube_geometry(rgb,depth,K,self.vision.T,self.vision.public_masks,self.vision.background_roi)
            else:
                mask=self.vision.public_masks['red_knife_grasp_part'];valid=np.asarray(mask,bool)&np.isfinite(depth)&(depth>.05)&(rgb[:,:,0]>1.4*rgb[:,:,1])&(rgb[:,:,0]>1.4*rgb[:,:,2]);yy,xx=np.where(valid)
                if len(xx)<25:raise SafetyError('visible red handle geometry sparse')
                zz=depth[yy,xx];cloud=np.c_[(xx-K[0,2])*zz/K[0,0],(yy-K[1,2])*zz/K[1,1],zz]@self.vision.T[:3,:3].T+self.vision.T[:3,3];origin=cloud.mean(0);_,_,V=np.linalg.svd(cloud-origin,full_matrices=False);uv=(cloud-origin)@V[1:].T;A=np.c_[2*uv,np.ones(len(uv))];beta=np.linalg.lstsq(A,(uv*uv).sum(1),rcond=None)[0];rad=float(np.sqrt(max(0,beta[2]+beta[:2]@beta[:2])));res=np.abs(np.linalg.norm(uv-beta[:2],axis=1)-rad);rms=float(np.sqrt((res*res).mean()))
                # A graspable part need not be cylindrical. Use its observed
                # trimmed midpoint and keep the cylinder diagnostic as uncertainty.
                center=(np.quantile(cloud,.05,axis=0)+np.quantile(cloud,.95,axis=0))/2;self.grasp_centers={'red_knife':center};self.grasp_geometry={'visible_part':'red handle, actual SAM + RGBD','center_m':center.tolist(),'radius_m':rad,'RMS_m':rms,'points':len(xx),'backside_unobserved':True}
            write(self.output/'INITIAL_FUNCTIONAL_GEOMETRY.json',self.grasp_geometry)
            if self.b['task'] in ('collect_cubes','sorting_cube_bimanual'):
                from visible_carry_witness_v61 import VisibleCarryWitness
                self.visible_witness=VisibleCarryWitness(rgb,self.vision.public_masks['yellow_cube'])
        if self.b['task']=='put_knife_board_new':
            from public_grasp_geometry_v55 import cloud
            from scipy.spatial.transform import Rotation
            stamp,rgb,depth,K=self.vision.camera.latest_public_capture;mask=self.vision.public_masks['red_knife_grasp_part'];mask=np.asarray(mask,bool)&(rgb[:,:,0]>1.4*rgb[:,:,1])&(rgb[:,:,0]>1.4*rgb[:,:,2]);pts=cloud(depth,K,self.vision.T,mask);center=pts.mean(0);_,singular,V=np.linalg.svd(pts-center,full_matrices=False)
            if singular[0]>1.5*singular[1]:
                exposed=pts[pts[:,2]>=np.quantile(pts[:,2],.6)];self.grasp_centers['red_knife']=np.median(exposed,axis=0);shaft=V[0];gstage=next(x for x in self.b['graph']['stages'] if x['phase']=='Grasp' and x['controller']=='neural_short');t=gstage['reference']['end_time_sec'];g=np.array([np.interp(t,self.b['times'],self.b['gripper'][:,i]) for i in range(2)]);_,R=self._shared_pose(np.asarray(self.b['qcurve'](t)),g,gstage['arm']);axis=R@np.asarray(self.pad_prior['by_arm'][gstage['arm']]['robot_pad_CAD']['closing_axis_TCP_local']);desired=axis-shaft*(axis@shaft)
                if np.linalg.norm(desired)>.05:
                    desired/=np.linalg.norm(desired);change,_=Rotation.align_vectors([desired],[axis]);self.knife_bias=np.eye(3);write(self.output/'VISIBLE_HANDLE_GRASP_AXIS.json',{'source':'actual visible red handle principal axis and exposed upper part','center_m':self.grasp_centers['red_knife'].tolist(),'handle_axis':shaft.tolist(),'closing_axis_target':desired.tolist(),'no_object_truth':True})
    def _reached(self,goal,pose):
        if self.b['task']=='open_drawer_and_place_object' and self.stage and self.stage['phase']=='Grasp':
            e=self._error(goal,pose);return bool(np.linalg.norm(e[:3])<=.013 and np.linalg.norm(e[3:])<=.10)
        return super()._reached(goal,pose)
    def rebase_public_part(self,s):
        if self.b['task']=='put_knife_board_new' and self.knife_bias is not None and s['cycle_id'] not in self.geometry_cycles:
            gstage=next(x for x in self.b['graph']['stages'] if x['phase']=='Grasp' and x['controller']=='neural_short');t=gstage['reference']['end_time_sec'];g=np.array([np.interp(t,self.b['times'],self.b['gripper'][:,i]) for i in range(2)]);p,R=self._shared_pose(np.asarray(self.b['qcurve'](t)),g,s['arm']);pad=np.asarray(self.pad_prior['by_arm'][s['arm']]['robot_pad_CAD']['pad_midpoint_TCP_local_m']);self.coarse_source=self.grasp_centers[s['object']]-(p+R@self.knife_bias@pad);self.geometry_cycles.add(s['cycle_id']);return

        if s['object'] not in self.grasp_centers or s['cycle_id'] in self.geometry_cycles:return
        if self.b['task']=='sorting_cube_bimanual' and s['cycle_id']=='c02':
            from public_grasp_geometry_v55 import cube_geometry
            self.capture();stamp,rgb,depth,K=self.vision.camera.latest_public_capture
            if not 0<=self.clock()-stamp<=self.cfg['safety']['max_age_sec']:raise SafetyError('handover re-grounding stale')
            self.grasp_centers,self.grasp_geometry=cube_geometry(rgb,depth,K,self.vision.T,self.vision.public_masks,self.vision.background_roi)
            write(self.output/'HANDOVER_PUBLIC_GEOMETRY.json',self.grasp_geometry)
        grasp=next(x for x in self.b['graph']['stages'] if x['cycle_id']==s['cycle_id'] and x['phase']=='Grasp' and x['controller']=='neural_short');t=grasp['reference']['end_time_sec'];g=np.array([np.interp(t,self.b['times'],self.b['gripper'][:,i]) for i in range(2)]);p,R=self._shared_pose(np.asarray(self.b['qcurve'](t)),g,s['arm']);prior=self.pad_prior['by_arm'][s['arm']]
        if self.b['task'] in ('collect_cubes','sorting_cube_bimanual'):
            from public_grasp_axes_v58 import top_face_alignment
            stamp,rgb,depth,K=self.vision.camera.latest_public_capture;correction,record=top_face_alignment(depth,K,self.vision.T,self.vision.public_masks[s['object']],self.grasp_geometry,R@np.asarray(prior['robot_pad_CAD']['closing_axis_TCP_local']));self.grasp_rotations[s['cycle_id']]=correction;R=correction@R;write(self.output/(s['cycle_id']+'_OBSERVED_GRASP_AXIS.json'),record)
        pad=np.asarray(prior['robot_pad_CAD']['pad_midpoint_TCP_local_m']);center=self.grasp_centers[s['object']];goal=center-R@pad
        if self.b['task'] in ('collect_cubes','sorting_cube_bimanual'):
            v=np.asarray(prior['robot_pad_CAD']['pad_vertices_TCP_local_m']);n=np.asarray(self.grasp_geometry['support_plane_normal']);o=np.asarray(self.grasp_geometry['support_plane_point_m']);vertices=v@R.T+goal;clear=float(((vertices-o)@n).min());raise_by=max(0.,.003-clear);goal+=raise_by*n
            height=self.grasp_geometry['cube_measurements']['yellow_cube']['observed_height_m'];interval=(v@R.T+goal-o)@n;overlap=min(height,float(interval.max()))-max(0.,float(interval.min()))
            if overlap<.003:raise SafetyError('table-clear gripper pad has insufficient visible object height overlap')
            write(self.output/'FINGER_PLANE_CLEARANCE.json',{'initial_clearance_m':clear,'geometry_derived_normal_adjustment_m':raise_by,'pad_object_height_overlap_m':overlap})
        self.coarse_source=goal-p;self.geometry_cycles.add(s['cycle_id']);write(self.output/(s['cycle_id']+'_FUNCTIONAL_GRASP_PLAN.json'),{'source':'actual public SAM/RGBD geometry + own robot-only pad CAD','center_m':center.tolist(),'coarse_source_m':self.coarse_source.tolist(),'hidden_object_truth':False})
    def capture(self):
        samples,obs=super().capture()
        if self.visible_witness and self.witness_anchor and self.stage and self.stage['phase'] in ('Lift','Rotate','Move','Align'):
            i=0 if self.stage['arm']=='left' else 1;pose=self._shared_pose(np.stack([v.q for v in samples]),self.command_g,self.stage['arm']);stamp,rgb,depth,K=self.vision.camera.latest_public_capture;anchor=self.witness_anchor;center=pose[0]+pose[1]@anchor['local_center'];height=self.grasp_geometry['cube_measurements']['yellow_cube']['observed_height_m'];point,qualified,record=self.visible_witness.measure(stamp,rgb,depth,K,self.vision.T,self.grasp_geometry,center,height,np.linalg.norm(pose[0]-anchor['TCP']),self.stage['cycle_id']);self.witness_point=point;self.witness_qualified=qualified
            if point is not None and (obs.roles.get('yellow_cube') is None or int(np.asarray(self.vision.public_masks.get('yellow_cube',[])).sum())<6):
                obs.roles['yellow_cube']=point;obs.source+='+actual sparse colour RGBD surface witness (not SAM or body pose)'
            cert=self.certificates.get((self.stage['arm'],self.stage['object']))
            if point is not None and cert is not None and self.command_g[i]<=cert['g_command']+.1:
                cert['state']=True;cert['state_source']='fresh observed colour/depth body point maintains previously confirmed control; not inferred pose';self.current_holding=True
        return samples,obs
    def remember_closure(self,s,samples,obs):
        i=0 if s['arm']=='left' else 1;self.pending={'role':s['object'],'arm':s['arm'],'cycle':s['cycle_id'],'g':float(self.command_g[i]),'point':obs.roles.get(s['object'])};self.current_holding=None
        if self.visible_witness:
            pose=self._shared_pose(np.stack([v.q for v in samples]),self.command_g,s['arm']);center=self.grasp_centers[s['object']];self.witness_anchor={'TCP':pose[0].copy(),'local_center':pose[1].T@(center-pose[0])};self.visible_witness.recent.clear()
        self.pre_motion=(obs.stamp,obs.rgb.copy(),[{'q':v.q.tolist(),'qdot':v.qdot.tolist(),'TCP_m':v.tcp.tolist(),'stamp':v.stamp} for v in samples])
        f=self.output/'PRE_LIFT_PUBLIC_RGB.png';Image.fromarray(obs.rgb).save(f);write(self.output/'PRE_LIFT_PUBLIC_OBSERVATION.json',{'RGB':str(f.resolve()),'RGB_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'stamp':obs.stamp,'public_robot':self.pre_motion[2]})
    def confirm_consequence(self,s,samples,obs):
        if s['phase']=='Pull':return # Drawer is fixed: final released/open relation, not carried-object attachment.
        if self.visible_witness and not self.witness_qualified and self.witness_point is not None:
            # A continuous Code node captures only at boundaries. Collect real
            # endpoint frames before applying the existing temporal criterion.
            for _ in range(4):
                self.sleep(.08);samples,obs=self.capture()
                if self.witness_qualified:break
        if self.visible_witness and self.witness_qualified and self.witness_point is not None:
            i=0 if s['arm']=='left' else 1;tcp=samples[i].tcp if i==0 else (self.T@np.r_[samples[i].tcp,1])[:3];cert={'object':s['object'],'cycle':s['cycle_id'],'index':i,'g_command':float(self.command_g[i]),'aperture_m':None,'object_TCP_distance_m':float(np.linalg.norm(self.witness_point-tcp)),'state':True,'source':'fresh temporal colour/RGBD body witness after own nonzero Lift; control permission, not measured contact'};self.certificates[(s['arm'],s['object'])]=cert;self.accepted_grasp=cert;self.current_holding=True;self.holding_source='fresh temporal RGBD colour body witness';self._manipulated.add(s['object']);self.attachment=None;write(self.output/(s['cycle_id']+'_CARRY_SENSOR_PERMISSION.json'),self.visible_witness.records[-1]);return
        if s['phase']!='Lift' and self.pre_motion:
            from dataclasses import replace
            obs=replace(obs,history=[self.pre_motion]+list(obs.history[-2:]))
        after=Executor.semantics(self,s,['holding'],(samples,obs))
        if self.visual_facts.get('holding') is not True:raise SafetyError('post-Lift holding UNKNOWN/false; no carry')
        i=0 if s['arm']=='left' else 1;point=after[1].roles.get(s['object']);tcp=after[0][i].tcp if i==0 else (self.T@np.r_[after[0][i].tcp,1])[:3]
        if point is None:raise SafetyError('current confirmed object role unavailable')
        cert={'object':s['object'],'cycle':s['cycle_id'],'index':i,'g_command':float(self.command_g[i]),'aperture_m':after[0][i].aperture,'object_TCP_distance_m':float(np.linalg.norm(point-tcp)),'state':True,'source':'current API visual motion confirmation'}
        self.tool_confirmations[(s['arm'],s['object'])]=(after[1].stamp,after[1].rgb.copy(),[{'q':v.q.tolist(),'qdot':v.qdot.tolist(),'TCP_m':v.tcp.tolist(),'stamp':v.stamp} for v in after[0]])
        self.certificates[(s['arm'],s['object'])]=cert;self.accepted_grasp=cert;self.current_holding=True;self._manipulated.add(s['object']);self.attachment=None
        if self.b['task']=='hanging_mug':
            from rigid_surface_registration_v62 import register_surface,rigid
            from public_grasp_geometry_v55 import cloud
            data=np.load(self.cfg['known_object_registration_prior']);self.reference_object_attachment=data['reference_attachment'];self.capture();stamp,rgb,depth,K=self.vision.camera.latest_public_capture;mask=self.vision.public_masks['mug'];points=cloud(depth,K,self.vision.T,mask);own=self._shared_pose(np.stack([v.q for v in self.robot.samples()]),self.command_g,s['arm']);pose,record=register_surface(points,data['local_surface'],rigid(*own)@self.reference_object_attachment);self.registered_object_attachment=np.linalg.inv(rigid(*own))@pose
            rack_points=cloud(depth,K,self.vision.T,self.vision.public_masks['rack']);rack_pose,rack_record=register_surface(rack_points,data['rack_surface'],data['rack_reference_pose']);self.target_relative_object_goal=rack_pose@data['object_goal_in_target'];self.object_registration_data=data
            write(self.output/'OBSERVED_OBJECT_REGISTRATION.json',dict(record,source_stamp=stamp,attachment=self.registered_object_attachment.tolist(),source='actual SAM/RGBD registered to declared fixed known CAD; latched estimate, not current simulator body pose'))
        # Visible medians change under occlusion; they certify visibility, not
        # rigid attachment coordinates. Preserve the cycle's initial retargeting.
    def semantics(self,s,facts,before):
        if s.get('hold_at_end') and (s['arm'],s['object']) in self.tool_confirmations:
            from dataclasses import replace
            past=self.tool_confirmations[(s['arm'],s['object'])]
            if past[0]<before[1].stamp:before=(before[0],replace(before[1],history=[past]+list(before[1].history[-2:])))
        from api_stage_judge_v61 import RetryableAPIError
        for attempt in range(3):
            try:return super().semantics(s,facts,before)
            except RetryableAPIError:
                if attempt==2:raise
                self.sleep(.5);before=self.capture()
    def command(self,s,*args,**kw):
        if s.get('inspect_grasp'):
            from scipy.spatial.transform import Rotation
            samples,obs=args[1],args[2];q=np.stack([v.q for v in samples]);p,R=self._shared_pose(q,self.command_g,s['arm']);view=np.asarray(self.vision.T[:3,3])-p;view/=np.linalg.norm(view);axis=R[:,2];z_axis=axis-view*(axis@view);z_axis/=np.linalg.norm(z_axis);y_axis=np.cross(z_axis,view);y_axis/=np.linalg.norm(y_axis)
            turn=np.cross(axis,z_axis);angle=np.arctan2(np.linalg.norm(turn),axis@z_axis);desired_R=Rotation.from_rotvec(turn/max(np.linalg.norm(turn),1e-12)*min(angle,np.pi/6)).as_matrix()@R;offset=self.witness_anchor['local_center'] if self.witness_anchor else np.zeros(3);goal=(p+R@offset-desired_R@offset,desired_R);correction=self.grasp_rotations.pop(s['cycle_id'],None)
            try:return self.stream_code(s,samples,obs,[goal],self.command_g.copy())
            finally:
                if correction is not None:self.grasp_rotations[s['cycle_id']]=correction
        if s['phase']=='Approach':return self.approach_segment(s,*args,**kw)
        if s['phase']=='Wait':
            samples,obs=args[1],args[2];i=0 if s['arm']=='left' else 1;g=self.command_g.copy();g[i]=1.;p,R=self._shared_pose(np.asarray(self.b['qcurve'](s['reference']['end_time_sec'])),g,s['arm']);goal=(p+self.coarse_target,R);correction=self.grasp_rotations.pop(s['cycle_id'],None)
            try:return self.stream_code(s,samples,obs,[goal],g)
            finally:
                if correction is not None:self.grasp_rotations[s['cycle_id']]=correction

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
    def _reference_command(self,s,elapsed,samples,obs,dt):
        if s.get('closure_at_actual_entry'):
            i=0 if s['arm']=='left' else 1
            if self._phase!=s['stage_id']:
                self._phase=s['stage_id'];self.closure_q=np.stack([v.q for v in samples]);self._last_cartesian_goal=self._shared_pose(self.closure_q,self.command_g,s['arm'])
            span=s['reference']['end_time_sec']-s['reference']['start_time_sec'];u=min(1.,elapsed*self.cfg['hardware_reference_speed_scale']/max(span,1e-9));t=s['reference']['start_time_sec']+u*span;g=self.command_g.copy();g[i]=np.interp(t,self.b['times'],self.b['gripper'][:,i]);q=self.closure_q.copy();q[1-i]=samples[1-i].q
            self.safety.validate(q,g,samples,dt,self.robot,self.T);self.robot.command(q,g,[i==0,i==1]);self._hardware_attempts+=1;self._hardware_acknowledged+=1;self.command_index+=1;self.command_g=g;self.safety.last_target=q.copy();self.sleep(dt)
            return q,u,{'controller':'close at actual feasible contact-entry own pose; no push into cabinet','reference_time_sec':t}
        return self.base_reference_command(s,elapsed,samples,obs,dt)
    def base_reference_command(self,s,elapsed,samples,obs,dt):
        if s['phase']=='Push' and self.b['graph']['logical_task']=='sweep_with_stationary_pan':
            start=s['reference']['start_time_sec'];span=s['reference']['end_time_sec']-start;u=min(1.,getattr(self,'_phase_elapsed',0.)/max(span,1e-9));original=self.coarse_target.copy();pan=self.initial_roles.get('dustpan');reference=self.nearest_ref(s['reference']['end_time_sec'])['surface_centroids_world_m'].get('dustpan')
            if pan is None or reference is None:raise SafetyError('current public dustpan or demonstration recipient reference missing')
            destination=np.asarray(pan)-np.asarray(reference);self.coarse_target=(1-u)*self.coarse_source+u*destination
            try:return super()._reference_command(s,elapsed,samples,obs,dt)
            finally:self.coarse_target=original

        if not s.get('empty_gripper_clearance'):
            target,progress,record=super()._reference_command(s,elapsed,samples,obs,dt)
            if s['phase']=='Pull':record['executed_actual_entry_relative_TCP_m']=self._last_cartesian_goal[0].tolist()
            return target,progress,record
        if self._empty_hand_anchor is None:raise SafetyError('clearance requires released hand anchor')
        move=next((x for x in self.b['graph']['stages'] if x['cycle_id']==s['cycle_id'] and x['phase']=='Move'),None)
        if move is None:raise SafetyError('clearance requires declared reverse transport corridor')
        times=AllCodeStageExecutor._reference_times(self,move)[::-1];goals=[]
        for t in times:
            u=(t-move['reference']['start_time_sec'])/(move['reference']['end_time_sec']-move['reference']['start_time_sec']);p,R=self._shared_pose(np.asarray(self.b['qcurve'](t)),self.command_g,s['arm']);goals.append((p+(1-u)*self.coarse_source+u*self.coarse_target,R))
        return self.stream_code(s,samples,obs,goals,self.command_g.copy())
    def approach_segment(self,s,elapsed,samples,obs,dt):
        AllCodeStageExecutor._ground_red_cycle(self,s,obs);self.rebase_public_part(s)
        actual=np.stack([x.q for x in samples]);i=0 if s['arm']=='left' else 1;g=self.command_g.copy();g[i]=max(g[i],self.b['initial_g'][i])
        times=AllCodeStageExecutor._reference_times(self,s);goals=[]
        for t in times:
            p,R=self._shared_pose(np.asarray(self.b['qcurve'](t)),g,s['arm']);goals.append((p+self.coarse_source,R))
        return self.stream_code(s,samples,obs,goals,g)
    def stream_code(self,s,samples,obs,goals,g):
        if self.knife_bias is not None and s['phase']=='Approach':goals=[(p,R@self.knife_bias) for p,R in goals]
        correction=self.grasp_rotations.get(s['cycle_id'])
        if correction is not None:goals=[(p,correction@R) for p,R in goals]
        actual=np.stack([x.q for x in samples]);i=0 if s['arm']=='left' else 1
        c=self.cfg['trajectory'];safe=self.cfg['safety'];limits=Limits(safe['joint_min_rad'],safe['joint_max_rad'],safe['workspace_min_m'],safe['workspace_max_m'],velocity=c['velocity_rad_sec'],acceleration=c['acceleration_rad_sec2'],step=.009,rate_hz=c['rate_hz'],feedback_age=safe['max_age_sec'],tracking=safe['max_tracking_rad'],position_tolerance=.003,rotation_tolerance=.02,settle_seconds=c['settle_seconds'],maximum_lateness=c['maximum_lateness_sec'],feedback_velocity=safe['max_velocity_rad_sec'],actual_step_limit=safe['max_step_rad'])
        trajectory=plan_stage(self.robot.fk,actual,goals,g,s['arm'],limits,T_left_from_right=self.T,clock=self.clock)
        fresh,frame=self.capture()
        if not 0<=self.clock()-frame.stamp<=safe['max_age_sec'] or np.max(abs(np.stack([x.q for x in fresh])-actual))>limits.start_drift:raise SafetyError('planned approach start changed/stale')
        owner=self
        class Dispatch:
            def samples(self):owner.guard_limits();v=owner.robot.samples();owner.safety.samples(v);return v
            def command(self,q,g,flags):
                owner.guard_limits();owner._hardware_attempts+=1;write(owner.output/'COMMAND_INTENT.json',{'stage_id':s['stage_id'],'attempt_count':owner._hardware_attempts,'q_target_rad':q.tolist(),'gripper_command':g.tolist(),'flags':flags});owner.robot.command(q,g,flags);owner._hardware_acknowledged+=1;owner.command_index+=1;owner.command_g=g.copy();owner.safety.last_target=q.copy()
        def stop():
            for hook in self.stops:hook.stop()
        report=stream_stage(Dispatch(),trajectory,limits,stop,clock=self.clock,sleep=self.sleep,maximum_stall_sec=c['maximum_stall_sec']);self.code_reports.append({'stage_id':s['stage_id'],**report});write(self.output/'CODE_SEGMENTS.json',self.code_reports)
        self._phase=s['stage_id'];self._last_cartesian_goal=goals[-1];self._phase_elapsed=s['reference']['end_time_sec']-s['reference']['start_time_sec']
        return trajectory.q[-1],1.,{'controller':'complete demonstrated object-relative Approach path planned once','goal_TCP_shared_m':goals[-1][0].tolist(),'segment':report}
    def _step_ik(self,actual_q,g,arm,goal):
        if self.knife_bias is not None and self.stage['phase'] in ('Grasp','Lift','Move','Align'):
            goal=(goal[0],goal[1]@self.knife_bias);self._last_cartesian_goal=goal
        if getattr(self,'registered_object_attachment',None) is not None and self.stage['phase'] in ('Move','Insert'):
            from rigid_surface_registration_v62 import rigid
            if self.stage['phase']=='Insert':
                if self.command_index%10==0 or not getattr(self,'insert_registration_started',False):
                    from public_grasp_geometry_v55 import cloud
                    from rigid_surface_registration_v62 import register_surface
                    stamp,rgb,depth,K=self.vision.camera.latest_public_capture;current=self._shared_pose(actual_q,g,arm);points=cloud(depth,K,self.vision.T,self.vision.public_masks['mug']);pose,record=register_surface(points,self.object_registration_data['local_surface'],rigid(*current)@self.registered_object_attachment);self.registered_object_attachment=np.linalg.inv(rigid(*current))@pose;self.insert_registration_started=True
                native_end=self._shared_pose(np.asarray(self.b['qcurve'](self.stage['reference']['end_time_sec'])),g,arm);feedforward=np.linalg.inv(rigid(*native_end))@rigid(*goal);desired=self.target_relative_object_goal@np.linalg.inv(self.registered_object_attachment)@feedforward
            else:desired=rigid(*goal)@self.reference_object_attachment@np.linalg.inv(self.registered_object_attachment)
            goal=(desired[:3,3],desired[:3,:3]);self._last_cartesian_goal=goal

        correction=self.grasp_rotations.get(self.stage['cycle_id']) if self.stage else None
        if correction is not None and self.stage['phase'] in ('Grasp','Lift','Move','Align'):
            goal=(goal[0],correction@goal[1]);self._last_cartesian_goal=goal
        if self.stage and self.stage['phase']=='Pull':
            if not hasattr(self,'pull_actual_pose'):
                self.pull_actual_pose=self._shared_pose(actual_q,g,arm);self.pull_reference_pose=self._shared_pose(np.asarray(self.b['qcurve'](self.stage['reference']['start_time_sec'])),g,arm)
            p0,R0=self.pull_reference_pose;p1,R1=self.pull_actual_pose;goal=(p1+R1@R0.T@(goal[0]-self.coarse_target-p0),R1@R0.T@goal[1]);self._last_cartesian_goal=goal
        return super()._step_ik(actual_q,g,arm,goal)
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
                if (s['phase']=='Lift' and not s.get('defer_holding_confirmation')) or s.get('inspect_grasp') or (s.get('diagnostic_consequence_only') and s['phase']!='Lift'):self.confirm_consequence(s,*self.capture())
            complete=True
        except Exception as e:
            failure=str(e)
            for stop in self.stops:
                try:stop.stop();stopped.append({'stopped':True})
                except Exception as e:stopped.append({'error':str(e)})
        r={'schema':'graph-self-adapter.night-task-run.v1','task':self.b['task'],'logical_task':self.b['graph']['logical_task'],'graph_completed':complete,'failure':failure,'graph_stages':len(self.b['graph']['stages']),'stage_trace':self.stage_trace,'trace':self.trace,'judge_records':self.judge.records,'fixed_z':True,'latent_updates':0,'command_attempts':self._hardware_attempts,'command_acknowledgments':self._hardware_acknowledged,'stop_results':stopped,'hardware_qualified':False}
        if self.visible_witness:write(self.output/'VISIBLE_CARRY_WITNESSES.json',self.visible_witness.records)
        write(self.output/'RESULT.json',r);return r
SharedCupExecutor=NightExecutor
SharedSAMVision=AutoTaskVision
