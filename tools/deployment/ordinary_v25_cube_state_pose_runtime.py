"""Visual holding state separated from measured pose and explicit uncertain inferred control."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from prove.xtrainer.runtime import Executor
from ordinary_v25_cube_stationary_judge_runtime import StationaryJudgeExecutor
from ordinary_v25_cube_pad_runtime import CubePadVision
from ordinary_v25_cube_colour_observer import prototype
class StatePoseVision(CubePadVision):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.prototypes={};self.context_provider=lambda:None;self.witness_records=[];self.latest_witness=None
    def capture(self):
        obs=super().capture()
        if not self.prototypes:self.prototypes={k:prototype(obs.rgb,v) for k,v in self.public_masks.items() if k!='bowl'}
        context=self.context_provider()
        if not context or not context['stage'] or context['geometry'] is None or context['stage']['phase'] not in ('Lift','Move','Align'):return obs
        stage=context['stage'];role=stage['object'];i=0 if stage['arm']=='left' else 1;s=context['samples'][i];p=s.tcp;R=s.rotation
        if i:p=(context['T']@np.r_[p,1])[:3];R=context['T'][:3,:3]@R
        stamp,rgb,depth,K=self.camera.latest_public_capture;arr=rgb.astype(float);total=arr.sum(2);colour=(total>30)&(np.linalg.norm(arr/np.maximum(total[:,:,None],1)-self.prototypes[role],axis=2)<.12)&np.isfinite(depth)&(depth>.05)&(depth<3);y,x=np.where(colour);z=depth[y,x];points=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]@self.T[:3,:3].T+self.T[:3,3]
        height=context['geometry']['cube_measurements'][role]['observed_height_m'];pad=context['pad'][stage['arm']]['robot_pad_CAD'];v=np.asarray(pad['pad_vertices_TCP_local_m']);axis=np.asarray(pad['closing_axis_TCP_local']);verts=np.concatenate([v,v+.043*axis,v-.043*axis])@R.T+p;lo=verts.min(0)-height;hi=verts.max(0)+height;inside=np.all((points>=lo)&(points<=hi),axis=1);n=np.asarray(context['geometry']['support_plane_normal_left_base']);base=np.asarray(context['geometry']['support_plane_point_left_base_m']);floor=base[2]-((points[:,:2]-base[:2])@n[:2])/n[2];inside&=points[:,2]-floor>=height+.010
        mask=np.zeros(depth.shape,bool);mask[y[inside],x[inside]]=True;components,count=label(mask,np.ones((3,3)));measured=None
        for j in range(1,count+1):
            yy,xx=np.where(components==j)
            if len(xx)>=12:
                zz=depth[yy,xx];cloud=np.c_[(xx-K[0,2])*zz/K[0,0],(yy-K[1,2])*zz/K[1,1],zz]@self.T[:3,:3].T+self.T[:3,3];measured=np.median(cloud,axis=0);break
        # Sparse witnesses are real depth pixels, not a measured body pose or fake SAM result.
        obs.roles=dict(obs.roles);obs.roles[role]=measured;obs.source='actual_initial_SAM+public_RGBD_colour_witnesses; measured_pose_None_when_sparse'
        record={'role':role,'stage_id':stage['stage_id'],'stamp':stamp,'mask_source':'public_RGBD_colour_in_actor_roi','sparse_elevated_witness_count':int(inside.sum()),'witness_pixels_xy':np.c_[x[inside],y[inside]].tolist(),'witness_depth_points_m':points[inside].tolist(),'measured_body_point_m':None if measured is None else measured.tolist(),'measured_pose_minimum_points':12,'sparse_witnesses_are_not_measured_pose':True,'raw_SAM_pixels':int(self.public_masks[role].sum())};self.latest_witness=record;self.witness_records.append(record);write(self.evidence_root/'STATE_POSE_WITNESSES.json',self.witness_records)
        return obs
class StatePoseExecutor(StationaryJudgeExecutor):
    def __init__(self,bundle,robot,*args,**kwargs):
        super().__init__(bundle,robot,*args,**kwargs);self.pad=json.loads(Path(self.cfg['cube_robot_pad_prior']).read_text())['by_arm'];self.vision.context_provider=lambda:{'stage':self.stage,'samples':self.robot.samples(),'T':self.T,'geometry':getattr(self,'geometry',None),'pad':self.pad};self.inferred_records=[]
    def start(self):
        super().start();stamp,rgb,depth,K=self.vision.camera.latest_public_capture;mask=self.vision.public_masks['bowl'];y,x=np.where(mask&np.isfinite(depth)&(depth>0));z=depth[y,x];points=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]@self.vision.T[:3,:3].T+self.vision.T[:3,3];n=np.asarray(self.geometry['support_plane_normal_left_base']);base=np.asarray(self.geometry['support_plane_point_left_base_m']);h=(points-base)@n;rim=points[h>=np.quantile(h,.9)-.004];A=np.c_[2*rim[:,:2],np.ones(len(rim))];beta=np.linalg.lstsq(A,(rim[:,:2]**2).sum(1),rcond=None)[0];center=beta[:2];rad=np.linalg.norm(rim[:,:2]-center,axis=1);self.bowl_region={'center_xy_m':center.tolist(),'inner_radius_m':float(np.quantile(rad,.05)-.002),'source':'public initial actual SAM/RGBD visible rim conservative circle'};write(self.output/'PUBLIC_TARGET_SAFE_REGION.json',self.bowl_region)
    def capture(self):
        samples,obs=super().capture()
        if self.lift_verified and self.stage and self.stage['phase'] in ('Lift','Move','Align') and self.accepted_grasp:
            w=self.vision.latest_witness or {};i=self.accepted_grasp['index']
            if w.get('role')==self.stage['object'] and w.get('sparse_elevated_witness_count',0)>0 and self.command_g[i]<=self.accepted_grasp['g_command']+.1:
                self.current_holding=True;self.holding_source='prior genuine visual holding + current actual elevated colour witness; measured pose may be None'
        return samples,obs
    def refresh_lift_attachment(self,stage,samples,obs):
        if self.pre_lift_frame:obs.history=[self.pre_lift_frame]+list(obs.history[-2:])
        witness=self.vision.latest_witness or {};stage=dict(stage);stage['state_pose_channels']={'holding_query':'actual causal RGB/support disappearance/own nonzero TCP motion and unique elevated colour witnesses; unclear remains UNKNOWN','measured_pose':None if obs.roles.get(stage['object']) is None else 'actual current12point component','sparse_public_witnesses':witness}
        after=Executor.semantics(self,stage,['holding'],(samples,obs))
        if self.visual_facts.get('holding') is not True:raise SafetyError('current visual holding UNKNOWN/false; no inferred control')
        role=stage['object'];row=self.geometry['cube_measurements'][role];height=row['observed_height_m'];center=np.asarray(row['observed_center_m']).copy();center[2]=row['local_observed_table_z_m']+height/2;anchor=self.accepted_grasp_pose[stage['cycle_id']];grasp=next(s for s in self.b['graph']['stages'] if s['cycle_id']==stage['cycle_id'] and s['phase']=='Grasp');demo_grasp=self._shared_pose(np.asarray(self.b['qcurve'](grasp['reference']['end_time_sec'])),self.command_g,stage['arm']);reference_center=center-self.static_source_env;local_actual=anchor[1].T@(center-anchor[0]);local_ref=demo_grasp[1].T@(reference_center-demo_grasp[0]);actual=self._actual_pose(after[0],stage['arm']);demo_lift=self._shared_pose(np.asarray(self.b['qcurve'](stage['reference']['end_time_sec'])),self.command_g,stage['arm'])
        self.attachment={'role':role,'a_actual':local_actual,'a_ref':local_ref,'R_demo_end':demo_lift[1].copy(),'R_actual_grasp':actual[1].copy(),'source':'inferred_from_public_grasp_entry_body_geometry_and_actual_TCP; not current measurement','uncertainty_xy_m':height/2+.003,'uncertainty_z_m':height/2+.003};self.coarse_source=(actual[0]+actual[1]@local_actual)-(demo_lift[0]+demo_lift[1]@local_ref);self.provisional=False;self.lift_verified=True;self.pre_lift_frame=None;self.current_holding=True
        record={'stage_id':stage['stage_id'],'holding_state_source':self.judge.records[-1]['actual_root_identity'],'measured_body_pose':None if after[1].roles.get(role) is None else 'actual dense component','inferred_attachment_local_m':local_actual.tolist(),'source':self.attachment['source'],'uncertainty_xy_m':self.attachment['uncertainty_xy_m'],'not_adaptation_feedback':True};self.inferred_records.append(record);write(self.output/'INFERRED_CONTROL_ATTACHMENT.json',self.inferred_records)
    def _code_goals_and_gripper(self,stage,pose,obs):
        if stage['phase']=='Move' and self.attachment:
            align=next(s for s in self.b['graph']['stages'] if s['cycle_id']==stage['cycle_id'] and s['phase']=='Align');native=self._shared_pose(np.asarray(self.b['qcurve'](align['reference']['end_time_sec'])),self.command_g,stage['arm']);_,object_goal=self.attached_goal(native,self.coarse_target);height=self.geometry['cube_measurements'][stage['object']]['observed_height_m'];radius=height/np.sqrt(2)+self.attachment['uncertainty_xy_m']+.003;clearance=self.bowl_region['inner_radius_m']-np.linalg.norm(object_goal[:2]-np.asarray(self.bowl_region['center_xy_m']))-radius;write(self.output/(stage['cycle_id']+'_INFERRED_TARGET_CONTAINMENT.json'),{'object_goal_xy_m':object_goal[:2].tolist(),'object_envelope_radius_plus_uncertainty_m':radius,'bowl_region':self.bowl_region,'containment_margin_m':float(clearance),'no_fake_measured_pose':True})
            if clearance<0:raise SafetyError('target bowl does not safely contain inferred object envelope; stop/observe required')
        return super()._code_goals_and_gripper(stage,pose,obs)
    def run(self):
        result=super().run();result.update(execution_variant='cube_visual_state_measured_pose_separation_v26',mask_source='actual_initial_SAM+public_RGBD_colour_witnesses+separate_inferred_control_attachment',state_pose_witnesses=self.vision.witness_records,inferred_attachment_records=self.inferred_records);write(self.output/'RESULT.json',result);return result
SharedCupExecutor=StatePoseExecutor
SharedSAMVision=StatePoseVision
