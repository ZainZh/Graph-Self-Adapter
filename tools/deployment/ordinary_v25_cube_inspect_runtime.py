"""Bounded own-CAD camera-gap inspection before current colour/Lift verification."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from ordinary_v25_cube_stationary_judge_runtime import StationaryJudgeExecutor
from ordinary_v25_cube_pad_runtime import CubePadVision
from ordinary_v25_cube_colour_observer import prototype,observe
from all_code_stage_trajectory import plan_stage,stream_stage
class ColourVision(CubePadVision):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.prototypes={};self.context_provider=lambda:None;self.colour_records=[]
    def capture(self):
        obs=super().capture()
        if not self.prototypes:self.prototypes={role:prototype(obs.rgb,self.public_masks[role]) for role in self.public_masks if role!='bowl'}
        context=self.context_provider()
        if not context or not context['stage'] or context['stage']['phase'] not in ('InspectHold','Lift','Move','Align'):return obs
        stage=context['stage'];role=stage['object'];i=0 if stage['arm']=='left' else 1;sample=context['samples'][i];tcp=sample.tcp;R=sample.rotation
        if i:tcp=(context['T']@np.r_[tcp,1])[:3];R=context['T'][:3,:3]@R
        stamp,rgb,depth,K=self.camera.latest_public_capture;geometry=context['geometry']
        if geometry is None:return obs
        height=geometry['cube_measurements'][role]['observed_height_m'];pad=context['pad'][stage['arm']]['robot_pad_CAD'];anchor=None
        if context['attachment'] is not None:anchor=tcp+R@context['attachment']['a_actual']
        point,record,mask=observe(rgb,depth,K,self.T,self.prototypes[role],tcp,R,pad,height,geometry,anchor=anchor)
        record.update(role=role,stage_id=stage['stage_id'],actual_frame_stamp=stamp,raw_SAM_mask_pixels=int(np.asarray(self.public_masks[role],bool).sum()))
        self.colour_records.append(record);write(self.evidence_root/'COLOUR_OBSERVER_RECORDS.json',self.colour_records)
        if point is not None:
            obs.roles=dict(obs.roles);obs.roles[role]=point;obs.source='actual_initial_SAM+public_RGBD_colour_in_actor_roi';np.savez_compressed(self.evidence_root/('COLOUR_MASK_'+str(self.frame_count)+'.npz'),mask=mask,stamp=np.asarray(stamp));record['selected_role_geometry_source']='public_RGBD_colour_in_actor_roi'
        else:record['selected_role_geometry_source']='actual_SAM_if_available; colour_observer_UNKNOWN'
        return obs
class InspectExecutor(StationaryJudgeExecutor):
    def __init__(self,bundle,robot,*args,**kwargs):
        super().__init__(bundle,robot,*args,**kwargs);self.pad=json.loads(Path(self.cfg['cube_robot_pad_prior']).read_text())['by_arm'];self.vision.context_provider=lambda:{'stage':self.stage,'samples':self.robot.samples(),'T':self.T,'geometry':getattr(self,'geometry',None),'pad':self.pad,'attachment':self.attachment};self.inspection_reports=[]
    def capture(self):
        samples,obs=super().capture()
        if self.provisional and not self.lift_verified and self.stage and self.stage['phase']=='InspectHold':self.current_holding=True;self.holding_source='provisional bounded information-gathering only; not verified holding'
        return samples,obs
    def inspect(self,stage,samples,obs):
        actual=np.stack([s.q for s in samples]);p,R=self._actual_pose(samples,stage['arm']);axis=R@np.asarray(self.pad[stage['arm']]['robot_pad_CAD']['closing_axis_TCP_local']);c=axis[:2]/np.linalg.norm(axis[:2]);view=self.vision.T[:3,3]-p;view=view[:2]/np.linalg.norm(view[:2]);wanted=np.array([-view[1],view[0]]);angles=[float(np.arctan2(c[0]*w[1]-c[1]*w[0],c@w)) for w in (wanted,-wanted)];angles.sort(key=abs);g=self.command_g.copy();g[self.accepted_grasp['index']]=self.accepted_grasp['g_command'];trajectory=None;failure=[]
        for angle in angles:
            newR=Rotation.from_rotvec([0,0,angle]).as_matrix()@R
            try:trajectory=plan_stage(self.robot.fk,actual,[(p,R),(p,newR)],g,stage['arm'],self.trajectory_limits,T_left_from_right=self.T,clock=self.clock);break
            except SafetyError as error:failure.append(str(error))
        if trajectory is None:raise SafetyError('camera-gap yaw has no reachable bounded own-CAD IK pose')
        inspect_stage=dict(stage);inspect_stage.update(stage_id=stage['cycle_id']+'.inspect_hold',phase='InspectHold',skill='own_CAD_camera_gap_yaw',object_part='current_cube_visible_body',done=['own_pose'],invariants=['finite'],holding_contract='provisional_for_bounded_active_observation')
        original=self.stage;self.stage=inspect_stage;write(self.output/'NODE_STATUS.json',{'task':self.b['task'],'stage_id':inspect_stage['stage_id'],'phase':'InspectHold','purpose':'camera LOS perpendicular to closing axis; no private object pose'});owner=self
        class Dispatch:
            def samples(self):owner.guard_limits();rows=owner.robot.samples();owner.safety.samples(rows);return rows
            def command(self,q,g,flags):
                owner.guard_limits();owner._hardware_attempts+=1;owner.robot.command(q,g,flags);owner._hardware_acknowledged+=1;owner.command_index+=1;owner.command_g=g.copy();owner.safety.last_target=q.copy()
        def stop():
            for hook in self.stops:hook.stop()
        self.lift_active=True;self.lift_active_started=self.clock()
        try:report=stream_stage(Dispatch(),trajectory,self.trajectory_limits,stop,clock=self.clock,sleep=self.sleep,maximum_stall_sec=self.maximum_stall_sec)
        finally:self.lift_active=False;self.stage=original
        report.update(stage_id=inspect_stage['stage_id'],camera_gap_yaw_rad=angle,actual_TCP_position_preserved=True,closed_g_preserved=True,source='own robot CAD closing axis + actual camera calibration LOS, shortest reachable equivalent gap direction',rejected_IK_goals=failure);self.inspection_reports.append(report);self.stage_trace.append({'stage_id':inspect_stage['stage_id'],'completed':True,'commands':report['command_attempts'],'information_gathering_only':True});write(self.output/'INSPECTION_REPORTS.json',self.inspection_reports)
    def refresh_lift_attachment(self,stage,samples,obs):
        self.inspect(stage,samples,obs);samples,obs=self.capture()
        record=self.vision.colour_records[-1] if self.vision.colour_records else {}
        if record.get('component_pixels',0)<12:raise SafetyError('active observation did not yield12 true coloured body/depth pixels; holding UNKNOWN')
        return super().refresh_lift_attachment(stage,samples,obs)
    def run(self):
        result=super().run();result.update(execution_variant='cube_active_camera_gap_observation_v26',graph_stages=len(self.b['graph']['stages'])+sum(s['phase']=='Lift' for s in self.b['graph']['stages']),mask_source='actual_initial_SAM_plus_public_RGBD_colour_in_actor_roi',colour_observer_records=self.vision.colour_records,inspection_reports=self.inspection_reports);write(self.output/'RESULT.json',result);return result
SharedCupExecutor=InspectExecutor
SharedSAMVision=ColourVision
