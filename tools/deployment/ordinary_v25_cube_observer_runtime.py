"""Bounded provisional Grasp→Lift observer unit; all five cubes retained."""
import numpy as np
from pathlib import Path
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from prove.xtrainer.runtime import Executor
from object_centered_runtime import ObjectCenteredExecutor
from ordinary_v25_repair_runtime import RepairExecutor,EvidenceVision
from ordinary_v25_runtime import OrdinaryV25Executor
from all_code_stage_runtime import ExactFKCacheRobot
from stage_trajectory import Limits
from multitask_ordinary_route import role_contract
from ordinary_v25_cube_observer_route import SCHEMA
from three_task_v26_public_io import PublicRobot
class ObserverExecutor(RepairExecutor):
    def __init__(self,bundle,robot,*args,**kwargs):
        if bundle['graph']['schema']!=SCHEMA:raise SafetyError('explicit V26 observer graph required')
        ObjectCenteredExecutor.__init__(self,bundle,ExactFKCacheRobot(PublicRobot(robot,source='sim_real_public_contract')),*args,**kwargs)
        self.accepted_grasp=None;self._registered_cycles=set();self.semantic_log=[];self.roles=role_contract(bundle['graph'],bundle['refs']);self.roles['SAM_roles']=[r for r in self.roles['SAM_roles'] if r!='table_patch']
        self.accepted_grasp_pose={};self.lift_progress={};self.relative_lift_records=[];self.all_code_stage_reports=[];self.stage_timing={};self.provisional=False;self.lift_verified=False;self.pre_lift_frame=None
        cfg=self.cfg['trajectory'];s=self.cfg['safety'];self.maximum_stall_sec=cfg['maximum_stall_sec']
        self.trajectory_limits=Limits(s['joint_min_rad'],s['joint_max_rad'],s['workspace_min_m'],s['workspace_max_m'],velocity=cfg['velocity_rad_sec'],acceleration=cfg['acceleration_rad_sec2'],step=.009,rate_hz=cfg['rate_hz'],feedback_age=s['max_age_sec'],tracking=s['max_tracking_rad'],position_tolerance=.003,rotation_tolerance=.02,settle_seconds=cfg['settle_seconds'],maximum_lateness=cfg['maximum_lateness_sec'],feedback_velocity=s['max_velocity_rad_sec'],actual_step_limit=s['max_step_rad'])
    def capture(self):
        samples,obs=super().capture()
        if self.provisional and not self.lift_verified and self.stage and self.stage['phase'] in ('Grasp','Lift'):
            cert=self.accepted_grasp
            if cert and self.command_g[cert['index']]<=cert['g_command']+.1:
                if self.stage['phase']=='Lift' and self.clock()-self.stage_started>60:raise SafetyError('provisional one-Lift budget exhausted')
                self.current_holding=True;self.holding_source='provisional_grasp_permission_for_one_bounded_Lift_not_attachment'
        return samples,obs
    def semantics(self,stage,facts,before):
        if stage['phase']!='Grasp':return super().semantics(stage,facts,before)
        after=Executor.semantics(self,stage,facts,before)
        if self.visual_facts.get('holding') is not True:raise SafetyError('provisional visual Grasp UNKNOWN/false')
        i=0 if stage['arm']=='left' else 1;self.accepted_grasp={'object':stage['object'],'cycle':stage['cycle_id'],'index':i,'g_command':float(self.command_g[i]),'aperture_m':None,'object_TCP_distance_m':.10,'source':'visual provisional one-Lift permission; no fabricated current role point'}
        self.accepted_grasp_pose[stage['cycle_id']]=self._actual_pose(after[0],stage['arm']);self.provisional=True;self.lift_verified=False;self.current_holding=True;self._manipulated.add(stage['object']);self.attachment=None
        self.semantic_log.append({'stage_id':stage['stage_id'],'holding':'provisional_next_Lift_only','current_role_point_available':after[1].roles.get(stage['object']) is not None})
        return after
    def command(self,stage,elapsed,samples,obs,dt):
        if stage['phase']=='Lift' and not self.pre_lift_frame:self.pre_lift_frame=obs.history[-1]
        return super().command(stage,elapsed,samples,obs,dt)
    def refresh_lift_attachment(self,stage,samples,obs):
        anchor=self.accepted_grasp_pose[stage['cycle_id']];pose=self._actual_pose(samples,stage['arm'])
        if np.linalg.norm(pose[0]-anchor[0])<.003 or obs.roles.get(stage['object']) is None:raise SafetyError('fresh Lift own motion/object geometry unknown')
        if self.pre_lift_frame:obs.history=[self.pre_lift_frame]+list(obs.history[-2:])
        after=Executor.semantics(self,stage,['holding'],(samples,obs))
        if self.visual_facts.get('holding') is not True or after[1].roles.get(stage['object']) is None:raise SafetyError('fresh lifted holding UNKNOWN/false')
        point=np.asarray(after[1].roles[stage['object']]);actual=self._actual_pose(after[0],stage['arm']);end=stage['reference']['end_time_sec'];ref=self.nearest_ref(end);refpoint=ref['surface_centroids_world_m'].get(stage['object'])
        if refpoint is None:raise SafetyError('declared Lift-end visible reference missing')
        nominal=self._shared_pose(np.asarray(self.b['qcurve'](end)),self.command_g,stage['arm']);self.attachment={'role':stage['object'],'a_actual':actual[1].T@(point-actual[0]),'a_ref':nominal[1].T@(np.asarray(refpoint)-nominal[0]),'R_demo_end':nominal[1].copy(),'R_actual_grasp':actual[1].copy()};self.coarse_source=point-np.asarray(refpoint);self.accepted_grasp['object_TCP_distance_m']=float(np.linalg.norm(point-actual[0]));self.provisional=False;self.lift_verified=True;self.pre_lift_frame=None
        if not hasattr(self,'attachment_updates'):self.attachment_updates=[]
        self.attachment_updates.append({'stage_id':stage['stage_id'],'source':'fresh visible Lift object and verified visual holding; own nonzero Lift; no Grasp role point fabricated','actual_Lift_distance_m':float(np.linalg.norm(actual[0]-anchor[0])),'semantic_record':self.judge.records[-1]})
    def run(self):
        result=super().run();result.update(execution_variant='ordinary_provisional_grasp_observed_lift_v26',graph_schema=SCHEMA);write(self.output/'RESULT.json',result);return result
SharedCupExecutor=ObserverExecutor
SharedSAMVision=EvidenceVision
