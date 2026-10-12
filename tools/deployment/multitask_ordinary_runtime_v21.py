"""V21: Lift own robot relative SE(3), not an occluded object-centroid path."""
import numpy as np
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from multitask_ordinary_runtime import OrdinaryExecutor,SharedSAMVision

VARIANT='multitask_ordinary_relative_lift_v21'


def relative_lift_goal(actual_anchor,reference_start,reference_current):
    p0,R0=actual_anchor;pr,Rr=reference_start;pt,Rt=reference_current
    rebase=np.asarray(R0)@np.asarray(Rr).T
    return np.asarray(p0)+rebase@(np.asarray(pt)-np.asarray(pr)),rebase@np.asarray(Rt)


class RelativeLiftExecutor(OrdinaryExecutor):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.accepted_grasp_pose={};self.lift_progress={};self.relative_lift_records=[]
    def _actual_pose(self,samples,arm):
        i=0 if arm=='left' else 1;p=samples[i].tcp.copy();R=samples[i].rotation.copy()
        return ((self.T@np.r_[p,1])[:3],self.T[:3,:3]@R) if i else (p,R)
    def semantics(self,stage,facts,before):
        after=super().semantics(stage,facts,before)
        if stage['phase']=='Grasp' and self.visual_facts.get('holding') is True:self.accepted_grasp_pose[stage['cycle_id']]=self._actual_pose(after[0],stage['arm'])
        return after
    def command(self,stage,elapsed,samples,obs,dt):
        if stage['phase']!='Lift':return super().command(stage,elapsed,samples,obs,dt)
        cert=self.accepted_grasp;cycle=stage['cycle_id'];arm=stage['arm'];index=0 if arm=='left' else 1
        if cert is None or cert['cycle']!=cycle or self.holding(samples,stage) is not True:raise SafetyError('relative Lift actual holding unknown/false')
        anchor=self.accepted_grasp_pose.get(cycle)
        if anchor is None:raise SafetyError('accepted actual Grasp TCP anchor missing')
        start=stage['reference']['start_time_sec'];end=stage['reference']['end_time_sec'];pose=self._actual_pose(samples,arm)
        first=stage['stage_id'] not in self.lift_progress
        if first:self.lift_progress[stage['stage_id']]=0.;self._last_cartesian_goal=None
        elif self._last_cartesian_goal is not None and self._reached(self._last_cartesian_goal,pose):self.lift_progress[stage['stage_id']]+=dt
        t=min(end,start+self.lift_progress[stage['stage_id']]*self.cfg['hardware_reference_speed_scale']*stage.get('reference_speed_scale',1.))
        g=self.command_g.copy();g[index]=cert['g_command']
        reference_start=self._shared_pose(np.asarray(self.b['qcurve'](start)),g,arm);reference_now=self._shared_pose(np.asarray(self.b['qcurve'](t)),g,arm)
        goal=relative_lift_goal(anchor,reference_start,reference_now);self._last_cartesian_goal=goal;self._goal_reached=self._reached(goal,pose)
        actual=np.stack([s.q for s in samples]);target,_,error=self._step_ik(actual,g,arm,goal)
        fresh,frame=self.capture();self.safety.validate(target,g,fresh,dt,self.robot,self.T);latest=self.robot.samples();self.safety.samples(latest)
        if not 0<=self.clock()-frame.stamp<=self.cfg['safety']['max_age_sec']:raise SafetyError('relative Lift fresh frame stale before dispatch')
        if np.max(abs(target-np.stack([s.q for s in latest])))>.01+1e-9:raise SafetyError('relative Lift actual pose changed during IK')
        if self.holding(latest,stage) is not True:raise SafetyError('relative Lift fresh holding anomaly before dispatch')
        self._hardware_attempts+=1;write(self.output/'COMMAND_INTENT.json',{'stage_id':stage['stage_id'],'attempt_count':self._hardware_attempts,'acknowledged_count':self._hardware_acknowledged,'q_target_rad':target.tolist(),'gripper_command':g.tolist(),'flags':[index==0,index==1],'partial_motion_possible_without_ack':True})
        self.robot.command(target,g,[index==0,index==1]);self._hardware_acknowledged+=1;self.command_index+=1;self.command_g=g.copy();self.safety.last_target=target.copy();self.sleep(dt)
        progress=(t-start)/max(end-start,1e-9)
        record={'controller':'accepted_actual_Grasp_TCP_relative_robot_Lift_SE3','reference_time_sec':t,'first_query_exact_reference_START':first,'goal_TCP_shared_m':goal[0].tolist(),'goal_rotation_shared':goal[1].tolist(),'actual_grasp_anchor_shared_m':anchor[0].tolist(),'object_centroid_not_used_for_Lift_goal':True,'locked_gripper_command':float(g[index]),'Cartesian_error_before_command':error.tolist()}
        self.relative_lift_records.append(record);return target,progress,record
    def refresh_lift_attachment(self,stage,samples,obs):
        # Real positive motion support is required; tentative API permission is not contact truth.
        if self.holding(samples,stage) is not True:raise SafetyError('relative Lift empty/unknown holding at endpoint')
        evidence=getattr(self,'comotion_record',None)
        if not evidence or evidence.get('role')!=stage['object'] or evidence.get('nonzero_evidence') is not True or evidence.get('disagreement_m',float('inf'))>.006:raise SafetyError('relative Lift actual SAM/TCP positive comotion unknown')
        super().refresh_lift_attachment(stage,samples,obs)
        # Anchor subsequent relative rotations at this actual reached Lift endpoint.
        actual=self._actual_pose(samples,stage['arm']);reference=self._shared_pose(np.asarray(self.b['qcurve'](stage['reference']['end_time_sec'])),self.command_g,stage['arm'])
        self.attachment['R_demo_end']=reference[1].copy();self.attachment['R_actual_grasp']=actual[1].copy()
        self.attachment_updates[-1].update(rotation_anchor_source='actual Lift-end own TCP/R plus declared reference Lift-end; ensures next Move starts continuously',reason='positive nonzero actual visible SAM/TCP motion after relative robot Lift; estimated attachment, not contact')
    def run(self):
        result=super().run();result.update(execution_variant=VARIANT,relative_Lift_records=self.relative_lift_records,Lift_contract='actual accepted Grasp TCP times relative demonstrated robot SE3; no occluded-centroid lateral correction')
        write(self.output/'RESULT.json',result);return result
