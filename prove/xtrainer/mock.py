"""Explicit synthetic I/O fixtures. Never selected as a real robot/judge fallback."""
import numpy as np
from .io import RobotSample
from .perception import Observation

class MockClock:
    def __init__(self):self.t=0.
    def __call__(self):return self.t
    def sleep(self,duration):self.t+=duration

class MockRobot:
    def __init__(self,bundle,clock):
        self.q=bundle['initial_q'].copy();self.g=bundle['initial_g'].copy();self.clock=clock
        self.commands=0;self.calls=[]
    def samples(self):
        return [RobotSample(self.clock(),self.q[i].copy(),np.zeros(6),self.q[i,:3].copy(),np.eye(3),
                            float(self.g[i]*.04),'synthetic_mock_actual_encoders') for i in range(2)]
    def fk(self,q,g,arm):
        self.calls.append('get_fk');return np.asarray(q)[0 if arm=='left' else 1,:3].copy(),np.eye(3)
    def command(self,q,g,flags):
        self.calls.append(('command_joint_state',flags));self.commands+=1
        for index,on in enumerate(flags):
            if on:self.q[index]=q[index];self.g[index]=g[index]

class MockVision:
    def __init__(self,robot,clock):self.robot=robot;self.clock=clock;self.context=None;self.attachments={}
    def set_context(self,stage,elapsed,bundle):self.context=(stage,elapsed,bundle)
    def capture(self):
        if self.context is None:
            return Observation(self.clock(),np.zeros((32,48,3),np.uint8),{},'synthetic_mock_fixture')
        stage,elapsed,b=self.context
        # Static synthetic point fixtures make source/target deltas exactly zero.
        t=stage['reference']['start_time_sec'];ref=b['frames'][int(np.argmin(abs(b['frame_times']-t)))]
        roles={role:None if p is None else np.asarray(p).copy() for role,p in ref['surface_centroids_world_m'].items()}
        if stage['target']=='table_handover_region':roles[stage['target']]=np.asarray(b['refs']['handover_anchors'][stage['cycle_id']]['centroid_world_m'])
        # Synthetic rigid co-motion evidence uses the current fake FK after grasp.
        index=0 if stage['arm']=='left' else 1
        if stage['object'] in self.attachments:
            ai,offset=self.attachments[stage['object']]
            roles[stage['object']]=self.robot.q[ai,:3]+offset
        return Observation(self.clock(),np.full((32,48,3),64,np.uint8),roles,'synthetic_mock_fixture_not_hardware')

class MockJudge:
    def __init__(self,vision=None,robot=None):self.records=[];self.vision=vision;self.robot=robot
    def ask(self,stage,facts,*,before,recapture,heartbeat=None):
        judged={key:True for key in facts};after=recapture()
        if self.vision and 'holding' in facts:
            index=0 if stage['arm']=='left' else 1;point=after[1].roles.get(stage['object'])
            if point is not None:self.vision.attachments[stage['object']]=(index,np.asarray(point)-self.robot.q[index,:3])
        if self.vision and 'released' in facts:self.vision.attachments.pop(stage['object'],None)
        self.records.append({'stage_id':stage['stage_id'],'requested_facts':facts,'facts':judged,
                             'identity':'explicit_synthetic_test_judge_not_Root','actual_agent_inference':False})
        return judged,after

class MockStop:
    def __init__(self):self.calls=0
    def stop(self):self.calls+=1
