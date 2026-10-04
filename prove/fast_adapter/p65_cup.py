"""P65 cup stage routing around entirely unchanged frozen P64 modules.

Simple approach/transport stages use the original code controller. Only the
critical final approach emits a P64 neural position command. The first-EEF
coordinate frame and public history are never rebased at the stage boundary.
"""
import numpy as np
import torch

from prove.fast_adapter.p64_cup import NOMINAL_ROTATION
from prove.fast_adapter.p64_cup_policy import PHASES
from prove.fast_adapter.p64_cup_runtime import CupRuntime
from prove.fast_adapter.robot_geometry_env_p45_v02 import GeometryTaskController

ARMS=('fixed','online_history','online_local')
STAGES=('observe','clearance','align','descend','approach','close','transport','hold')


class RoutedCupController(GeometryTaskController):
    """Original gates/gripper decisions, neural position only in approach."""
    def __init__(self,start_pos,start_R,policy):
        super().__init__('cup',start_pos,start_R)
        self.policy=policy.eval().requires_grad_(False)
        self.routing_records=[]

    def command(self,t,estimate,proprio):
        before=self.phase
        code_p,code_R,closed,stage=super().command(t,estimate,proprio)
        after=self.phase
        qualified=estimate is not None and all(np.isfinite(x).all() for x in estimate)
        neural=bool(t>=12 and before=='approach' and qualified)
        record=dict(t=int(t),phase_before=before,phase_after=after,route='neural' if neural else 'code',
            code_target=np.asarray(code_p).copy(),code_rotation=np.asarray(code_R).copy(),
            gripper_closed=bool(closed),adaptation_frozen=bool(self.adaptation_frozen),
            used=neural,stage=stage)
        if not neural:
            record['target']=np.asarray(code_p).copy();self.routing_records.append(record)
            return code_p,code_R,closed,stage
        center,rotation,size=estimate
        anchor=self.anchor[0] if self.anchor is not None else center+rotation[:,0]*(size[0]-.003)-rotation[:,2]*.004
        device=next(self.policy.parameters()).device
        with torch.no_grad():
            values=[torch.as_tensor(x,device=device,dtype=torch.float32)[None] for x in (center,size,self.start_pos,anchor)]
            phase=torch.tensor([PHASES.index('approach')],device=device)
            target=self.policy(*values,phase)[0].cpu().numpy()
        record.update(target=target.copy(),center=np.asarray(center).copy(),size=np.asarray(size).copy(),
            start=self.start_pos.copy(),anchor=np.asarray(anchor).copy(),neural_phase='approach',phase='approach')
        self.routing_records.append(record)
        self.last_target=(target.copy(),NOMINAL_ROTATION.copy())
        return target,NOMINAL_ROTATION.copy(),closed,stage


class StageLocalCupRuntime(CupRuntime):
    """Schedule unchanged P64 updates; retain original anchor/public support."""
    def __init__(self,checkpoint,model_dir,arm,device='cuda'):
        if arm not in ARMS:raise ValueError('Unknown P65 arm')
        super().__init__(checkpoint,model_dir,'fixed',device)
        self.initialize_stage_schedule(arm)

    def initialize_stage_schedule(self,arm):
        if arm not in ARMS:raise ValueError('Unknown P65 arm')
        self.p65_arm=arm;self.previous_phase=None;self.local_entries=0
        self.original_anchor=None;self.phase_records=[]
        self.exit_records=[];self.critical_closed=False

    def update_phase(self,public,rgb,masks,t,phase,*,adaptation_frozen=False):
        if phase not in STAGES:raise ValueError('Unknown controller stage')
        if self.critical_closed or adaptation_frozen or phase in ('close','transport','hold'):
            raise ValueError('No encoder/latent update permitted after closing')
        entry=phase=='approach' and self.previous_phase!='approach'
        reset=entry and self.p65_arm=='online_local'
        history_before=len(self.history)
        if reset:
            self.local_entries+=1
            if self.local_entries!=1:raise RuntimeError('Unexpected repeated approach entry')
            self.z=torch.zeros_like(self.z)
        active=self.p65_arm=='online_history' or (self.p65_arm=='online_local' and phase=='approach')
        self.arm='translation_size5' if active else 'fixed'
        # Exactly one current-frame encoder/feedback call; no replay, extra
        # observation, truth geometry, or anchor reset at critical entry.
        admitted=super().update(public,rgb,masks,t)
        if self.original_anchor is None:self.original_anchor=self.anchor.detach().clone()
        if not torch.equal(self.anchor,self.original_anchor):raise RuntimeError('Episode anchor changed')
        record=self.records[-1]
        if (not active) and (record['U'] or bool(self.z.abs().any())):
            raise RuntimeError('Latent changed outside its authorized stage')
        if self.p65_arm=='online_local' and record['U'] and phase!='approach':
            raise RuntimeError('Local adaptation escaped critical stage')
        boundary=dict(t=int(t),arm=self.p65_arm,phase_before=phase,update_enabled=active,
            local_entry_reset=reset,history_before=history_before,history_after=len(self.history),
            readonly_episode_anchor=True,U=record['U'])
        self.phase_records.append(boundary);record['stage_schedule']=boundary
        self.previous_phase=phase
        return admitted

    def end_critical(self,t,reason='approach_to_close'):
        """Clear live latent, retaining frozen decoded target and archived trace."""
        if self.critical_closed:raise RuntimeError('Critical stage already closed')
        before=self.z.detach().clone()
        self.z=torch.zeros_like(self.z)
        self.critical_closed=True
        event=dict(t=int(t),reason=reason,z_before=before.cpu().numpy().copy(),
            z_after=self.z.cpu().numpy().copy(),cached_estimate_retained=True,
            archived_prediction_unchanged=True)
        self.exit_records.append(event)
        return event
