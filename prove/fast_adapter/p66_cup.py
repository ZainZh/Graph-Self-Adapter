"""P66 fixed-reference state shared by live runtime and sparse public replay."""
import numpy as np
import torch
from prove.fast_adapter.p64_cup import adapt
from prove.fast_adapter.p61_adaptation import corrected_components
from prove.fast_adapter.p60_contour import causal_support
from prove.fast_adapter.p64_cup_runtime import CupRuntime

ARMS=('frame_fixed','reference_fixed','entry_once','continuous')
PHASES=('observe','clearance','align','descend','approach','close','transport','hold')


class ReferenceState:
    """No simulator/model access: consumes causal public readout + contours."""
    def __init__(self,arm,device='cpu',adapter=adapt):
        if arm not in ARMS:raise ValueError('Unknown P66 arm')
        self.arm=arm;self.adapter=adapter;self.z=torch.zeros(8,device=device)
        self.anchor=None;self.latest=None;self.g_ref=None;self.ref_time=None
        self.entry_seen=False;self.entry_initialized=False;self.ended=False
        self.last_t=-1;self.records=[];self.estimate=None;self.exit_records=[]

    def step(self,raw,anchor,support,*,t,phase,admitted,goal_valid):
        if phase not in PHASES:raise ValueError('Unknown controller phase')
        if self.ended or phase in ('close','transport','hold'):raise ValueError('No update after critical exit')
        if t<=self.last_t:raise ValueError('Strictly causal observations required')
        if raw.shape!=(8,) or not torch.isfinite(raw).all():raise ValueError('Finite raw geometry8 required')
        if anchor.shape!=(4,4) or not torch.isfinite(anchor).all():raise ValueError('Finite4x4 original anchor required')
        if self.z.shape!=(8,) or not torch.isfinite(self.z).all() or bool(self.z[3:6].abs().any()):
            raise ValueError('Finite position/size-only latent required')
        if any(m.t>t for m in support):raise ValueError('Future feedback is forbidden')
        self.last_t=t
        if self.anchor is None:self.anchor=anchor.detach().clone()
        if not torch.equal(self.anchor,anchor):raise ValueError('Original episode anchor must remain fixed')
        if admitted:self.latest=(int(t),raw.detach().clone())
        critical=phase=='approach';entered=critical and not self.entry_seen
        if entered:
            self.entry_seen=True;self.z=torch.zeros_like(self.z)
        latched=False
        if critical and self.arm!='frame_fixed' and self.g_ref is None and self.latest is not None:
            self.ref_time,self.g_ref=self.latest[0],self.latest[1].clone()
            latched=True
        before=self.z.detach().clone();event=None;inner=[]
        base=raw
        qualified=bool(goal_valid)
        if critical and self.arm!='frame_fixed':
            base=self.g_ref;qualified=base is not None
            if qualified and self.arm in ('entry_once','continuous') and len(support)>=3:
                if not self.entry_initialized:
                    event='entry';self.entry_initialized=True
                elif self.arm=='continuous' and admitted:event='continuous'
                if event:
                    self.z,inner=self.adapter(base,self.z,self.anchor,support,arm='translation_size5')
        if self.z.shape!=(8,) or not torch.isfinite(self.z).all() or bool(self.z[3:6].abs().any()):
            raise RuntimeError('Adapter changed orientation or produced invalid latent')
        if not critical or self.arm in ('frame_fixed','reference_fixed'):
            if bool(self.z.abs().any()):raise RuntimeError('Nonzero z outside an updating arm/stage')
        if base is None:
            self.estimate=None;reason='no_admitted_historical_reference_hold'
        elif qualified:
            with torch.no_grad():
                p,R,size=corrected_components(base,self.z)
                self.estimate=tuple(x.detach().cpu().numpy() for x in
                    (self.anchor[:3,:3]@p+self.anchor[:3,3],self.anchor[:3,:3]@R,size))
            reason='reference_retained_despite_current_mask_absence' if critical and self.arm!='frame_fixed' and not goal_valid else 'qualified'
        else:self.estimate=None;reason='current_goal_mask_absent_hold'
        record=dict(t=int(t),arm=self.arm,phase_before=phase,raw_geometry8=raw.detach().cpu().numpy().copy(),
            geometry8=raw.detach().cpu().numpy().copy(),decoder_base=None if base is None else base.detach().cpu().numpy().copy(),
            g_ref=None if self.g_ref is None else self.g_ref.cpu().numpy().copy(),ref_time=self.ref_time,
            z_before=before.cpu().numpy().copy(),z_after=self.z.detach().cpu().numpy().copy(),
            T_world_anchor=self.anchor.cpu().numpy().copy(),admitted=bool(admitted),goal_valid=bool(goal_valid),
            S=len(support),support_times=[int(m.t) for m in support],U=len(inner),inner=inner,event=event,
            entry_seen=self.entry_seen,entry_initialized=self.entry_initialized,entry_at_this_frame=entered,
            reference_latched_this_frame=latched,latest_admitted_time=None if self.latest is None else self.latest[0],
            initialization_pending=critical and self.arm in ('entry_once','continuous') and not self.entry_initialized,
            initialization_pending_reason=('no_admitted_reference' if self.g_ref is None else 'fewer_than_three_supports')
                if critical and self.arm in ('entry_once','continuous') and not self.entry_initialized else None,
            reference_available=self.g_ref is not None,qualification_reason=reason,
            decoded=None if self.estimate is None else tuple(x.copy() for x in self.estimate))
        if record['U'] not in (0,3):raise RuntimeError('Original update budget changed')
        self.records.append(record);return self.estimate

    def finish(self,t,reason='approach_to_close'):
        if self.ended:raise RuntimeError('Already ended')
        event=dict(t=int(t),reason=reason,z_before=self.z.cpu().numpy().copy(),z_after=np.zeros(8),
            cached_estimate_retained=True,archive_retained=True)
        self.z=torch.zeros_like(self.z);self.ended=True;self.exit_records.append(event)
        return event


class ReferenceCupRuntime(CupRuntime):
    """Unchanged P64 raw forward/measurement with independent P66 decoder state."""
    def __init__(self,checkpoint,model_dir,arm,device='cuda'):
        super().__init__(checkpoint,model_dir,'fixed',device)
        self.reference=ReferenceState(arm,device=device)
        self.p66_records=[]

    def update_phase(self,public,rgb,masks,t,phase):
        if self.reference.ended:raise ValueError('Runtime closed')
        # Always retain the original current-frame encoder and visible-mask
        # extraction; the parent's own z stays zero and cannot drive reference.
        admitted=super().update(public,rgb,masks,t)
        raw_record=self.records[-1]
        self._estimate=self.reference.step(self.geometry,self.anchor,causal_support(self.history),
            t=t,phase=phase,admitted=admitted,goal_valid=bool(raw_record['role_valid'][2]))
        record=self.reference.records[-1];record['role_valid']=raw_record['role_valid']
        self.p66_records.append(record)
        return admitted

    def end_critical(self,t,reason='approach_to_close'):
        return self.reference.finish(t,reason)
