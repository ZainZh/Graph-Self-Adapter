"""Apple body-center grasp from current actual SAM RGBD plus robot-only pad CAD."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from ordinary_v25_repair_runtime import RepairExecutor,EvidenceVision
class AppleVision(EvidenceVision):
    def _tracker(self):
        base=super()._tracker();owner=self
        class Tracker:
            def initialize(self,*args):
                masks=base.initialize(*args);owner.public_masks=masks;return masks
            def track(self,*args):
                masks=base.track(*args);owner.public_masks=masks;return masks
        return Tracker()
def visible_apple_sphere(rgb,depth,K,T,mask):
    mask=binary_erosion(np.asarray(mask,bool),iterations=2)&np.isfinite(depth)&(depth>0)
    mask &= (rgb[:,:,0].astype(float)>1.15*rgb[:,:,1])&(rgb[:,:,0].astype(float)>1.15*rgb[:,:,2])
    y,x=np.nonzero(mask)
    if len(x)<100:raise SafetyError('fresh red apple foreground depth insufficient for bounded near-sphere fit')
    z=depth[y,x];camera=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z];cloud=camera@T[:3,:3].T+T[:3,3];idx=np.ones(len(cloud),bool)
    for _ in range(4):
        pts=cloud[idx];origin=pts.mean(axis=0);v=pts-origin;A=np.c_[2*v,np.ones(len(v))];beta=np.linalg.lstsq(A,(v*v).sum(axis=1),rcond=None)[0];center=origin+beta[:3];radius=np.sqrt(max(0,beta[3]+np.dot(beta[:3],beta[:3])));residual=abs(np.linalg.norm(cloud-center,axis=1)-radius);idx=residual<=max(.0015,float(np.quantile(residual,.8)))
    rms=float(np.sqrt(np.mean(residual[idx]**2)));condition=float(np.linalg.cond(A))
    if not .02<=radius<=.045 or rms>.0025 or condition>120:raise SafetyError('observed apple near-sphere hypothesis is uncertain')
    return center,{'center_m':center.tolist(),'radius_m':float(radius),'fit_RMS_m':rms,'fit_condition':condition,'foreground_points':len(cloud),'inliers':int(idx.sum()),'backside_observed':False,'center_is_estimate_not_truth':True}
class ApplePadExecutor(RepairExecutor):
    def start(self):
        super().start();stamp,rgb,depth,K=self.vision.camera.latest_public_capture
        if not 0<=self.clock()-stamp<=self.cfg['safety']['max_age_sec']:raise SafetyError('apple initialization actual RGBD too stale')
        self.body_center,self.body_fit=visible_apple_sphere(rgb,depth,K,self.vision.T,self.vision.public_masks['apple'])
        prior=json.loads(Path(self.cfg['apple_robot_pad_prior']).read_text());self.pad=np.asarray(prior['robot_pad_CAD']['pad_midpoint_TCP_local_m']);self.pad_prior=prior;self.pad_rebased=False
    def _ground_red_cycle(self,stage,obs):
        super()._ground_red_cycle(stage,obs)
        if self.pad_rebased:return
        grasp=next(s for s in self.b['graph']['stages'] if s['phase']=='Grasp');t=grasp['reference']['end_time_sec'];g=np.array([np.interp(t,self.b['times'],self.b['gripper'][:,i]) for i in range(2)]);p,R=self._shared_pose(np.asarray(self.b['qcurve'](t)),g,stage['arm'])
        previous=self.coarse_source.copy();self.coarse_source=self.body_center-(p+R@self.pad);self.pad_rebased=True
        # Frozen environment features remain original cycle-latched offsets minus same Code handled offset; neither dynamic geometry nor changed grasp point enters NN12:18.
        write(self.output/'APPLE_PAD_GRASP_PLAN.json',{'source':'fresh actual SAM RGBD near-sphere estimate + frozen robot-only pad CAD','body_fit':self.body_fit,'pad_TCP_local_m':self.pad.tolist(),'previous_surface_source_shift_m':previous.tolist(),'new_body_center_source_shift_m':self.coarse_source.tolist(),'static_NN_source_cue_m':self.static_source_env.tolist(),'Code_handled_NN_source_cue_m':self.code_static_source_handled.tolist(),'frozen_weights_unchanged':True,'object_truth_read':False})
SharedCupExecutor=ApplePadExecutor
SharedSAMVision=AppleVision
