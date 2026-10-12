"""Minimal repair: restore frozen baseline Grasp cue; observed local support-plane Lift height."""
import json
from pathlib import Path
import numpy as np
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from ordinary_v25_cube_vertical_runtime import VerticalLiftExecutor
from ordinary_v25_cube_pad_runtime import CubePadVision,cube_centers
from ordinary_v25_cube_observer_runtime import ObserverExecutor
from ordinary_v25_runtime import OrdinaryV25Executor
class BaselineVerticalExecutor(VerticalLiftExecutor):
    def start(self):
        ObserverExecutor.start(self);stamp,rgb,depth,K=self.vision.camera.latest_public_capture
        if not 0<=self.clock()-stamp<=self.cfg['safety']['max_age_sec']:raise SafetyError('initial cube local-plane RGBD stale')
        self.initial_geometry_stamp=stamp;roles=sorted({s['object'] for s in self.b['graph']['stages']});centers,self.geometry=cube_centers(rgb,depth,K,self.vision.T,self.vision.public_masks,roles,self.cfg['cube_table_ROI'])
        x0,y0,x1,y1=self.cfg['cube_table_ROI'];yy,xx=np.mgrid[y0:y1,x0:x1];z=depth[yy,xx];ok=np.isfinite(z)&(z>0);cam=np.c_[(xx[ok]-K[0,2])*z[ok]/K[0,0],(yy[ok]-K[1,2])*z[ok]/K[1,1],z[ok]];pts=cam@self.vision.T[:3,:3].T+self.vision.T[:3,3];point=pts.mean(0);_,_,vh=np.linalg.svd(pts-point);normal=vh[-1];normal*=1 if normal[2]>0 else -1
        rms=float(np.sqrt(np.mean(((pts-point)@normal)**2)))
        if normal[2]<.9 or rms>.001:raise SafetyError('observed cube table support plane unresolved')
        for role,center in centers.items():
            local_z=point[2]-(normal[:2]@(center[:2]-point[:2]))/normal[2];row=self.geometry['cube_measurements'][role];top_z=2*center[2]-self.geometry['table_z_m'];height=float(top_z-local_z)
            if not .015<=height<=.055:raise SafetyError('local cube body height invalid')
            row.update(observed_height_m=height,local_observed_table_z_m=float(local_z),height_source='left-base calibrated RGBD plane evaluated at this cube XY; no shared scalar height')
        self.geometry.update(support_plane_point_left_base_m=point.tolist(),support_plane_normal_left_base=normal.tolist(),support_plane_RMS_m=rms,gravity_Lift_axis_left_base=[0,0,1],baseline_Grasp_environment_cue_restored=True)
        write(self.output/'CUBE_INITIAL_PUBLIC_GEOMETRY.json',self.geometry)
    def _ground_red_cycle(self,stage,obs):
        # Restore task frozen baseline object/reference paired cue exactly; do not apply speculative body-centered grasp shift.
        return OrdinaryV25Executor._ground_red_cycle(self,stage,obs)
SharedCupExecutor=BaselineVerticalExecutor
SharedSAMVision=CubePadVision
