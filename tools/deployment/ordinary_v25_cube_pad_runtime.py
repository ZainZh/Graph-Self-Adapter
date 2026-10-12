"""All five cube body centers from visible top faces + measured plane; distal actual robot pad geometry."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from ordinary_v25_cube_observer_runtime import ObserverExecutor
from ordinary_v25_apple_pad_runtime import AppleVision
class CubePadVision(AppleVision):pass
def cube_centers(rgb,depth,K,T,masks,roles,roi):
    def cloud(mask):
        y,x=np.where(np.asarray(mask,bool)&np.isfinite(depth)&(depth>0));z=depth[y,x];cam=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z];return cam@T[:3,:3].T+T[:3,3]
    x0,y0,x1,y1=roi;mask=np.zeros(depth.shape,bool);mask[y0:y1,x0:x1]=True;tablecloud=cloud(mask)
    if len(tablecloud)<100:raise SafetyError('cube tabletop visible depth insufficient')
    table=float(np.median(tablecloud[:,2]));rms=float(np.sqrt(np.mean((tablecloud[:,2]-table)**2)))
    if rms>.001:raise SafetyError('cube table ROI is not a validated horizontal plane')
    centers={};records={}
    for role in roles:
        pts=cloud(binary_erosion(masks[role],iterations=1))
        if len(pts)<100:raise SafetyError('cube fresh SAM geometry sparse: '+role)
        topz=float(np.quantile(pts[:,2],.95));top=pts[pts[:,2]>=topz-.0015];height=topz-table
        if len(top)<30 or not .015<=height<=.055:raise SafetyError('cube visible top/body height uncertain: '+role)
        xy=(np.quantile(top[:,:2],.01,axis=0)+np.quantile(top[:,:2],.99,axis=0))/2
        centers[role]=np.r_[xy,(topz+table)/2];records[role]={'observed_center_m':centers[role].tolist(),'observed_height_m':height,'top_face_pixels':len(top),'cloud_points':len(pts),'hypothesis':'visible top face XY midpoint plus observed top-to-table halfheight; backside not observed','static_unmanipulated_initial_prior':True}
    return centers,{'table_z_m':table,'table_plane_RMS_m':rms,'cube_measurements':records}
class CubePadExecutor(ObserverExecutor):
    def start(self):
        super().start();stamp,rgb,depth,K=self.vision.camera.latest_public_capture
        if not 0<=self.clock()-stamp<=self.cfg['safety']['max_age_sec']:raise SafetyError('cube initial RGBD geometry stale')
        self.initial_geometry_stamp=stamp;roles=sorted({s['object'] for s in self.b['graph']['stages']});self.centers,self.geometry=cube_centers(rgb,depth,K,self.vision.T,self.vision.public_masks,roles,self.cfg['cube_table_ROI'])
        self.prior=json.loads(Path(self.cfg['cube_robot_pad_prior']).read_text());self.pad_cycles=set()
        write(self.output/'CUBE_INITIAL_PUBLIC_GEOMETRY.json',self.geometry)
    def _ground_red_cycle(self,stage,obs):
        super()._ground_red_cycle(stage,obs)
        if stage['cycle_id'] in self.pad_cycles:return
        grasp=next(s for s in self.b['graph']['stages'] if s['cycle_id']==stage['cycle_id'] and s['phase']=='Grasp');t=grasp['reference']['end_time_sec'];g=np.array([np.interp(t,self.b['times'],self.b['gripper'][:,i]) for i in range(2)]);p,R=self._shared_pose(np.asarray(self.b['qcurve'](t)),g,stage['arm']);prior=self.prior['by_arm'][stage['arm']];pad=np.asarray(prior['distal_pad_TCP_local_m']);center=self.centers[stage['object']];goal=center-R@pad;vertices=np.asarray(prior['robot_pad_CAD']['pad_vertices_TCP_local_m']);clearance=float((vertices@R.T+goal)[:,2].min())-self.geometry['table_z_m']
        if clearance<.003:raise SafetyError('observed cube center grasp finger CAD tabletop clearance below3mm')
        self.coarse_source=center-(p+R@pad);self.pad_cycles.add(stage['cycle_id']);write(self.output/(stage['cycle_id']+'_PAD_PLAN.json'),{'source':'current actual SAM visible top+measured plane and true robot common distal pad overlap','center_m':center.tolist(),'distal_pad_TCP_local_m':pad.tolist(),'Code_source_shift_m':self.coarse_source.tolist(),'full_finger_table_clearance_m':clearance,'NN_environment_cue_minus_same_Code_handled_cue_m':(self.static_source_env-self.code_static_source_handled).tolist(),'static_scene_prior_age_sec':self.clock()-self.initial_geometry_stamp if hasattr(self,'initial_geometry_stamp') else None,'backside_unobserved':True,'no_object_truth':True})
SharedCupExecutor=CubePadExecutor
SharedSAMVision=CubePadVision
