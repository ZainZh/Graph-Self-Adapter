"""Re-ground an occluded tabletop ROI using current RGBD and an initial validated plane."""
import numpy as np
from pathlib import Path
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write
from autonomous_stage_runtime import AutonomousVision
from cup_v25_current_goalset_runtime import SharedCupExecutor,SharedSAMVision as BaseVision

def current_plane_windows(depth,K,T,center,normal,original_box):
    h,w=depth.shape;y,x=np.mgrid[:h,:w];z=np.asarray(depth,float)
    valid=np.isfinite(z)&(z>0);camera=np.stack([(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z],axis=-1)
    points=camera@T[:3,:3].T+T[:3,3];valid &= np.abs((points-center)@normal)<=.003
    # Plane prior selects candidates only; each is independently fitted below.
    candidates=[];cx=(original_box[0]+original_box[2])/2;cy=(original_box[1]+original_box[3])/2
    for yy in range(0,h-24,12):
        for xx in range(0,w-32,12):
            fraction=float(valid[yy:yy+24,xx:xx+32].mean())
            if fraction>=.95:candidates.append(((xx+16-cx)**2+(yy+12-cy)**2,[xx,yy,xx+32,yy+24]))
    return [box for _,box in sorted(candidates)]

class VisibleTableVision(BaseVision):
    def _table_plane(self,depth,K,T,box):
        prior=getattr(self,'validated_table_prior',None)
        try:
            cloud,mask=AutonomousVision._table_plane(depth,K,T,box)
            if prior is not None and np.quantile(np.abs((cloud-prior[0])@prior[1]),.9)>.003:
                raise SafetyError('current ROI plane differs from initial visible table plane')
            if prior is None:
                center=np.median(cloud,axis=0);_,_,vectors=np.linalg.svd(cloud-center,full_matrices=False);normal=vectors[-1]
                self.validated_table_prior=(center.copy(),normal.copy())
            return cloud,mask
        except SafetyError as error:
            if prior is None:raise
            candidates=current_plane_windows(depth,K,T,prior[0],prior[1],box)
            for selected in candidates[:40]:
                try:
                    cloud,mask=AutonomousVision._table_plane(depth,K,T,selected)
                    if np.quantile(np.abs((cloud-prior[0])@prior[1]),.9)>.003:continue
                except SafetyError:continue
                record={'original_roi':[int(x) for x in box],'selected_current_roi':selected,'reason':str(error),'current_validated_pixels':len(cloud),'initial_plane_used_only_as_candidate_prior':True,'fresh_current_depth_refit':True,'original_3mm_plane_test_preserved':True,'stage':self.stage_context()}
                self.table_relocations=getattr(self,'table_relocations',[])+[record]
                write(Path(self.evidence_root)/'TABLE_ROI_RELOCATIONS.json',self.table_relocations)
                return cloud,mask
            raise SafetyError('no fresh visible tabletop ROI meets original3mm geometry contract')
    def capture(self):
        obs=super().capture()
        # Background geometry remains available through cloud(), not semantic entity centroids.
        obs.roles={name:point for name,point in obs.roles.items() if name!='table_patch'}
        return obs

SharedSAMVision=VisibleTableVision
