"""Fresh sparse RGBD body evidence after own nonzero lift; never a body pose measurement."""
from collections import deque
import numpy as np
from scipy.ndimage import label
from public_grasp_geometry_v55 import cloud

class VisibleCarryWitness:
    def __init__(self,rgb,mask):
        pixels=np.asarray(rgb,float)[np.asarray(mask,bool)];pixels=pixels[pixels.sum(1)>30];chroma=pixels/np.maximum(pixels.sum(1,keepdims=True),1)
        self.prototype=np.median(chroma,axis=0);distance=np.linalg.norm(chroma-self.prototype,axis=1);self.threshold=min(.10,max(.025,float(np.quantile(distance,.95))*1.5));self.recent=deque(maxlen=12);self.records=[]
    def measure(self,stamp,rgb,depth,K,T,plane,center_prior,height,tcp_displacement,cycle):
        arr=np.asarray(rgb,float);total=arr.sum(2);chroma=arr/np.maximum(total[:,:,None],1);mask=(total>30)&(np.linalg.norm(chroma-self.prototype,axis=2)<=self.threshold)&np.isfinite(depth)&(depth>.05)&(depth<3)
        yy,xx=np.where(mask);zz=depth[yy,xx];p=np.c_[(xx-K[0,2])*zz/K[0,0],(yy-K[1,2])*zz/K[1,1],zz]@T[:3,:3].T+T[:3,3];n=np.asarray(plane['support_plane_normal']);o=np.asarray(plane['support_plane_point_m']);radius=np.sqrt(3)*height/2+.010;valid=(np.linalg.norm(p-center_prior,axis=1)<radius)&((p-o)@n>height+.010)
        actual=np.zeros(depth.shape,bool);actual[yy[valid],xx[valid]]=True;components,count=label(actual,np.ones((3,3)));best=None
        for i in range(1,count+1):
            m=components==i;number=int(m.sum())
            if number<3:continue
            pts=cloud(depth,K,T,m);point=np.median(pts,axis=0)
            if best is None or number>best[0]:best=(number,point)
        number=0 if best is None else best[0];point=None if best is None else best[1];self.recent.append((stamp,number,cycle));past=[r for r in self.recent if r[2]==cycle and 0<=stamp-r[0]<=.5 and r[1]>=3]
        qualified=bool(point is not None and tcp_displacement>=height and len(past)>=3 and stamp-past[0][0]>=.19 and sum(r[1] for r in past)>=12)
        record={'stamp':stamp,'cycle':cycle,'actual_visible_depth_points':number,'recent_positive_frames':len(past),'recent_actual_points':sum(r[1] for r in past),'nonzero_own_lift_m':float(tcp_displacement),'visible_surface_point_m':None if point is None else point.tolist(),'qualified_control_permission':qualified,'association_prior_m':np.asarray(center_prior).tolist(),'association_prior_is_not_measurement':True,'body_pose_measured':False,'contact_measured':False,'source':'initial actual SAM colour prototype + current RGBD + own TCP; no object truth'};self.records.append(record);return point,qualified,record
