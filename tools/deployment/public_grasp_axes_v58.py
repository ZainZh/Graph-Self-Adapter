"""Antipodal grasp axes estimated from observed top surfaces; no object pose labels."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from public_grasp_geometry_v55 import cloud

def top_face_alignment(depth,K,T,mask,plane,closing_axis):
    pts=cloud(depth,K,T,mask);n=np.asarray(plane['support_plane_normal']);o=np.asarray(plane['support_plane_point_m']);h=(pts-o)@n;top=pts[h>=np.quantile(h,.95)-.0015]
    ex=np.array([1.,0,0]);ex-=n*(ex@n);ex/=np.linalg.norm(ex);ey=np.cross(n,ex);uv=top@np.stack([ex,ey],axis=1);hull=ConvexHull(uv);corners=uv[hull.vertices];edges=np.roll(corners,-1,axis=0)-corners
    best=None
    for edge in edges:
        angle=np.arctan2(edge[1],edge[0]);c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s],[s,c]]);p=uv@R;extent=p.max(0)-p.min(0);area=np.prod(extent)
        if best is None or area<best[0]:best=(area,angle,extent)
    angle=best[1];axis_angle=np.arctan2(closing_axis@ey,closing_axis@ex);delta=(angle-axis_angle+np.pi/4)%(np.pi/2)-np.pi/4
    return Rotation.from_rotvec(n*delta).as_matrix(),{'axis_source':'minimum area rectangle of observed top surface boundary','rotation_rad':float(delta),'observed_extent_m':best[2].tolist(),'no_simulator_orientation':True}
