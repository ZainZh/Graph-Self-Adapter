"""Observed tabletop plane and graspable part geometry; no object poses or contact truth."""
import numpy as np
from scipy.ndimage import binary_erosion
from prove.xtrainer.io import SafetyError

def cloud(depth,K,T,mask):
    y,x=np.where(np.asarray(mask,bool)&np.isfinite(depth)&(depth>.05)&(depth<3));z=depth[y,x]
    p=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]
    return p@T[:3,:3].T+T[:3,3]

def cube_geometry(rgb,depth,K,T,masks,roi):
    x0,y0,x1,y1=map(int,roi);patch=np.zeros(depth.shape,bool);patch[y0:y1,x0:x1]=True;p=cloud(depth,K,T,patch)
    if len(p)<100:raise SafetyError('observed tabletop plane sparse')
    o=np.median(p,axis=0)
    for _ in range(3):
        _,_,V=np.linalg.svd(p-o,full_matrices=False);n=V[-1];n*=1 if n[2]>0 else -1;r=(p-o)@n;p=p[np.abs(r)<max(.0015,float(np.quantile(np.abs(r),.9)))];o=p.mean(0)
    rms=float(np.sqrt(np.mean(((p-o)@n)**2)))
    if n[2]<.9 or rms>.0015:raise SafetyError('tabletop plane hypothesis uncertain')
    pts=cloud(depth,K,T,binary_erosion(masks['yellow_cube'],iterations=1));height=(pts-o)@n
    if len(pts)<100:raise SafetyError('observed cube surface sparse')
    h=float(np.quantile(height,.95));top=pts[height>=h-.0015]
    if len(top)<30 or not .015<h<.055:raise SafetyError('cube top-to-local-plane height uncertain')
    top_center=(np.quantile(top,.01,axis=0)+np.quantile(top,.99,axis=0))/2;top_center+=(h-(top_center-o)@n)*n;center=top_center-n*h/2
    return {'yellow_cube':center},{'support_plane_point_m':o.tolist(),'support_plane_normal':n.tolist(),'table_plane_RMS_m':rms,'cube_measurements':{'yellow_cube':{'observed_center_m':center.tolist(),'observed_height_m':h,'top_face_pixels':len(top),'cloud_points':len(pts)}},'backside_unobserved':True}
