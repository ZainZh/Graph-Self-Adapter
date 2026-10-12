"""Partial observed RGBD surface to frozen known-object CAD; returns an estimate, not truth."""
import numpy as np
from scipy.spatial import cKDTree
from prove.xtrainer.io import SafetyError

def rigid(p,R):
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=p;return T

def register_surface(observed,model,initial):
    if len(observed)<100:raise SafetyError('visible object registration surface sparse')
    obs=np.asarray(observed);obs=obs[::max(1,len(obs)//1200)];tree=cKDTree(model);T=initial.copy();records=[]
    for k in range(35):
        local=(obs-T[:3,3])@T[:3,:3];distance,index=tree.query(local);keep=distance<=max(.002,float(np.quantile(distance,.85)));a=model[index[keep]]@T[:3,:3].T+T[:3,3];b=obs[keep];ca=a.mean(0);cb=b.mean(0);U,_,V=np.linalg.svd((a-ca).T@(b-cb));R=V.T@U.T
        if np.linalg.det(R)<0:V[-1]*=-1;R=V.T@U.T
        delta=rigid(cb-R@ca,R);T=delta@T;records.append(float(np.sqrt(np.mean(distance[keep]**2))))
    distance,_=tree.query((obs-T[:3,3])@T[:3,:3]);rms=float(np.sqrt(np.mean(np.sort(distance)[:int(.85*len(distance))]**2)))
    if rms>.004:raise SafetyError('partial object/CAD registration uncertain')
    return T,{'visible_points':len(obs),'trimmed_RMS_m':rms,'iterations':len(records),'body_pose_is_estimate':True,'known_CAD_prior':True}
