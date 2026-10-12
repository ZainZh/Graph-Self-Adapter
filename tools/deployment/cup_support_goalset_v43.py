"""Bounded wrist tilt plus yaw jointly solve visible cup/box support and robot reachability."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from cup_v25_full_support_envelope import hull2,normals
from cup_visible_surfaces_v39 import observed_surfaces


def solve_goalset(fk,q,g,cup_cloud,box_cloud,red_cloud,nominal,limits,*,maximum_iterations=160):
    q=np.asarray(q,float);g=np.asarray(g,float);start,R=fk(q,g,'left');start=np.asarray(start);R=np.asarray(R)
    cup,cm=observed_surfaces(cup_cloud);red,rm=observed_surfaces(red_cloud)
    vertices=cup[ConvexHull(cup).vertices];box=np.asarray(box_cloud);top=box[box[:,2]>=np.quantile(box[:,2],.5)];blo,bhi=np.quantile(top[:,:2],[.02,.98],axis=0)
    topz=float(np.median(box[box[:,2]>=np.quantile(box[:,2],.8),2]));occupied=hull2(red[:,:2]);cache={};records=[]
    def state(x):
        key=np.asarray(x).tobytes()
        if key not in cache:
            full=q.copy();full[0]=x[:6];p,rot=fk(full,g,'left');Q=Rotation.from_euler('xyz',x[6:9]).as_matrix()
            projected=(vertices-start)@Q.T+np.asarray(p);cache[key]=(full,np.asarray(p),np.asarray(rot),Q,projected)
        return cache[key]
    def equality(x):
        full,p,rot,Q,cloud=state(x)
        return np.r_[(cloud[:,2].min()-topz-.020)*20,Rotation.from_matrix((Q@R)@rot.T).as_rotvec()]
    def inside(x):
        pts=state(x)[4]
        return np.r_[(pts[:,:2].min(axis=0)-blo-.010)*20,(bhi-.010-pts[:,:2].max(axis=0))*20,.55-np.linalg.norm(x[6:8])]
    def objective(x):
        center=np.median(state(x)[4],axis=0)
        return float(np.sum((x[:6]-q[0])**2)+.3*np.sum(x[6:]**2)+.2*np.sum((center[:2]-nominal)**2)/.05**2)
    bounds=list(zip(limits.joint_min,limits.joint_max))+[(-.55,.55),(-.55,.55),(-np.pi,np.pi)]
    axes=list(normals(occupied));proxy=np.median(cup,axis=0);axes.sort(key=lambda n:-float((proxy[:2]-np.median(red[:,:2],axis=0))@n))
    for branch,n in enumerate(axes):
        redmax=float((occupied@n).max())
        def separate(x,n=n,redmax=redmax):return float((state(x)[4][:,:2]@n).min()-redmax)*20
        result=minimize(objective,np.r_[q[0],0.,0.,0.],method='SLSQP',bounds=bounds,constraints=[{'type':'eq','fun':equality},{'type':'ineq','fun':inside},{'type':'ineq','fun':separate}],options={'maxiter':maximum_iterations,'ftol':1e-9})
        x=result.x;eq=float(np.max(np.abs(equality(x))));margin=float(inside(x).min());sep=separate(x);valid=eq<=1e-4 and margin>=-1e-6 and sep>=-1e-6
        records.append({'branch':branch,'success':bool(result.success),'message':str(result.message),'equal_residual':eq,'box_and_tilt_margin':margin,'red_separation':sep,'feasible':valid})
        if valid:
            full,p,rot,Q,cloud=state(x)
            report={'feasible':True,'source':'same current visible cup convex boundary, box top, red exclusion and own FK/joints','orientation_mode':'yaw plus bounded31.5degree wrist tilt; full current visible surface moves rigidly','relative_xyz_deg':np.rad2deg(x[6:9]).tolist(),'goal_q_rad':full.tolist(),'goal_TCP_m':p.tolist(),'goal_rotation':rot.tolist(),'cup_component':cm,'red_component':rm,'box_lower_xy_m':blo.tolist(),'box_upper_xy_m':bhi.tolist(),'box_margin_m':.010,'release_clearance_m':.020,'projected_cup_contour_xy_m':hull2(cloud[:,:2]).tolist(),'red_exclusion_contour_xy_m':occupied.tolist(),'optimizer_branches':records,'FK_evaluations':len(cache),'hidden_mesh_or_COM':False}
            return (p,rot),report
    return None,{'feasible':False,'reason':'no bounded joint/full-visible-support/tilt solution','optimizer_branches':records,'FK_evaluations':len(cache),'cup_component':cm,'red_component':rm}
