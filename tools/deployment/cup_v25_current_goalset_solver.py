"""Current-entry joint/visible-support/yaw planning; no neural adaptation or grid."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from cup_v25_full_support_envelope import hull2,normals
from cup_v25_visible_support_geometry import dominant_visible_surface
from prove.xtrainer.io import SafetyError


def solve_goalset(fk,q,g,cup_cloud,box_cloud,red_cloud,nominal,limits,*,maximum_iterations=140):
    q=np.asarray(q,float);g=np.asarray(g,float);start,R=fk(q,g,'left')
    cup,cm=dominant_visible_surface(cup_cloud);red,rm=dominant_visible_surface(red_cloud)
    proxy=np.median(cup,axis=0);delta=np.asarray(start)-proxy
    white=hull2(cup[:,:2])-proxy[:2];occupied=hull2(red[:,:2]);box=np.asarray(box_cloud,float)
    top=box[box[:,2]>=np.quantile(box[:,2],.5)];blo,bhi=np.quantile(top[:,:2],[.02,.98],axis=0)
    surface_top=box[box[:,2]>=np.quantile(box[:,2],.8)]
    topz=float(np.median(surface_top[:,2]));bottom=float(cup[:,2].min())
    goalz=float(start[2]+topz+.020-bottom)
    nominal=np.asarray(nominal,float);cache={};records=[];choices=[]
    def state(x):
        key=np.asarray(x,float).tobytes()
        if key not in cache:
            full=q.copy();full[0]=x[:6];p,rotation=fk(full,g,'left')
            Q=Rotation.from_euler('z',x[6]).as_matrix();center=np.asarray(p)[:2]-(Q@delta)[:2]
            footprint=white@Q[:2,:2].T+center
            cache[key]=(full,np.asarray(p),np.asarray(rotation),Q,center,footprint)
        return cache[key]
    def equality(x):
        _,p,rotation,Q,_,_=state(x)
        return np.r_[(p[2]-goalz)*20,Rotation.from_matrix((Q@R)@rotation.T).as_rotvec()]
    def box_constraints(x):
        footprint=state(x)[5]
        return np.r_[footprint.min(axis=0)-blo-.010,bhi-.010-footprint.max(axis=0)]*20
    def objective(x):
        center=state(x)[4]
        return float(np.sum((x[:6]-q[0])**2)+.3*x[6]**2+.2*np.sum((center-nominal)**2)/.05**2)
    bounds=list(zip(np.asarray(limits.joint_min),np.asarray(limits.joint_max)))+[(-np.pi,np.pi)]
    seed=np.r_[q[0],0.]
    # Each outward red edge defines a legitimate disjoint halfspace. Solve the
    # joint goal continuously within that support relation, without motion.
    axes=list(normals(occupied))
    axes.sort(key=lambda n: -float((proxy[:2]-np.median(red[:,:2],axis=0))@n))
    for branch,n in enumerate(axes):
        redmax=float((occupied@n).max())
        def separation(x,n=n,redmax=redmax):return float((state(x)[5]@n).min()-redmax)*20
        constraints=[{'type':'eq','fun':equality},{'type':'ineq','fun':box_constraints},{'type':'ineq','fun':separation}]
        result=minimize(objective,seed,method='SLSQP',bounds=bounds,constraints=constraints,
                        options={'ftol':1e-9,'maxiter':maximum_iterations,'disp':False})
        x=result.x;full,p,rotation,Q,center,footprint=state(x)
        equal=float(np.max(np.abs(equality(x))));margin=float(np.min(box_constraints(x)));sep=float(separation(x))
        valid=equal<=1e-4 and margin>=-1e-6 and sep>=-1e-6
        records.append({'branch':branch,'optimizer_success':bool(result.success),'message':str(result.message),
                        'iterations':int(result.nit),'verified_equal_residual':equal,'verified_box_constraint':margin,
                        'verified_red_separation':sep,'feasible':valid})
        if valid:
            choices.append((objective(x),full,p,Q@R,center,footprint,x[6],n,redmax))
            # A current-state feasible solution is sufficient; no geometry/yaw
            # scan across centimetres, private poses, or simulator episodes.
            break
    report={'source':'fresh public dominant cup/red2–98% contours+box top+own FK/joints',
            'cup_component':cm,'red_component':rm,'proxy_center_m':proxy.tolist(),
            'box_lower_xy_m':blo.tolist(),'box_upper_xy_m':bhi.tolist(),'box_margin_m':.010,
            'observed_box_top_m':topz,'observed_cup_bottom_m':bottom,'release_clearance_m':.020,
            'nominal_visible_goal_xy_m':nominal.tolist(),'current_TCP_m':np.asarray(start).tolist(),
            'current_rotation':R.tolist(),'optimizer_branches':records,'FK_evaluations':len(cache),
            'frame_anchor_consistent':True,'hidden_COM_or_full_shape_verified':False,'NN_updated':False,
            'source_changed_during_episode':False,'cm_grid_search':False}
    if not choices:report.update(feasible=False,reason='no current joint/visible-support/yaw solution under original bounds');return None,report
    _,full,p,rotation,center,footprint,yaw,n,b=min(choices,key=lambda row:row[0])
    report.update(feasible=True,goal_q_rad=full.tolist(),goal_TCP_m=p.tolist(),goal_rotation=rotation.tolist(),
                  goal_visible_proxy_xy_m=center.tolist(),relative_yaw_deg=float(np.rad2deg(yaw)),
                  projected_cup_contour_xy_m=footprint.tolist(),red_exclusion_contour_xy_m=occupied.tolist(),
                  separating_axis=n.tolist(),separating_bound_m=b,
                  verified_box_clearances_m={'lower':(footprint.min(axis=0)-blo).tolist(),'upper':(bhi-footprint.max(axis=0)).tolist()})
    return (p,rotation),report
