"""Solve current visible cup translation into box support excluding current red.
The feasible set is computed from contour inequalities and separating axes;
there is no centimetre grid search or hidden object/COM input.
"""
import numpy as np
from scipy.spatial import ConvexHull
from cup_v25_visible_support_geometry import dominant_visible_surface
from prove.xtrainer.io import SafetyError

BOX_MARGIN_M=.010

def hull2(points):
    points=np.asarray(points,float)
    if len(points)<12:raise SafetyError('visible support contour insufficient')
    low,high=np.quantile(points,[.02,.98],axis=0)
    keep=points[(points>=low).all(axis=1)&(points<=high).all(axis=1)]
    if len(keep)<12:raise SafetyError('robust visible contour insufficient')
    return keep[ConvexHull(keep).vertices]

def clip(poly,n,b):
    out=[]
    for i,current in enumerate(poly):
        previous=poly[i-1];dc=float(n@current-b);dp=float(n@previous-b)
        if (dc<=1e-10)!=(dp<=1e-10):out.append(previous+(current-previous)*(dp/(dp-dc)))
        if dc<=1e-10:out.append(current)
    return np.asarray(out,float).reshape(-1,2)

def nearest(poly,point):
    point=np.asarray(point,float)
    if all(np.cross(poly[(i+1)%len(poly)]-poly[i],point-poly[i])>=-1e-9 for i in range(len(poly))):return point.copy()
    rows=[]
    for i,a in enumerate(poly):
        b=poly[(i+1)%len(poly)];d=b-a;t=np.clip((point-a)@d/max(d@d,1e-20),0,1);rows.append(a+t*d)
    return min(rows,key=lambda x:float(np.sum((x-point)**2))).copy()

def normals(poly):
    for i,a in enumerate(poly):
        d=poly[(i+1)%len(poly)]-a
        n=np.array([d[1],-d[0]])
        if np.linalg.norm(n)>1e-8:yield n/np.linalg.norm(n)

def plan_support(cup_cloud,box_cloud,red_cloud,nominal):
    cup,cm=dominant_visible_surface(cup_cloud);red,rm=dominant_visible_surface(red_cloud)
    box=np.asarray(box_cloud,float)
    if len(box)<200 or not np.isfinite(box).all():raise SafetyError('current box top unavailable')
    top=box[box[:,2]>=np.quantile(box[:,2],.5)]
    boxlo,boxhi=np.quantile(top[:,:2],[.02,.98],axis=0)
    center=np.median(cup,axis=0);white=hull2(cup[:,:2])-center[:2];occupied=hull2(red[:,:2])
    wlo,whi=white.min(axis=0),white.max(axis=0)
    low=boxlo+BOX_MARGIN_M-wlo;high=boxhi-BOX_MARGIN_M-whi
    report={'source':'current actual SAM+RGBD dominant 2–98percent visible contours; no object truth/COM',
            'box_margin_m':BOX_MARGIN_M,'cup_component':cm,'red_component':rm,
            'visible_cup_center_m':center.tolist(),'white_relative_contour_xy_m':white.tolist(),
            'red_exclusion_contour_xy_m':occupied.tolist(),'box_top_lower_xy_m':boxlo.tolist(),'box_top_upper_xy_m':boxhi.tolist(),
            'translation_lower_xy_m':low.tolist(),'translation_upper_xy_m':high.tolist(),'nominal_visible_goal_xy_m':np.asarray(nominal).tolist(),
            'hidden_full_shape_or_COM_verified':False,'continuous_contour_solution_not_cm_search':True}
    if np.any(low>=high):report.update(feasible=False,reason='whole observed cup envelope cannot fit box with original10mm margin');return None,report
    rectangle=np.array([[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]])
    choices=[]
    for n in list(normals(white))+list(normals(occupied)):
        # Non-overlap of convex projected contours is equivalent to at least
        # one separating-axis halfspace. Exact polygon clipping gives each
        # feasible region; project the nominal relation into it continuously.
        wp=white@n;rp=occupied@n
        for direction,bound in [(n,float(rp.min()-wp.max())),(-n,float(wp.min()-rp.max()))]:
            poly=clip(rectangle,direction,bound)
            if len(poly)<3:continue
            candidate=nearest(poly,nominal)
            separation=float(bound-direction@candidate)
            if separation>=-1e-8:choices.append((float(np.sum((candidate-nominal)**2)),candidate,poly,direction,bound))
    if not choices:report.update(feasible=False,reason='conservative observed cup/red contours have no disjoint supported placement');return None,report
    report['feasible_translation_regions_xy_m']=[row[2].tolist() for row in choices]
    _,target,region,n,b=min(choices,key=lambda row:row[0])
    placed=white+target
    report.update(feasible=True,safe_object_goal_xy_m=target.tolist(),correction_xy_m=(target-nominal).tolist(),
                  projected_cup_contour_xy_m=placed.tolist(),feasible_region_xy_m=region.tolist(),
                  separating_axis=n.tolist(),separating_bound_m=b,
                  verified_visible_box_clearances_m={'lower':(placed.min(axis=0)-boxlo).tolist(),'upper':(boxhi-placed.max(axis=0)).tolist()},
                  red_exclusion_verified_for_observed_contours=True)
    return target,report
