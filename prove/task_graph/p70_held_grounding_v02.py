"""Bounded public dual-view held-book fit using a fixed native-book shape prior.
Revision on opened training diagnostics, not fresh or general 6D perception.
Stable held-book vertical correction is physically bounded; no new sensors.
No training, physics, current private poses, shape fitting or restarts.
"""
import itertools
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from prove.task_graph.p69_book_silhouette import project


def fit_held_book(public, prior, reference):
    prior=np.asarray(prior,float)
    if prior.shape!=(4,4) or not np.isfinite(prior).all():raise ValueError('Invalid public propagated prior')
    half=np.asarray(reference['box_halfsize']);center=np.asarray(reference['box_center'])
    local=center+np.asarray(list(itertools.product((-1,1),repeat=3)))*half
    limit=float(2*half.min());bounds=np.r_[limit,limit,float(half.min()/4),np.full(3,.35)]
    views=[]
    for v in range(2):
        masks=np.asarray(public['visible_masks'][v],bool);book=masks[1];occ=masks[0]|masks[2]
        contour=book&~binary_erosion(book)&~binary_dilation(occ,iterations=2)
        yy,xx=np.nonzero(contour)
        if len(xx)<25:continue
        ids=np.linspace(0,len(xx)-1,min(128,len(xx))).round().astype(int)
        observed=np.column_stack((xx[ids],yy[ids]))
        views.append((v,observed,distance_transform_edt(~book),distance_transform_edt(~occ)))
    if len(views)<2:return dict(valid=False,reason='two_qualified_views_required',qualified_views=len(views),truth_inputs=False)
    def pose(d):
        T=prior.copy();T[:3,3]+=d[:3];T[:3,:3]=Rotation.from_rotvec(d[3:]).as_matrix()@prior[:3,:3];return T
    def residual(d,include_prior=True):
        T=pose(d);world=local@T[:3,:3].T+T[:3,3];parts=[]
        for v,obs,dt,occ_dt in views:
            uv=project(world,public['camera_K'][v],public['camera_pose'][v]);hull=uv[ConvexHull(uv).vertices]
            a=hull;b=np.roll(hull,-1,axis=0);edge=b-a
            # Observed non-occluded boundary to nearest projected box segment.
            t=np.clip(((obs[:,None,:]-a)*edge).sum(2)/(edge*edge).sum(1),0,1)
            dist=np.linalg.norm(obs[:,None,:]-(a+t[:,:,None]*edge),axis=2).min(1)
            parts.append(dist/3/np.sqrt(len(dist)))
            # Fixed-count perimeter samples permit finite differences through hull changes.
            lengths=np.linalg.norm(edge,axis=1);cum=np.r_[0,np.cumsum(lengths)]
            s=(np.arange(128)+.5)*cum[-1]/128;idx=np.searchsorted(cum[1:],s)
            pts=a[idx]+((s-cum[idx])/lengths[idx])[:,None]*edge[idx]
            coords=np.array([pts[:,1],pts[:,0]])
            distance=map_coordinates(dt,coords,order=1,mode='constant',cval=448.)
            visible=np.clip(map_coordinates(occ_dt,coords,order=1,mode='constant',cval=0.)/2,0,1)
            parts.append(distance*visible/3/np.sqrt(128))
        if include_prior:parts.append(.02*d/bounds)
        return np.concatenate(parts)
    initial=residual(np.zeros(6),False)
    fit=least_squares(residual,np.zeros(6),bounds=(-bounds,bounds),max_nfev=100,diff_step=1e-4,x_scale=bounds)
    final=residual(fit.x,False)
    return dict(valid=bool(fit.success),reason='converged' if fit.success else 'bounded_fit_not_converged',body_pose=pose(fit.x).tolist(),prior_body_pose=prior.tolist(),delta=fit.x.tolist(),bounds=bounds.tolist(),nfev=int(fit.nfev),initial_objective=float(initial@initial),final_objective=float(final@final),active_bounds=fit.active_mask.tolist(),qualified_views=len(views),truth_inputs=False,shape_fitted=False,restarts=0,method='dual_direction_contour_distances_occlusion_excluded_plus_light_prior')
