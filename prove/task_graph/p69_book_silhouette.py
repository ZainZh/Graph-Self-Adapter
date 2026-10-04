"""Single bounded native-book box fit from view0 silhouette; not generic 3D vision.

Runtime inputs are current visible mask and camera calibration only. The shape,
settled height, source pose and raster support bias are fixed demo/train priors.
"""
import itertools
import numpy as np
from scipy.optimize import least_squares

DIRECTIONS=np.column_stack((np.cos(np.arange(16)*2*np.pi/16),np.sin(np.arange(16)*2*np.pi/16)))


def rz(yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def project(points,K,T_world_camera):
    points=np.asarray(points,float);K=np.asarray(K,float);T=np.asarray(T_world_camera,float)
    if K.shape!=(3,3) or T.shape!=(4,4) or not all(np.isfinite(a).all() for a in (points,K,T)):raise ValueError('Invalid public calibration')
    camera=(points-T[:3,3])@T[:3,:3]
    optical=camera*np.array([1,-1,-1])
    if np.any(optical[:,2]<=0):raise ValueError('Book behind camera')
    pixels=optical@K.T
    return pixels[:,:2]/pixels[:,2:]


def mask_support(mask):
    mask=np.asarray(mask,bool)
    if mask.ndim!=2 or mask.sum()<25:raise ValueError('Missing/small visible book mask')
    yy,xx=np.nonzero(mask)
    return (np.column_stack((xx,yy))@DIRECTIONS.T).max(0)


def corners(reference,delta):
    center=np.asarray(reference['box_center'],float);half=np.asarray(reference['box_halfsize'],float)
    source=np.asarray(reference['source_body_pose'],float);delta=np.asarray(delta,float)
    if source.shape!=(4,4) or center.shape!=(3,) or half.shape!=(3,) or np.any(half<=0):raise ValueError('Invalid fixed native box prior')
    local=center+np.asarray(list(itertools.product((-1,1),repeat=3)))*half
    rotation=rz(delta[2])@source[:3,:3];position=source[:3,3]+np.r_[delta[:2],0.]
    return local@rotation.T+position


def projected_support(reference,delta,K,T_world_camera):
    return (project(corners(reference,delta),K,T_world_camera)@DIRECTIONS.T).max(0)


def calibrate(mask,K,T_world_camera,source_body_pose,box_center,box_halfsize):
    reference=dict(schema='prove.p69-native-book-silhouette.v1',box_center=list(box_center),box_halfsize=list(box_halfsize),
        source_body_pose=np.asarray(source_body_pose,float).tolist(),directions=DIRECTIONS.tolist(),
        scope='Fixed native upright book shape and settled demo Z; not general 3D perception.',
        fit=dict(bounds=[.15,.15,float(np.pi/3)],max_nfev=80,variables=['dx_m','dy_m','yaw_rad'],method='trf',initialization='single planar estimate; no grid/random restarts'))
    reference['support_bias_pixels']=(mask_support(mask)-projected_support(reference,np.zeros(3),K,T_world_camera)).tolist()
    return reference


def fit_book_pose(mask,K,T_world_camera,reference,initial_delta=(0.,0.,0.)):
    if reference.get('schema')!='prove.p69-native-book-silhouette.v1':raise ValueError('Wrong native book reference')
    if not np.allclose(reference['directions'],DIRECTIONS):raise ValueError('Support directions changed')
    try:observed=mask_support(mask)
    except ValueError:
        return dict(valid=False,success=False,reason='missing_or_small_visible_book_mask',truth_inputs=False,search_restarts=0,nfev=0)
    bias=np.asarray(reference['support_bias_pixels'],float)
    start=np.asarray(initial_delta,float);bounds=np.array([.15,.15,np.pi/3])
    if start.shape!=(3,) or not np.isfinite(start).all():raise ValueError('Invalid planar initialization')
    start=np.clip(start,-bounds+1e-10,bounds-1e-10)
    residual=lambda d:projected_support(reference,d,K,T_world_camera)+bias-observed
    fit=least_squares(residual,start,bounds=(-bounds,bounds),max_nfev=80,method='trf')
    if not fit.success:
        return dict(valid=False,success=False,reason='bounded_fit_not_converged',nfev=int(fit.nfev),truth_inputs=False,search_restarts=0,residual_rms_pixels=float(np.sqrt(np.mean(fit.fun**2))))
    source=np.asarray(reference['source_body_pose'],float);current=source.copy()
    current[:3,:3]=rz(fit.x[2])@source[:3,:3];current[:2,3]+=fit.x[:2]
    delta_world=current@np.linalg.inv(source)
    return dict(valid=True,reason='ok',body_xy=current[:2,3].tolist(),body_yaw=float(np.arctan2(current[1,0],current[0,0])),
        delta_xy=fit.x[:2].tolist(),delta_yaw=float(fit.x[2]),source_xy=source[:2,3].tolist(),
        body_pose=current.tolist(),delta_world4=delta_world.tolist(),success=bool(fit.success),nfev=int(fit.nfev),
        initial_delta=start.tolist(),residual_rms_pixels=float(np.sqrt(np.mean(fit.fun**2))),
        residual_max_pixels=float(np.abs(fit.fun).max()),active_bounds=fit.active_mask.tolist(),
        truth_inputs=False,search_restarts=0)
