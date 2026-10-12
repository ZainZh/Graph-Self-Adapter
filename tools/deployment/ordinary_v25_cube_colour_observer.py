"""Public colour/body-depth observer in full possible own finger CAD envelope; not SAM."""
import numpy as np
from scipy.ndimage import label
from prove.xtrainer.io import SafetyError

def prototype(rgb,mask):
    pixels=np.asarray(rgb)[np.asarray(mask,bool)].astype(float);pixels=pixels[pixels.sum(1)>30]
    if len(pixels)<12:raise SafetyError('initial actual SAM colour prototype sparse')
    return np.median(pixels/pixels.sum(1,keepdims=True),axis=0)

def observe(rgb,depth,K,T,proto,tcp,R,pad,height,plane,travel=.043,anchor=None):
    arr=np.asarray(rgb,float);total=arr.sum(2);chroma=arr/np.maximum(total[:,:,None],1)
    mask=(total>30)&(np.linalg.norm(chroma-proto,axis=2)<.12)&np.isfinite(depth)&(depth>.05)&(depth<3)
    yy,xx=np.where(mask);z=depth[yy,xx];pts=np.c_[(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z]@T[:3,:3].T+T[:3,3]
    vertices=np.asarray(pad['pad_vertices_TCP_local_m']);axis=np.asarray(pad['closing_axis_TCP_local']);allv=np.concatenate([vertices,vertices+travel*axis,vertices-travel*axis])@R.T+tcp
    lo=allv.min(0)-height;hi=allv.max(0)+height;inside=np.all((pts>=lo)&(pts<=hi),axis=1)
    n=np.asarray(plane['support_plane_normal_left_base']);base=np.asarray(plane['support_plane_point_left_base_m']);floor=base[2]-((pts[:,:2]-base[:2])@n[:2])/n[2]
    inside &= pts[:,2]-floor>=height+.010
    actual=np.zeros(depth.shape,bool);actual[yy[inside],xx[inside]]=True;components,count=label(actual,np.ones((3,3)));candidates=[]
    for i in range(1,count+1):
        y,x=np.where(components==i)
        if len(x)<12:continue
        z=depth[y,x];p=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]@T[:3,:3].T+T[:3,3];point=np.median(p,axis=0)
        candidates.append((point,len(x),[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1],components==i))
    diagnostic={'mask_source':'public_RGBD_colour_in_actor_roi','actual_points_after_CAD_and_floor_filter':int(actual.sum()),'qualified_component_count':len(candidates),'minimum_real_component_points':12,'actor_all_gripper_g0_to1_CAD_roi_min_m':lo.tolist(),'actor_all_gripper_g0_to1_CAD_roi_max_m':hi.tolist(),'no_interpolated_points':True,'not_SAM_mask':True}
    if not candidates:return None,diagnostic,None
    target=tcp if anchor is None else anchor;candidates.sort(key=lambda c:np.linalg.norm(c[0]-target))
    if len(candidates)>1 and abs(np.linalg.norm(candidates[0][0]-target)-np.linalg.norm(candidates[1][0]-target))<.005:
        diagnostic['ambiguity']='same-colour components cannot be associated uniquely';return None,diagnostic,None
    point,pixels,box,mask=candidates[0];diagnostic.update(visible_surface_point_m=point.tolist(),component_pixels=pixels,component_box=box,point_is_visible_surface_not_body_center=True)
    return point,diagnostic,mask
