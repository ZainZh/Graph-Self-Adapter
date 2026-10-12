"""Local functional-part candidate: observed handle tube + robot-only pad CAD.

No object mesh, body pose, segmentation truth or contact is read. Hidden tube
backside is an explicit geometric hypothesis for Root review, not a measurement.
"""
import numpy as np
from prove.xtrainer.io import SafetyError


def observed_patch_cloud(depth,mask,K,T,box):
    depth=np.asarray(depth);mask=np.asarray(mask,bool);K=np.asarray(K);T=np.asarray(T)
    if mask.shape!=depth.shape or K.shape!=(3,3) or T.shape!=(4,4):raise SafetyError('invalid handle RGBD geometry')
    x0,y0,x1,y1=map(int,box)
    if not (0<=x0<x1<=depth.shape[1] and 0<=y0<y1<=depth.shape[0]):raise SafetyError('handle patch outside image')
    roi=np.zeros_like(mask);roi[y0:y1,x0:x1]=True
    y,x=np.where(mask&roi&np.isfinite(depth)&(depth>.05)&(depth<3))
    if len(x)<12:raise SafetyError('visible handle patch depth insufficient')
    z=depth[y,x];points=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]
    return points@T[:3,:3].T+T[:3,3]


def tube_relation(points,camera_position,*,minimum_span_m=.008,maximum_radial_extent_m=.016):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<12 or not np.isfinite(points).all():raise SafetyError('invalid observed tube cloud')
    center=np.median(points,axis=0);_,singular,V=np.linalg.svd(points-center,full_matrices=False);tangent=V[0]
    if tangent[2]>0:tangent=-tangent # declared downward insertion axis
    span=float(np.ptp((points-center)@tangent));radial=(points-center)-np.outer((points-center)@tangent,tangent)
    extent=float(np.quantile(np.linalg.norm(radial,axis=1),.9))
    if span<minimum_span_m or singular[0]<1.5*max(singular[1],1e-9) or extent>maximum_radial_extent_m:
        raise SafetyError('Root patch does not support a narrow tube hypothesis')
    closing=np.asarray(camera_position)-center;closing-=tangent*np.dot(closing,tangent)
    if np.linalg.norm(closing)<1e-5:raise SafetyError('tube viewing axis ambiguous')
    closing/=np.linalg.norm(closing)
    return {'center':center,'tangent_down':tangent,'closing_axis':closing,'span_m':span,'radial_extent_m':extent,
            'backside_observed':False,'hypothesis':'opposed pad clamp across manually grounded visible outer tube; hidden backside unverified'}


def pad_relation_from_robot_mesh(model,data,arm='left'):
    """Robot mesh vertices only; inner 15% of each named finger pad surface."""
    tcp=data.site(arm+'_tcp');R=tcp.xmat.reshape(3,3);clouds=[]
    for suffix in ('J1_7_03_00_collision','J1_8_03_00_collision'):
        geom=model.geom(arm+'_'+suffix);mesh=int(geom.dataid[0]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
        if mesh<0 or count<3:raise SafetyError('robot finger CAD unavailable')
        vertices=model.mesh_vert[start:start+count].copy();gid=geom.id
        world=vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
        clouds.append((world-tcp.xpos)@R)
    direction=R.T@data.xaxis[model.joint(arm+'_gripper1').id];direction/=np.linalg.norm(direction)
    if np.dot(direction,np.mean(clouds[1],axis=0)-np.mean(clouds[0],axis=0))<0:direction=-direction
    faces=[]
    for index,cloud in enumerate(clouds):
        projection=cloud@direction;threshold=np.quantile(projection,.85 if index==0 else .15)
        faces.append(np.mean(cloud[projection>=threshold] if index==0 else cloud[projection<=threshold],axis=0))
    return {'pad_midpoint_TCP_local_m':(faces[0]+faces[1])/2,'closing_axis_TCP_local':direction,
            'inner_pad_points_TCP_local_m':np.array(faces),'pad_vertices_TCP_local_m':np.concatenate(clouds),'source':'robot named collision pad mesh inner-face CAD; not current object truth'}


def candidate_goal(relation,pads):
    closing=relation['closing_axis'];insertion=relation['tangent_down']
    # Map robot tool +Z insertion and CAD jaw closing direction to observed tube axes.
    local_z=np.array([0.,0.,1.]);local_x=pads['closing_axis_TCP_local'].copy();local_x-=local_z*np.dot(local_x,local_z);local_x/=np.linalg.norm(local_x)
    local_y=np.cross(local_z,local_x);world_y=np.cross(insertion,closing)
    R=np.column_stack((closing,world_y,insertion))@np.column_stack((local_x,local_y,local_z)).T
    tcp=relation['center']-R@pads['pad_midpoint_TCP_local_m']
    return tcp,R


def lift_goal_from_visible_height(pose,cloud,table_points,*,clearance_fraction=.35):
    """Lift enough for observed lower surface to clear observed tabletop by a fraction of object visible height."""
    cloud=np.asarray(cloud);table=np.asarray(table_points)
    if len(cloud)<12 or len(table)<12:raise SafetyError('visible object/table height unknown')
    table_z=float(np.median(table[:,2]));low,high=np.quantile(cloud[:,2],[.05,.95]);height=float(high-low)
    if height<.01 or height>.2:raise SafetyError('observed cup height invalid')
    rise=max(0.,table_z+clearance_fraction*height-low)
    if rise>.12:raise SafetyError('height-derived lift outside local candidate scope')
    p=np.asarray(pose[0]).copy();p[2]+=rise
    return (p,np.asarray(pose[1]).copy()),{'observed_table_z_m':table_z,'observed_low_z_m':float(low),'observed_height_m':height,'requested_rise_m':rise,'clearance_fraction':clearance_fraction}
