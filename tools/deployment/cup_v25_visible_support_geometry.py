"""Current observed cup surface/TCP support planning, independent of jaw sensors."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from prove.xtrainer.io import SafetyError

NEIGHBOR_RADIUS_M=.006
MINIMUM_POINTS=200
MINIMUM_DOMINANT_FRACTION=.70
SUPPORT_CLEARANCE_M=.010

def dominant_visible_surface(points):
    cloud=np.asarray(points,float)
    if cloud.ndim!=2 or cloud.shape[1]!=3 or len(cloud)<MINIMUM_POINTS or not np.isfinite(cloud).all():
        raise SafetyError('current cup visible surface insufficient')
    pairs=cKDTree(cloud).query_pairs(NEIGHBOR_RADIUS_M,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(cloud),len(cloud)))
    count,labels=connected_components(graph,directed=False)
    sizes=np.bincount(labels);surface=cloud[labels==sizes.argmax()]
    if len(surface)<MINIMUM_POINTS or len(surface)/len(cloud)<MINIMUM_DOMINANT_FRACTION:
        raise SafetyError('current cup surface disconnected/ambiguous')
    return surface,{'input_points':len(cloud),'dominant_points':len(surface),'component_count':int(count),
                    'dominant_fraction':len(surface)/len(cloud),'neighbor_radius_m':NEIGHBOR_RADIUS_M}

def support_geometry(cup_cloud,box_cloud,tcp,rotation):
    surface,report=dominant_visible_surface(cup_cloud)
    box=np.asarray(box_cloud,float)
    if box.ndim!=2 or box.shape[1]!=3 or len(box)<200 or not np.isfinite(box).all():
        raise SafetyError('current box support surface insufficient')
    top=box[box[:,2]>=np.quantile(box[:,2],.8)]
    top_height=float(np.median(top[:,2]))
    # The observed minimum is retained as a measured visible boundary, not
    # inferred hidden bottom/contact. Root visibility qualification is separate.
    lowest=float(surface[:,2].min())
    report.update(current_TCP_m=np.asarray(tcp).tolist(),current_rotation=np.asarray(rotation).tolist(),
                  current_box_top_m=top_height,lowest_observed_cup_surface_m=lowest,
                  observed_cup_to_TCP_bottom_offset_m=lowest-float(np.asarray(tcp)[2]),
                  visible_cup_surface_z_quantiles_m=np.quantile(surface[:,2],[0,.01,.05,.5,.95,1]).tolist(),
                  visible_cup_xy_quantiles_m=np.quantile(surface[:,:2],[.02,.5,.98],axis=0).tolist(),
                  target_visible_surface_clearance_m=SUPPORT_CLEARANCE_M,
                  proposed_lower_delta_z_m=top_height+SUPPORT_CLEARANCE_M-lowest,
                  source='fresh actual SAM mask + metric RGBD + own robot TCP',
                  contact_observed=False,hidden_bottom_inferred=False,original_final_goal_unchanged=True)
    return report
