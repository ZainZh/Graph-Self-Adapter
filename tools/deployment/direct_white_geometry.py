"""Observed upright support geometry; no object CAD, pose truth or contact input."""
import numpy as np
from prove.xtrainer.io import SafetyError


def upright_support_prior(cup_cloud,handle_point,table_cloud):
    cup=np.asarray(cup_cloud);table=np.asarray(table_cloud);handle=np.asarray(handle_point)
    if len(cup)<12 or len(table)<12 or handle.shape!=(3,) or not np.isfinite(cup).all():raise SafetyError('initial upright observation unknown')
    center=np.median(cup,axis=0);table_z=float(np.median(table[:,2]));low,high=np.quantile(cup[:,2],[.05,.95])
    return {'handle_minus_cup_visible_m':(handle-center).tolist(),'upright_handle_above_table_m':float(handle[2]-table_z),
            'observed_visible_height_m':float(high-low),'visible_extent_radius_m':float(np.quantile(np.linalg.norm(cup-handle,axis=1),.99)),
            'initial_table_z_m':table_z,'source':'initial upright actual SAM cup/handle/table observation; partial view estimated',
            'airborne_attachment_not_used_as_support_target':True}


def direct_targets(prior,box_cloud,box_anchor,reference_final_white,reference_static_box):
    box=np.asarray(box_cloud);anchor=np.asarray(box_anchor)
    if len(box)<12 or anchor.shape!=(3,) or not np.isfinite(box).all():raise SafetyError('actual SAM box support unknown')
    # Robust upper face; source is current predicted box cloud, not object mesh.
    z=box[:,2];lo,hi=np.quantile(z,[.8,.95]);face=box[(z>=lo)&(z<=hi)];top=float(np.median(face[:,2]))
    delta=np.asarray(reference_final_white)[:2]-np.asarray(reference_static_box)[:2]
    cup_xy=anchor[:2]+delta;handle_xy=cup_xy+np.asarray(prior['handle_minus_cup_visible_m'])[:2]
    support=np.r_[handle_xy,top+prior['upright_handle_above_table_m']]
    clearance=top+prior['visible_extent_radius_m']+.25*prior['observed_visible_height_m']
    return {'handle_support_goal_m':support.tolist(),'transport_handle_height_m':float(clearance),'observed_box_top_z_m':top,
            'white_visible_goal_xy_m':cup_xy.tolist(),'declared_white_relative_box_xy_m':delta.tolist(),
            'reference_box_anchor_contract':'declared scene-initial fixed visible box anchor paired with declared final white goal; not occlusion-shifted reference box',
            'support_relationship_source':'locked initial upright SAM handle/table/cup relation; no airborne a_actual target'}


def tcp_at_handle(handle_point,rotation,pad_local):return (np.asarray(handle_point)-np.asarray(rotation)@np.asarray(pad_local),np.asarray(rotation).copy())
