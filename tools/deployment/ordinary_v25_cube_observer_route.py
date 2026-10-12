"""Explicit provisional Grasp then fresh observed Lift route; same actions and original goal."""
from multitask_ordinary_route import ordinary_route
SCHEMA='graph-self-adapter.ordinary-provisional-observer-route.v26'
def observer_route(parent):
    graph=ordinary_route(parent);graph['schema']=SCHEMA
    graph['observation_contract']='Grasp supported visual permission is provisional for exactly one immediately following bounded Lift. Fresh visible elevated holding plus real own nonzero motion must verify before Move; UNKNOWN stops.'
    for stage in graph['stages']:
        stage['holding_contract']='none' if stage['phase'] in ('Approach','Release','Wait','Complete') else 'visual_provisional_then_lift_verified'
        stage['object_part']='visible_cube_body_between_opposed_finger_pads'
        if stage['phase']=='Grasp':stage['postreply_guard']='fresh robot joint/TCP and static target; manipulated SAM centroid occlusion is uncertainty, not rigid drift truth'
        if stage['phase']=='Lift':stage['provisional_maximum_wall_sec']=60;stage['semantic_facts']=['holding'];stage['verification_before_next']='fresh actual RGB-D/SAM elevated object with own nonzero TCP motion; UNKNOWN stops'
    return graph
