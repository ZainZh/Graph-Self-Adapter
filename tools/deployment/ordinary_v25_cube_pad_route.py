"""Same provisional observer schema with explicit distal/body-center geometry and real agent identity."""
from ordinary_v25_cube_observer_route import observer_route as parent_route
ACTUAL_OBSERVER='Codex stage observer /root/apple_cube_v25'
def observer_route(graph):
    route=parent_route(graph);route['actual_codex_observer_identity']=ACTUAL_OBSERVER;route['grasp_geometry']='fresh visible cube top face center and observed height plus true robot distal flat pad overlap; no object mesh truth'
    for stage in route['stages']:
        stage['object_part']='observed_cube_body_center_at_true_distal_flat_pad_contact'
        stage['semantic_observer_fallback_identity']=ACTUAL_OBSERVER
    return route
