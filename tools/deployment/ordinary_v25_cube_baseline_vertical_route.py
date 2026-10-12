from ordinary_v25_cube_vertical_route import observer_route as parent_route
def observer_route(graph):
    route=parent_route(graph);route['grasp_geometry']='original frozen task baseline paired static source cue; no new body-centered pad shift';route['height_geometry']='current measured support plane at each observed object XY in left-base frame, plus fixed gravity-axis Lift clearance'
    for stage in route['stages']:stage['object_part']='original_baseline_grasp_region_verified_only_by_current_Lift'
    return route
