from ordinary_v25_cube_stationary_judge_route import observer_route as parent_route
def observer_route(graph):
    route=parent_route(graph);route['perception_contract']='initial actual SAM plus explicit current public RGBD colour_in_actor_roi after camera-gap inspection; min12 true depth pixels, local support plane elevated, current visual co-motion'
    for stage in route['stages']:
        if stage['phase']=='Lift':stage['execution_subnodes']=[{'stage_id':stage['cycle_id']+'.inspect_hold','phase':'InspectHold','actor':stage['actor'],'object':stage['object'],'target':'camera_visible_finger_gap','entry':'own pure-Lift complete; grasp still provisional','goal':'closing axis perpendicular to calibrated camera LOS, shortest reachable world-up yaw; actual TCP position/closedg preserved','holding':'tentative information-gathering permission only','budget_active_sec':60,'failure':'IK/feedback limit or no12actualbodypoints or semanticUNKNOWN; neverMove'}]
    route['total_declared_nodes_including_subnodes']=45
    return route
