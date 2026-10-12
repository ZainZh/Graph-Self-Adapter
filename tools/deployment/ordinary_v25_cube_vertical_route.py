from ordinary_v25_cube_pad_route import observer_route as parent_route
def observer_route(graph):
    route=parent_route(graph);route['Lift_relation']='own accepted Grasp TCP + current observed cube height +10mm gravity clearance, constant XY/R; no demonstrated lateral sweep'
    for stage in route['stages']:
        if stage['phase']=='Lift':stage['controller_geometry']='gravity_vertical_from_actual_Grasp_XY_R';stage['object_relation_goal']='clear_of_table_then_fresh_visual_holding_verified'
    return route
