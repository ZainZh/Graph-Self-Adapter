from ordinary_v25_cube_stationary_judge_route import observer_route as parent_route
def observer_route(graph):
    route=parent_route(graph);route['Lift_perception']='one actual Codex visible current body bbox re-prompts SAME existing actual SAM role ID after motion; other IDs/masks retained, real fresh RGBD then visual holding and original drift guard; no objectpoint fabrication'
    for stage in route['stages']:
        if stage['phase']=='Lift':stage['current_body_SAM_relocalization_budget']=1
    return route
