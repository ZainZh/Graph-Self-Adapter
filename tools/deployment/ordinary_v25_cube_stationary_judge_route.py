from ordinary_v25_cube_baseline_vertical_route import observer_route as parent_route
def observer_route(graph):
    route=parent_route(graph);route['semantic_provider']='actual Codex stage observer /root/apple_cube_v25 direct, no automatic API'
    for stage in route['stages']:
        if stage['phase']=='Lift':stage.update(active_plan_stream_budget_sec=60,stationary_judge_wait_budget_sec=300,provisional_maximum_wall_sec=None,no_next_motion_until_fresh_positive_verification=True)
    return route
