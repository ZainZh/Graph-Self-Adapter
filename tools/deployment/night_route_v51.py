import copy
from multitask_special_route import special_route
PROFILES={
'open_drawer_only':('open_drawer_and_place_object',('c01',)),
'hang_single_mug':('hanging_mug',('c01',)),
'nest_two_cups':('stack_cups',('c01',)),
'sort_one_yellow_cube':('sorting_cube_bimanual',('c01','c02')),
'sweep_single_block':('sweep_objects',('c01','c02')),
'put_knife_board_new':('put_knife_board_new',('c01',)),
'collect_single_cube':('collect_cubes',('c01',))}
def compile_task(parent,name):
    raw=copy.deepcopy(parent);cycles=PROFILES[name][1];raw['stages']=[s for s in raw['stages'] if s['cycle_id'] in cycles]
    graph=special_route(raw);graph['logical_task']=name;graph['variant_cycles']=list(cycles)
    for s in graph['stages']:
        if s['phase']=='Grasp':s.update(holding_contract='none',done=['own_pose','own_progress'],invariants=['finite'])
        if s['phase'] in ('Lift','Pull'):s.update(holding_contract='none',invariants=['finite'])
    for i,s in enumerate(graph['stages']):
        if s['phase']=='Move' and i>0 and graph['stages'][i-1]['phase']=='Grasp':
            s.update(holding_contract='none',invariants=['finite'],diagnostic_consequence_only=True)
    return graph
