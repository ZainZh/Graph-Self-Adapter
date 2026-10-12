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
def compile_task(parent,name,bundle=None):
    raw=copy.deepcopy(parent);cycles=PROFILES[name][1];raw['stages']=[s for s in raw['stages'] if s['cycle_id'] in cycles]
    graph=special_route(raw);graph['logical_task']=name;graph['variant_cycles']=list(cycles)
    for s in graph['stages']:
        if s['phase']=='Grasp':s.update(holding_contract='none',done=['own_pose','own_progress'],invariants=['finite'])
        if s['phase'] in ('Lift','Pull'):s.update(holding_contract='none',invariants=['finite'])
    for i,s in enumerate(graph['stages']):
        if s['phase']=='Move' and i>0 and graph['stages'][i-1]['phase']=='Grasp':
            s.update(holding_contract='none',invariants=['finite'],diagnostic_consequence_only=True)
    if name=='sort_one_yellow_cube':
        for s in graph['stages']:
            if s['phase']=='Wait':s['empty_gripper_clearance']=True
    if bundle is not None:
        import numpy as np
        stages=[]
        for s in graph['stages']:
            if s['phase']=='Approach':
                i=0 if s['arm']=='left' else 1;ref=s['reference'];times=bundle['times'];g=bundle['gripper'][:,i]
                initial=float(np.interp(ref['start_time_sec'],times,g));ids=np.where((times>ref['start_time_sec'])&(times<=ref['end_time_sec'])&(g<initial-.03))[0]
                if len(ids):
                    j=max(0,int(ids[0])-1);boundary=max(ref['start_time_sec'],float(times[j]))
                    if boundary>ref['start_time_sec'] and boundary<ref['end_time_sec']:
                        suffix=copy.deepcopy(s);suffix.update(stage_id=s['cycle_id']+'.contact_entry',phase='Grasp',controller='code_reference_retarget',holding_contract='none',done=['own_pose','own_progress'],invariants=['finite'],semantic_facts=[],boundary_reason='demonstration starts closing before old Approach end; explicit coordinated contact-entry suffix')
                        suffix['reference']['start_time_sec']=boundary;s['reference']['end_time_sec']=boundary
                        stages.extend([s,suffix]);continue
            stages.append(s)
        graph['stages']=stages
        for a,b in zip(stages,stages[1:]):a['next']=b['stage_id']
        stages[-1]['next']=None
    return graph
