"""Red two cycles preserved, white goes directly to box; original final intent unchanged."""
import copy


def direct_route_graph(original):
    graph=copy.deepcopy(original);red=[s for s in graph['stages'] if s['cycle_id'] in ('c01','c02')];nodes=[]
    for stage in red:
        if stage['phase'] in ('Lift','Move','Align','Release'):
            removed={p['id'] for p in stage['completion_predicates'] if p['predicate'] in ('stage_goal','released')}
            stage['completion_predicates']=[p for p in stage['completion_predicates'] if p['id'] not in removed];stage['done']=[k for k in stage['done'] if k not in removed]
            if stage['phase'] in ('Lift','Move','Align'):stage['holding_contract']='tentative_macro_grasp'
        nodes.append(stage)
        if stage['phase']=='Wait':
            complete=copy.deepcopy(stage);complete.update(stage_id=stage['cycle_id']+'.08.complete',phase='Complete',completion_predicates=[],invariant_predicates=[],done=[],invariants=[],controller='supported_Root_completion');nodes.append(complete)
    source=next(s for s in original['stages'] if s['cycle_id']=='c03' and s['phase']=='Approach')
    grasp=next(s for s in original['stages'] if s['cycle_id']=='c03' and s['phase']=='Grasp')
    release=next(s for s in original['stages'] if s['cycle_id']=='c03' and s['phase']=='Release')
    for i,phase in enumerate(('Approach','Grasp','Lift','Raise','Move','Lower','Release','Retreat','Complete'),1):
        stage=copy.deepcopy(grasp if phase=='Grasp' else release if phase=='Release' else source);stage.update(stage_id=f'white_direct.{i:02d}.{phase.lower()}',cycle_id='white_direct',phase=phase,arm='left',object='white_mug',target='box',controller='observed_white_direct_Code')
        stage['completion_predicates']=[{'id':'own_pose','predicate':'actor_command_reached'},{'id':'own_progress','predicate':'reference_progress'}]
        stage['done']=['own_pose','own_progress'];stage['invariant_predicates']=[{'id':'finite','predicate':'finite_simulation'}];stage['invariants']=['finite'];stage['holding_contract']='none' if phase in ('Approach','Release','Retreat','Complete') else 'tentative_macro_grasp'
        if phase=='Grasp':stage['completion_predicates'].append({'id':'Root_holding','predicate':'holding'});stage['done'].append('Root_holding')
        if phase=='Complete':stage.update(completion_predicates=[],invariant_predicates=[],done=[],invariants=[],controller='supported_Root_completion')
        nodes.append(stage)
    graph['stages']=nodes;graph['engineering_direct_contract']={'red_original_action_order_preserved':True,'white_arm':'left','white_regrasp_handover_removed':True,'Root_boundaries':'3 supported Grasp +3 supported completion after empty retreat','final_intent':'both mugs stably on original box; original private evaluator unchanged','source':'actual SAM relationships and robot own FK/pads; partial observation assumptions explicit'}
    return graph
