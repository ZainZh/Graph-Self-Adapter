"""Pure interface tests: no simulator, checkpoints, or scientific evidence."""
from dataclasses import replace
import pytest
from prove.task_graph import TaskGraph, GroundedSkill, RoleBinding, InputReference, GraphExecutor, RuleRouter


def document(skills=('Move', 'Move')):
    return {'schema_version': 'prove.task_graph.v1', 'graph_id': 'fixture', 'task_id': 'fixture_task',
            'entry_stage': 's0', 'supervisor_inputs': ['ready', 'done', 'held', 'wrong_target', 'object_lost', 'local_error', 'local_failure'],
            'nodes': [dict(id=f's{i}', skill=skill, actor='robot', object='object', object_part=None,
                target=None, goal_relation='fixture_relation', preconditions=[dict(signal='ready',op='eq',value=True)],
                invariants=[], termination=[dict(signal='done',op='eq',value=True)],
                next_stage=f's{i+1}' if i+1<len(skills) else None,
                evidence=dict(source='synthetic_fixture',start_step=i,end_step=i+1,provenance='unit_test_only'),
                step_budget=4,no_progress_budget=2) for i,skill in enumerate(skills)]}


def ground(graph, index=0):
    stage = graph.nodes[index]
    roles = {}
    for name in ('actor','object','object_part','target'):
        ref = getattr(stage,name)
        visible = ref is not None and name != 'actor'
        roles[name] = RoleBinding(ref, ref is not None,
            mask=InputReference('fixture_mask','vision','unit_test') if visible else None,
            visible=True if visible else None)
    return GroundedSkill.bind(stage, roles, graph_id=graph.graph_id, task_id=graph.task_id,instance_id='case0')


def setup(skills=('Move',), doc=None):
    graph = TaskGraph.from_dict(doc or document(skills))
    router = RuleRouter()
    calls = []
    for skill in set(skills):
        router.register(skill, kind=router.required_kind(skill), revision='fixed_fixture_v1',
                        call=lambda payload: calls.append(payload) or [0.1], compatible=lambda g: True)
    return graph, GraphExecutor(graph,router,instance_id='case0'), calls


@pytest.mark.parametrize('mutation',[
    lambda d:d['nodes'][0].update(skill='Invented'),
    lambda d:d['nodes'][0].update(next_stage='missing'),
    lambda d:d['nodes'][1].update(next_stage='s0'),
    lambda d:d['nodes'][0].update(next_stage=None),
    lambda d:d['nodes'][1].update(id='s0'),
    lambda d:d['nodes'][0].update(step_budget=0),
    lambda d:d['nodes'][0].update(surprise='code'),
    lambda d:d['nodes'][0]['termination'][0].update(signal='undeclared'),
    lambda d:d['nodes'][0]['termination'][0].update(op='eval'),
    lambda d:d['nodes'][0]['evidence'].update(start_step=10),
])
def test_strict_schema_and_topology(mutation):
    doc=document(); mutation(doc)
    with pytest.raises(ValueError): TaskGraph.from_dict(doc)


def test_repeated_skills_and_boundary_do_not_call_backend():
    graph, executor, calls=setup(('Move','Move'))
    result=executor.tick(ground(graph),{'ready':True,'done':True})
    assert result.state=='complete' and result.next_stage=='s1' and calls==[]
    result=executor.tick(ground(graph,1),{'ready':True,'done':True})
    assert result.graph_complete
    assert executor.tick(ground(graph),{} ) is result
    assert calls==[]


@pytest.mark.parametrize('skill,kind',[('Move','code'),('Approach','code'),('Release','code'),('Grasp','fixed_policy'),('Lift','fixed_policy'),('Push','fixed_policy'),('Pull','fixed_policy'),('Rotate','fixed_policy'),('Align','fast_adapter'),('Insert','fast_adapter')])
def test_rules(skill,kind):
    graph,executor,calls=setup((skill,))
    assert RuleRouter.required_kind(skill)==kind
    assert executor.tick(ground(graph),{'ready':True}).state=='continue'
    assert len(calls)==1
    with pytest.raises(ValueError):
        RuleRouter().register(skill,kind='invented',revision='x',call=lambda p:None,compatible=lambda g:True)


def test_capability_gap_and_incompatible_adapter_never_fall_back():
    graph=TaskGraph.from_dict(document(('Align',)))
    for registered in (False,True):
        router=RuleRouter(); calls=[]
        if registered:
            router.register('Align',kind='fast_adapter',revision='x',call=calls.append,compatible=lambda g:False)
        executor=GraphExecutor(graph,router,instance_id='case0')
        result=executor.tick(ground(graph),{'ready':True})
        assert result.reason=='capability_gap' and result.replan_requested and not calls
        assert executor.tick(ground(graph),{'done':True}) is result


def test_entry_precondition_not_rechecked_but_invariant_is():
    doc=document(('Move',));doc['nodes'][0]['invariants']=[dict(signal='held',op='eq',value=True)]
    graph,executor,calls=setup(doc=doc)
    executor.tick(ground(graph),{'ready':True,'held':True},progress=True)
    assert executor.tick(ground(graph),{'ready':False,'held':True},progress=True).state=='continue'
    result=executor.tick(ground(graph),{'held':False,'done':True})
    assert result.reason=='invariant_violated' and len(calls)==2


@pytest.mark.parametrize('signal',['object_lost','wrong_target'])
def test_structural_fault_beats_completion(signal):
    graph,executor,calls=setup()
    result=executor.tick(ground(graph),{'ready':True,'done':True,signal:True})
    assert result.state=='structural_failure' and not calls


def test_missing_precondition_waits_bounded_without_action():
    graph,executor,calls=setup()
    for _ in range(2):
        assert executor.tick(ground(graph),{'done':True}).reason=='precondition_unknown'
    assert executor.tick(ground(graph),{'done':True}).reason=='repeated_no_progress'
    assert not calls


def test_missing_termination_never_completes_or_uses_backend_claims():
    graph,executor,calls=setup()
    for _ in range(2):
        assert executor.tick(ground(graph),{'ready':True}).state=='continue'
    assert executor.tick(ground(graph),{'ready':True}).reason=='repeated_no_progress'
    assert len(calls)==2


def test_step_budget_with_progress_is_structural_failure():
    graph,executor,calls=setup()
    for _ in range(4):
        assert executor.tick(ground(graph),{'ready':True},progress=True).state=='continue'
    result=executor.tick(ground(graph),{'ready':True},progress=True)
    assert result.state=='structural_failure' and result.replan_requested and len(calls)==4


def test_supervisor_truth_not_forwarded_to_backend():
    graph,executor,calls=setup(('Align',))
    executor.tick(ground(graph),{'ready':True,'done':False,'gt_pose':[999,999,999]})
    payload=calls[0]
    assert 'gt_pose' not in str(payload) and 'supervisor' not in payload
    assert payload['target']['present'] is False
    assert payload['object']['present'] is True
    assert payload['object']['roi'] is None
    with pytest.raises(ValueError): InputReference('pose.json','supervisor_gt','test')
    with pytest.raises(ValueError): RoleBinding(None,True)


@pytest.mark.parametrize('field,value',[('instance_id','other'),('task_id','other'),('graph_id','other'),('goal_relation','other')])
def test_grounding_binding_mismatch_stops(field,value):
    graph,executor,calls=setup()
    result=executor.tick(replace(ground(graph),**{field:value}),{'ready':True,'done':True})
    assert result.state=='structural_failure' and not calls


def test_wrong_semantic_role_rejected_before_backend():
    graph,executor,calls=setup()
    result=executor.tick(replace(ground(graph),object=RoleBinding('other',True)),{'ready':True})
    assert result.reason=='grounding_role_mismatch' and not calls


def test_missing_invariant_and_non_boolean_truth_cannot_complete():
    doc=document(('Move',));doc['nodes'][0]['invariants']=[dict(signal='held',op='eq',value=True)]
    graph,executor,calls=setup(doc=doc)
    assert executor.tick(ground(graph),{'ready':True,'done':True}).reason=='invariant_unknown'
    assert not calls
    graph,executor,calls=setup()
    assert executor.tick(ground(graph),{'ready':True,'done':1}).state=='continue'


def test_local_adapter_correction_continues_and_structural_fault_stops():
    graph,executor,calls=setup(('Align',))
    for _ in range(2):
        result=executor.tick(ground(graph),{'ready':True,'local_error':True},progress=True)
        assert result.state=='local_failure' and result.action==[0.1]
        assert not result.replan_requested and not executor.stopped
    result=executor.tick(ground(graph),{'ready':True,'local_error':True,'wrong_target':True})
    assert result.state=='structural_failure' and len(calls)==2


def test_current_progress_resets_stall_before_failure_check():
    graph,executor,calls=setup()
    for _ in range(2): executor.tick(ground(graph),{'ready':True})
    result=executor.tick(ground(graph),{'ready':True},progress=True)
    assert result.state=='continue' and executor.stalled==0 and len(calls)==3


@pytest.mark.parametrize('visible',[False,None])
def test_adapter_requires_visible_entities_not_just_role_identity(visible):
    graph,executor,calls=setup(('Align',))
    grounded=ground(graph)
    grounded=replace(grounded,object=replace(grounded.object,visible=visible))
    assert grounded.object.present
    result=executor.tick(grounded,{'ready':True})
    assert result.reason=='capability_gap' and not calls


def test_undeclared_local_error_is_ignored():
    doc=document(('Align',)); doc['supervisor_inputs'].remove('local_error')
    graph,executor,calls=setup(('Align',),doc=doc)
    assert executor.tick(ground(graph),{'ready':True,'local_error':True}).state=='continue'
