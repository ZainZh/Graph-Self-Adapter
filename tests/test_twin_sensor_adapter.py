from types import SimpleNamespace
import json
import numpy as np
from prove.task_graph.twin_sensor_adapter import sensor_world_class
from prove.task_graph.twin_sensor_judge import JudgeConfig


class SpyWorld:
    forbidden = {'_held_by_arm', 'object_specs', '_current_held', '_current_loaded', 'body', 'contacts'}
    def __getattribute__(self, name):
        if name in object.__getattribute__(self, 'forbidden'):
            raise AssertionError('hidden access: ' + name)
        return object.__getattribute__(self, name)

    def __init__(self):
        self.data = SimpleNamespace(time=0., ctrl=np.zeros(16), qpos=np.zeros(16), qvel=np.zeros(16), qfrc_bias=np.zeros(16))
        self.model = SimpleNamespace(numeric=lambda _: SimpleNamespace(data=[.037]), opt=SimpleNamespace(timestep=.001))
        self.arm_actuator_indices = np.arange(12)
        self.arm_qpos_indices = self.arm_dof_indices = np.arange(12)
        self.reconstruction = {}
        self.fingers = [(SimpleNamespace(qposadr=[12+i], dofadr=[12+i]), 12+i, i//2, (-1, 1)[i%2]) for i in range(4)]
        self._finger_targets = dict.fromkeys(range(12,16), 0.)
        self._previous_gripper_command = np.array([.5, .5])

    def numerical_safety(self):
        return {'simulation_finite': True}

    def proprioception(self):
        return SimpleNamespace(joint_pos=np.zeros((2, 6)))

    def visible_stage_facts(self, *args, **kwargs):
        raise AssertionError('legacy judge accessed')

    stage_goal_facts = visible_stage_facts

    def step_command_callback(self, callback, duration, *args, **kwargs):
        self.command = callback(self.data.time)
        self.data.time += duration
        return self.command

    def contact_evidence(self):
        assert self.seal.exists()
        return {"roles": {"object": {"original_independent_contact": True}}}

    def finalize_evaluator(self, task_outcome):
        assert self.seal.exists(), 'sensor prediction not sealed before evaluation'
        assert self.contact_evidence()['roles']['object']['original_independent_contact']
        return {'independent_evaluation': True}


def make_world(tmp_path):
    physics = SimpleNamespace(TwinWorld=SpyWorld, ARM_KP=np.ones(12), ARM_KD=np.ones(12))
    seal = tmp_path / 'SEAL.json'
    cls = sensor_world_class(physics, config=JudgeConfig(), references={}, judge_mode='sensor_only', seal_path=seal, timeout_sec=.2)
    world = cls(); world.seal = seal
    return world


def test_no_hidden_judge_contact_or_object_access_and_proprio_only_servo(tmp_path):
    world = make_world(tmp_path)
    assert world.visible_stage_facts('obj', None, arm='left')['bilateral_contact'] is None
    assert world.stage_goal_facts('obj', 'target', arm='left', goal_relation='inside')['goal_inside'] is None
    assert world.contact_evidence()['roles'] == {}
    world._servo(np.zeros((2,6)), np.ones(2), np.zeros((2,6)))
    assert np.isfinite(world.data.ctrl).all()


def test_unknown_holding_stops_motion_and_seals_before_independent_evaluator(tmp_path):
    world = make_world(tmp_path)
    world.sensor_stage = {'stage_id':'Lift', 'phase':'Lift', 'object':'obj', 'arm':'left', 'holding_contract':'bilateral_contact', 'invariant_predicates':[{'predicate':'holding'}]}
    world.sensor_runtime = SimpleNamespace(state=SimpleNamespace(failure=None), trace=[])
    attempted = lambda _: (np.ones((2,6)), np.zeros(2), np.ones((2,6)))
    world.step_command_callback(attempted, .1)
    assert np.array_equal(world.command[0], np.zeros((2,6)))
    assert world.sensor_motion_blocked and world.sensor_runtime.state.failure is None
    world.step_command_callback(attempted, .2)
    world.step_command_callback(attempted, .1)
    assert world.sensor_runtime.state.failure == 'unknown_stop'
    assert world.finalize_evaluator({})['independent_evaluation']
    seal = json.loads(world.seal.read_text())
    assert seal['sealed_before_finalize_evaluator'] is True
    assert seal['motion_gates'][-1]['reason'] == 'unknown_stop'
    assert world.contact_evidence()['roles'] == {}
    assert world.sensor_evaluator_context is False


def test_requested_goal_relation_aliases_and_visible_handover_without_contacts(tmp_path):
    from prove.task_graph.twin_sensor_judge import ArmSensors, SensorPacket
    world = make_world(tmp_path)
    def ingest(t, z, gap=.04, command=.02):
        arm = ArmSensors(np.zeros(6), np.zeros(6), np.array([0.,0.,z]), np.eye(3),
                         np.zeros(2), gap, command, 'test', .08, 0., t, 'measured_encoder')
        world.sensor_judge.ingest(SensorPacket(t,t, {'obj':np.tile([0.,0.,z], (10,1))}, {'left':arm}, {}),t)
        world.data.time=t
    ingest(0., 0.)
    world.stage_goal_facts('obj','target',arm='left',goal_relation='lifted')
    ingest(.2, .05)
    facts=world.stage_goal_facts('obj','target',arm='left',goal_relation='lifted')
    assert facts['goal_lifted'] is True and facts['lifted'] is True and facts['stage_goal_complete'] is True
    for relation in ('inside','nested_with_support','hanging'):
        assert world.stage_goal_facts('obj','target',arm='left',goal_relation=relation)['stage_goal_complete'] is None
    for t in (.3,.5,.7,.9):
        ingest(t,.05,gap=.08,command=.08)
        world.sensor_judge.handover('obj','left',[0.,0.,.05])
    assert world.sensor_judge.handover('obj','left',[0.,0.,.05]) is True
    assert world.sensor_judge.diagnostics['goal_handover']['physical_contact_verified'] is False


def test_evaluator_private_context_resets_when_evaluation_raises(tmp_path, monkeypatch):
    world=make_world(tmp_path)
    def fail(self, outcome):
        assert self.seal.exists()
        assert self.contact_evidence()['roles']['object']['original_independent_contact']
        raise RuntimeError('independent evaluator error')
    monkeypatch.setattr(SpyWorld,'finalize_evaluator',fail)
    import pytest
    with pytest.raises(RuntimeError,match='independent evaluator error'):
        world.finalize_evaluator({})
    assert world.contact_evidence()['roles']=={} and not world.sensor_evaluator_context


def test_execution_guard_requires_actual_bound_sources_and_authorized_scope(tmp_path):
    from tools.task_graph import twin_sensor_rsi as runner
    scope=tmp_path/'scope.json'; freeze=tmp_path/'freeze.json'
    scope.write_text(json.dumps({'execution_authorized':False}))
    freeze.write_text(json.dumps({'schema':'graph-self-adapter.sensor-execution-freeze.v1',
                                 'scope_sha256':runner.digest(scope),'bindings':[]}))
    import pytest
    with pytest.raises(ValueError,match='not authorized'):
        runner.validate_execution(scope,freeze)
    scope.write_text(json.dumps({'execution_authorized':True}))
    freeze.write_text(json.dumps({'schema':'graph-self-adapter.sensor-execution-freeze.v1',
                                 'scope_sha256':runner.digest(scope),'bindings':[]}))
    with pytest.raises(ValueError,match='missing sensor source binding'):
        runner.validate_execution(scope,freeze)


def test_batch_worker_errors_retained_without_rerun(tmp_path):
    from tools.task_graph.twin_sensor_validate import run_one
    output=tmp_path/'used';output.mkdir()
    result=run_one({'task':'put_knife_board_new','trial':0,'judge_mode':'sensor_only',
                    'kind':'development','output':str(output)}, {}, tmp_path/'campaign.json')
    assert result['strict_success'] is False and result['error'].startswith('FileExistsError:')


def test_motion_gate_respects_none_and_held_or_supported_contracts(tmp_path):
    from prove.task_graph.twin_sensor_adapter import sensor_holding_contract, carrying_gate_required
    world=make_world(tmp_path)
    world.sensor_stage={'phase':'Lift','holding_contract':'none'}
    world.step_command_callback(lambda _: (np.ones((2,6)), np.zeros(2), np.zeros((2,6))),.1)
    assert np.array_equal(world.command[0],np.ones((2,6)))
    stage={'phase':'Align','holding_contract':'held_or_supported','invariant_predicates':[{'predicate':'holding'}]}
    assert carrying_gate_required(stage)
    assert sensor_holding_contract(stage,{'bilateral_contact':False,'object_support_contact':True}) is True
    assert sensor_holding_contract(stage,{'bilateral_contact':False,'object_support_contact':None}) is None
    assert sensor_holding_contract(stage,{'bilateral_contact':False,'object_support_contact':False}) is False


def test_shadow_prediction_covers_declared_sensor_kinds_not_progress_or_actor():
    from tools.task_graph.twin_sensor_shadow import sensor_predicate_predictions
    judge=SimpleNamespace(packet=SimpleNamespace(timestamp_sec=1.), handover=lambda *args: True)
    stage={'stage_id':'s','object':'obj','target':'target','arm':'left',
           'holding_contract':'held_or_supported','goal_relation':'supported',
           'completion_predicates':[{'id':'hold','predicate':'holding'},
                                    {'id':'release','predicate':'released'},
                                    {'id':'goal','predicate':'stage_goal','goal_relation':'accessible_to_right_arm'},
                                    {'id':'progress','predicate':'reference_progress'},
                                    {'id':'actor','predicate':'actor_command_reached'}]}
    values={'bilateral_contact':None,'object_support_contact':None,'released':None,'lifted':False,'supported_contact':None}
    result=sensor_predicate_predictions(stage,values,judge,{}, {})
    assert set(result)=={'hold','release','goal'}
    assert result['hold']['value'] is None and result['release']['value'] is None
    assert result['goal']['value'] is True


def test_handover_holding_contract_uses_declared_visible_table_alias_only():
    from prove.task_graph.twin_sensor_adapter import sensor_holding_contract
    stage={'holding_contract':'held_or_supported','target':'table_handover_region',
           'goal_relation':'accessible_to_right_arm'}
    values={'bilateral_contact':False,'object_support_contact':None,'object_table_support_contact':True}
    assert sensor_holding_contract(stage,values) is True
    values['object_table_support_contact']=None
    assert sensor_holding_contract(stage,values) is None
