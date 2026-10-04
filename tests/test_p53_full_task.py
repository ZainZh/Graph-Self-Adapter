import copy
import pytest
from dataclasses import asdict
from prove.task_graph.p52_stage_policy import StagePolicy,Config
from tools.task_graph.p53_full_task import neural_graph,check_packages


def graph():
    return dict(schema='prove.p53-short-taskgraph.v3',graph_id='qualified_teacher',prefix_nodes=[dict(id=f's{i}') for i in range(1,7)],shared_code_budget=120,nodes=[
        dict(id='s7',skill='Align',backend='public_code_pose_path',next='s8',reference_start=0,reference_end=1,target_rule='raise_current_xy_orientation_to_public_max_z_then_reference41',termination='public_endpoint_reached'),
        dict(id='s8',skill='Approach',backend='public_code_pose_path',next='s9',reference_start=2,reference_end=2,target_rule='public_reference57_plus_one_position_tolerance_z',termination='public_endpoint_reached_and_near_ready'),
        dict(id='s9',skill='Insert',backend='private_training_micro_teacher',next=None,max_actions=24,termination='original_goal_and_contact_hold3')])


def test_neural_child_has_actual_backend_without_mutating_teacher_parent():
    parent=graph();saved=copy.deepcopy(parent);child=neural_graph(parent)
    assert parent==saved
    assert child['nodes'][2]['backend']=='frozen_stage_policy'
    assert child['nodes'][:2]==parent['nodes'][:2]
    assert child['nodes'][2]['max_actions']==24


def test_final_branch_requires_identical_actual_warmup_data_and_budget():
    config=Config(raw_dim=12,appearance_dim=4,role_dim=4,hidden=8,latent=2)
    model=StagePolicy(config)
    bc=dict(schema='p53_short_insert_skill_v01',branch='bc',common_warmup_sha256='warm',data_sha256='data',horizon=24,total_outer_updates=3000,config=asdict(config),model=model.state_dict())
    joint=dict(bc,branch='joint');check_packages(bc,joint,'data','warm')
    with pytest.raises(ValueError):check_packages(bc,dict(joint,data_sha256='other'),'data','warm')
    with pytest.raises(ValueError):check_packages(bc,dict(joint,horizon=180),'data','warm')
