import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from prove.task_graph.precision_action_policy import (
    PHASES, PrecisionActionController, PrecisionActionMLP, decode_action, encode_state,
    load_frozen_head, sample_examples, training_contract,
)
from tools.task_graph.train_precision_action_policy import teacher_precheck


def inputs():
    slot_R = Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix()
    eef_R = slot_R @ Rotation.from_rotvec([.1, -.2, .3]).as_matrix()
    slot_p = np.array([.3, -.1, .2])
    eef_p = slot_p + slot_R @ np.array([.05, -.02, .08])
    return (slot_p, slot_R, np.array([.019, .016])), (eef_p, eef_R)


def test_teacher_labels_match_actual_unchanged_controller():
    result = teacher_precheck(training_contract())
    assert result['passed'] and result['cases'] == 32


def test_rotation_increment_is_left_composed_in_slot_frame():
    estimate, proprio = inputs()
    features = encode_state(estimate, proprio, .06, 'approach')
    action = np.r_[-features[:3], -features[3:6] / .5]
    action[2] += (.06 + .004) / .2
    p, R = decode_action(action, estimate[1], proprio)
    np.testing.assert_allclose(p, estimate[0] + estimate[1][:, 2] * .064, atol=2e-8)
    np.testing.assert_allclose(R, estimate[1], atol=3e-8)


def test_global_frame_equivariance():
    estimate, proprio = inputs()
    W = Rotation.from_euler('xyz', [-.4, .6, .2]).as_matrix()
    shift = np.array([.7, -.5, .1])
    transformed_estimate = (W @ estimate[0] + shift, W @ estimate[1], estimate[2])
    transformed_proprio = (W @ proprio[0] + shift, W @ proprio[1])
    np.testing.assert_allclose(encode_state(estimate, proprio, .06, 'insert'),
                               encode_state(transformed_estimate, transformed_proprio, .06, 'insert'), atol=1e-7)
    action = np.array([.2, -.1, .3, .5, -.2, .1])
    p, R = decode_action(action, estimate[1], proprio)
    p2, R2 = decode_action(action, transformed_estimate[1], transformed_proprio)
    np.testing.assert_allclose(p2, W @ p + shift, atol=1e-12)
    np.testing.assert_allclose(R2, W @ R, atol=1e-12)


class ConstantHead(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.register_buffer('action', torch.as_tensor(action, dtype=torch.float32))
        self.calls = 0

    def forward(self, features):
        self.calls += 1
        return self.action[None].expand(len(features), -1)


def test_precision_commands_come_from_model_not_analytic_teacher():
    estimate, proprio = inputs()
    model = ConstantHead(np.zeros(6))
    controller = PrecisionActionController(model, *proprio, length=.06)
    p, R, closed, stage = controller.command(12, estimate, proprio)
    np.testing.assert_allclose(p, proprio[0])
    np.testing.assert_allclose(R, proprio[1], atol=1e-12)
    assert not np.allclose(p, estimate[0] + estimate[1][:, 2] * .1)
    assert closed and stage == 'neural align' and model.calls == 1
    assert controller.records[-1]['command_source'] == 'frozen_precision_action_mlp'


def test_prefix_and_missing_geometry_do_not_query_model():
    estimate, proprio = inputs()
    model = ConstantHead(np.zeros(6))
    controller = PrecisionActionController(model, *proprio, length=.06)
    p, _, closed, _ = controller.command(0, estimate, proprio)
    np.testing.assert_allclose(p, proprio[0] + proprio[1] @ [.018, 0, 0])
    p, R, _, _ = controller.command(12, None, proprio)
    np.testing.assert_allclose(p, proprio[0])
    np.testing.assert_allclose(R, proprio[1])
    assert model.calls == 0 and closed


def test_no_fallback_on_nonfinite_model_output():
    estimate, proprio = inputs()
    controller = PrecisionActionController(ConstantHead(np.full(6, np.nan)), *proprio, length=.06)
    with pytest.raises(ValueError, match='finite'):
        controller.command(12, estimate, proprio)
    assert not controller.records


def test_same_model_responds_to_changed_estimate_via_public_features():
    torch.manual_seed(11)
    estimate, proprio = inputs()
    model = PrecisionActionMLP()
    controller = PrecisionActionController(model, *proprio, length=.06)
    p, _, _, _ = controller.command(12, estimate, proprio)
    changed = (estimate[0] + [.01, -.005, .002], estimate[1], estimate[2])
    p2, _, _, _ = controller.command(13, changed, proprio)
    assert np.linalg.norm(p - p2) > 1e-6
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_balanced_sampling_and_deterministic_stream():
    x, y, rows = sample_examples(np.random.default_rng(11), 256)
    xx, yy, _ = sample_examples(np.random.default_rng(11), 256)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(y, yy)
    assert x.shape == (256, 13) and y.shape == (256, 6)
    np.testing.assert_array_equal(np.bincount(rows['phase']), [64] * 4)
    assert training_contract()['training']['updates'] == 3000


def test_frozen_checkpoint_contract_and_parameters(tmp_path):
    path = tmp_path / 'head.pt'
    torch.save(dict(schema='prove.p50-precision-action-checkpoint.v1', state_dict=PrecisionActionMLP().state_dict(),
                    contract=training_contract()), path)
    model = load_frozen_head(path)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 19078


@pytest.mark.parametrize('phase', PHASES)
def test_onehot_and_invalid_reflection(phase):
    estimate, proprio = inputs()
    features = encode_state(estimate, proprio, .06, phase)
    np.testing.assert_array_equal(features[-4:], np.eye(4)[PHASES.index(phase)])
    reflected = (estimate[0], np.diag([1., 1., -1.]), estimate[2])
    with pytest.raises(ValueError, match='proper rotation'):
        encode_state(reflected, proprio, .06, phase)
