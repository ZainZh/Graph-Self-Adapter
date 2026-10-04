
from __future__ import annotations

import torch

from tools.fast_adapter.p18_experiment import (
    P18FeatureBatch,
    P18ModelConfig,
    P18TransitionRecord,
    ablate_dino,
    action_imitation_loss,
    adapt_latent,
    bounded_axis_probe_actions,
    build_gradient_system,
    build_history_system,
    build_hybrid_system,
    encode_transition_records,
    observable_action_design_rank,
    physical_support_fingerprint,
    prepare_latent_for_arm,
    projected_oracle_action,
    sample_p18_positions,
    sample_p18_shifts,
    step_p18_shift,
    swapped_latent,
    trainable_parameter_groups,
)


def feature_batch(batch_size: int = 4, seed: int = 0, cfg: P18ModelConfig | None = None) -> P18FeatureBatch:
    cfg = cfg or P18ModelConfig()
    g = torch.Generator().manual_seed(seed)
    obj = torch.randn(batch_size, cfg.obj_dino_dim, generator=g)
    target = torch.randn(batch_size, cfg.target_dino_dim, generator=g)
    proprio = torch.zeros(batch_size, cfg.proprio_dim)
    prev = torch.randn(batch_size, cfg.action_dim, generator=g).clamp(-1, 1)
    roi = torch.rand(batch_size, cfg.roi_dim, generator=g)
    base = torch.tanh(torch.randn(batch_size, cfg.action_dim, generator=g))
    return P18FeatureBatch(obj, target, proprio, prev, roi, base)


def support_records(system, batch_size: int = 4, k: int = 3):
    current = [feature_batch(batch_size, seed=10 + i, cfg=system.cfg) for i in range(k)]
    future = [feature_batch(batch_size, seed=50 + i, cfg=system.cfg) for i in range(k)]
    actions = [bounded_axis_probe_actions(device="cpu")[i].expand(batch_size, -1) for i in range(k)]
    return encode_transition_records(system, current, actions, future)


def test_primary_support_actions_have_frozen_order_and_full_rank_by_k5():
    actions = bounded_axis_probe_actions()
    expected = torch.tensor([[0.25, 0.0], [0.0, 0.25], [-0.25, 0.0], [0.0, -0.25], [0.18, 0.18]])
    assert torch.equal(actions, expected)
    assert not observable_action_design_rank(actions[:2])["identifiable_full_affine_response"]
    assert observable_action_design_rank(actions[:3])["identifiable_full_affine_response"]
    assert observable_action_design_rank(actions[:5])["identifiable_full_affine_response"]


def test_straight_line_base_actions_are_rank_diagnostic_not_identifiable():
    straight = torch.tensor([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0], [0.5, 0.0]])
    diag = observable_action_design_rank(straight)
    assert diag["rank"] == 2
    assert not diag["identifiable_full_affine_response"]


def test_residual_actor_is_bounded_and_dino_features_are_causal_inputs():
    torch.manual_seed(1)
    system = build_gradient_system()
    batch = feature_batch(5, seed=2, cfg=system.cfg)
    z = system.z0.unsqueeze(0).expand(5, -1)
    action, delta, _u = system.act(batch, z)
    assert torch.all(delta.abs() <= system.cfg.max_residual + 1.0e-6)
    zero_batch = ablate_dino(batch, "zero")
    zero_action, _zero_delta, _zero_u = system.act(zero_batch, z)
    assert not torch.allclose(action, zero_action)


def test_inner_update_changes_only_z_when_eval_frozen():
    torch.manual_seed(3)
    system = build_gradient_system()
    records = support_records(system, batch_size=3, k=3)
    z_init = system.initial_latent(records, batch_size=3, device=torch.device("cpu"))
    system.freeze_for_eval()
    before = {name: tensor.detach().clone() for name, tensor in system.state_dict().items()}
    z, metrics = adapt_latent(system, records, z_init, steps=3, create_graph=False, detach_initial=True)
    assert len(metrics) == 3
    assert not torch.allclose(z.detach(), z_init.detach())
    for name, tensor in system.state_dict().items():
        assert torch.equal(before[name], tensor.detach()), name
    assert all(param.grad is None for param in system.parameters())


def test_hybrid_is_jointly_trainable_and_outer_loss_reaches_history_predictor_actor_projector():
    torch.manual_seed(4)
    system = build_hybrid_system()
    records = support_records(system, batch_size=4, k=3)
    z_init = system.initial_latent(records, batch_size=4, device=torch.device("cpu"))
    z, _metrics = adapt_latent(system, records, z_init, steps=3, create_graph=True, detach_initial=False)
    query = feature_batch(4, seed=99, cfg=system.cfg)
    oracle = torch.zeros(4, system.cfg.action_dim)
    loss, _ = action_imitation_loss(system, query, z, oracle, residual_reg=0.0)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.projector.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.dynamics.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.history.parameters())
    assert system.log_inner_alpha.grad is not None and system.log_inner_alpha.grad.abs().sum() > 0


def test_history_and_hybrid_parameter_groups_reflect_separate_baselines():
    history = build_history_system()
    hybrid = build_hybrid_system()
    hg = trainable_parameter_groups(history)
    yg = trainable_parameter_groups(hybrid)
    assert hg["history"] > 0 and hg["dynamics"] == 0 and hg["log_inner_alpha"] == 0
    assert yg["history"] > 0 and yg["dynamics"] > 0 and yg["log_inner_alpha"] == 1
    assert yg["total"] > hg["total"]


def test_swapped_latent_preserves_current_history_init_and_uses_other_delta():
    z_init = torch.tensor([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    adapted = z_init + torch.tensor([[0.1, 0.2], [1.0, 1.2], [2.0, 2.2]])
    perm = torch.tensor([2, 0, 1])
    swapped = swapped_latent(z_init, adapted, perm)
    assert torch.allclose(swapped, z_init + (adapted - z_init)[perm])
    assert not torch.allclose(swapped, adapted[perm])


def test_physical_support_fingerprint_excludes_encoded_features():
    rgb = torch.arange(2 * 4 * 4 * 3, dtype=torch.uint8).reshape(2, 4, 4, 3)
    next_rgb = rgb.flip(1)
    action = bounded_axis_probe_actions()[:2]
    roi = torch.ones(2, 4)
    proprio = torch.zeros(2, 2)
    first = physical_support_fingerprint(rgb=rgb, action=action, next_rgb=next_rgb, roi_geometry=roi, proprio=proprio)
    # Model-specific encoded u is intentionally not part of this fingerprint.
    _encoded_u_a = torch.randn(2, 256)
    _encoded_u_b = torch.randn(2, 256)
    second = physical_support_fingerprint(rgb=rgb, action=action, next_rgb=next_rgb, roi_geometry=roi, proprio=proprio)
    assert first == second


def test_prepare_latent_for_arms_uses_k_prefixes():
    torch.manual_seed(5)
    system = build_hybrid_system()
    records = support_records(system, batch_size=2, k=5)
    z0, m0 = prepare_latent_for_arm(system, "hybrid", records, batch_size=2, k=0, create_graph=False, detach_initial=True)
    z3, m3 = prepare_latent_for_arm(system, "hybrid", records, batch_size=2, k=3, create_graph=False, detach_initial=True)
    assert m0 == []
    assert len(m3) == 3
    assert not torch.allclose(z0, z3)



def test_p18_shift_strata_follow_root_fixed_bands():
    train = sample_p18_shifts(256, 1001, "train")
    det = torch.linalg.det(train.M)
    assert torch.all(det > 0)
    assert torch.all(train.b.abs() <= 0.040001)

    mild = sample_p18_shifts(256, 1002, "mild")
    sx_mild = torch.linalg.norm(mild.M[:, :, 0], dim=-1)
    assert torch.all((sx_mild > 0.6) & (sx_mild < 1.4))
    assert torch.all(mild.b.abs() <= 0.040001)

    strong = sample_p18_shifts(256, 1003, "strong")
    assert torch.all(strong.b.abs() <= 0.040001)
    # Strong rotations/cross-couplings should produce off-diagonal response in most domains.
    offdiag = torch.maximum(strong.M[:, 0, 1].abs(), strong.M[:, 1, 0].abs())
    assert float((offdiag > 0.1).float().mean()) > 0.8


def test_projected_oracle_is_bounded_like_adapter_not_unfair_unbounded_inverse():
    cfg = None
    state, target = sample_p18_positions(16, 2001)
    shift = sample_p18_shifts(16, 2002, "strong")
    base = target.sub(state).mul(0.5).clamp(-1.0, 1.0)
    oracle = projected_oracle_action(state, target, shift, cfg=cfg)
    assert torch.all(oracle.abs() <= 1.000001)
    assert torch.all((oracle - base).abs() <= 0.150001)
    next_state = step_p18_shift(state, shift, oracle)
    assert next_state.shape == state.shape
