from __future__ import annotations

import torch

from prove.fast_adapter import Toy2DConfig, Toy2DDomain, Toy2DVectorEnv, sample_toy2d_domains
from tools.fast_adapter.p16_experiment import (
    P16FastConfig,
    build_fast_system,
    build_history_system,
    history_config,
    collect_common_base_records,
    adapt_fast,
    support_loss_per_domain,
    support_displacement_outer_mse,
    history_update_context,
    rollout_fast_arrays,
    ls_fit_delta,
    ls_predict,
    SupportRecord,
)
from tools.fast_adapter.p15_experiment import repeat_domain


def test_p16_support_loss_mean_over_s_then_sum_over_domains_batch_invariant():
    device = torch.device("cpu")
    cfg = P16FastConfig(config_id="feature_fast", method="feature_fast")
    torch.manual_seed(0)
    system = build_fast_system(cfg, device=device)
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    one_domain = sample_toy2d_domains(1, 123, env.config, device=device)
    rec1 = collect_common_base_records(system, env, one_domain, support_seed=456, max_s=3)
    z1 = system.z0.unsqueeze(0).clone().detach().requires_grad_(True)
    loss1 = support_loss_per_domain(system, rec1, z1, cfg)["total"].sum()
    grad1 = torch.autograd.grad(loss1, z1, retain_graph=True)[0]

    rec4 = []
    for rec in rec1:
        rec4.append(SupportRecord(
            u=rec.u.repeat(4, 1),
            action=rec.action.repeat(4, 1),
            u_next=rec.u_next.repeat(4, 1),
            state=rec.state.repeat(4, 1),
            next_state=rec.next_state.repeat(4, 1),
        ))
    z4 = system.z0.unsqueeze(0).expand(4, -1).clone().detach().requires_grad_(True)
    loss4 = support_loss_per_domain(system, rec4, z4, cfg)["total"].sum()
    grad4 = torch.autograd.grad(loss4, z4)[0]
    assert torch.allclose(grad1[0], grad4[0], atol=1e-6, rtol=1e-5)


def test_p16_s_u_separation_reuses_same_prefix_and_u0_noop():
    device = torch.device("cpu")
    cfg = P16FastConfig(config_id="displacement_fast", method="displacement_fast")
    torch.manual_seed(1)
    system = build_fast_system(cfg, device=device)
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(4, 10, env.config, device=device)
    records = collect_common_base_records(system, env, domain, support_seed=20, max_s=5)
    z0, metrics0 = adapt_fast(system, records, s=3, u_passes=0, config=cfg, create_graph=False)
    assert metrics0 == []
    assert torch.allclose(z0, system.z0.unsqueeze(0).expand(4, -1))
    z3, metrics3 = adapt_fast(system, records, s=3, u_passes=3, config=cfg, create_graph=False)
    assert len(metrics3) == 3
    assert all(m["support_records"] == 3 for m in metrics3)
    assert not torch.allclose(z3, z0)


def test_p16_displacement_target_detached_action_detached_and_z0_reg_grad():
    device = torch.device("cpu")
    cfg = P16FastConfig(config_id="displacement_fast", method="displacement_fast")
    torch.manual_seed(2)
    system = build_fast_system(cfg, device=device)
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(3, 11, env.config, device=device)
    records = collect_common_base_records(system, env, domain, support_seed=22, max_s=3)
    # Observed executed actions/physical targets are fixed observations.
    for rec in records:
        assert not rec.action.requires_grad
        assert not rec.delta_x.requires_grad
    z = system.z0.unsqueeze(0).expand(3, -1).clone().detach().requires_grad_(True) + 0.1
    loss = support_loss_per_domain(system, records[:3], z, cfg)["total"].sum()
    loss.backward()
    assert system.z0.grad is not None
    assert float(system.z0.grad.norm()) > 0.0


def test_p16_query_gradient_reaches_displacement_predictor_through_z_without_physical_outer():
    device = torch.device("cpu")
    cfg = P16FastConfig(config_id="displacement_fast", method="displacement_fast")
    torch.manual_seed(3)
    system = build_fast_system(cfg, device=device)
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(5, 33, env.config, device=device)
    records = collect_common_base_records(system, env, domain, support_seed=44, max_s=3)
    z, _ = adapt_fast(system, records, s=3, u_passes=3, config=cfg, create_graph=True)
    qobs = env.reset(5, 55, domain=domain)
    from tools.fast_adapter.p16_experiment import query_loss_fast
    loss, _ = query_loss_fast(system, env, domain, qobs, z, horizon=3, residual_reg=0.0)
    loss.backward()
    dyn_grad = sum(float(p.grad.detach().abs().sum()) for p in system.dynamics.parameters() if p.grad is not None)
    assert dyn_grad > 0.0


def test_p16_history_consumes_support_once_and_query_context_frozen():
    device = torch.device("cpu")
    cfg = history_config()
    torch.manual_seed(4)
    system = build_history_system(cfg, device=device)
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(2, 77, env.config, device=device)
    records = collect_common_base_records(system, env, domain, support_seed=88, max_s=3)
    c0 = history_update_context(system, records, s=0, batch_size=2, device=device)
    c3 = history_update_context(system, records, s=3, batch_size=2, device=device)
    assert not torch.allclose(c0, c3)
    frozen = c3.detach().clone()
    qdomain = repeat_domain(domain, 2)
    qobs = env.reset(4, 99, domain=qdomain)
    from tools.fast_adapter.p16_experiment import rollout_history_arrays
    _ = rollout_history_arrays(system, env, qdomain, qobs, frozen.repeat_interleave(2, 0), c0.repeat_interleave(2, 0), horizon=2)
    assert torch.allclose(c3.detach(), frozen)


def test_p16_first_query_action_change_is_expert_independent():
    device = torch.device("cpu")
    cfg = P16FastConfig(config_id="feature_fast", method="feature_fast")
    torch.manual_seed(5)
    system = build_fast_system(cfg, device=device)
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(2, 101, env.config, device=device)
    qdomain = repeat_domain(domain, 2)
    z_frozen = system.z0.detach().unsqueeze(0).expand(4, -1).clone()
    z_changed = z_frozen + 0.25
    qobs = env.reset(4, 202, domain=qdomain)
    arrays = rollout_fast_arrays(system, env, qdomain, qobs, z_changed, z_frozen, arm="correct_z", horizon=2)
    original_change = arrays["first_action_change_vs_frozen"].clone()
    def shifted_expert(obs, domain):
        return torch.ones_like(obs.state) * -0.75
    env.expert_action = shifted_expert  # type: ignore[method-assign]
    qobs2 = env.reset(4, 202, domain=qdomain)
    arrays2 = rollout_fast_arrays(system, env, qdomain, qobs2, z_changed, z_frozen, arm="correct_z", horizon=2)
    assert torch.allclose(original_change, arrays2["first_action_change_vs_frozen"])
    assert arrays["first_action_change_vs_frozen"].shape == (4,)


def test_p16_ls_recovers_known_parameters_and_predicts_disjoint_transition():
    # S=3 with distinct actions per axis; construct directly from observable a and delta_x.
    u = torch.zeros(1, 256)
    z = torch.zeros(1, 32)
    actions = [torch.tensor([[0.0, -0.5]]), torch.tensor([[0.5, 0.0]]), torch.tensor([[1.0, 0.5]])]
    gain = torch.tensor([[1.2, 0.8]])
    bias = torch.tensor([[0.05, -0.02]])
    records = []
    state = torch.zeros(1, 2)
    for a in actions:
        delta = gain * a + bias
        records.append(SupportRecord(u=u, action=a, u_next=u, state=state, next_state=state + delta))
    fit = ls_fit_delta(records)
    assert torch.all(fit["rank"] >= 2)
    assert torch.allclose(fit["gain_hat"], gain, atol=1e-4)
    assert torch.allclose(fit["bias_hat"], bias, atol=1e-4)
    heldout = torch.tensor([[0.25, -0.25]])
    pred, identifiable = ls_predict(fit, heldout)
    assert bool(identifiable.item())
    assert torch.allclose(pred, gain * heldout + bias, atol=1e-4)


def test_p16_ls_rank_deficient_marked_unidentifiable():
    u = torch.zeros(1, 256)
    a = torch.tensor([[0.25, 0.25]])
    records = [SupportRecord(u=u, action=a, u_next=u, state=torch.zeros(1, 2), next_state=torch.ones(1, 2) * i) for i in [0.1, 0.2, 0.3]]
    fit = ls_fit_delta(records)
    # Constant action leaves slope/intercept unidentifiable even with three rows.
    assert not bool(((fit["rank"] >= 2).all(dim=-1)).item())
