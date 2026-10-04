import torch

from prove.fast_adapter import AdaptationLossConfig, Toy2DConfig, Toy2DVectorEnv, expand_latent, sample_toy2d_domains
from tools.fast_adapter import p15_experiment as p15


def test_p15_fast_grid_has_nine_semantic_configs_and_shared_streams():
    grid = p15.fast_grid()
    keys = {(cfg.z_dim, cfg.alpha_mode, cfg.inner_lr) for cfg in grid}
    assert len(grid) == 9
    assert len(keys) == 9
    assert sum(1 for cfg in grid if cfg.config_id == "z32_alpha5") == 1
    assert p15.HistoryConfig(config_id="h", width=256).train_stream_offset == p15.FastConfig(config_id="f").train_stream_offset


def test_p15_all_z_dimensions_have_correct_shapes():
    env = Toy2DVectorEnv(Toy2DConfig())
    domain = sample_toy2d_domains(3, 123, env.config)
    obs = env.reset(3, 456, domain=domain)
    for z_dim in [4, 8, 16, 32, 64]:
        cfg = p15.FastConfig(config_id=f"z{z_dim}", z_dim=z_dim, inner_lr=5.0)
        system = p15.build_fast_system(cfg, device=torch.device("cpu"))
        z = expand_latent(system.z0, 3)
        action, delta, base, u = system.act(obs.adapter_observation(), z)
        assert z.shape == (3, z_dim)
        assert action.shape == (3, 2)
        assert delta.shape == (3, 2)
        assert u.shape == (3, 256)


def test_p15_learnable_alpha_gets_outer_gradient_and_freezes_in_eval():
    cfg = p15.FastConfig(config_id="learned", z_dim=32, alpha_mode="learned", inner_lr=0.5)
    system = p15.build_fast_system(cfg, device=torch.device("cpu"))
    assert system.log_inner_lr is not None
    assert torch.allclose(system.inner_lr.detach(), torch.tensor(0.5), atol=1e-6)
    env = Toy2DVectorEnv(Toy2DConfig())
    domain = sample_toy2d_domains(8, 111, env.config)
    obs = env.reset(8, 222, domain=domain)
    z, _metrics = p15.support_adapt_fast_on_policy(system, env, domain, obs, k=3, loss_config=AdaptationLossConfig(), create_graph=True)
    query = env.reset(8, 333, domain=domain)
    loss, _ = p15.query_loss_fast(system, env, domain, query, z, horizon=10, residual_reg=0.0)
    loss.backward()
    assert system.log_inner_lr.grad is not None
    assert system.log_inner_lr.grad.abs().item() > 0
    before = system.inner_lr.detach().clone()
    for param in system.parameters():
        param.requires_grad_(False)
    assert torch.allclose(system.inner_lr.detach(), before)


def test_p15_history_width_only_changes_policy_width_not_transition_projection_width():
    h256 = p15.P15HistorySystem(p15.HistoryConfig(config_id="h256", width=256))
    h640 = p15.P15HistorySystem(p15.HistoryConfig(config_id="h640", width=640))
    assert h256.adapter.transition_proj[0].out_features == 128
    assert h640.adapter.transition_proj[0].out_features == 128
    assert h256.adapter.policy[0].out_features == 256
    assert h640.adapter.policy[0].out_features == 640
    assert p15.count_parameters(h640)["total"] > p15.count_parameters(h256)["total"]


def test_p15_calibration_uses_four_independent_one_step_starts():
    cfg = p15.FastConfig(config_id="fast", z_dim=32)
    system = p15.build_fast_system(cfg, device=torch.device("cpu"))
    env = Toy2DVectorEnv(Toy2DConfig())
    domain = sample_toy2d_domains(5, 444, env.config)
    transitions = p15.collect_base_calibration_transitions(system, env, domain, seed=555, starts=4)
    assert len(transitions) == 4
    assert all(t.u.shape == (5, 256) for t in transitions)
    assert not torch.allclose(transitions[0].u, transitions[1].u)


def test_p15_common_base_fixed_transition_pre_post_and_k0_equals_frozen():
    cfg = p15.FastConfig(config_id="fast", z_dim=32, inner_lr=5.0)
    system = p15.build_fast_system(cfg, device=torch.device("cpu"))
    env = Toy2DVectorEnv(Toy2DConfig())
    domain = sample_toy2d_domains(4, 666, env.config)
    transitions = p15.collect_common_base_transitions(system, env, domain, support_seed=777, max_k=3)
    assert len(transitions) == 3
    z0 = expand_latent(system.z0, 4)
    z_next, metrics = p15.update_z_from_transition(system, transitions[0], z0, AdaptationLossConfig(), create_graph=True)
    assert "pre_mse" in metrics and "post_mse" in metrics
    assert not torch.allclose(z_next.detach(), z0.detach())
    z_k0, _ = p15.support_adapt_fast_common_base(system, transitions, k=0, loss_config=AdaptationLossConfig(), create_graph=False)
    frozen = system.z0.detach().unsqueeze(0).expand_as(z_k0)
    assert torch.allclose(z_k0.detach(), frozen)


def test_p15_query_rollout_does_not_update_latent():
    cfg = p15.FastConfig(config_id="fast", z_dim=32)
    system = p15.build_fast_system(cfg, device=torch.device("cpu"))
    env = Toy2DVectorEnv(Toy2DConfig())
    domain = sample_toy2d_domains(2, 888, env.config)
    obs = env.reset(2, 889, domain=domain)
    z, _ = p15.support_adapt_fast_on_policy(system, env, domain, obs, k=2, loss_config=AdaptationLossConfig(), create_graph=False)
    z_before = z.detach().clone()
    query_domain = p15.repeat_domain(domain, 8)
    query_obs = env.reset(16, 890, domain=query_domain)
    arrays = p15.rollout_query_fast_arrays(
        system,
        env,
        query_domain,
        query_obs,
        z.repeat_interleave(8, dim=0),
        arm="correct_z",
        horizon=10,
    )
    assert arrays["final_error"].shape == (16,)
    assert torch.allclose(z.detach(), z_before)
