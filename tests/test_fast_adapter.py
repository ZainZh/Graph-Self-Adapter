import dataclasses

import torch

from prove.fast_adapter import (
    AdaptationLossConfig,
    AdapterObservation,
    FastAdapter,
    FastAdapterSystem,
    BaseSkillPolicy,
    FrozenVisualEncoder,
    InteractionEncoder,
    ScriptedNominalPolicy2D,
    DynamicsPredictor,
    Toy2DConfig,
    Toy2DVectorEnv,
    expand_latent,
    inner_update_z,
    sample_toy2d_domains,
)


def build_system(max_residual=0.15):
    torch.manual_seed(7)
    return FastAdapterSystem(
        encoder=InteractionEncoder(obj_dim=2, tar_dim=2, proprio_dim=2, action_dim=2),
        base_policy=ScriptedNominalPolicy2D(),
        adapter=FastAdapter(max_residual=max_residual),
        dynamics=DynamicsPredictor(),
        inner_lr=0.05,
    )


def make_batch(batch_size=4, seed=10):
    env = Toy2DVectorEnv(Toy2DConfig())
    domain = sample_toy2d_domains(batch_size, seed + 1, env.config)
    obs = env.reset(batch_size, seed, domain=domain)
    return env, domain, obs


def support_step(system, env, domain, obs, z):
    action, delta, base, u = system.act(obs.adapter_observation(), z)
    next_obs = env.step(obs, domain, action)
    u_next = system.encode(next_obs.adapter_observation())
    updated = inner_update_z(
        system=system,
        u=u,
        action_observed=next_obs.prev_action,
        z=z,
        u_next=u_next,
        loss_config=AdaptationLossConfig(),
        create_graph=True,
    )
    return next_obs, updated, action, delta, base, u, u_next


def test_shapes_and_bounded_residual():
    system = build_system(max_residual=torch.tensor([0.15, 0.1]))
    env, domain, obs = make_batch(batch_size=5)
    z = expand_latent(system.z0, 5)
    action, delta, base, u = system.act(obs.adapter_observation(), z)
    assert u.shape == (5, 256)
    assert z.shape == (5, 32)
    assert action.shape == (5, 2)
    assert base.shape == (5, 2)
    assert torch.all(delta.abs() <= torch.tensor([0.150001, 0.100001]))


def test_inner_loop_changes_only_z_without_parameter_grads():
    system = build_system()
    env, domain, obs = make_batch(batch_size=3)
    z = expand_latent(system.z0, 3)
    before = {name: p.detach().clone() for name, p in system.named_parameters()}
    next_obs, updated, *_ = support_step(system, env, domain, obs, z)
    assert not torch.allclose(updated.z_next, z)
    for name, param in system.named_parameters():
        assert torch.equal(before[name], param.detach()), name
        assert param.grad is None, name


class ParametricFrozenBase(BaseSkillPolicy):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, obs, u):
        del u
        return torch.tanh(self.linear(obs.tar_feat - obs.obj_feat))


def test_outer_second_order_gradients_reach_trainable_modules_but_not_base_or_frozen_visual():
    torch.manual_seed(7)
    frozen_visual = FrozenVisualEncoder(in_dim=2, feature_dim=2)
    system = FastAdapterSystem(
        encoder=InteractionEncoder(obj_dim=2, tar_dim=2, proprio_dim=2, action_dim=2),
        base_policy=ParametricFrozenBase(),
        adapter=FastAdapter(),
        dynamics=DynamicsPredictor(),
        inner_lr=0.05,
    )
    env, domain, obs = make_batch(batch_size=4)

    def through_frozen_visual(toy_obs):
        return AdapterObservation(
            obj_feat=frozen_visual(toy_obs.state),
            tar_feat=frozen_visual(toy_obs.target),
            proprio=torch.zeros_like(toy_obs.state),
            prev_action=toy_obs.prev_action,
        )

    z = expand_latent(system.z0, 4)
    runtime_obs = through_frozen_visual(obs)
    action, _delta, _base, u = system.act(runtime_obs, z)
    next_obs = env.step(obs, domain, action)
    u_next = system.encode(through_frozen_visual(next_obs))
    updated = inner_update_z(
        system=system,
        u=u,
        action_observed=next_obs.prev_action,
        z=z,
        u_next=u_next,
        loss_config=AdaptationLossConfig(),
        create_graph=True,
    )

    query = env.reset(4, 202, domain=domain)
    q_action, q_delta, _q_base, _ = system.act(through_frozen_visual(query), updated.z_next)
    expert = env.expert_action(query, domain)
    loss = torch.nn.functional.mse_loss(q_action, expert) + 1.0e-3 * q_delta.square().mean()
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.dynamics.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.adapter.parameters())
    assert system.z0.grad is not None and system.z0.grad.abs().sum() > 0
    assert all(p.grad is None for p in system.base_policy.parameters())
    assert all(p.grad is None for p in frozen_visual.parameters())


def test_query_uses_new_initial_state_and_fixed_z():
    system = build_system()
    env, domain, obs = make_batch(batch_size=2)
    z = expand_latent(system.z0, 2)
    _, updated, *_ = support_step(system, env, domain, obs, z)
    query = env.reset(2, 999, domain=domain)
    assert not torch.allclose(query.state, obs.state)
    z_before_query = updated.z_next.detach().clone()
    for _ in range(3):
        action, _, _, _ = system.act(query.adapter_observation(), updated.z_next)
        query = env.step(query, domain, action)
    assert torch.allclose(updated.z_next.detach(), z_before_query)


def test_cross_domain_latent_resets_to_checkpoint_z0():
    system = build_system()
    env, domain, obs = make_batch(batch_size=1)
    z = expand_latent(system.z0, 1)
    _, updated, *_ = support_step(system, env, domain, obs, z)
    assert not torch.allclose(updated.z_next.detach(), system.z0.detach().unsqueeze(0))
    new_domain_z = expand_latent(system.z0, 1)
    assert torch.allclose(new_domain_z.detach(), system.z0.detach().unsqueeze(0))


def test_hidden_domain_parameters_are_not_runtime_inputs():
    fields = {field.name for field in dataclasses.fields(AdapterObservation)}
    assert fields == {"obj_feat", "tar_feat", "proprio", "prev_action"}
    env, domain, obs = make_batch(batch_size=2)
    runtime = obs.adapter_observation()
    assert not hasattr(runtime, "gain")
    assert not hasattr(runtime, "bias")
    assert domain.gain.shape == (2, 2)
    assert domain.bias.shape == (2, 2)


def test_zero_support_keeps_checkpoint_latent_and_positive_support_updates_it():
    system = build_system()
    env, domain, obs = make_batch(batch_size=2)
    z0 = expand_latent(system.z0, 2)
    assert torch.allclose(z0.detach(), system.z0.detach().expand(2, -1))
    _, updated, *_ = support_step(system, env, domain, obs, z0)
    assert not torch.allclose(updated.z_next.detach(), system.z0.detach().expand(2, -1))


def test_inner_update_is_batch_size_invariant_for_repeated_domain():
    system = build_system()
    torch.manual_seed(123)
    u = torch.randn(1, 256)
    u_next = torch.randn(1, 256)
    action = torch.randn(1, 2).clamp(-1, 1)
    z_single = expand_latent(system.z0, 1)
    single = inner_update_z(
        system=system,
        u=u,
        action_observed=action,
        z=z_single,
        u_next=u_next,
        loss_config=AdaptationLossConfig(),
        create_graph=True,
    ).z_next.detach()

    copies = 7
    z_batch = expand_latent(system.z0, copies)
    batch = inner_update_z(
        system=system,
        u=u.expand(copies, -1),
        action_observed=action.expand(copies, -1),
        z=z_batch,
        u_next=u_next.expand(copies, -1),
        loss_config=AdaptationLossConfig(),
        create_graph=True,
    ).z_next.detach()
    assert torch.allclose(batch[0:1], single, atol=1e-6)
    assert torch.allclose(batch, single.expand(copies, -1), atol=1e-6)



def test_runner_query_loss_has_outer_action_gradients_without_residual_reg():
    from tools.fast_adapter.train_toy2d import query_loss, support_adapt

    system = build_system()
    env, domain, obs = make_batch(batch_size=8, seed=303)
    z, _support_tail, _metrics = support_adapt(
        system,
        env,
        domain,
        obs,
        k=3,
        loss_config=AdaptationLossConfig(),
    )
    query = env.reset(8, 404, domain=domain)
    loss, metrics = query_loss(
        system,
        env,
        domain,
        query,
        z,
        horizon=10,
        residual_reg=0.0,
    )
    assert metrics["query_action_mse"] > 0.0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.adapter.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in system.dynamics.parameters())
    assert system.z0.grad is not None and system.z0.grad.abs().sum() > 0
