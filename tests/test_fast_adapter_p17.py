from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
from pathlib import Path

import torch

from prove.fast_adapter import Toy2DConfig, Toy2DVectorEnv, sample_toy2d_domains
from tools.fast_adapter.p15_experiment import repeat_domain, tensor_state_hash
from tools.fast_adapter.p16_experiment import P16HistoryConfig, SupportRecord, build_history_system, collect_common_base_records
from tools.fast_adapter.p17_experiment import (
    P17RetrofitConfig,
    P17RetrofittedHistorySystem,
    adapt_history_latent,
    build_retrofit_from_history_checkpoint,
    compute_z_init,
    evaluate_retrofit,
    history_action_from_z,
    make_swapped_correction,
    query_loss_corrected_history,
    rollout_history_z_arrays,
    sample_p17_domains,
    support_physical_fingerprint,
    support_prediction_terms,
    train_retrofit,
)


def _history_checkpoint(tmp_path: Path, *, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    cfg = P16HistoryConfig(config_id="history", width=256)
    system = build_history_system(cfg, device=torch.device("cpu"))
    path = tmp_path / f"history_seed{seed}.pt"
    torch.save({"system": system.state_dict(), "config": {"history_config": asdict(cfg)}}, path)
    return path


def test_p17_u0_matches_loaded_history_actions_exactly(tmp_path: Path):
    device = torch.device("cpu")
    checkpoint = _history_checkpoint(tmp_path, seed=17)
    model, _config = build_retrofit_from_history_checkpoint(checkpoint, device=device)
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(3, 100, env.config, device=device)
    records = collect_common_base_records(model.history, env, domain, support_seed=101, max_s=3)
    z_init = compute_z_init(model, records, batch_size=3, device=device)
    qobs = env.reset(3, 102, domain=domain)

    z_u0, metrics = adapt_history_latent(model, records, z_init, u_passes=0, cfg=P17RetrofitConfig(), create_graph=False)
    assert metrics == []
    assert torch.equal(z_u0, z_init)
    action, delta, base, u = history_action_from_z(model, qobs, z_u0)
    u_ref = model.history.encode(qobs)
    base_ref = model.history.base_action(qobs, u_ref)
    delta_ref = model.history.adapter(u_ref, z_init, base_ref)

    assert torch.equal(u, u_ref)
    assert torch.equal(base, base_ref)
    assert torch.equal(delta, delta_ref)
    assert torch.equal(action, base_ref + delta_ref)


def test_p17_training_and_eval_leave_history_hash_unchanged(tmp_path: Path):
    history_checkpoint = _history_checkpoint(tmp_path, seed=23)
    before_model, _config = build_retrofit_from_history_checkpoint(history_checkpoint, device=torch.device("cpu"))
    before_hash = tensor_state_hash(before_model.history)
    train_dir = tmp_path / "train"
    summary = train_retrofit(argparse.Namespace(
        history_checkpoint=str(history_checkpoint),
        train_seed=900000001,
        output_dir=str(train_dir),
        updates=1,
        batch_size=4,
        support_s_train=3,
        gradient_u_train=3,
        query_horizon=2,
        train_stream_offset=65_000_000,
        device="cpu",
        log_every=0,
    ))
    assert summary["history_freeze_audit"]["unchanged"]
    after_model, _config = build_retrofit_from_history_checkpoint(history_checkpoint, device=torch.device("cpu"))
    assert tensor_state_hash(after_model.history) == before_hash

    eval_dir = tmp_path / "eval"
    eval_summary = evaluate_retrofit(argparse.Namespace(
        checkpoint=str(train_dir / "checkpoint.pt"),
        output_dir=str(eval_dir),
        config_id=P17RetrofitConfig().config_id,
        train_seed=900000001,
        split="smoke",
        domain_seed=900000101,
        support_seed=900000201,
        calibration_seed=900000301,
        query_seed=900000401,
        num_domains=3,
        query_repeats=2,
        query_horizon=2,
        calibration_starts=1,
        u_values=[0, 1, 3],
        strata=["iid"],
        device="cpu",
    ))
    assert eval_summary["freeze_audit"]["unchanged"]


def test_p17_outer_action_loss_gradient_reaches_predictor_without_auxiliary():
    device = torch.device("cpu")
    torch.manual_seed(3)
    model = P17RetrofittedHistorySystem(build_history_system(P16HistoryConfig(), device=device))
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(5, 33, env.config, device=device)
    records = collect_common_base_records(model.history, env, domain, support_seed=44, max_s=3)
    z_init = compute_z_init(model, records, batch_size=5, device=device)
    z, _metrics = adapt_history_latent(model, records, z_init, u_passes=3, cfg=P17RetrofitConfig(), create_graph=True)
    qobs = env.reset(5, 55, domain=domain)
    loss, _query = query_loss_corrected_history(model, env, domain, qobs, z, horizon=3, residual_reg=0.0)
    loss.backward()
    predictor_grad = sum(float(p.grad.detach().abs().sum()) for p in model.predictor.parameters() if p.grad is not None)
    assert predictor_grad > 0.0


def test_p17_inner_signature_uses_records_and_z_only_with_detached_physics():
    signature = inspect.signature(adapt_history_latent)
    assert "domain" not in signature.parameters
    assert "env" not in signature.parameters
    device = torch.device("cpu")
    model = P17RetrofittedHistorySystem(build_history_system(P16HistoryConfig(), device=device))
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(2, 11, env.config, device=device)
    records = collect_common_base_records(model.history, env, domain, support_seed=12, max_s=3)
    assert all(not record.action.requires_grad and not record.delta_x.requires_grad for record in records)


def test_p17_support_update_is_batch_invariant_for_repeated_records():
    device = torch.device("cpu")
    torch.manual_seed(5)
    model = P17RetrofittedHistorySystem(build_history_system(P16HistoryConfig(), device=device))
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(1, 123, env.config, device=device)
    records1 = collect_common_base_records(model.history, env, domain, support_seed=456, max_s=3)
    z_init1 = compute_z_init(model, records1, batch_size=1, device=device)
    z1 = z_init1.detach().clone().requires_grad_(True)
    loss1 = support_prediction_terms(model, records1, z1, z_init1, P17RetrofitConfig())["total"].sum()
    grad1 = torch.autograd.grad(loss1, z1)[0]

    records4 = [
        SupportRecord(
            u=record.u.repeat(4, 1),
            action=record.action.repeat(4, 1),
            u_next=record.u_next.repeat(4, 1),
            state=record.state.repeat(4, 1),
            next_state=record.next_state.repeat(4, 1),
        )
        for record in records1
    ]
    z_init4 = z_init1.repeat(4, 1)
    z4 = z_init4.detach().clone().requires_grad_(True)
    loss4 = support_prediction_terms(model, records4, z4, z_init4, P17RetrofitConfig())["total"].sum()
    grad4 = torch.autograd.grad(loss4, z4)[0]
    assert torch.allclose(grad4, grad1.repeat(4, 1), atol=1e-6, rtol=1e-5)


def test_p17_swapped_correction_preserves_current_history_init():
    z_init = torch.tensor([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])
    correction = torch.tensor([[0.1, 0.2], [1.0, 1.2], [2.0, 2.2]])
    z_adapted = z_init + correction
    perm = torch.tensor([2, 0, 1])
    swapped = make_swapped_correction(z_init, z_adapted, perm)
    assert torch.allclose(swapped, z_init + correction[perm])
    assert not torch.allclose(swapped, z_adapted[perm])


def test_p17_domain_strata_bands_are_valid():
    device = torch.device("cpu")
    toy = Toy2DConfig()
    iid = sample_p17_domains(128, 1, toy, stratum="iid", device=device)
    assert torch.all((iid.gain >= 0.75) & (iid.gain <= 1.25))
    assert torch.all((iid.bias >= -0.075) & (iid.bias <= 0.075))
    gain = sample_p17_domains(128, 2, toy, stratum="gain_ood", device=device)
    assert torch.all(((gain.gain >= 0.6) & (gain.gain <= 0.75)) | ((gain.gain >= 1.25) & (gain.gain <= 1.4)))
    assert torch.all((gain.bias >= -0.075) & (gain.bias <= 0.075))
    bias = sample_p17_domains(128, 3, toy, stratum="bias_ood", device=device)
    assert torch.all((bias.gain >= 0.75) & (bias.gain <= 1.25))
    assert torch.all((bias.bias.abs() >= 0.075) & (bias.bias.abs() <= 0.09))
    joint = sample_p17_domains(128, 4, toy, stratum="joint_ood", device=device)
    assert torch.all(((joint.gain >= 0.6) & (joint.gain <= 0.75)) | ((joint.gain >= 1.25) & (joint.gain <= 1.4)))
    assert torch.all((joint.bias.abs() >= 0.075) & (joint.bias.abs() <= 0.09))


def test_p17_rollout_actions_are_bounded():
    device = torch.device("cpu")
    torch.manual_seed(7)
    model = P17RetrofittedHistorySystem(build_history_system(P16HistoryConfig(), device=device))
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_p17_domains(2, 200, env.config, stratum="joint_ood", device=device)
    records = collect_common_base_records(model.history, env, domain, support_seed=201, max_s=3)
    z_init = compute_z_init(model, records, batch_size=2, device=device)
    z, _metrics = adapt_history_latent(model, records, z_init, u_passes=3, cfg=P17RetrofitConfig(), create_graph=False)
    qdomain = repeat_domain(domain, 3)
    qobs = env.reset(6, 202, domain=qdomain)
    arrays = rollout_history_z_arrays(model, env, qdomain, qobs, z.repeat_interleave(3, 0), z_init.repeat_interleave(3, 0), arm="corrected_u3", horizon=4)
    assert float(arrays["action_abs_max"].max()) <= env.config.action_limit + 1.0e-6


def test_p17_physical_support_fingerprint_ignores_encoded_u_for_shared_evidence():
    device = torch.device("cpu")
    torch.manual_seed(8)
    history_a = build_history_system(P16HistoryConfig(), device=device)
    torch.manual_seed(9)
    history_b = build_history_system(P16HistoryConfig(), device=device)
    env = Toy2DVectorEnv(Toy2DConfig(), device=device)
    domain = sample_toy2d_domains(4, 301, env.config, device=device)
    records_a = collect_common_base_records(history_a, env, domain, support_seed=302, max_s=3)
    records_b = collect_common_base_records(history_b, env, domain, support_seed=302, max_s=3)
    assert support_physical_fingerprint(records_a) == support_physical_fingerprint(records_b)
    assert all(torch.equal(a.action, b.action) for a, b in zip(records_a, records_b))
    assert any(not torch.equal(a.u, b.u) for a, b in zip(records_a, records_b))
