"""P38 bounded numerical qualification: direct control, sealed paired tests."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.causal_mapping import (DirectController, Identification,
    nominal, draw_domains, effect, fit_ls)

ROOT = Path('evaluations/fast_adapter_p38_causal_v01')


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generator(seed):
    return torch.Generator(device='cuda').manual_seed(seed)


def uniform(shape, g, low=-1., high=1.):
    return low+(high-low)*torch.rand(shape, generator=g, device='cuda')


def support(e, g, nsteps=5, noise=.005, excitation=True):
    actions = torch.tensor([[.7, 0.], [0., .7], [-.7, -.7], [.5, -.5], [-.4, .6]], device='cuda')[:nsteps]
    actions = actions[None].expand(len(e), -1, -1).clone()
    if not excitation:
        actions.zero_()
    state, goal = uniform((len(e), nsteps, 2), g), uniform((len(e), nsteps, 2), g)
    delta = effect(e, actions)+noise*torch.randn(actions.shape, generator=g, device='cuda')
    return state, goal, actions, delta


def controller_train():
    out = ROOT/'mapping'/'controller'; out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(3801); g = generator(3802)
    net = DirectController().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    began = time.time(); log = []
    for step in range(1, 8001):
        e = draw_domains(512, g, 'cuda', 'id')
        relation = uniform((512, 2), g)
        action = net(relation, e)
        # Differentiable numerical task loss, privileged dynamics TRAIN ONLY.
        wanted = .6*relation
        loss = (effect(e, action)-wanted).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1 or step % 1000 == 0:
            row = {'step': step, 'loss': float(loss.detach()), 'seconds': time.time()-began}
            log.append(row); print(row, flush=True)
    torch.save(net.cpu().state_dict(), out/'model.pt')
    save_json(out/'TRAIN.json', {'log': log, 'sha256': sha(out/'model.pt'), 'training_domain': 'id_only'})


def controller():
    net = DirectController().cuda()
    net.load_state_dict(torch.load(ROOT/'mapping/controller/model.pt', weights_only=True))
    return net.eval().requires_grad_(False)


def train(kind, seed, steps):
    out = ROOT/'mapping'/kind/f'seed{seed}'; out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed); g = generator(380000+seed)
    model = Identification(kind).cuda(); actor = controller()
    opt = torch.optim.Adam(model.parameters(), lr=.0007)
    began = time.time(); records = []
    for step in range(1, steps+1):
        e = draw_domains(64, g, 'cuda')
        k = (1, 2, 3, 5)[step % 4]
        obs = support(e, g, nsteps=k)
        z, _ = model.adapt(*obs, updates=k, meta=True)
        relation = uniform((64, 2), g)
        estimated = model.response(z)
        a = actor(relation, estimated)
        task_loss = (effect(e, a)-.6*relation).square().mean()
        qa = uniform((64, 4, 2), g, -.7, .7)
        qs, qg = uniform((64, 4, 2), g), uniform((64, 4, 2), g)
        prediction_loss = (model.predict(qs, qg, qa, z)-effect(e, qa)).square().mean()
        loss = task_loss+.2*prediction_loss
        assert torch.isfinite(loss), (kind, seed, step)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step()
        if step == 1 or step % 500 == 0 or step == steps:
            row = {'step': step, 'task_loss': float(task_loss.detach()),
                   'prediction_loss': float(prediction_loss.detach()),
                   'z_rms': float((z-model.z0).detach().square().mean().sqrt()),
                   'rate_mean': float(model.rate.detach().mean()), 'seconds': time.time()-began}
            records.append(row); save_json(out/'PROGRESS.json', row); print(kind, seed, row, flush=True)
    torch.save(model.cpu().state_dict(), out/'model.pt')
    save_json(out/'TRAIN.json', {'kind': kind, 'seed': seed, 'updates': steps,
              'log': records, 'model_hash': sha(out/'model.pt'),
              'controller_hash': sha(ROOT/'mapping/controller/model.pt')})


def trajectory(actor, estimated, initial, goal, e, model=None, mode='fixed', g=None):
    x = initial.clone(); z = model.initial(len(x)) if model is not None else None
    raw = {'state': [x.detach().cpu()], 'action': [], 'z': [], 'response': []}
    sx, sg, sa, sy = [], [], [], []
    for t in range(8):
        with torch.no_grad():
            ee = model.response(z) if mode == 'online' else estimated
            action = actor(goal-x, ee)
            new_x = x+effect(e, action)
        raw['action'].append(action.cpu()); raw['response'].append(ee.detach().cpu())
        if mode == 'online':
            sx.append(x); sg.append(goal); sa.append(action)
            sy.append((new_x-x)+.005*torch.randn(x.shape, generator=g, device='cuda'))
            fields = [torch.stack(v[-4:], 1) for v in (sx, sg, sa, sy)]
            z, _ = model.update(z, *fields, meta=False)
            raw['z'].append(z.cpu())
        x = new_x; raw['state'].append(x.cpu())
    raw = {k: torch.stack(v, 1) for k, v in raw.items() if v}
    raw['goal'] = goal.cpu()
    return raw


def evaluate(kind, seed, split, phase):
    out = ROOT/'mapping'/kind/f'seed{seed}'
    model = Identification(kind).cuda()
    model.load_state_dict(torch.load(out/'model.pt', weights_only=True))
    model.eval().requires_grad_(False); actor = controller()
    target = out/phase/split; target.mkdir(parents=True, exist_ok=False)
    n = 128 if phase == 'dev' else 512
    seed_map = {'native': 0, 'id': 1, 'ood': 2}
    g = generator((381000 if phase == 'dev' else 389000)+seed_map[split])
    e = draw_domains(n, g, 'cuda', split)
    obs = support(e, g, nsteps=3)
    x, goal = uniform((n, 2), g, -.6, .6), uniform((n, 2), g, -.6, .6)
    z, losses = model.adapt(*obs, updates=3)
    with torch.no_grad():
        responses = {'frozen': nominal(n, 'cuda'), 'correct': model.response(z),
                     'swapped': model.response(z.roll(1, 0)), 'least_squares': fit_ls(obs[2], obs[3]),
                     'oracle_learned': e}
    raw = {name: trajectory(actor, ee, x, goal, e) for name, ee in responses.items()}
    if kind != 'history':
        for updates in (1, 5, 10, 20):
            zz, _ = model.adapt(*obs, updates=updates)
            raw[f'updates{updates}'] = trajectory(actor, model.response(zz), x, goal, e)
        zero_obs = list(obs); zero_obs[3] = torch.zeros_like(obs[3])
        zz, _ = model.adapt(*zero_obs, updates=3)
        raw['zero_feedback'] = trajectory(actor, model.response(zz), x, goal, e)
    raw['online'] = trajectory(actor, None, x, goal, e, model, mode='online', g=g)
    # Seal runtime predictions/actions before attaching evaluator dynamics/labels.
    torch.save({'observations': [v.cpu() for v in obs], 'latent': z.detach().cpu(),
                'trajectories': raw}, target/'RUNTIME.pt')
    save_json(target/'SEAL.json', {'runtime_hash': sha(target/'RUNTIME.pt'),
              'model_hash': sha(out/'model.pt'), 'time': time.time(), 'n_domains': n})
    torch.save({'dynamics': e.cpu()}, target/'EVALUATOR.pt')
    rows = {}
    for name, trace in raw.items():
        errors = (trace['state']-trace['goal'][:, None]).norm(dim=-1)
        # Final, not ever-hit, success. Domain is the statistical unit.
        rows[name] = {'success': (errors[:, -1] < .03).tolist(),
                      'error': errors[:, -1].tolist(), 'error_curve': errors.mean(0).tolist()}
    with torch.enable_grad():
        zs = z.detach().requires_grad_(True)
        aa = actor(goal-x, model.response(zs))
        sensitivities = []
        for coordinate in range(2):
            sensitivities.append(torch.autograd.grad(aa[:, coordinate].sum(), zs, retain_graph=True)[0])
        jac = torch.stack(sensitivities, 1).norm(dim=(1, 2))
    with torch.no_grad():
        qa = uniform((n, 10, 2), g, -.7, .7)
        qs, qg = uniform((n, 10, 2), g), uniform((n, 10, 2), g)
        pred_before = model.predict(qs, qg, qa, model.initial(n))
        pred_after = model.predict(qs, qg, qa, z)
        target_delta = effect(e, qa)
        diagnostics = {'heldout_mse_before': float((pred_before-target_delta).square().mean()),
          'heldout_mse_after': float((pred_after-target_delta).square().mean()),
          'response_rmse': float((model.response(z)-e).square().mean().sqrt()),
          'action_delta_rms': float((aa-actor(goal-x, nominal(n, 'cuda'))).square().mean().sqrt()),
          'jacobian_action_wrt_z': float(jac.mean()),
          'latent_between_domain_rms': float(z.std(dim=0).square().mean().sqrt()),
          'loss_curve': [float(v.mean()) for v in losses]}
    save_json(target/'RESULT.json', {'kind': kind, 'seed': seed, 'phase': phase,
              'split': split, 'n': n, 'rows': rows, 'diagnostics': diagnostics})
    print(kind, seed, phase, split, {k: round(np.mean(v['success'])*100, 2) for k,v in rows.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['controller', 'train', 'eval'])
    p.add_argument('--kind', choices=['structured', 'effect', 'absolute', 'relation', 'history'])
    p.add_argument('--seed', type=int, default=11); p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--phase', choices=['dev', 'final'], default='dev')
    a = p.parse_args(); torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    if a.mode == 'controller': controller_train()
    elif a.mode == 'train': train(a.kind, a.seed, a.steps)
    else:
        for split in ('native', 'id', 'ood'): evaluate(a.kind, a.seed, split, a.phase)
