"""P37: frozen P35/P36 candidates, fresh paired full-task simulator evaluation.

The observation boundary returns RGB-derived features and proprioception only.
Evaluator state is recorded separately and read after the trajectory is sealed.
"""
from pathlib import Path
import argparse
import gc
import json
import os
import time
import numpy as np
import torch

from tools.fast_adapter.p33_common import dump, digest, base_model, state_hash
from tools.fast_adapter.p35_common import model as view_model
from tools.fast_adapter.p36_common import model as geometry_model
from tools.fast_adapter.p35_runtime import MultiViewState
from tools.fast_adapter.p36_runtime import GeometryRoleState
from tools.fast_adapter.p34_visual import RoleVisual, calibration
from tools.fast_adapter.p35_visual import NativeVisual
from tools.fast_adapter.p35_camera import CameraRig
from prove.fast_adapter.object_interaction_env import ObjectEnv, object_config, mujoco
from prove.fast_adapter.held_hammer_env import HeldHammerEnv, PreparationFailed

ROOT = Path('evaluations/fast_adapter_p37_full_success_v01')
TASKS = ('mug-handle-lift', 'hammer-block')
VIEWS = ('g224', 'g448', 'g224w', 'g448w')
GEOMETRIES = ('no_explicit_bbox_geometry', 'p34_existing_bbox_geometry', 'p36_explicit_relation_geometry')
MOUNT = Path('evaluations/fast_adapter_p35_resolution_views_v01/preflight/WRIST_MOUNT.json')


def variants():
    out = [dict(kind='base', representation='base', seed=None, mode='fixed')]
    for kind, reps in (('p35', VIEWS), ('p36', GEOMETRIES)):
        for rep in reps:
            for seed in (11, 22):
                for mode in ('fixed', 'continuous'):
                    out.append(dict(kind=kind, representation=rep, seed=seed, mode=mode))
    return out


def build_manifest():
    conditions = []
    for ti, task in enumerate(TASKS):
        for si, split in enumerate(('train', 'geometry_ood')):
            for i in range(8):
                seed = 37000000 + ti * 100000 + si * 1000 + i
                conditions.append(dict(task=task, split=split, condition=i,
                    config=object_config(task, seed, split), reset_seed=seed+500000))
    jobs = []
    for ti, task in enumerate(TASKS):
        for split in ('train', 'geometry_ood'):
            for v in variants():
                tag = f"{task}__{split}__{v['kind']}__{v['representation']}__s{v['seed']}__{v['mode']}"
                # Round-robin high/low resolution work; all workers own disjoint jobs.
                jobs.append(dict(id=tag, task=task, split=split, **v))
    for i, job in enumerate(jobs):
        job['worker'] = i % 3
    return dict(conditions=conditions, jobs=jobs, episodes=928, conditions_per_task_split=8)


class LiveVision:
    """One DINO instance, shared by the unchanged low/native frontends."""
    def __init__(self, task, model_dir):
        import tools.fast_adapter.p33_visual as historical_visual
        historical_visual.DINO = model_dir  # asset location only; historical file bytes unchanged
        self.low = RoleVisual(task)
        self.native = NativeVisual.__new__(NativeVisual)
        for k in ('front', 'spatial_projection', 'context_projection'):
            setattr(self.native, k, getattr(self.low, k))
        self.task = task

    @torch.inference_mode()
    def observe(self, envs, rigs, q, job):
        rgb = np.stack([e.rgb() for e in envs])
        low = self.low.encode(rgb, q)
        ob = {**low, 'proprio': torch.as_tensor(q, device='cuda')}
        if job['kind'] == 'p35':
            view = job['representation']
            size = 448 if '448' in view else 224
            g = low
            if size == 448:
                frames, cals = zip(*(r.render(size, 'global') for r in rigs))
                g = self.native.encode_view(np.stack(frames), q, cals, self.task)
            if view.endswith('w'):
                frames, cals = zip(*(r.render(size, 'wrist') for r in rigs))
                w = self.native.encode_view(np.stack(frames), q, cals, self.task)
            else:
                w = {k: torch.zeros_like(g[k]) for k in ('tokens', 'boxes', 'presence')}
            ob['target_tokens'], ob['target_boxes'] = low['tokens'], low['boxes']
            for k in ('tokens', 'boxes', 'presence'):
                ob[k] = torch.stack((g[k], w[k]), 1)
        return ob, rgb


def cpu_ob(ob):
    return {k: v.detach().cpu().clone() for k, v in ob.items()}


def new_model(job):
    if job['kind'] == 'base':
        return None
    if job['kind'] == 'p35':
        return view_model(job['task'], job['representation'], job['seed'], 'relation_attention')
    return geometry_model(job['task'], job['seed'], job['representation'])


def new_state(net, n, job):
    if net is None:
        return None
    cls = MultiViewState if job['kind'] == 'p35' else GeometryRoleState
    return cls(net, n, job['mode'])


def execute(job, conditions, vision, base, horizon=200):
    net = new_model(job)
    weights = dict(base=state_hash(base), adapter=state_hash(net) if net is not None else None)
    envs, rigs, valid, qs, preparation = [], [], [], [], []
    cls = ObjectEnv if job['task'] == TASKS[0] else HeldHammerEnv
    start = time.time()
    try:
        for i, c in enumerate(conditions):
            e = cls(c['config']); envs.append(e)
            try:
                q = e.reset(c['reset_seed'])
                valid.append(i); qs.append(q)
                preparation.append(dict(passed=True, steps=e.preparation['steps'] if hasattr(e, 'preparation') else 0))
            except PreparationFailed:
                preparation.append(dict(passed=False, steps=e.preparation['steps']))
        ee = [envs[i] for i in valid]
        if not ee:
            return dict(valid=[], preparation=preparation, job=job, conditions=conditions,
                        horizon=horizon, weights_before=weights, weights_after=weights), [], net
        if job['kind'] == 'p35' and job['representation'] != 'g224':
            mount = json.loads(MOUNT.read_text())
            rigs = [CameraRig(e, mount) for e in ee]
        vision.low.cal = calibration(ee[0])
        ob, rgb = vision.observe(ee, rigs, np.stack(qs), job)
        n = len(ee); learner = new_state(net, n, job)
        previous = torch.zeros(n, 4, device='cuda')
        observations = [cpu_ob(ob)]; states = [[e.state() for e in ee]]
        actions, base_actions, choices, scores, latents, counts, diagnostics = [], [], [], [], [], [], []
        frames = [torch.as_tensor(rgb.copy())]; frame_steps = [0]
        for t in range(horizon):
            with torch.no_grad():
                ba = base(ob['visual'], ob['proprio'], previous)
                if job['task'] == 'hammer-block': ba[:, 3] = .9
                if learner is None:
                    a, idx, sc = ba, torch.zeros(n, dtype=torch.long, device='cuda'), torch.zeros(n, 9, device='cuda')
                else:
                    latents.append(learner.z.cpu().clone()); counts.append(learner.updates)
                    a, idx, sc = learner.act(ob, ba)
                a = a.clamp(-1, 1).detach().clone()
                if job['task'] == 'hammer-block': a[:, 3] = .9
            q = np.stack([e.step(x) for e, x in zip(ee, a.cpu().numpy())])
            no, rgb = vision.observe(ee, rigs, q, job)
            if learner is not None:
                d = learner.observe(a, no)
                diagnostics.append({k: v.cpu() if torch.is_tensor(v) else v for k, v in d.items()})
            actions.append(a.cpu()); base_actions.append(ba.detach().cpu()); choices.append(idx.cpu()); scores.append(sc.cpu())
            observations.append(cpu_ob(no)); states.append([e.state() for e in ee])
            if t+1 in (50, 100, 150, horizon):
                frames.append(torch.as_tensor(rgb.copy())); frame_steps.append(t+1)
            ob, previous = no, a
            if (t+1) % 50 == 0:
                print('STEP', job['id'], t+1, 'seconds', round(time.time()-start, 1), flush=True)
        raw = {k: torch.stack([o[k] for o in observations], 1) for k in observations[0]}
        raw.update(actions=torch.stack(actions, 1), base_actions=torch.stack(base_actions, 1),
                   choices=torch.stack(choices, 1), scores=torch.stack(scores, 1),
                   states={k: torch.tensor(np.array([[s[j][k] for s in states] for j in range(n)])) for k in states[0][0]},
                   frames=torch.stack(frames, 1), frame_steps=frame_steps,
                   valid=valid, preparation=preparation, conditions=conditions, job=job, horizon=horizon,
                   weights_before=weights, diagnostics=diagnostics, seconds=time.time()-start)
        if learner is not None:
            raw.update(z_before_action=torch.stack(latents, 1), update_counts=counts,
                       z_final=learner.z.cpu(), updates=learner.updates)
        raw['weights_after'] = dict(base=state_hash(base), adapter=state_hash(net) if net is not None else None)
        assert raw['weights_after'] == weights
        # The evaluator metrics are kept out of every observation and latent update.
        return raw, [e.metrics for e in ee], net
    finally:
        for r in rigs: r.close()
        for e in envs: e.close()


def audit(raw, base, net):
    """Replay every action/latent from stored public inputs, and every success state."""
    job = raw['job']; n = len(raw['valid']); h = raw['horizon']
    out = dict(passed=True, action_max_error=0., score_max_error=0., z_max_error=0., physical_states=0,
               choices_equal=True, physical_success={}, weights_frozen=raw['weights_before']==raw['weights_after'])
    if not n: return out
    learner = new_state(net, n, job)
    previous = torch.zeros(n, 4, device='cuda')
    keys = ('visual', 'proprio', 'tokens', 'boxes', 'presence', 'target_tokens', 'target_boxes')
    for t in range(h):
        ob = {k: raw[k][:, t].cuda() for k in keys if k in raw}
        no = {k: raw[k][:, t+1].cuda() for k in keys if k in raw}
        with torch.no_grad():
            ba = base(ob['visual'], ob['proprio'], previous)
            if job['task'] == 'hammer-block': ba[:, 3] = .9
            if learner is None: a = ba
            else:
                out['z_max_error'] = max(out['z_max_error'], float((learner.z.cpu()-raw['z_before_action'][:,t]).abs().max()))
                assert learner.updates == raw['update_counts'][t]
                a, ix, score = learner.act(ob, ba)
                out['choices_equal'] &= torch.equal(ix.cpu(), raw['choices'][:,t])
                out['score_max_error'] = max(out['score_max_error'], float((score.cpu()-raw['scores'][:,t]).abs().max()))
            a = a.clamp(-1,1)
            if job['task'] == 'hammer-block': a[:,3] = .9
            out['action_max_error'] = max(out['action_max_error'], float((a.cpu()-raw['actions'][:,t]).abs().max()))
            actual = raw['actions'][:,t].cuda()
            if learner is not None: learner.observe(actual, no)
            previous = actual
    assert out['choices_equal'] and max(out[k] for k in ('action_max_error','score_max_error','z_max_error')) < 1e-4, out
    for j, i in enumerate(raw['valid']):
        c = raw['conditions'][i]
        cls = ObjectEnv if job['task'] == TASKS[0] else HeldHammerEnv
        e = cls(c['config'], render=False)
        try:
            e.reset(c['reset_seed']); ss=[]
            for t in range(1,h+1):
                for k,v in raw['states'].items(): getattr(e.env.data,k)[:] = v[j,t].numpy()
                mujoco.mj_forward(e.env.model,e.env.data)
                _, m = e.env.evaluate_state(e.env._get_obs(),raw['actions'][j,t-1].numpy())
                ss.append(bool(m['success'])); out['physical_states'] += 1
            out['physical_success'][str(i)] = dict(success=any(ss), terminal_success=ss[-1], first_success=next((t+1 for t,x in enumerate(ss) if x),None))
        finally: e.close()
    return out


def finish(raw, metrics, net, base, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir/'TRAJECTORY.pt'; torch.save(raw, path)
    ref = dict(path=str(path), sha256=digest(path))
    dump(outdir/'TRAJECTORY_SEAL.json',dict(trajectory=ref, sealed_at=time.time(), evaluation_summary_not_yet_read=True))
    rows=[]
    for i,c in enumerate(raw['conditions']):
        j = raw['valid'].index(i) if i in raw['valid'] else None
        mm = metrics[j] if j is not None else []
        first = next((t+1 for t,m in enumerate(mm) if m['success']),None)
        rows.append(dict(**c, **{k:raw['job'][k] for k in ('kind','representation','seed','mode')},
            preparation=raw['preparation'][i], success=first is not None,
            terminal_success=bool(mm[-1]['success']) if mm else False, first_success=first,
            source=ref, source_row=j, updates=raw.get('updates',0) if j is not None else 0))
    dump(outdir/'OUTCOMES.json', dict(rows=rows, metrics=metrics, trajectory=ref))
    aa = audit(raw,base,net)
    for r in rows:
        if r['source_row'] is not None:
            assert all(r[k]==v for k,v in aa['physical_success'][str(r['condition'])].items())
    aa['trajectory']=ref; dump(outdir/'AUDIT.json',aa)
    dump(outdir/'COMPLETE.json',dict(job=raw['job'],episodes=len(rows),successful=sum(r['success'] for r in rows),
        seconds=raw.get('seconds',0), audit_passed=True, outcomes_sha256=digest(outdir/'OUTCOMES.json')))
    print('COMPLETE', raw['job']['id'], sum(r['success'] for r in rows), '/', len(rows), flush=True)


def verify_freeze():
    freeze=json.loads((ROOT/'FREEZE.json').read_text())
    for b in freeze['bindings']: assert digest(b['path'])==b['sha256'], b['path']


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--dino',required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.smoke:
        jobs=[]
        for task in TASKS:
            jobs += [dict(task=task,kind='p35',representation='g448w',seed=11,mode='continuous',id=task+'_vision_smoke'),
                     dict(task=task,kind='p36',representation=GEOMETRIES[-1],seed=11,mode='continuous',id=task+'_geometry_smoke')]
        manifest=None
    else:
        verify_freeze();manifest=json.loads((ROOT/'MANIFEST.json').read_text())
        jobs=[j for j in manifest['jobs'] if j['worker']==args.worker]
    vision=base=None;task=None
    for job in jobs:
        outdir=ROOT/('smoke' if args.smoke else 'runs')/job['id']
        if (outdir/'COMPLETE.json').exists(): continue
        if not args.smoke and (outdir/'TRAJECTORY.pt').exists():
            raise RuntimeError(f'Preserve prior trajectory and diagnose incomplete job: {outdir}')
        if job['task']!=task:
            del vision,base;gc.collect();torch.cuda.empty_cache()
            task=job['task'];vision=LiveVision(task,args.dino);base=base_model(task)
        conditions = ([dict(task=task,split='training_smoke',condition=0,
                       config=object_config(task,290000,'train'),reset_seed=290100)] if args.smoke else
                      [c for c in manifest['conditions'] if c['task']==task and c['split']==job['split']])
        raw,metrics,net=execute(job,conditions,vision,base,horizon=8 if args.smoke else 200)
        finish(raw,metrics,net,base,outdir)
        del raw,metrics,net;gc.collect();torch.cuda.empty_cache()
    print('WORKER_COMPLETE', args.worker, flush=True)


if __name__=='__main__': main()
