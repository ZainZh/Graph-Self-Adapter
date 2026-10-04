"""P64 task heads, public-residual GN and train-only fixed-budget fitting.

This module never accesses evaluator state or development data. A task owns
its train-case loader and loss, including the frozen action decoder and the
0.25 fixed-query term. Runtime loads only the new P64 checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from prove.fast_adapter.p61_encoder import P61RelationEncoder, _sha256

TASK_DIMS = {'cup': 5, 'book': 9}


@dataclass(frozen=True)
class P64EncoderOutput:
    u: Tensor
    geometry: Tensor
    role_tokens: Tensor


class P64RoleEncoder(nn.Module):
    """Exact P61 official LiLa/ROI/relation trunk, new position/size head.

    Cup: surface center in first-EEF anchor, log outer half sizes.
    Book: held origin in current EEF, caddy origin in first-EEF anchor,
    log book axis scales. No predicted or adapted orientation coordinates.
    ``backbone`` injection is reserved for lightweight structural tests.
    """
    def __init__(self, lila_source=None, lila_checkpoint=None, *, task: str,
                 geometry_mean: Sequence[float] | Tensor,
                 p61_checkpoint=None, finetune_lila=True, max_views=4,
                 adapter_microbatch=8, backbone=None):
        super().__init__()
        if task not in TASK_DIMS:
            raise ValueError(f'Unknown P64 task: {task}')
        mean = torch.as_tensor(geometry_mean, dtype=torch.float32)
        if mean.shape != (TASK_DIMS[task],) or not torch.isfinite(mean).all():
            raise ValueError('geometry_mean must be finite and match task dimension')
        self.task = task
        self.initialization_binding = None
        self.backbone = backbone or P61RelationEncoder(
            lila_source, lila_checkpoint, finetune_lila=finetune_lila,
            geometry_mean8=[0.] * 8, max_views=max_views,
            adapter_microbatch=adapter_microbatch)
        if p61_checkpoint is not None:
            path = Path(p61_checkpoint).resolve()
            package = torch.load(path, map_location='cpu', weights_only=True)
            if (package.get('schema') != 'prove.p61-relation-checkpoint.v1'
                    or package.get('training', {}).get('arm') != 'task_trained_lila'):
                raise ValueError('Initialization requires task-trained P61 checkpoint')
            for field in ('source_sha256', 'checkpoint_sha256'):
                if package['lila_binding'][field] != self.backbone.lila_binding[field]:
                    raise ValueError(f'P61 official LiLa binding mismatch: {field}')
            self.backbone.load_state_dict(package['state_dict'], strict=True)
            self.initialization_binding = {'path': str(path), 'sha256': _sha256(path)}
        self.backbone.set_lila_trainable(finetune_lila)
        self.backbone.geometry_head.requires_grad_(False)
        # Head initialization is seed11 independently of caller RNG consumption.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            self.geometry_head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256),
                                               nn.GELU(), nn.Linear(256, TASK_DIMS[task]))
            nn.init.normal_(self.geometry_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.geometry_head[-1].bias)
        self.register_buffer('geometry_mean', mean.clone())

    def forward(self, **public_inputs):
        output = self.backbone(**public_inputs)
        geometry = self.geometry_mean + self.geometry_head(output.u)
        if not torch.isfinite(geometry).all():
            raise FloatingPointError('Non-finite P64 geometry')
        return P64EncoderOutput(output.u, geometry, output.role_tokens)

    def parameter_groups(self):
        official = [p for module in (self.backbone.fusion, self.backbone.visual_adapter)
                    for p in module.parameters() if p.requires_grad]
        ids = {id(p) for p in official}
        return {'lila': official, 'task': [p for p in self.parameters()
                                         if p.requires_grad and id(p) not in ids]}

    def constructor(self):
        return dict(lila_source=self.backbone.lila_binding['source'],
                    lila_checkpoint=self.backbone.lila_binding['checkpoint'],
                    task=self.task, geometry_mean=self.geometry_mean.detach().cpu(),
                    finetune_lila=True, max_views=self.backbone.max_views,
                    adapter_microbatch=self.backbone.adapter_microbatch)


@dataclass(frozen=True)
class P64InnerConfig:
    steps: int = 3
    damping: float = 1e-4
    step_scale: float = .65
    step_clip: float = 1.


def adapt_z(residual_from_z: Callable[[Tensor], Tensor | None] | None,
            start_z: Tensor, *, create_graph=False, config=P64InnerConfig(),
            residual_factory=None):
    """Three trace-normalized GN steps, arbitrary 1-D latent, full outer graph.

    A factory may freeze discrete hull memberships at each linearization. It
    receives detached z and returns a differentiable residual callable. None
    or empty measurements preserve z without fabricating observations.
    """
    if config != P64InnerConfig():
        raise ValueError('P64 fixes three GN steps, damping1e-4, scale.65, clip1')
    if start_z.ndim != 1 or not start_z.numel() or not start_z.is_floating_point():
        raise ValueError('start_z must be a nonempty floating vector')
    if not torch.isfinite(start_z).all():
        raise FloatingPointError('Non-finite initial latent')
    z = start_z if create_graph else start_z.detach()
    records = []
    identity = torch.eye(z.numel(), dtype=z.dtype, device=z.device)
    with torch.enable_grad():
        for _ in range(config.steps):
            callback = residual_factory(z.detach()) if residual_factory else residual_from_z
            if callback is None:
                break
            if not z.requires_grad:
                z = z.requires_grad_(True)
            residual = callback(z)
            if residual is None or residual.numel() == 0:
                break
            if residual.ndim != 1 or not torch.isfinite(residual).all():
                raise FloatingPointError('Public residual must be finite 1-D vector')
            jacobian = torch.func.jacfwd(callback)(z)
            if not torch.isfinite(jacobian).all():
                raise FloatingPointError('Non-finite public residual Jacobian')
            metric = jacobian.T @ jacobian / residual.numel()
            gradient = jacobian.T @ residual / residual.numel()
            curvature = (metric.trace() / z.numel()).clamp_min(1e-12)
            change = torch.linalg.solve(metric / curvature + config.damping * identity,
                                        gradient / curvature)
            norm = change.square().sum().clamp_min(1e-24).sqrt()
            change = change * (config.step_clip / norm).clamp(max=1.)
            next_z = z - config.step_scale * change
            if not torch.isfinite(next_z).all():
                raise FloatingPointError('Non-finite GN latent')
            records.append(dict(loss=float((.5 * residual.square().mean()).detach()),
                                gradient_norm=float(gradient.norm().detach()),
                                curvature_scale=float(curvature.detach()),
                                delta_norm=float((next_z-z).norm().detach()),
                                active_dim=z.numel(), measurements=residual.numel()))
            z = next_z if create_graph else next_z.detach()
    return (z if create_graph else z.detach()), records


def _state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_training_freeze(freeze_path, model, bindings):
    """Admit only a root-reviewed exact freeze, checking all file bytes."""
    path = Path(freeze_path).resolve()
    freeze = json.loads(path.read_text())
    review_path = path.parent / 'ROOT_TRAINING_REVIEW.json'
    review = json.loads(review_path.read_text())
    if (review.get('approved') is not True or review.get('reviewer') != 'root'
            or review.get('freeze_sha256') != _sha256(path)):
        raise ValueError('Missing exact root training review')
    if (freeze.get('schema') != 'prove.p64-training-freeze.v1'
            or freeze.get('task') != model.task or freeze.get('bindings') != dict(bindings)):
        raise ValueError('P64 freeze task or provenance bindings mismatch')
    for key, expected in dict(supervised_steps=500, meta_steps=100, batch_size=2,
                              seed=11, official_lr=1e-5, task_lr=1e-4).items():
        if freeze.get('training', {}).get(key) != expected:
            raise ValueError(f'P64 freeze training constant mismatch: {key}')
    files = freeze.get('files', {})
    resolved = {}
    for name, expected in files.items():
        source = Path(name).resolve()
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f'P64 frozen file changed: {name}')
        resolved[str(source)] = expected
    required = [__file__, model.initialization_binding['path'],
                model.backbone.lila_binding['source'], model.backbone.lila_binding['checkpoint']]
    if any(str(Path(name).resolve()) not in resolved for name in required):
        raise ValueError('Freeze must bind shared code, P61 checkpoint and official LiLa files')
    if resolved[str(Path(model.initialization_binding['path']).resolve())] != model.initialization_binding['sha256']:
        raise ValueError('Initialized P61 checkpoint differs from freeze')
    return dict(path=str(path), sha256=_sha256(path), review_path=str(review_path),
                review_sha256=_sha256(review_path))


def train_encoder(model, train_cases, loss_callback, *, checkpoint_path,
                  bindings: Mapping[str, Any], freeze_path, device='cuda'):
    """Fixed 500 supervised +100 meta steps; batch2, seed11, final-only save.

    Callers construct the model under seed11 and pass train cases exclusively.
    Callback ``(model, case, phase) -> scalar loss`` owns temporal sampling.
    ``bindings`` must bind source, checkpoint and cache manifests. No dev
    callback or checkpoint selection API exists.
    """
    output = Path(checkpoint_path)
    if output.exists() or output.with_suffix('.training.jsonl').exists():
        raise FileExistsError('P64 refuses checkpoint/run overwrite')
    if not train_cases or not model.initialization_binding:
        raise ValueError('P64 fitting requires train cases and frozen P61 initialization')
    if not all(key in bindings for key in ('source', 'checkpoint', 'cache')):
        raise ValueError('Missing source/checkpoint/cache provenance bindings')
    if not model.backbone.finetune_lila:
        raise ValueError('P64 fitting must train both official LiLa modules')
    freeze_binding = verify_training_freeze(freeze_path, model, bindings)
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    model.to(device).train()
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = model.parameter_groups()
    optimizer = torch.optim.AdamW([dict(params=groups['lila'], lr=1e-5),
                                   dict(params=groups['task'], lr=1e-4)], weight_decay=.01)
    modules = {name: getattr(model.backbone, name) for name in ('fusion', 'visual_adapter')}
    proof = {name: dict(initial_sha256=_state_hash(module), max_gradient_norm=0.)
             for name, module in modules.items()}
    old_head_hash = _state_hash(model.backbone.geometry_head)
    schedule = hashlib.sha256()
    logpath = output.with_suffix('.training.jsonl')
    for phase, count in (('supervised', 500), ('meta', 100)):
        for step in range(count):
            indices = rng.integers(len(train_cases), size=2).tolist()
            schedule.update(json.dumps([phase, step, indices]).encode())
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index in indices:
                loss = loss_callback(model, train_cases[index], phase)
                if loss.ndim or not torch.isfinite(loss):
                    raise FloatingPointError(f'Invalid {phase} loss at {step}')
                (loss / 2).backward()
                losses.append(float(loss.detach()))
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('Non-finite training gradients')
            for name, module in modules.items():
                norm = sum(float(p.grad.detach().square().sum()) for p in module.parameters()
                           if p.grad is not None) ** .5
                proof[name]['max_gradient_norm'] = max(proof[name]['max_gradient_norm'], norm)
            nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            record = dict(phase=phase, step=step+1, episode_indices=indices,
                          mean_loss=sum(losses)/2)
            with logpath.open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            if step == 0 or (step+1) % 10 == 0:
                print(json.dumps(record), flush=True)
    for name, module in modules.items():
        proof[name]['final_sha256'] = _state_hash(module)
        if (proof[name]['max_gradient_norm'] <= 0 or
                proof[name]['initial_sha256'] == proof[name]['final_sha256']):
            raise AssertionError(f'Official {name} did not receive effective updates')
    if _state_hash(model.backbone.geometry_head) != old_head_hash:
        raise AssertionError('Unused P61 geometry head changed')
    summary = dict(supervised_steps=500, meta_steps=100, batch_size=2, seed=11,
                   official_lr=1e-5, task_lr=1e-4, weight_decay=.01,
                   selection='final fixed budget', official_modules=proof,
                   sample_schedule_sha256=schedule.hexdigest(),
                   old_head_unchanged=True, all_gradients_finite=True)
    package = dict(schema='prove.p64-role-checkpoint.v1', constructor=model.constructor(),
                   lila_binding=model.backbone.lila_binding,
                   initialization_binding=model.initialization_binding,
                   bindings=dict(bindings), training_freeze=freeze_binding, training=summary,
                   state_dict={n:v.detach().cpu() for n,v in model.state_dict().items()})
    with output.open('xb') as stream:
        torch.save(package, stream)
    return summary


def load_encoder(checkpoint_path, device='cuda', *, expected_bindings=None, expected_sha256=None):
    """Runtime requires a new trained P64 checkpoint; never runs P61 predictions."""
    if expected_sha256 is not None and _sha256(Path(checkpoint_path)) != expected_sha256:
        raise ValueError('P64 runtime checkpoint hash mismatch')
    package = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if expected_bindings is not None and package.get('bindings') != dict(expected_bindings):
        raise ValueError('P64 runtime provenance mismatch')
    if package.get('schema') != 'prove.p64-role-checkpoint.v1':
        raise ValueError('Runtime requires a P64 task checkpoint')
    training = package['training']
    for key, expected in dict(supervised_steps=500, meta_steps=100, batch_size=2,
                              seed=11, selection='final fixed budget').items():
        if training.get(key) != expected:
            raise ValueError(f'P64 training contract mismatch: {key}')
    model = P64RoleEncoder(**package['constructor'])
    for field in ('source_sha256', 'checkpoint_sha256'):
        if model.backbone.lila_binding[field] != package['lila_binding'][field]:
            raise ValueError(f'Official LiLa source changed: {field}')
    model.load_state_dict(package['state_dict'], strict=True)
    model.initialization_binding = package['initialization_binding']
    return model.to(device).eval().requires_grad_(False)
