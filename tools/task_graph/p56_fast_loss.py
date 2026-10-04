"""Mathematically equivalent P53 full-stage loss with cached public encodings.

This module is an optional performance candidate. It neither changes the
frozen trainer nor introduces an experiment, checkpoint, or rollout. The
history predictor is batched; all second-order inner gradients are retained.
"""
import torch


def _prediction_errors(model, encoded, actions, next_features, z):
    """Per-transition P53 feedback error for one or many consecutive steps."""
    c = model.config
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
        actions = actions.unsqueeze(0)
        next_features = {key: value.unsqueeze(0) for key, value in next_features.items()}
    count = encoded.shape[0]
    latent = z.expand(count, -1)
    prediction = model.predictor(torch.cat((encoded, latent, actions), -1))
    cut = c.views * 3 * c.appearance_dim
    appearance = prediction[..., :cut].reshape(count, c.views, 3, c.appearance_dim)
    boxes = prediction[..., cut:cut + c.views * 12].reshape(count, c.views, 3, 4)
    proprio = prediction[..., cut + c.views * 12:]
    target_appearance = ((next_features['raw'].detach() - model.raw_mean) / model.raw_scale)[..., :c.appearance_dim]
    target_boxes = (next_features['boxes'].detach() - model.box_mean) / model.box_scale
    target_proprio = (next_features['proprio'].detach() - model.proprio_mean) / model.proprio_scale
    mask = next_features['visible'].detach()
    app_error = (appearance - target_appearance).square().mean(-1)
    box_error = (boxes - target_boxes).square().mean(-1)
    return (((app_error + box_error) * mask).sum((-1, -2)) / mask.sum((-1, -2)).clamp_min(1)
            + (proprio - target_proprio).square().mean(-1))


def _auxiliary(model, encoded, action, next_features, z):
    """Exact P53 isolated predictor auxiliary, with encoder and latent detached."""
    c = model.config
    prediction = model.predictor(torch.cat((encoded.detach(), z.detach(), action.detach()), -1))
    cut = c.views * 3 * c.appearance_dim
    appearance = prediction[..., :cut].reshape(c.views, 3, c.appearance_dim)
    boxes = prediction[..., cut:cut + c.views * 12].reshape(c.views, 3, 4)
    proprio = prediction[..., cut + c.views * 12:]
    with torch.no_grad():
        target_appearance = ((next_features['raw'] - model.raw_mean) / model.raw_scale)[..., :c.appearance_dim]
        target_boxes = (next_features['boxes'] - model.box_mean) / model.box_scale
        target_proprio = (next_features['proprio'] - model.proprio_mean) / model.proprio_scale
        mask = next_features['visible']
    per_role = (appearance - target_appearance).square().mean(-1) + (boxes - target_boxes).square().mean(-1)
    return ((per_role * mask).sum((-1, -2)) / mask.sum((-1, -2)).clamp_min(1)
            + (proprio - target_proprio).square().mean(-1)).mean()


def fast_full_stage_loss(model, features, executed_actions, expert_actions, *, horizon,
                         meta=True, fixed_bc_weight=.5, auxiliary_weight=.1):
    """P53 full_stage_loss contract with shared per-frame encode and batched history.

    A complete native stage starts at t0; expert labels remain outer-only.
    Feedback uses the actual executed action and the following public frame.
    """
    length = len(executed_actions)
    if not 1 <= length <= horizon:
        raise ValueError('Stage exceeds declared training horizon')
    if executed_actions.shape != (length, 6) or expert_actions.shape != (length, 6):
        raise ValueError('Expected real executed/expert OSC6 sequences')
    if set(features) != {'raw', 'boxes', 'visible', 'proprio'} or any(len(x) != length + 1 for x in features.values()):
        raise ValueError('Expected exactly T+1 public feature observations')
    at = lambda t: {key: value[t] for key, value in features.items()}
    # The final frame is a target only. One encoder call per action frame
    # replaces repeated action/auxiliary/history encoder calls.
    encoded = torch.stack([model.encode(at(t)) for t in range(length)])
    z0 = model.initial(at(0))
    z = z0
    online_bc = []
    fixed_bc = []
    auxiliary = []
    latents = []
    for t in range(length):
        nxt = at(t + 1)
        online_action = torch.tanh(model.policy(torch.cat((encoded[t], z), -1)))
        fixed_action = torch.tanh(model.policy(torch.cat((encoded[t], z0), -1)))
        online_bc.append((online_action - expert_actions[t]).square().mean())
        fixed_bc.append((fixed_action - expert_actions[t]).square().mean())
        auxiliary.append(_auxiliary(model, encoded[t], executed_actions[t], nxt, z))
        latents.append(z)
        if meta:
            if not z.requires_grad:
                z = z.detach().requires_grad_(True)
            start = max(0, t + 1 - model.config.history)
            history_next = {key: value[start + 1:t + 2] for key, value in features.items()}
            errors = _prediction_errors(model, encoded[start:t + 1],
                                        executed_actions[start:t + 1], history_next, z)
            inner_loss = errors.mean(0).sum() + model.config.anchor * (z - z0).square().mean(-1).sum()
            gradient, = torch.autograd.grad(inner_loss, z, create_graph=True)
            z = z - model.config.inner_lr * gradient
    online_bc = torch.stack(online_bc).mean()
    fixed_bc = torch.stack(fixed_bc).mean()
    auxiliary = torch.stack(auxiliary).mean()
    control = ((1 - fixed_bc_weight) * online_bc + fixed_bc_weight * fixed_bc) if meta else fixed_bc
    return control + auxiliary_weight * auxiliary, dict(control=control, online_bc=online_bc,
        fixed_bc=fixed_bc, auxiliary=auxiliary, action_latents=torch.stack(latents), final_z=z)
