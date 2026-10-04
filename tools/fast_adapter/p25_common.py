"""P25 small development trial: shared inputs, phase-spanning feedback."""
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.feedback_adapter import FeedbackAdapter
from tools.fast_adapter.p24_common import load_base, weight_hash, PARENT

R = Path('evaluations/fast_adapter_p25_feedback_v01')
SOURCE = Path('evaluations/fast_adapter_p24_common_prior_v01')
VARIANTS = ('original', 'anchored', 'physical', 'history')
SEEDS = (11, 22)
FIELDS = ('visual', 'next_visual', 'proprio', 'next_proprio', 'previous_action', 'actions')
WINDOWS = ((0, 8), (8, 24), (24, 32))


def select_feedback(proprio, actions):
    """One completed transition per elapsed window; no goal/contact truth."""
    response_error = ((proprio[:, 1:33]-proprio[:, :32])/.008-actions[:, :32]).square().sum(-1)
    selected = []
    for lo, hi in WINDOWS:
        selected.append(response_error[:, lo:hi].argmax(1)+lo)
    return torch.stack(selected, 1)


def transitions(visual, proprio, actions, indices):
    row = torch.arange(len(visual), device=visual.device)[:, None]
    a = actions[row, indices]
    prev = actions[row, (indices-1).clamp_min(0)] * (indices > 0).unsqueeze(-1)
    return dict(visual=visual[row, indices], next_visual=visual[row, indices+1],
        proprio=proprio[row, indices], next_proprio=proprio[row, indices+1],
        previous_action=prev, actions=a)


def load_adapter(seed, variant, device='cuda'):
    net = FeedbackAdapter(variant).to(device).eval().requires_grad_(False)
    state = torch.load(R/'models'/f'seed{seed}'/f'{variant}.pt', map_location=device, weights_only=True)
    net.load_state_dict(state['state'])
    return net


def make_bank():
    # Each domain has six independently reset query trajectories from old TRAINING runs.
    paths = [SOURCE/'training_refresh'/f'seed{s}'/f'step{step}/BANK.pt'
             for s in (11, 22, 33) for step in (10000, 30000)]
    values = []
    for path in paths:
        b = torch.load(path, weights_only=True)
        values.append({'visual': b['query_visual'].float(), 'proprio': b['proprio'][:, 6:].float(),
                       'actions': b['query_action'].float(), 'expert': b['expert_label'].float()})
    return {k: torch.stack([v[k] for v in values]) for k in values[0]}, paths
