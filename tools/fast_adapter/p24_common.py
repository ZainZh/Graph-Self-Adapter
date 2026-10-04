"""Common paths and frozen frontend loading for P24."""
from pathlib import Path
import hashlib
import torch
from prove.fast_adapter.skill_adapter import SkillAdapter
from prove.fast_adapter.visual_proprio_adapter import VisualBase

R = Path('evaluations/fast_adapter_p24_common_prior_v01')
PARENT = Path('evaluations/fast_adapter_p22_geometry_v01')
KS = (0, 1, 2, 3, 5)
SEEDS = (11, 22, 33)


def weight_hash(net):
    h = hashlib.sha256()
    for key, value in net.state_dict().items():
        h.update(key.encode()); h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def load_base(device='cuda'):
    base = VisualBase().to(device).eval().requires_grad_(False)
    base.load_state_dict(torch.load(PARENT / 'models/base/model.pt', map_location=device, weights_only=True)['state'])
    return base


def load_adapter(seed, name='meta', checkpoint=None, device='cuda'):
    net = SkillAdapter(history=name == 'history').to(device).eval().requires_grad_(False)
    path = checkpoint or R / 'models' / f'seed{seed}' / f'{name}.pt'
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True)['state'])
    return net
