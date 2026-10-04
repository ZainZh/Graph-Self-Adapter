"""New P31 paths. Never write to the sealed P29/P30 cohorts."""
import json
from pathlib import Path
import torch
from tools.fast_adapter.p30v2_common import digest, dump, state_hash, base_model, base_path, P29
from prove.fast_adapter.continuous_adapter import ObjectAdapter
from prove.fast_adapter.interaction_stage_adapter import StageObserver

ROOT = Path('evaluations/fast_adapter_p31_interaction_stage_v01')
TASK = 'mug-handle-lift'
OLD = Path('evaluations/fast_adapter_p30_online_v02')
FIELDS = ('visual', 'proprio', 'actions', 'teacher', 'near')


def protocol():
    return json.loads((ROOT/'PROTOCOL.json').read_text())


def observer():
    net = StageObserver().cuda()
    net.load_state_dict(torch.load(ROOT/'observer.pt', map_location='cpu', weights_only=True)['state'])
    return net.eval().requires_grad_(False)


def adapter(seed, variant):
    blob = torch.load(ROOT/'models'/f'seed{seed}'/f'{variant}.pt', map_location='cpu', weights_only=True)
    net = ObjectAdapter().cuda()
    net.load_state_dict(blob['state'])
    return net.eval().requires_grad_(False)


def source_paths():
    return [ROOT/'PROTOCOL.json', ROOT/'EXECUTION_AMENDMENT.json', Path('prove/fast_adapter/interaction_stage_adapter.py'),
            Path('prove/fast_adapter/continuous_adapter.py'), Path('prove/fast_adapter/object_interaction_env.py'),
            Path('tools/fast_adapter/p29_common.py'), Path('tools/fast_adapter/p22_data.py'),
            Path('tools/fast_adapter/p31_common.py'), Path('tools/fast_adapter/p31_data.py'),
            Path('tools/fast_adapter/p31_train.py'), Path('tools/fast_adapter/p31_eval.py'),
            base_path(TASK), P29/TASK/'data/PCA.pt']


def bindings(paths):
    return [{'path': str(p), 'sha256': digest(p)} for p in paths]
