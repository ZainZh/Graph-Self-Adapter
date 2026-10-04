import hashlib
import json
from pathlib import Path
import torch
from prove.fast_adapter.continuous_adapter import ObjectSkill, ObjectAdapter

ROOT = Path('evaluations/fast_adapter_p30_online_v02')
P29 = Path('evaluations/fast_adapter_p29_object_transfer_v01/tasks')
TASKS = ('mug-handle-lift', 'hammer-block')


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(net):
    h = hashlib.sha256()
    for k,v in sorted(net.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def base_path(task):
    return P29/task/'models/base_final.pt' if task == TASKS[0] else ROOT/task/'base.pt'


def base_model(task):
    net = ObjectSkill().cuda()
    net.load_state_dict(torch.load(base_path(task),map_location='cpu',weights_only=True)['state'])
    return net.eval().requires_grad_(False)


def bank_path(task): return ROOT/task/'TRAIN.pt'


def load_bank(task):
    return {k:v.cuda() for k,v in torch.load(bank_path(task),weights_only=True).items()}


def adapters(task, seed):
    out = {}
    for kind in ('gradient','history'):
        blob = torch.load(ROOT/task/'models'/f'seed{seed}'/f'{kind}.pt',map_location='cpu',weights_only=True)
        net = ObjectAdapter(history=kind=='history').cuda()
        net.load_state_dict(blob['state']); net.eval().requires_grad_(False)
        out[kind] = (net,blob)
    return out
