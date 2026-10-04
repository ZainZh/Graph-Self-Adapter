"""Isolated P34 paths and source bindings."""
from tools.fast_adapter.p33_common import dump,digest,bindings,DINO,base_model,base_path,state_hash
from pathlib import Path
import json
import torch
ROOT=Path('evaluations/fast_adapter_p34_role_tokens_v01')
TASKS=('mug-handle-lift','hammer-block')
FAMILIES=('flat','role_attention','relation_mlp','relation_attention')
H=4
def protocol():return json.loads((ROOT/'PROTOCOL.json').read_text())
def load_data(task,device='cuda'):
    p=ROOT/task/'data/BANK.pt' if task==TASKS[0] else Path('evaluations/fast_adapter_p33_relation_dynamics_v01/data/BANK.pt')
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in torch.load(p,weights_only=True).items()}
def stats_path(task):return ROOT/task/'data/STATS.pt'
def source_paths(task):
    return [ROOT/'PROTOCOL.json',base_path(task),Path('prove/fast_adapter/role_dynamics.py'),
            Path('prove/fast_adapter/relational_dynamics.py'),Path(__file__),
            Path('tools/fast_adapter/p34_visual.py'),Path('tools/fast_adapter/p33_visual.py'),
            Path('tools/fast_adapter/p29_common.py'),Path('prove/fast_adapter/lila_frontend.py'),
            Path('prove/fast_adapter/object_interaction_env.py'),Path('prove/fast_adapter/held_hammer_env.py'),
            stats_path(task)]
def model(task,seed,family,device='cuda'):
    from prove.fast_adapter.role_dynamics import RoleDynamics
    p=ROOT/task/'models'/f'seed{seed}'/f'{family}.pt'
    n=RoleDynamics(family,task).to(device);n.load_state_dict(torch.load(p,weights_only=True,map_location='cpu')['state'])
    return n.eval().requires_grad_(False)
