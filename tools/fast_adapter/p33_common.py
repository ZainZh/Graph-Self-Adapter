"""P33 isolated visual-dynamics pilot; never changes historical cohorts."""
import hashlib
import json
from pathlib import Path
import torch
from tools.fast_adapter.p30v2_common import base_model, base_path, state_hash

ROOT=Path('evaluations/fast_adapter_p33_relation_dynamics_v01')
P30=Path('evaluations/fast_adapter_p30_online_v02/hammer-block')
P29=Path('evaluations/fast_adapter_p29_object_transfer_v01/tasks/hammer-block')
TASK='hammer-block'
DINO='/home/zheng/github/LiLa-WAM/dinov3_pretrain/dinov3-vitl16-pretrain-lvd1689m'
H=4

def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def bindings(paths):return [{'path':str(p),'sha256':digest(p)} for p in paths]
def protocol():return json.loads((ROOT/'PROTOCOL.json').read_text())

def source_paths():
    return [ROOT/'PROTOCOL.json',base_path(TASK),P29/'data/PCA.pt',
      Path('prove/fast_adapter/relational_dynamics.py'),Path('tools/fast_adapter/p33_visual.py'),
      Path('tools/fast_adapter/p33_common.py'),Path('prove/fast_adapter/held_hammer_env.py'),
      Path('prove/fast_adapter/object_interaction_env.py'),Path('prove/fast_adapter/lila_frontend.py'),
      Path('prove/fast_adapter/visual_alignment.py'),Path('tools/fast_adapter/p29_common.py'),
      Path('prove/fast_adapter/continuous_adapter.py'),
      Path('data/sources/lila-wam-reference-20260923/VISUAL_WEIGHTS.pt')]

def load_data(device='cuda'):
    b=torch.load(ROOT/'data/BANK.pt',weights_only=True)
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in b.items()}

def model(seed,family,device='cuda'):
    from prove.fast_adapter.relational_dynamics import RelationalDynamics
    b=torch.load(ROOT/'models'/f'seed{seed}'/f'{family}.pt',weights_only=True,map_location='cpu')
    n=RelationalDynamics(family).to(device);n.load_state_dict(b['state'])
    return n.eval().requires_grad_(False)
