"""P35 bounded visual-input comparison; shared low-global feedback remains fixed."""
from pathlib import Path
import json
import torch
from tools.fast_adapter.p34_common import load_data as previous_data
from tools.fast_adapter.p33_common import dump,digest,bindings,base_model,base_path,state_hash,DINO
ROOT=Path('evaluations/fast_adapter_p35_resolution_views_v01')
TASKS=('mug-handle-lift','hammer-block');VIEWS=('g224','g448','g224w','g448w')
FAMILIES=('flat','relation_attention');H=4
def protocol():return json.loads((ROOT/'PROTOCOL.json').read_text())
def load_data(task,view,device='cuda'):
    old=previous_data(task,'cpu');keep=(old['object_ids']<24)|((old['object_ids']>=48)&(old['object_ids']<56))
    b={k:v[keep] for k,v in old.items() if torch.is_tensor(v)}
    b['target_tokens']=b['tokens'];b['target_boxes']=b['boxes']
    out=ROOT/task/'data'
    g=torch.load(out/'global448.pt',weights_only=True) if '448' in view else b
    if view.endswith('w'):
        w=torch.load(out/('wrist448.pt' if '448' in view else 'wrist224.pt'),weights_only=True)
    else:w={k:torch.zeros_like(g[k]) for k in ('tokens','boxes','presence')}
    for k in ('tokens','boxes','presence'):b[k]=torch.stack((g[k],w[k]),2)
    return {k:v.to(device) for k,v in b.items()}
def stats_path(task,view):return ROOT/task/'data'/f'STATS_{view}.pt'
def model(task,view,seed,family,device='cuda'):
    from prove.fast_adapter.multiview_role_dynamics import MultiViewRoleDynamics
    n=MultiViewRoleDynamics(family,task).to(device)
    p=ROOT/task/'models'/view/f'seed{seed}'/f'{family}.pt'
    n.load_state_dict(torch.load(p,weights_only=True,map_location='cpu')['state'])
    return n.eval().requires_grad_(False)
def source_paths(task,view):
    return [ROOT/'PROTOCOL.json',ROOT/'preflight/WRIST_MOUNT.json',base_path(task),stats_path(task,view),
      ROOT/task/'data/RECEIPT.json',Path(__file__),Path('tools/fast_adapter/p35_visual.py'),Path('tools/fast_adapter/p35_camera.py'),
      Path('prove/fast_adapter/multiview_role_dynamics.py'),Path('prove/fast_adapter/role_dynamics.py'),
      Path('prove/fast_adapter/relational_dynamics.py')]
