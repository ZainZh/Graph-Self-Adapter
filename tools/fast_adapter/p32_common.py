"""P32 IO and frozen parents. No writes to P29/P30/P31."""
import json
from pathlib import Path
import torch
from tools.fast_adapter.p31v2_common import ROOT as P31, TASK, P29, base_model, base_path, observer, bindings, digest, dump, state_hash
from prove.fast_adapter.learned_interaction_loss import LossAdapter

ROOT = Path('evaluations/fast_adapter_p32_learned_loss_v01')
DINO = '/home/zheng/github/LiLa-WAM/dinov3_pretrain/dinov3-vitl16-pretrain-lvd1689m'


def protocol():
    return json.loads((ROOT/'PROTOCOL.json').read_text())


def source_paths():
    visual = json.loads((P31/'VISUAL_FREEZE.json').read_text())['files']
    return [ROOT/'PROTOCOL.json', P31/'data/own.pt', P31/'observer.pt',
            P29/TASK/'data/PCA.pt', base_path(TASK),
            Path('prove/fast_adapter/learned_interaction_loss.py'),
            Path('prove/fast_adapter/continuous_adapter.py'),
            Path('prove/fast_adapter/interaction_stage_adapter.py'),
            Path('prove/fast_adapter/object_interaction_env.py'),
            Path('tools/fast_adapter/p29_common.py'),
            Path('tools/fast_adapter/p31v2_common.py'),
            Path('tools/fast_adapter/p32_common.py'),
            Path('tools/fast_adapter/p32_train.py')]+[Path(x['path']) for x in visual]


def make_pair(seed):
    nets={}
    for family in ('predictive','learned'):
        torch.manual_seed(seed)
        nets[family]=LossAdapter(family).cuda()
    shared=lambda n:{k:v for k,v in n.state_dict().items() if not k.startswith(('predictor.','energy.'))}
    a,b=[shared(n) for n in nets.values()]
    assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
    counts={k:sum(p.numel() for p in n.parameters()) for k,n in nets.items()}
    assert abs(counts['predictive']-counts['learned'])/counts['learned']<.01
    return nets,counts


def adapter(seed,family):
    blob=torch.load(ROOT/'models'/f'seed{seed}'/f'{family}.pt',map_location='cpu',weights_only=True)
    n=LossAdapter(family).cuda();n.load_state_dict(blob['state'])
    return n.eval().requires_grad_(False)
