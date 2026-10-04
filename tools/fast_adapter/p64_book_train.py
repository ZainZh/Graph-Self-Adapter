"""P64 book task loss callbacks; bounded shared trainer requires root freeze."""
import argparse
import json
from pathlib import Path
import torch
from prove.fast_adapter.p64_book_bridge import BookPriors,adapt_book_z,frozen_book_action
from prove.fast_adapter.p64_book_runtime import encoder_inputs,decode_measurements
from prove.fast_adapter.p64_common import P64RoleEncoder,train_encoder
from tools.fast_adapter.p64_book_data import sha


def load_public(folder):
    return torch.load(Path(folder)/'public.pt',weights_only=True)


def prediction(model,public):
    device=next(model.parameters()).device
    return model(**encoder_inputs(public,device),image_hw=(448,448)).geometry[0]


class BookLoss:
    def __init__(self,priors,policy):
        self.priors=priors;self.policy=policy.eval().requires_grad_(False)

    def __call__(self,model,case,phase):
        public=load_public(case['path']);q=prediction(model,public)
        label=torch.load(Path(case['path'])/'private.pt',weights_only=True)['geometry'].to(q)
        scale=q.new_tensor([.02]*6+[.25]*3)
        args=(self.priors,public['eef_transform'],public['bboxwh'],public['gripper'],public['T_world_anchor'])
        zeros=torch.zeros_like(q)
        with torch.no_grad():target_action=frozen_book_action(self.policy,label,zeros,*args)
        action_scale=q.new_tensor([.2]*3+[.1]*3)
        def query_loss(latent):
            geometry_loss=((q+latent*scale-label)/scale).square().mean()
            action=frozen_book_action(self.policy,q,latent,*args)
            action_loss=((action[:6]-target_action[:6])/action_scale).square().mean()
            return geometry_loss+action_loss
        fixed=query_loss(zeros)
        if phase=='supervised':return fixed
        if len(case['support_paths'])!=2:raise ValueError('meta query needs exactly two earlier frames')
        z=torch.zeros_like(q)
        for path,support_t in zip(case['support_paths'],case['meta_support_times']):
            support=load_public(path);g=prediction(model,support)
            measurements=decode_measurements(support['measurements'])
            # Runtime entry t=0 only establishes the episode estimate.
            if support_t>=1 and any(m.t==support_t for m in measurements):
                z,_=adapt_book_z(g,z,self.priors,measurements,create_graph=True)
        # Query pixels only feed its encoder; query mask is not an update target.
        online=query_loss(z)
        return online+.25*fixed


def cases_for_training(manifest):
    by_cell={};cases=[]
    for record in manifest['records']:
        if record['split']!='train':continue
        folder=Path(record['path'])
        for name in ('public','private'):
            if sha(folder/(name+'.pt'))!=record[name+'_sha256']:
                raise ValueError('frozen train cache payload changed: '+name)
        by_cell.setdefault(record['cell_index'],[]).append(record)
    for rows in by_cell.values():
        rows.sort(key=lambda r:r['t'])
        for i,row in enumerate(rows):
            expected=[r['path'] for r in rows[max(0,i-2):i]]
            if row['meta_support_paths']!=expected:raise ValueError('frozen causal support schedule changed')
            cases.append(dict(row,support_paths=expected))
    return cases

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--freeze',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--p61-checkpoint',required=True)
    p.add_argument('--lila-source',required=True);p.add_argument('--lila-checkpoint',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args()
    cache=Path(a.cache);manifest=json.loads((cache/'MANIFEST.json').read_text())
    freeze=json.loads(Path(a.freeze).read_text())
    priors=BookPriors(**json.loads((cache/'PRIORS.json').read_text())['values'])
    torch.manual_seed(11)
    model=P64RoleEncoder(a.lila_source,a.lila_checkpoint,task='book',geometry_mean=manifest['geometry_mean'],p61_checkpoint=a.p61_checkpoint)
    from prove.task_graph.p57_geometry_policy import load_geometry_policy
    root=Path(__file__).resolve().parents[2]/'evaluations/fast_adapter_p57_geometry_tokens_v01'
    policy,_,_=load_geometry_policy(root/'runs_b/HEAD.pt',root/'runs_a2/mask_geometry/HEAD.pt',device=a.device)
    cases=cases_for_training(manifest);meta=[c for c in cases if len(c['support_paths'])==2]
    loss=BookLoss(priors,policy)
    # Shared trainer accepts phase-specific schedule through deterministic case mapping.
    # Every supervised frame retained; meta cycles only causal query eligible cases.
    callback=lambda m,c,phase:loss(m,c if phase=='supervised' else meta[cases.index(c)%len(meta)],phase)
    train_encoder(model,cases,callback,checkpoint_path=a.checkpoint,bindings=freeze['bindings'],freeze_path=a.freeze,device=a.device)
