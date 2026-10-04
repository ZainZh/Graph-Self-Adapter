"""Qualify one physical branch bank for two independently sealed experiments."""
import json
from pathlib import Path
import torch
from tools.fast_adapter.p35_common import ROOT,bindings,digest,dump


def main():
    other=Path('evaluations/fast_adapter_p36_geometry_tokens_v01')
    for root in (ROOT,other):
        assert not (root/'branches/OUTCOMES.pt').exists(), 'shared protocol must precede outcomes'
        p=root/'branches/PREDICTION_SEAL.json'
        for b in json.loads(p.read_text())['bindings']:
            assert digest(b['path'])==b['sha256'],b['path']
    a=torch.load(ROOT/'branches/PREDICTIONS.pt',weights_only=True)['snapshots']
    b=torch.load(other/'branches/PREDICTIONS.pt',weights_only=True)['snapshots']
    key=lambda s:tuple(s[k] for k in ('task','role','object_id','t'))
    bb={key(s):s for s in b};assert len(a)==len(b)==len(bb)==96
    for s in a:
        assert key(s) in bb and torch.equal(s['candidates'],bb[key(s)]['candidates']),key(s)
    dump(ROOT/'branches/SHARED_BRANCH_SEAL.json',{
        'before_outcomes':True,'experiments':['P35','P36'],'snapshot_keys_equal':True,
        'all_nine_candidate_actions_exactly_equal':True,'snapshots':96,'physical_branches':864,
        'owner':'P35','outcomes_path':str(ROOT/'branches/OUTCOMES.pt'),
        'permission':'P36 may grade only its sealed choices after physical completion; no model changes',
        'bindings':bindings([ROOT/'branches/PREDICTION_SEAL.json',other/'branches/PREDICTION_SEAL.json',
            ROOT/'branches/PREDICTIONS.pt',other/'branches/PREDICTIONS.pt',Path(__file__)])})
    print('P35_P36_SHARED_BRANCHES_QUALIFIED',flush=True)


if __name__=='__main__':main()
