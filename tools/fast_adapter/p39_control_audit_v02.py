"""Post-seal checks of common probes, latent controls, and matched appearance."""
import json
import torch
from tools.fast_adapter.p39_tokens import ROOT,js,sha

def main():
    summary=dict(batches=0,common_probe_max_action_error=0.,common_probe_max_state_error=0.,
                 same_appearance_pairs=0,matching_initial_rgb_pairs=0,matching_initial_mask_pairs=0,
                 paired_initial_position_max_error=0.,fixed_controls_checked=0,source_bindings=[])
    for p in sorted((ROOT/'final').glob('*/seed*/*/*_RUNTIME.pt')):
        split=p.parent.name;kind=p.parent.parent.parent.name;arm=p.name.removesuffix('_RUNTIME.pt')
        r=torch.load(p,weights_only=True,map_location='cpu')
        base=torch.load(ROOT/'final/references/seed0'/split/'base_RUNTIME.pt',weights_only=True,map_location='cpu')
        ae=float((r['actions'][:,:3]-base['actions'][:,:3]).abs().max())
        pe=float((r['qpos'][:,:4]-base['qpos'][:,:4]).abs().max())
        summary['common_probe_max_action_error']=max(summary['common_probe_max_action_error'],ae)
        summary['common_probe_max_state_error']=max(summary['common_probe_max_state_error'],pe)
        assert ae<1e-7 and pe<1e-8,(str(p),ae,pe)
        if arm in ('mean','first3'):
            t=0 if arm=='mean' else 3
            assert torch.equal(r['z'][:,t:],r['z'][:,t:t+1].expand_as(r['z'][:,t:])),str(p)
            summary['fixed_controls_checked']+=1
        if kind not in ('references','history') and arm!='mean':assert torch.count_nonzero(r['z'][:,0])==0
        if split=='paired':
            config=json.loads(p.with_name(arm+'_EVALUATOR.json').read_text())['configs']
            for i in range(0,64,2):
                for key in ('origin','radius','tool_radius','direction','distance','mass','camera','colors','background','gap','lateral'):
                    assert config[i][key]==config[i+1][key],(i,key)
                summary['same_appearance_pairs']+=1
                summary['matching_initial_rgb_pairs']+=r['rgb_hashes'][0][i]==r['rgb_hashes'][0][i+1]
                summary['matching_initial_mask_pairs']+=r['mask_hashes'][0][i]==r['mask_hashes'][0][i+1]
                error=float((r['positions'][i,0]-r['positions'][i+1,0]).abs().max())
                summary['paired_initial_position_max_error']=max(summary['paired_initial_position_max_error'],error)
                assert error<1e-8,(str(p),i,error)
        summary['batches']+=1;summary['source_bindings'].append(dict(path=str(p),sha256=sha(p)))
    assert summary['batches']==210
    summary['passed']=True;js(ROOT/'summary/CONTROL_AUDIT.json',summary)
    print({k:v for k,v in summary.items() if k!='source_bindings'})

if __name__=='__main__':torch.set_num_threads(2);main()
