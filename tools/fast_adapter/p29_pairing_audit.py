"""Verify identical initial physical/proprio inputs across all sealed query arms."""
import argparse,json,time
import torch
from tools.fast_adapter.p29_common import R,task_root,save_json,digest
from tools.fast_adapter.p29_eval import STRATA

def main(task):
    root=task_root(task);rnd=json.loads((root/'FINAL_FREEZE.json').read_text())['round'];count=0;mx=0.;rows=[]
    for st in STRATA:
        folder=root/'final'/rnd/st;d=json.loads((folder/'RESULT.json').read_text())
        for off in range(0,d['conditions'],8):
            base=torch.load(folder/f'base_{off:03d}.pt',weights_only=True)
            for row in [r for r in d['rows'] if r['offset']==off]:
                b=torch.load(row['path'],weights_only=True)
                assert b['reset_seeds']==base['reset_seeds']
                for k in ('qpos','qvel','mocap_pos','mocap_quat'):assert torch.equal(b['states'][k][:,0],base['states'][k][:,0]),(st,off,row['arm'],k)
                assert torch.equal(b['proprio'][:,0],base['proprio'][:,0])
                assert torch.equal(b['audit_rgb'][:,0],base['audit_rgb'][:,0]),(st,off,row['arm'],'sampled_rgb')
                error=float((b['visual'][:,0]-base['visual'][:,0]).abs().max());mx=max(mx,error);count+=len(b['visual'])
            rows.append({'stratum':st,'offset':off})
    # Physical/proprio and stored initial pixels are required exact. Visual feature
    # differences are reported without choosing a post hoc pass threshold.
    out=R/'summary'/f'INITIAL_PAIRING_AUDIT_{task}.json';assert not out.exists()
    save_json(out,{'time':time.time(),'task':task,'round':rnd,'physical_proprio_initial_equal':True,'stored_initial_rgb_equal':True,'rgb_sampling':'first2 of8 conditions in every block; all conditions have physical/proprio checks','query_episodes':count,'max_initial_visual_feature_difference':mx,'initial_visual_features_exactly_equal':mx==0,'blocks':rows})
    print(task,'initial pairing',count,'visual max difference',mx,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);a=p.parse_args();main(a.task)
