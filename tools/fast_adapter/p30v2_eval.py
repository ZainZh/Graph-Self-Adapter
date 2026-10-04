"""Paired closed-loop development evaluation; no learning across test objects."""
import argparse
import json
import time
import torch
import numpy as np
from tools.fast_adapter.p30v2_common import *
from tools.fast_adapter.p30v2_data import collect_block
from tools.fast_adapter.p29_common import Visual
from prove.fast_adapter.object_interaction_env import object_config
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL


def evaluate(task,seed,dino):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    cfg=json.loads((ROOT/'PROTOCOL.json').read_text())['evaluation']
    out=ROOT/task/'development'/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    nets=adapters(task,seed);base=base_model(task)
    visual=Visual(dino,task,torch.load(P29/task/'data/PCA.pt',weights_only=True))
    paths=[ROOT/'PROTOCOL.json',ROOT/task/'models'/f'seed{seed}'/'gradient.pt',
           ROOT/task/'models'/f'seed{seed}'/'history.pt',base_path(task),P29/task/'data/PCA.pt',
           Path(__file__),Path('tools/fast_adapter/p30v2_data.py'),Path('tools/fast_adapter/p30v2_common.py'),
           Path('prove/fast_adapter/continuous_state.py'),Path('prove/fast_adapter/continuous_adapter.py'),
           Path('prove/fast_adapter/held_hammer_env.py'),Path('prove/fast_adapter/object_interaction_env.py')]
    dump(out/'SOURCE_FREEZE.json',{'before_evaluation':True,'model_seed':seed,
          'bindings':[{'path':str(p),'sha256':digest(p)} for p in paths]})
    before={'base':state_hash(base),**{k:state_hash(n[0]) for k,n in nets.items()}}
    rows=[];began=time.time()
    for si,split in enumerate(cfg['strata']):
        origin=TASKS.index(task)*10000+si*1000
        for offset in range(0,cfg['conditions_each'],8):
            ids=np.arange(offset,min(offset+8,cfg['conditions_each']))
            configs=[object_config(task,cfg['object_seed_origin']+origin+int(i),split) for i in ids]
            resets=cfg['reset_seed_origin']+origin+ids
            initial=None
            for arm in cfg['arms']:
                net=None;mode=None;z=None
                if arm!='base':
                    net,blob=nets['history' if arm=='history' else 'gradient']
                    mode='fixed' if arm=='fixed_mean' else arm
                    if arm=='fixed_mean':z=blob['mean_z'].cuda()[None]
                b,m,prep=collect_block(configs,resets,visual,'evaluate',base,net,mode,z)
                current={k:v[:,0] for k,v in b['states'].items()}
                if initial is None:initial=current
                else:
                    for key in initial:assert torch.equal(initial[key],current[key]),(split,arm,key)
                path=out/f'{split}_{offset:03d}_{arm}.pt';torch.save(b,path)
                binding={'path':str(path),'sha256':digest(path)}
                new=[]
                for j,i in enumerate(ids):
                    ms=m[j];steps=next((t+1 for t,x in enumerate(ms) if x['success']),None)
                    row={'task':task,'model_seed':seed,'split':split,'condition':int(i),'arm':arm,
                         'success':steps is not None,'first_success':steps,'terminal_success':bool(ms[-1]['success']),
                         'updates':b['updates'],'preparation':prep[j],
                         'max_target_displacement':max(x['target_displacement'] for x in ms),
                         'held_steps':sum(x['bilateral_handle']>0 for x in ms),
                         'head_contact_steps':sum(x['head_target_contact']>0 for x in ms),
                         'source':binding,'source_row':j,'reset_seed':int(resets[j]),'config':configs[j]}
                    new.append(row)
                dump(path.with_suffix('.json'),{'trajectory_sealed_before_summary':binding,'rows':new,'metrics':m})
                rows.extend(new)
                print('EVAL',task,seed,split,offset,arm,sum(r['success'] for r in new),'/',len(new),flush=True)
    after={'base':state_hash(base),**{k:state_hash(n[0]) for k,n in nets.items()}}
    assert after==before
    dump(out/'RESULT.json',{'phase':'development','task':task,'seed':seed,'rows':rows,
         'network_weights_unchanged':True,'weight_hashes':before,'seconds':time.time()-began,
         'unique_conditions':len(cfg['strata'])*cfg['conditions_each'],'episodes':len(rows)})
    print('EVALUATION_COMPLETE',task,seed,'episodes',len(rows),'seconds',time.time()-began,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--dino',default=DEFAULT_MODEL);a=p.parse_args();evaluate(a.task,a.seed,a.dino)
