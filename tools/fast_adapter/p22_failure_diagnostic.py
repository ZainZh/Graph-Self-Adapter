"""Post-hoc failure locations on sealed trajectories; no training or new trials."""
from pathlib import Path
import json
import numpy as np
import torch
from prove.fast_adapter.geometry_insert import geometry
from tools.fast_adapter.p18_run import save_json

R=Path('evaluations/fast_adapter_p22_geometry_v02')

def main():
    summary={}
    for st in ('id','position','scale'):
        entries={}
        for arm in ('base','history','u0','u3'):
            rows=[]
            for seed in ([0] if arm=='base' else [11,22,33]):
                out=R/'final'/st/f'seed{seed}'
                result=json.loads((out/'RESULT.json').read_text())['metrics'][arm]
                raw=torch.load(out/f'{arm}_RAW.pt',weights_only=False)
                for i,success in enumerate(result['success']):
                    if success:continue
                    cfg=geometry(raw['seed_base']+i,st);p=raw['positions'][i,-1];f=result['final_three_checks'][i][-1]
                    row={'seed':seed,'domain':i,'xy_error_mm':float(np.linalg.norm(p[:2]-cfg['xy'])*1000),
                         'not_aligned':not f['inside_xy'],'contact_count':f['contact_count'],
                         'penetration_failure':f['max_penetration']>.0015*cfg['scale'],
                         'bottom_above_slot_top':f['bottom_z']>.068*cfg['scale']}
                    if st=='position':
                        axis=int(np.argmax(np.abs(cfg['xy'])))
                        row['ends_toward_training_center']=bool((p[axis]-cfg['xy'][axis])*cfg['xy'][axis]<0)
                    rows.append(row)
            entries[arm]={'failed_model_domain_trials':len(rows),
              'not_aligned':sum(x['not_aligned'] for x in rows),
              'bottom_above_slot_top':sum(x['bottom_above_slot_top'] for x in rows),
              'penetration_failures':sum(x['penetration_failure'] for x in rows),
              'mean_failed_xy_error_mm':float(np.mean([x['xy_error_mm'] for x in rows])) if rows else None,
              'ends_toward_training_center':sum(x.get('ends_toward_training_center',False) for x in rows) if st=='position' else None,
              'rows':rows}
        summary[st]=entries
    save_json(R/'summary/FAILURE_DIAGNOSTIC.json',{'posthoc':True,'new_rollouts':0,'new_training':0,
      'interpretation_limit':'Describes where failures end; does not isolate visual encoder, compression, actor or objective as a unique cause. Repeated model-domain trials are correlated.',
      'strata':summary})
    print(json.dumps({s:{a:{k:v for k,v in x.items() if k!='rows'} for a,x in arms.items()} for s,arms in summary.items()}),flush=True)

if __name__=='__main__':main()
