import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from prove.fast_adapter.object_interaction_env import ObjectEnv,object_config,TASKS
R=Path('evaluations/fast_adapter_p29_object_transfer_v01')
def run(tag,n):
 out=R/'qualification'/tag;out.mkdir(parents=True,exist_ok=False)
 src=Path('prove/fast_adapter/object_interaction_env.py')
 (out/'SOURCE_FREEZE.json').write_text(json.dumps({'source':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'n_per_task_per_split':n,'horizon':200,'time':time.time()},indent=2))
 rows=[]
 for task in TASKS:
  for split in ('canonical','train','geometry_ood'):
   for i in range(n):
    cfg=object_config(task,2900100+i,split);env=ObjectEnv(cfg,render=i==0);env.reset(2900200+i)
    if i==0:Image.fromarray(env.rgb()).save(out/f'{task}_{split}_start.png')
    frames=[]
    for t in range(200):
     a=env.teacher();env.step(a)
     if i==0 and t in (30,60,100,199):Image.fromarray(env.rgb()).save(out/f'{task}_{split}_t{t}.png')
    m=env.metrics
    row={'task':task,'split':split,'i':i,'cfg':cfg,'success_any':any(x['success'] for x in m),'terminal_success':bool(m[-1]['success']),'bilateral':sum(x['bilateral_handle'] for x in m),'head_contacts':sum(x['head_target_contact'] for x in m),'max_lift':max(x['lift'] for x in m),'target_max':max(x['target_displacement'] for x in m),'phase':env.phase,'metrics':m}
    rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('cfg','metrics')}),flush=True);env.close()
 (out/'RESULT.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--n',type=int,default=2);a=p.parse_args();run(a.tag,a.n)
