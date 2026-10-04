"""Engineering-only physical qualification; no independent test claims."""
import argparse,json,os
from pathlib import Path
import numpy as np
from PIL import Image
from prove.fast_adapter.contact_skills import ContactSkill,EngineeringControl,configuration,STEPS

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--revision',default='v01');p.add_argument('--count',type=int,default=2);a=p.parse_args()
    out=Path('evaluations/fast_adapter_p40_skills_v01')/('engineering_'+a.revision)/a.task;out.mkdir(parents=True,exist_ok=False)
    rows=[]
    forces=(.5,1.5,5.) if a.task=='cup' else (.25,.7,1.5) if a.task=='hammer' else (.7,2.,4.5)
    for split in ('train','physical','geometry'):
        for i in range(a.count):
            cfg=configuration(a.task,40000+i,split)
            for force in forces:
                e=ContactSkill(cfg);policy=EngineeringControl(a.task,force);trace=[];frames=[]
                try:
                    for t in range(STEPS+1):
                        o,rgb,mask=e.public();s=e.state()
                        trace.append(dict(obs=o,state=s))
                        if i==0 and t in (0,25,75,100):frames.append(rgb)
                        if t==STEPS:break
                        act=policy(o,t);assert np.isfinite(act).all();trace[-1]['action']=act;e.step(act)
                    row=dict(task=a.task,split=split,seed=cfg['seed'],force=force,**e.evaluate(),config=cfg)
                    rows.append(row);name=f'{split}_{i}_{force}'
                    np.save(out/(name+'.npy'),np.array(trace,dtype=object),allow_pickle=True)
                    if frames:Image.fromarray(np.concatenate(frames,1)).save(out/(name+'.jpg'))
                    print(row,flush=True)
                finally:e.close()
    (out/'RESULTS.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
