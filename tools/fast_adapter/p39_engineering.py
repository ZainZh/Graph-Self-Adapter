"""Result-free engineering qualification for P39 (not final evidence)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from prove.fast_adapter.planar_contact import PlanarPush, configuration, controller, STEPS

def main():
    import os
    out=Path('evaluations/fast_adapter_p39_tokens_v01')/os.environ.get('P39_ENGINEERING','engineering_v03');out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for split in ('train','resistance','geometry'):
        for i in range(6):
            cfg=configuration(39000+i,split)
            for arm in ('base','oracle'):
                env=PlanarPush(cfg);trace=[];frames=[]
                response=np.stack([20/(np.array(cfg['damping'])+2.),-20*(np.array(cfg['friction'])+.005)/(np.array(cfg['damping'])+2.)],-1)
                for t in range(STEPS+1):
                    obs,rgb,masks=env.public();s=env.state()
                    trace.append(dict(t=t,positions=obs['positions'].tolist(),radii=obs['radii'].tolist(),state={k:v.tolist() if hasattr(v,'tolist') else v for k,v in s.items()}))
                    if i==0 and t in (0,20,40):frames.append(rgb)
                    if t==STEPS:break
                    a=controller(obs,response if arm=='oracle' else None);trace[-1]['action']=a.tolist();env.step(a)
                error=float(np.linalg.norm(s['object']-env.goal));speed=float(np.linalg.norm(s['qvel'][2:]))
                row=dict(split=split,seed=cfg['seed'],arm=arm,error=error,speed=speed,success=error<.0025 and speed<.008,trace=trace,config=cfg)
                rows.append(row);print({k:v for k,v in row.items() if k not in ('trace','config')},flush=True)
                if frames:Image.fromarray(np.concatenate(frames,axis=1)).save(out/f'{split}_{arm}.jpg')
                env.close()
    (out/'RESULT.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
