"""Result-free engineering geometry and oracle checks, not learned evidence."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from prove.fast_adapter.geometry_insert import GeometryInsert,geometry,initial_position,domain
from tools.fast_adapter.p22_data import R
from tools.fast_adapter.p18_run import save_json,digest

def main():
    out=R/'engineering_v02';out.mkdir(parents=True,exist_ok=False)
    results={};frames=[]
    for j,st in enumerate(('id','position','scale')):
        rows=[]
        for i in range(16):
            seed=2260000+j*1000+i;cfg=geometry(seed,st);env=GeometryInsert(cfg,render=True)
            env.set_domain(*domain(seed+100000));env.reset(initial_position(seed+200000))
            rgb=env.rgb()
            orange=(rgb[:,:,0]>1.9*rgb[:,:,1])&(rgb[:,:,0]>100)&(rgb[:,:,1]>25)
            yy,xx=np.nonzero(orange)
            visible=bool(len(xx)>20 and xx.min()>0 and xx.max()<223 and yy.min()>0 and yy.max()<223)
            if i<2:frames.append((st,cfg,rgb))
            for _ in range(48):env.step(env.expert_action())
            rows.append({'seed':seed,'geometry':cfg,'success':env.streak>=3,'peg_visible_initially':visible,'final':env.facts()})
            env.close()
        results[st]={'successes':sum(x['success'] for x in rows),'n':16,'episodes':rows}
    canvas=Image.new('RGB',(3*448,2*490),'white');draw=ImageDraw.Draw(canvas)
    for i,(st,cfg,rgb) in enumerate(frames):
        x=(i//2)*448;y=(i%2)*490;canvas.paste(Image.fromarray(rgb).resize((448,448)),(x,y))
        draw.text((x+8,y+451),f'{st}  scale={cfg["scale"]:.3f}  xy={np.round(cfg["xy"],3)}',fill='black')
    canvas.save(out/'geometry_examples.png')
    passed=all(x['successes']>=15 and all(r['peg_visible_initially'] for r in x['episodes']) for x in results.values())
    save_json(out/'RESULT.json',{'passed':passed,'results':results,'protocol_sha256':digest(R/'PROTOCOL.json'),
      'claim':'Engineering oracle feasibility only; no learned-policy results'})
    print(json.dumps({'passed':passed,'successes':{k:v['successes'] for k,v in results.items()}}),flush=True)
    if not passed:raise RuntimeError('Geometry oracle qualification failed')

if __name__=='__main__':main()
