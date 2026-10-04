"""Result-free same-appearance, opposite-resistance diagnostic for P39."""
import copy
import json
import numpy as np
from prove.fast_adapter.planar_contact import PlanarPush,controller,STEPS
from tools.fast_adapter.p39_tokens import ROOT,cfg,probes,js

def pairs(first,npairs):
    result=[]
    for i in range(npairs):
        common=cfg(first+i,'geometry');rng=np.random.default_rng(first+i+77000)
        low=copy.deepcopy(common);high=copy.deepcopy(common)
        low.update(damping=rng.uniform(1.,2.5,2).tolist(),friction=rng.uniform(.01,.04,2).tolist())
        high.update(damping=rng.uniform(6.,10.,2).tolist(),friction=rng.uniform(.1,.18,2).tolist())
        for j,c in enumerate((low,high)):
            c.update(seed=2*(first+i)+j,pair=i,split='paired',regime='low' if j==0 else 'high')
        result += [low,high]
    return result

def engineering():
    out=ROOT/'paired_engineering';out.mkdir(exist_ok=False)
    configs=pairs(399500,4);rows=[]
    for c in configs:
        for arm in ('base','oracle'):
            en=PlanarPush(c);response=np.stack((20/(np.array(c['damping'])+2),-20*(np.array(c['friction'])+.005)/(np.array(c['damping'])+2)),-1)
            trace=[]
            try:
                for t in range(STEPS+1):
                    obs,rgb,mask=en.public();state=en.state()
                    trace.append(dict(positions=obs['positions'].tolist(),state={k:v.tolist() if hasattr(v,'tolist') else v for k,v in state.items()}))
                    if t==STEPS:break
                    a=probes(obs,t) if t<3 else controller(obs,response if arm=='oracle' else None);trace[-1]['action']=a.tolist();en.step(a)
                error=float(np.linalg.norm(state['object']-en.goal));speed=float(np.linalg.norm(state['qvel'][2:]))
                row=dict(config=c,arm=arm,error=error,speed=speed,success=error<.0025 and speed<.008,trace=trace);rows.append(row)
                print(c['pair'],c['regime'],arm,error,row['success'],flush=True)
            finally:en.close()
    js(out/'RESULT.json',dict(scope='engineering only; includes3 common probes; never final evidence',rows=rows))

if __name__=='__main__':engineering()
