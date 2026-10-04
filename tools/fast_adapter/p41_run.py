"""Bounded P41 engineering/development. No force adaptation or final claim."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from prove.fast_adapter.geometry3d_env import GeometrySkill,configuration
from prove.fast_adapter.geometry3d_adapter import GeometryLatent,command

ROOT=Path('evaluations/fast_adapter_p41_geometry3d_v01')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def rollout(task,split,seed,arm,out,video=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);cfg=configuration(task,seed,split)
    sources=['prove/fast_adapter/geometry3d_env.py','prove/fast_adapter/geometry3d_adapter.py','tools/fast_adapter/p41_run.py']
    js(out/'PRE_RUN.json',dict(cfg=cfg,arm=arm,steps=190,sources={s:sha(s) for s in sources},time=time.time(),scope='engineering' if arm=='oracle' else 'development'))
    for s in sources:
        target=out/'source'/s;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(s).read_bytes())
    e=GeometrySkill(cfg);n=GeometryLatent(task,arm);traces=[];frames=[];anchor=None;observations=[]
    try:
        if task=='insert':
            if arm=='oracle':e.tool_calibration={'length':cfg['length']}
            else:e.calibrate_held_geometry()
        for t in range(190):
            o,rgb,mask=e.observe();observations.append(o)
            if arm=='oracle':est=e.oracle()
            else:
                if task!='cup' or t<105:n.update(o,t)
                est=n.estimate()
            length=e.tool_calibration.get('length',0.)
            if task=='cup' and t==105 and est is not None:
                p,R,size=est;anchor=(p+R[:,0]*(size[0]-.003)-R[:,2]*.004,R.copy())
            p,R,closed,stage=command(task,t,est,e.start_pos,e.start_R,length,anchor)
            traces.append(dict(t=t,qpos=e.data.qpos.copy(),qvel=e.data.qvel.copy(),command_p=p.copy(),command_R=R.copy(),closed=closed,stage=stage,estimate=est))
            if video:frames.append(np.concatenate([e.render_global(),rgb],axis=1))
            e.step(p,R,closed)
        terminal=dict(qpos=e.data.qpos.copy(),qvel=e.data.qvel.copy())
        np.save(out/'TERMINAL.npy',terminal,allow_pickle=True)
        # Raw predictions/actions first; only then evaluator-only success.
        np.save(out/'TRACE.npy',np.array(traces,dtype=object),allow_pickle=True)
        np.save(out/'OBS.npy',np.array(observations,dtype=object),allow_pickle=True)
        js(out/'SEAL.json',dict(trace=sha(out/'TRACE.npy'),observations=sha(out/'OBS.npy'),terminal=sha(out/'TERMINAL.npy')))
        facts=e.facts();held=[];penetrations=[]
        for state in traces[-9:]+[terminal]:
            e.data.qpos[:]=state['qpos'];e.data.qvel[:]=state['qvel'];e.mj.mj_forward(e.model,e.data)
            held.append(e.facts()['success']);penetrations.append(max([max(0.,-float(c.dist)) for c in e.data.contact]+[0.]))
        facts.update(success=bool(all(held) and max(penetrations)<.002),last10_successes=sum(held),max_last10_penetration=max(penetrations))
        visible=sum(o['corners'] is not None for o in observations)
        result=dict(task=task,split=split,seed=seed,arm=arm,**facts,valid_frames=visible,
                    scope='engineering' if arm=='oracle' else 'development',input_boundary=('Engineering Oracle: target geometry truth' if arm=='oracle' else 'Visible target part mask, calibrated wrist camera, own SE3; insertion held length fitted from external-view mask'),force_adaptation=False,
                    tool_calibration={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in e.tool_calibration.items()})
        js(out/'RESULT.json',result)
        if video:
            np.savez_compressed(out/'FRAMES.npz',frames=np.array(frames));from PIL import Image
            Image.fromarray(frames[0]).save(out/'start.jpg');Image.fromarray(frames[-1]).save(out/'end.jpg')
        print(json.dumps(result),flush=True)
        return result
    finally:e.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['cup','insert'],required=True);p.add_argument('--split',default='id');p.add_argument('--seed',type=int,default=410001);p.add_argument('--arm',default='oracle');p.add_argument('--out',required=True);p.add_argument('--video',action='store_true');a=p.parse_args()
    rollout(a.task,a.split,a.seed,a.arm,a.out,a.video)
