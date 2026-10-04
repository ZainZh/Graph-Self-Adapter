"""P43 same-checkpoint physical comparison, sealed before evaluator facts."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from prove.fast_adapter.geometry3d_env import GeometrySkill,configuration
from prove.fast_adapter.geometry3d_adapter import command,pose
from prove.fast_adapter.neural_geometry_adapter import NeuralGeometryLatent,load_model
from prove.fast_adapter.neural_geometry_visual import FrozenRoleVision,observe_roles

SOURCE=['prove/fast_adapter/geometry3d_env.py','prove/fast_adapter/geometry3d_adapter.py',
        'prove/fast_adapter/neural_geometry_adapter.py','prove/fast_adapter/neural_geometry_visual.py',
        'prove/fast_adapter/lila_frontend.py','prove/fast_adapter/visual_alignment.py',
        'tools/fast_adapter/p43_evaluate.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def parameters_sha(model):
    h=hashlib.sha256()
    for name,param in model.state_dict().items():h.update(name.encode());h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rollout(task,split,seed,arm,out,model,vision,bindings,scope='development',video=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);cfg=configuration(task,seed,split)
    js(out/'PRE_RUN.json',dict(cfg=cfg,arm=arm,steps=190,sources={s:sha(s) for s in SOURCE},bindings=bindings,
        time=time.time(),scope=scope,neural_input='Current and initial visible RGB role features, corners, camera/proprio; no truth'))
    for source in SOURCE:
        target=out/'source'/source;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(source).read_bytes())
    e=GeometrySkill(cfg);adapter=NeuralGeometryLatent(model,arm=arm);traces=[];observations=[];frames=[];anchor=None
    before_hash=parameters_sha(model)
    try:
        if task=='insert':e.calibrate_held_geometry()
        for t in range(190):
            obs,rgb,masks=observe_roles(e)
            if (task!='cup' or t<105) and obs['corners'] is not None:
                obs['local']=vision.encode(rgb[None],masks[None]).cpu().numpy()[0]
                adapter.update(obs,t)
            else:obs['local']=np.zeros(48,np.float32)
            observations.append(obs)
            estimate=adapter.estimate();length=e.tool_calibration.get('length',0.)
            if task=='cup' and t==105 and estimate is not None:
                p,R,size=estimate;anchor=(p+R[:,0]*(size[0]-.003)-R[:,2]*.004,R.copy())
            p,R,closed,stage=command(task,t,estimate,e.start_pos,e.start_R,length,anchor)
            evidence=None
            if adapter.records and adapter.records[-1]['t']==t:
                evidence=dict(adapter.records[-1]);pre=pose(np.array(evidence['pre_update_points']))
                p0,R0,_,_=command(task,t,pre,e.start_pos,e.start_R,length,anchor)
                evidence.update(same_state_action_position_delta=float(np.linalg.norm(p-p0)),
                    same_state_action_rotation_delta=float(np.linalg.norm(Rotation.from_matrix(R@R0.T).as_rotvec())))
            traces.append(dict(t=t,qpos=e.data.qpos.copy(),qvel=e.data.qvel.copy(),command_p=p.copy(),command_R=R.copy(),
                               closed=closed,stage=stage,estimate=estimate,adaptation=evidence))
            if video:frames.append(np.concatenate([e.render_global(),rgb],axis=1))
            e.step(p,R,closed)
        terminal=dict(qpos=e.data.qpos.copy(),qvel=e.data.qvel.copy())
        np.save(out/'TERMINAL.npy',terminal,allow_pickle=True);np.save(out/'TRACE.npy',np.array(traces,dtype=object),allow_pickle=True)
        np.save(out/'OBS.npy',np.array(observations,dtype=object),allow_pickle=True)
        after_hash=parameters_sha(model)
        js(out/'SEAL.json',dict(trace=sha(out/'TRACE.npy'),observations=sha(out/'OBS.npy'),terminal=sha(out/'TERMINAL.npy'),
                               neural_parameters_before=before_hash,neural_parameters_after=after_hash))
        if before_hash!=after_hash:raise RuntimeError('Online update changed neural weights')
        facts=e.facts();held=[];penetrations=[]
        for state in traces[-9:]+[terminal]:
            e.data.qpos[:]=state['qpos'];e.data.qvel[:]=state['qvel'];e.mj.mj_forward(e.model,e.data)
            held.append(e.facts()['success']);penetrations.append(max([max(0.,-float(c.dist)) for c in e.data.contact]+[0.]))
        facts.update(success=bool(all(held) and max(penetrations)<.002),last10_successes=sum(held),max_last10_penetration=max(penetrations))
        changes=[r['adaptation'] for r in traces if r['adaptation'] is not None]
        result=dict(task=task,split=split,seed=seed,arm=arm,**facts,scope=scope,
            valid_frames=sum(o['corners'] is not None for o in observations),adaptation_updates=sum(c['U'] for c in changes),
            max_latent_change=max([c['latent_change'] for c in changes]+[0.]),
            max_same_state_action_position_delta=max([c['same_state_action_position_delta'] for c in changes]+[0.]),
            max_same_state_action_rotation_delta=max([c['same_state_action_rotation_delta'] for c in changes]+[0.]),
            neural_weights_unchanged=True,force_adaptation=False,
            tool_calibration={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in e.tool_calibration.items()})
        js(out/'RESULT.json',result)
        if video:np.savez_compressed(out/'FRAMES.npz',frames=np.array(frames))
        print(json.dumps(result),flush=True);return result
    finally:e.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--pca',required=True);p.add_argument('--out',required=True)
    p.add_argument('--task',required=True,choices=['cup','insert']);p.add_argument('--seeds',required=True);p.add_argument('--splits',required=True)
    p.add_argument('--arms',default='fixed,continuous,prefix');p.add_argument('--scope',choices=['development','independent_validation'],default='development')
    p.add_argument('--freeze',required=True);p.add_argument('--model-dir',default='assets/dinov3');p.add_argument('--device',default='cuda');p.add_argument('--video-first',action='store_true')
    args=p.parse_args();torch.set_num_threads(2)
    freeze=json.loads(Path(args.freeze).read_text());seeds=[int(s) for s in args.seeds.split(',')];splits=args.splits.split(',')
    if len(seeds)!=len(splits):raise ValueError('Each seed requires its frozen distribution')
    if args.scope=='independent_validation':
        if freeze.get('phase')!='independent_validation' or freeze.get('checkpoint_sha256')!=sha(args.checkpoint):raise ValueError('Independent validation requires matching result-free checkpoint freeze')
        if freeze.get('seeds')!=seeds or freeze.get('splits')!=splits or freeze.get('arms')!=args.arms.split(','):raise ValueError('Evaluation scope differs from freeze')
    model,package=load_model(args.checkpoint,'cpu')
    if model.task!=args.task:raise ValueError('Task/checkpoint mismatch')
    vision=FrozenRoleVision(args.pca,model_dir=args.model_dir,device=args.device,microbatch=1)
    visual_receipt=Path('evaluations/fast_adapter_p43_neural_geometry_v01/visual/PREFLIGHT.json')
    if not visual_receipt.exists():raise ValueError('Missing frozen visual qualification receipt')
    bindings=dict(visual_preflight=dict(path=str(visual_receipt),sha256=sha(visual_receipt)),checkpoint=dict(path=args.checkpoint,sha256=sha(args.checkpoint)),pca=dict(path=args.pca,sha256=sha(args.pca)),freeze=dict(path=args.freeze,sha256=sha(args.freeze)),model_seed=package['seed'])
    for i,(seed,split) in enumerate(zip(seeds,splits)):
        for arm in args.arms.split(','):
            if arm not in ('fixed','continuous','prefix'):raise ValueError(arm)
            dest=Path(args.out)/f'{split}_{seed}_{arm}'
            if dest.exists():raise FileExistsError(f'Preserve existing result: {dest}')
            rollout(args.task,split,seed,arm,dest,model,vision,bindings,args.scope,video=args.video_first and i in (0,4))

if __name__=='__main__':main()
