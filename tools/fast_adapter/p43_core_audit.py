"""Replay P43 sealed neural decisions and physical trajectory; no new trials."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from prove.fast_adapter.geometry3d_adapter import command,pose
from prove.fast_adapter.geometry3d_env import GeometrySkill
from prove.fast_adapter.neural_geometry_adapter import load_model,NeuralGeometryLatent,observation_features,align_observation


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(cell,checkpoint,physics=False):
    cell=Path(cell);pre=json.loads((cell/'PRE_RUN.json').read_text());seal=json.loads((cell/'SEAL.json').read_text());result=json.loads((cell/'RESULT.json').read_text())
    for key,name in [('trace','TRACE.npy'),('observations','OBS.npy'),('terminal','TERMINAL.npy')]:
        if seal[key]!=sha(cell/name):raise ValueError(f'Seal mismatch: {cell/name}')
    if pre['bindings']['checkpoint']['sha256']!=sha(checkpoint):raise ValueError('Checkpoint mismatch')
    for path,digest in pre['sources'].items():
        if sha(path)!=digest:raise ValueError(f'Source changed after freeze: {path}')
        if sha(cell/'source'/path)!=digest:raise ValueError(f'Sealed source copy mismatch: {path}')
    model,_=load_model(checkpoint);n=NeuralGeometryLatent(model,pre['arm']);cfg=pre['cfg'];task=cfg['task']
    traces=np.load(cell/'TRACE.npy',allow_pickle=True);observations=np.load(cell/'OBS.npy',allow_pickle=True);terminal=np.load(cell/'TERMINAL.npy',allow_pickle=True).item()
    if len(traces)!=190 or len(observations)!=190 or [r['t'] for r in traces]!=list(range(190)):raise ValueError('Incomplete or nonconsecutive trajectory')
    env=GeometrySkill(cfg,render=False);anchor=None;maximum=dict(action_position=0.,action_rotation=0.,latent=0.,physical_qpos=0.,physical_qvel=0.)
    dependence=[];sustained=[];penetrations=[]
    length=result['tool_calibration'].get('length',0.)
    try:
        for obs,row in zip(observations,traces):
            t=row['t']
            if task!='cup' or t<105:n.update(obs,t)
            estimate=n.estimate()
            if task=='cup' and t==105 and estimate is not None:
                p,R,size=estimate;anchor=(p+R[:,0]*(size[0]-.003)-R[:,2]*.004,R.copy())
            p,R,closed,_=command(task,t,estimate,env.start_pos,env.start_R,length,anchor)
            maximum['action_position']=max(maximum['action_position'],float(np.max(np.abs(p-row['command_p']))))
            maximum['action_rotation']=max(maximum['action_rotation'],float(np.max(np.abs(R-row['command_R']))))
            if closed!=row['closed']:raise ValueError('Jaw action mismatch')
            if row['adaptation'] is not None:
                maximum['latent']=max(maximum['latent'],float(np.max(np.abs(n.z.numpy()[0]-np.array(row['adaptation']['z'])))))
            if t in (12,30,60,90) and obs['corners'] is not None and n.points is not None:
                current=observation_features(align_observation(obs,n.points));zero=current.copy();zero[:48]=0
                first=n.tensor(n.first)[None];reference=n.tensor(n.reference)[None]
                with torch.no_grad():
                    _,normal_offset=model.encode(first,n.tensor(current)[None],reference)
                    normal_estimate=pose(model.decode(reference,normal_offset,n.z).numpy()[0])
                    _,offset=model.encode(first,n.tensor(zero)[None],reference)
                    zero_estimate=pose(model.decode(reference,offset,n.z).numpy()[0])
                zp,zR,_,_=command(task,t,zero_estimate,env.start_pos,env.start_R,length,anchor)
                normal_p,normal_R,_,_=command(task,t,normal_estimate,env.start_pos,env.start_R,length,anchor)
                dependence.append(dict(t=t,current_dino_zero_position_delta=float(np.linalg.norm(zp-normal_p)),
                    current_dino_zero_rotation_delta=float(np.linalg.norm(Rotation.from_matrix(zR@normal_R.T).as_rotvec()))))
            if physics:
                maximum['physical_qpos']=max(maximum['physical_qpos'],float(np.max(np.abs(env.data.qpos-row['qpos']))))
                maximum['physical_qvel']=max(maximum['physical_qvel'],float(np.max(np.abs(env.data.qvel-row['qvel']))))
                env.step(row['command_p'],row['command_R'],row['closed'])
        if physics:
            maximum['physical_qpos']=max(maximum['physical_qpos'],float(np.max(np.abs(env.data.qpos-terminal['qpos']))))
            maximum['physical_qvel']=max(maximum['physical_qvel'],float(np.max(np.abs(env.data.qvel-terminal['qvel']))))
        for state in list(traces[-9:])+[terminal]:
            env.data.qpos[:]=state['qpos'];env.data.qvel[:]=state['qvel'];env.mj.mj_forward(env.model,env.data)
            sustained.append(env.facts()['success']);penetrations.append(max([max(0.,-float(c.dist)) for c in env.data.contact]+[0.]))
        recomputed=bool(all(sustained) and max(penetrations)<.002)
        passed=bool(max(maximum.values())<1e-6 and recomputed==result['success'] and seal['neural_parameters_before']==seal['neural_parameters_after'])
        return dict(cell=str(cell),passed=passed,maximum_error=maximum,success_recomputed=recomputed,full_physics_replay=physics,
                    dino_dependency=dependence,scope='Replay and zero-feature sensitivity on existing sealed trajectory; no new evaluation')
    finally:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--out',required=True);parser.add_argument('--physics-first',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);cells=sorted(Path(args.root).glob('*/RESULT.json'));rows=[]
    for i,path in enumerate(cells):
        rows.append(audit(path.parent,args.checkpoint,physics=args.physics_first and i<3));print(json.dumps(rows[-1]),flush=True)
    report=dict(passed=bool(rows) and all(r['passed'] for r in rows),cells=len(rows),audits=rows)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
