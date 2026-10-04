#!/usr/bin/env python3
"""Offline P51 opened-one-shot OSC supervision plus declared residual augmentation."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import h5py,numpy as np,torch
from scipy.spatial.transform import Rotation
from prove.task_graph.libero_precision_p51 import PrecisionMLP,PRECISE,features,project_progress


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def train(source,capture,graph_path,output,reference=None):
    torch.set_num_threads(1);torch.manual_seed(5101);rng=np.random.default_rng(5101)
    output.mkdir(parents=True,exist_ok=False)
    graph=json.loads(graph_path.read_text());names=json.loads((capture/'SUPERVISOR_NAMES.json').read_text())
    with np.load(capture/'frame_0000.npz') as z:
        k=names['site'].index('desk_caddy_1_back_contain_region');T=np.eye(4);T[:3,3]=z['supervisor_site_xpos'][k];T[:3,:3]=z['supervisor_site_xmat'][k].reshape(3,3)
    if reference is not None:T=np.asarray(json.loads(reference.read_text())['nominal_reference'],float)
    teacher_binding=json.loads((capture/'BINDING.json').read_text())
    if teacher_binding['source_sha256'] != sha(source):raise ValueError('teacher capture/source mismatch')
    with h5py.File(source) as f:actions=f['data/demo_0/actions'][()]
    paths={};raw=[];source_frames={}
    rows=[json.loads(x) for x in (capture/'FRAMES.jsonl').read_text().splitlines()]
    for stage in graph['nodes']:
        sid=stage['id']
        if sid not in PRECISE:continue
        start,end=stage['evidence']['start_step'],stage['evidence']['end_step']
        positions=[];rotations=[];grippers=[]
        for t in range(start,end):
            p=capture/f'frame_{t:04d}.npz'
            if sha(p)!=rows[t]['sha256']:raise ValueError('teacher frame hash mismatch')
            source_frames[p.name]=sha(p)
            with np.load(p) as z:
                if not np.array_equal(z['action_next'],actions[t]):raise ValueError('teacher frame/action alignment mismatch')
                positions.append(z['observation_robot0_eef_pos']);rotations.append(Rotation.from_quat(z['observation_robot0_eef_quat']).as_matrix());grippers.append(z['observation_robot0_gripper_qpos'])
        positions=np.asarray(positions);rotations=np.asarray(rotations)
        paths[sid]=dict(positions=positions.tolist(),rotations=rotations.tolist())
        for i,t in enumerate(range(start,end)):
            raw.append((sid,t,positions[i],rotations[i],grippers[i],positions,rotations,i/(end-start-1)))
    X=[];Y=[];cleanX=[];cleanY=[];stage_ids=[]
    for sid,t,p,R,grip,positions,rotations,progress in raw:
        local=np.r_[T[:3,:3].T@actions[t,:3],T[:3,:3].T@actions[t,3:6]]
        x=features(p,R,grip,T,progress,sid);cleanX.append(x);cleanY.append(local);stage_ids.append(sid)
        for sample in range(33):
            dp=np.zeros(3) if sample==0 else rng.uniform(-.012,.012,3)
            dr=np.zeros(3) if sample==0 else rng.uniform(-.07,.07,3)
            noisyR=Rotation.from_rotvec(dr).as_matrix()@R
            prog=progress if sample==0 else project_progress(p+dp,noisyR,positions,rotations)
            correction=np.r_[T[:3,:3].T@(-dp/.05),T[:3,:3].T@(-dr/.5)]
            correction=np.clip(correction,-.35,.35)
            X.append(features(p+dp,noisyR,grip,T,prog,sid));Y.append(local+correction)
    x=torch.tensor(np.asarray(X),dtype=torch.float32);y=torch.tensor(np.asarray(Y),dtype=torch.float32)
    model=PrecisionMLP();opt=torch.optim.Adam(model.parameters(),lr=.002)
    protocol=dict(schema='prove.p51-offline-osc-training.v1',source=str(source),source_sha256=sha(source),capture=str(capture),source_frame_hashes=source_frames,
                  graph=str(graph_path),graph_sha256=sha(graph_path),implementation={str(Path(__file__)):sha(__file__),str(ROOT/'prove/task_graph/libero_precision_p51.py'):sha(ROOT/'prove/task_graph/libero_precision_p51.py')},seed=5101,updates=1800,batch_size=256,learning_rate=.002,
                  alignment='public replay frame[t] pre-action state -> original source action[t]',
                  augmentation='32 pose residual neighbors + original; uniform world xyz +/- .012m, rotvec +/- .07rad; action correction -pose_error/.05 or .5, clipped .35; not pure BC',
                  nominal_reference=T.tolist(),runtime_inputs=['public EEF pose','public gripper qpos','reference transform','stage','monotone nearest-path progress'],
                  runtime_reference_mode='nominal_demo_reference engineering; Adapter disabled',simulator_rollouts=0)
    (output/'TRAINING_FREEZE.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for step in range(1800):
        idx=torch.randint(len(x),(256,));loss=(model(x[idx])-y[idx]).square().mean();opt.zero_grad();loss.backward();opt.step()
    model.eval()
    with torch.no_grad():pred=model(torch.tensor(np.asarray(cleanX))).numpy();augmse=float((model(x)-y).square().mean())
    report=dict(schema='prove.p51-offline-osc-fit.v1',original_pairs=len(raw),augmented_pairs=len(X),augmented_mse=augmse,
                original_mse=float(np.mean((pred-np.asarray(cleanY))**2)),
                stage_mse={s:float(np.mean((pred[np.array(stage_ids)==s]-np.asarray(cleanY)[np.array(stage_ids)==s])**2)) for s in PRECISE},
                interpretation='training-fit diagnostic only; no task-success evidence; fixed action model; residual-augmented teacher imitation',robot_rollouts=0)
    torch.save(dict(state_dict=model.state_dict(),nominal_reference=T.tolist(),paths=paths,training_freeze_sha256=sha(output/'TRAINING_FREEZE.json')),output/'action_head.pt')
    report['checkpoint_sha256']=sha(output/'action_head.pt');(output/'FIT_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','capture','graph','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reference',type=Path)
    a=p.parse_args();train(a.source,a.capture,a.graph,a.output,a.reference)
