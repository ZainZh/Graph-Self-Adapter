"""P70 demo-relative targets and explicitly synthetic teacher action distillation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from prove.task_graph.p70_grasp_policy_v02 import GraspPolicy,SCHEMA,ACTION_SCALES,relative_features

ROOT=Path(__file__).resolve().parents[2]
DEMO_SHA='92cf890a794a2c3bfe47cb68f0109f76901c3251f7331f6c03c10e258b30b94c'
CONFIG=dict(version='v02',equilibrium_correction='same-grip f(error)-f(zero)',gripper_sampling='independent left U(0,.045),right U(-.045,0)',seed=70010,updates=1200,batch_size=256,learning_rate=.002,synthetic_train_samples=30000,synthetic_validation_samples=3000,validation_seed=70011,
    position_range_m=.04,rotation_range_rad=.35,fine_fraction=1/3,fine_position_range_m=.004,fine_rotation_range_rad=.04,
    translation_step_cap_m=.01,rotation_step_cap_rad=.05,osc_translation_scale=.05,osc_rotation_scale=.5,
    pregrasp_frame=56,grasp_frame=67,hidden=[64,64],inputs='book-frame position/rotvec target error6 + gripper proprio2; no path index/progress',
    outputs='book-frame OSC6, rotated to world; explicit graph phase supplies open/close channel')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def extract_relations(demo,capture):
    import h5py
    if sha(demo)!=DEMO_SHA:raise ValueError('Wrong one-demo source')
    names=read(capture/'SUPERVISOR_NAMES.json');index=names['body'].index('black_book_1_main')
    binding=read(capture/'BINDING.json')
    if binding['source_sha256']!=DEMO_SHA:raise ValueError('Demo replay source differs')
    frames=[json.loads(s) for s in (capture/'FRAMES.jsonl').read_text().splitlines()]
    relations={};files={str(p):sha(p) for p in (demo,capture/'BINDING.json',capture/'FRAMES.jsonl',capture/'SUPERVISOR_NAMES.json')}
    with h5py.File(demo) as h:
        for label,t in [('pregrasp',56),('grasp',67)]:
            path=capture/f'frame_{t:04d}.npz'
            if sha(path)!=frames[t]['sha256']:raise ValueError('Changed replay frame')
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['action_next'],h['data/demo_0/actions'][t]):raise ValueError('Demo action alignment differs')
                book=np.eye(4);book[:3,3]=z['supervisor_body_xpos'][index];book[:3,:3]=z['supervisor_body_xmat'][index].reshape(3,3)
                eef=np.eye(4);eef[:3,3]=z['observation_robot0_eef_pos'];eef[:3,:3]=Rotation.from_quat(z['observation_robot0_eef_quat']).as_matrix()
                relations[label]=dict(frame=t,T_book_eef=(np.linalg.inv(book)@eef).tolist(),gripper_qpos=z['observation_robot0_gripper_qpos'].tolist(),demo_action=z['action_next'].tolist())
            files[str(path)]=sha(path)
    return dict(schema='prove.p70-demo-grasp-relations.v1',relations=relations,files=files,
        authority='Root preselected demo56 pregrasp and demo67 stable closed grasp; demo GT is fixed training prior, never current runtime object GT',
        semantic_note='Closing includes the relative descent between these fixed targets; no best-frame selection or absolute demo waypoint following')


def synthetic_samples(n,seed):
    rng=np.random.default_rng(seed);dp=rng.uniform(-.04,.04,(n,3));dr=rng.uniform(-.35,.35,(n,3))
    fine=np.arange(n)<n//3;dp[fine]=rng.uniform(-.004,.004,(fine.sum(),3));dr[fine]=rng.uniform(-.04,.04,(fine.sum(),3))
    grip=np.column_stack((rng.uniform(0,.045,n),rng.uniform(-.045,0,n)))
    x=np.concatenate((dp/.05,dr/.5,grip/.04),axis=1).astype(np.float32)
    def cap(v,limit):return v*np.minimum(1.,limit/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12))
    y=np.concatenate((cap(dp,.01)/.05,cap(dr,.05)/.5),axis=1).astype(np.float32)
    return x,y


def prepare(output,plan,demo,capture):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    relations=extract_relations(Path(demo).resolve(),Path(capture).resolve());dump(output/'DEMO_RELATIONS.json',relations)
    files=dict(relations['files'])
    for p in (Path(plan).resolve(),ROOT/'prove/task_graph/p70_grasp_policy_v02.py',Path(__file__).resolve(),output.resolve()/'DEMO_RELATIONS.json'):files[str(p)]=sha(p)
    freeze=dict(schema='prove.p70-grasp-training-freeze.v1',status='frozen_before_optimizer',config=CONFIG,files=files,
        authorization='Root explicit user-scoped training: seed70010,30000 synthetic rows,1200updates; no physical execution',
        data_kind='synthetic_relative_SE3_error_teacher_labels; not simulator trajectories or task success',physical_steps=0,rollout_validation='Required later on predeclared training states; not supplied by fit metrics')
    dump(output/'TRAINING_FREEZE.json',freeze)
    return freeze


def train(output):
    output=Path(output);freeze=read(output/'TRAINING_FREEZE.json')
    if freeze['config']!=CONFIG or freeze['status']!='frozen_before_optimizer':raise ValueError('Training contract changed')
    for name,digest in freeze['files'].items():
        if sha(name)!=digest:raise ValueError('Frozen input changed '+name)
    if (output/'final.pt').exists():raise FileExistsError(output/'final.pt')
    torch.set_num_threads(1);torch.manual_seed(70010);start=time.monotonic()
    X,Y=synthetic_samples(30000,70010);VX,VY=synthetic_samples(3000,70011)
    np.savez_compressed(output/'SYNTHETIC_DATA.npz',train_features=X,train_teacher_actions=Y,validation_features=VX,validation_teacher_actions=VY)
    x,y=torch.from_numpy(X),torch.from_numpy(Y);model=GraspPolicy();optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    generator=torch.Generator().manual_seed(70010);scales=torch.tensor(ACTION_SCALES)
    for step in range(1200):
        indices=torch.randint(len(x),(256,),generator=generator)
        loss=(((model(x[indices])-y[indices])/scales)**2).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    model.eval().requires_grad_(False)
    with torch.no_grad():prediction=model(torch.from_numpy(VX)).numpy()
    residual=prediction-VY
    # World/object frame equivariance and checkpoint actual-command wiring checked below.
    payload=dict(schema=SCHEMA,state_dict=model.state_dict(),config=CONFIG,freeze_sha256=sha(output/'TRAINING_FREEZE.json'),demo_relations_sha256=sha(output/'DEMO_RELATIONS.json'))
    torch.save(payload,output/'final.pt');loaded=GraspPolicy.load(output/'final.pt')
    with torch.no_grad():assert torch.equal(model(torch.from_numpy(VX[:8])),loaded(torch.from_numpy(VX[:8])))
    goal=np.asarray(read(output/'DEMO_RELATIONS.json')['relations']['grasp']['T_book_eef']);book=np.eye(4);current=goal.copy();current[:3,3]+=[.006,-.005,.008]
    action,record=loaded.command(book,current,[.03,-.03],goal,True)
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('z',.73).as_matrix();transform[:3,3]=[.4,-.2,.1]
    rotated,other=loaded.command(transform,transform@current,[.03,-.03],goal,True)
    np.testing.assert_allclose(rotated[:3],transform[:3,:3]@action[:3],atol=1e-7)
    np.testing.assert_allclose(rotated[3:6],transform[:3,:3]@action[3:6],atol=1e-7)
    assert action[6]==1 and loaded.command(book,current,[.03,-.03],goal,False)[0][6]==-1
    assert not any(p.requires_grad for p in loaded.parameters())
    grip_rng=np.random.default_rng(70012)
    grips=np.column_stack((grip_rng.uniform(0,.045,256),grip_rng.uniform(-.045,0,256)))
    zero_features=np.concatenate((np.zeros((256,6)),grips/.04),axis=1).astype(np.float32)
    with torch.no_grad():zero_output=loaded(torch.from_numpy(zero_features)).numpy()
    assert np.array_equal(zero_output,np.zeros_like(zero_output))
    closed_demo_grip=read(output/'DEMO_RELATIONS.json')['relations']['grasp']['gripper_qpos']
    zero_action,_=loaded.command(np.eye(4),goal,closed_demo_grip,goal,True)
    assert np.array_equal(zero_action[:6],np.zeros(6))
    # Fixed error / different grips: teacher is invariant. This measures learned
    # residual dependence, without selecting or retraining from the diagnostic.
    varied=zero_features.copy();varied[:,:6]=np.array([.1,-.08,.12,.05,-.03,.02],np.float32)
    with torch.no_grad():varied_output=loaded(torch.from_numpy(varied)).numpy()
    spread=varied_output-varied_output.mean(0)
    report=dict(schema='prove.p70-grasp-synthetic-fit-receipt.v2',
        zero_error_check=dict(grips=256,exact_zero=True,closed_demo_actual_command_zero=True),
        fixed_error_grip_dependence=dict(translation_step_spread_mm_max=float(np.linalg.norm(spread[:,:3],axis=1).max()*.05*1000),rotation_step_spread_deg_max=float(np.degrees(np.linalg.norm(spread[:,3:],axis=1).max()*.5))),elapsed_seconds=time.monotonic()-start,optimizer_steps=1200,seed=70010,
        train_synthetic_rows=30000,validation_synthetic_rows=3000,real_new_trajectories=0,physical_steps=0,task_success_evaluated=False,
        validation_action_component_rmse=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
        validation_equivalent_position_step_error_mm_mean=float(np.linalg.norm(residual[:,:3]*.05,axis=1).mean()*1000),
        validation_equivalent_rotation_step_error_deg_mean=float(np.degrees(np.linalg.norm(residual[:,3:]*.5,axis=1)).mean()),
        validation_position_step_error_mm_p95=float(np.quantile(np.linalg.norm(residual[:,:3]*.05,axis=1)*1000,.95)),
        actual_command_wiring=dict(passed=True,frame_equivariance=True,open_close_channels=True,frozen_reload_exact=True,sample=record),
        limits='Teacher distillation accuracy only; no physical grasp/entry qualification or success evidence.',
        files={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    dump(output/'TRAINING_RECEIPT.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path);p.add_argument('--demo',type=Path);p.add_argument('--capture',type=Path);a=p.parse_args()
    if a.command=='prepare':prepare(a.output,a.plan,a.demo,a.capture)
    else:train(a.output)
