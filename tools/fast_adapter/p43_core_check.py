"""Offline geometry/gradient/control contract checks, no simulator rollouts."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from prove.fast_adapter.geometry3d_env import camera_axes
from prove.fast_adapter.geometry3d_adapter import project,command
from prove.fast_adapter.neural_geometry_adapter import NeuralGeometry,NeuralGeometryLatent


def check():
    torch.set_num_threads(2);torch.manual_seed(43);reports=[]
    for task in ('cup','insert'):
        model=NeuralGeometry(task);weights={k:v.detach().clone() for k,v in model.state_dict().items()}
        fixed=NeuralGeometryLatent(model,'fixed');continuous=NeuralGeometryLatent(model,'continuous');prefix=NeuralGeometryLatent(model,'prefix')
        a,b=(.026,.023) if task=='insert' else (.026,.028)
        rotation=np.eye(3) if task=='insert' else Rotation.from_euler('x',90,degrees=True).as_matrix()
        center=np.array([0.,0.,.08]);points=np.array([[-a,b,0],[a,b,0],[a,-b,0],[-a,-b,0]])@rotation.T+center
        K=np.array([[350,0,224],[0,350,224],[0,0,1.]])
        for t in range(24):
            camera=np.array([.035*np.sin(t*.4),-.22+.02*np.cos(t*.4),.2])
            camera_R=camera_axes(camera,center)
            obs=dict(camera_position=camera,camera_rotation=camera_R,K=K,position=camera,rotation=rotation,
                     role_bbox=np.tile([150.,130.,270.,320.],(3,1)),role_visible=np.ones(3),local=np.sin(np.arange(48)*.2+t*.03).astype(np.float32))
            obs['corners']=project(points,obs)
            for candidate in (fixed,continuous,prefix):candidate.update(obs,t)
        errors={arm:float(np.sqrt(np.mean((actor.points-points)**2))) for arm,actor in [('fixed',fixed),('continuous',continuous),('prefix',prefix)]}
        action0=command(task,70,fixed.estimate(),np.array([0.,-.12,.2]),rotation,.05)[0]
        action1=command(task,70,continuous.estimate(),np.array([0.,-.12,.2]),rotation,.05)[0]
        unchanged=all(torch.equal(value,weights[key]) for key,value in model.state_dict().items())
        row=dict(task=task,corner_rmse=errors,weights_unchanged=unchanged,
            fixed_z_change=max(r['latent_change'] for r in fixed.records),continuous_z_change=max(r['latent_change'] for r in continuous.records),
            prefix_z_change_after12=max(r['latent_change'] for r in prefix.records if r['t']>=12),
            fixed_current_estimate_changes=float(np.linalg.norm(np.array(fixed.records[-1]['initial_z_points'])-np.array(fixed.records[0]['initial_z_points']))),
            command_position_change=float(np.linalg.norm(action1-action0)),
            support_loss_last_before=continuous.records[-1]['support_loss_before'],support_loss_last_after=continuous.records[-1]['support_loss_after'])
        assert unchanged and row['fixed_z_change']==0 and row['prefix_z_change_after12']==0
        assert row['continuous_z_change']>0 and row['command_position_change']>1e-4
        assert row['fixed_current_estimate_changes']>0
        assert errors['continuous']<errors['fixed']
        assert row['support_loss_last_after']<=row['support_loss_last_before']+1e-7
        reports.append(row)
    out=Path('evaluations/fast_adapter_p43_neural_geometry_v01/training/CORE_ENGINEERING_CHECK.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    record=dict(scope='offline synthetic observations only; untrained parameters; not physical or learned success evidence',passed=True,checks=reports,
       source_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('prove/fast_adapter/neural_geometry_adapter.py','tools/fast_adapter/p43_core_check.py')})
    out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__':check()
