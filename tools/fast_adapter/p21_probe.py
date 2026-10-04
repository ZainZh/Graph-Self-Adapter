"""Train-only projection and diagnostic geometry readouts, never runtime observers."""
from pathlib import Path
import json,time
import numpy as np
import torch
from prove.fast_adapter.insert_adapter import project
from prove.fast_adapter.lila_frontend import FRONTENDS
from tools.fast_adapter.p18_run import save_json,digest
R=Path('evaluations/fast_adapter_p21_frontends_v01')

def pca_fit(x, dim, randomized=False):
    x=x.float().cuda();mu=x.mean(0);x=x-mu
    if randomized:
        torch.manual_seed(21021)
        _,sv,basis=torch.pca_lowrank(x,q=80,center=False,niter=4)
        basis=basis[:,:dim];variance=sv[:dim].square()/(len(x)-1)
    else:
        cov=x.T@x/(len(x)-1);values,vectors=torch.linalg.eigh(cov)
        basis=vectors[:,-dim:];variance=values[-dim:]
    sd=variance.clamp_min(1e-6).sqrt()
    return {'mean':mu.cpu(),'basis':basis.cpu(),'sd':sd.cpu(),
            'retained_variance':float(variance.sum()/(x.square().sum()/(len(x)-1))),
            'fit_domains':[0,767]}

def ridge(train_x, train_y, eval_x):
    x=train_x.double();y=train_y.double();ev=eval_x.double()
    mu=x.mean(0);sd=x.std(0).clamp_min(1e-5)
    x=(x-mu)/sd;ev=(ev-mu)/sd
    x=torch.cat([x,torch.ones(len(x),1)],1)
    ev=torch.cat([ev,torch.ones(len(ev),1)],1)
    reg=torch.eye(x.shape[1],dtype=x.dtype)*.001;reg[-1,-1]=0
    w=torch.linalg.solve(x.T@x/len(x)+reg,x.T@y/len(x))
    return ev@w,{'mean':mu,'std':sd,'weight':w,'regularization':.001}

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    out=R/'diagnostic';out.mkdir(exist_ok=False)
    metas=[torch.load(R/'bank'/f'shard{i}/META.pt',weights_only=False) for i in range(4)]
    meta={k:torch.cat([m[k] for m in metas]) for k in metas[0] if isinstance(metas[0][k],torch.Tensor)}
    xyz=meta['positions_diagnostic_only'];before=[0,1,2,3,4,6,8,10,12];after=[1,2,3,4,5,7,9,11,13]
    dy=xyz[:,after]-xyz[:,before]
    metrics={};started=time.time()
    for name in FRONTENDS:
        parts=[torch.load(R/'bank'/f'shard{i}/{name}.pt',weights_only=True) for i in range(4)]
        f=torch.cat(parts);del parts
        pcas=[pca_fit(f[:768,:,i].flatten(0,1),64 if name=='lila' else 32,name=='lila') for i in range(f.shape[2])]
        # Batched projection avoids materializing all 49k-dimensional features in float32.
        values=torch.cat([project(part,pcas) for part in f.split(32)])
        dest=R/'frontends'/name/'bank';dest.mkdir(parents=True,exist_ok=False)
        torch.save(pcas,dest/'PCA.pt')
        bank={'support':values[:,:6],'query':values[:,6::2],'query_next':values[:,7::2]}
        bank.update({k:meta[k] for k in ['base','teacher','query_action']})
        torch.save(bank,dest/'TRAIN.pt')
        pred,pose_readout=ridge(values[:768].flatten(0,1),xyz[:768].flatten(0,1),values[768:].flatten(0,1))
        pred=pred.reshape(256,14,3);err=pred-xyz[768:]
        response_x=torch.cat([values[:,before],values[:,after]-values[:,before]],-1)
        dpred,motion_readout=ridge(response_x[:768].flatten(0,1),dy[:768].flatten(0,1),response_x[768:].flatten(0,1))
        dpred=dpred.reshape(256,9,3);derr=dpred-dy[768:]
        near=err[:,[10,11,12,13],:2]
        side_mask=xyz[768:,:,:2].abs()>.001
        side_correct=((pred[:,:,:2]>0)==(xyz[768:,:,:2]>0))[side_mask].double().mean()
        metrics[name]={'position_xy_rmse_mm':float(err[:,:,:2].square().mean().sqrt()*1000),
           'position_xyz_rmse_mm':float(err.square().mean().sqrt()*1000),
           'support_and_query_displacement_rmse_mm':float(derr.square().mean().sqrt()*1000),
           'near_contact_query_xy_rmse_mm':float(near.square().mean().sqrt()*1000),
           'left_right_sign_accuracy_outside1mm':float(side_correct),
           'projection_retained_variance':[p['retained_variance'] for p in pcas]}
        torch.save({'position_probe':pose_readout,'motion_probe':motion_readout,
                    'position_predictions':pred,'motion_predictions':dpred,
                    'inputs':'diagnostic-only; no export to control modules',
                    'validation_domain_ids':list(range(768,1024))},out/f'{name}_READOUT.pt')
        print(json.dumps({'frontend':name,**metrics[name],'seconds':time.time()-started}),flush=True)
        del f,values,bank
    for name in FRONTENDS:
        m=metrics[name];last=metrics['last']
        m['selection_score']=.5*(m['position_xy_rmse_mm']/last['position_xy_rmse_mm']+
              m['support_and_query_displacement_rmse_mm']/last['support_and_query_displacement_rmse_mm'])
    selected=min(('middle17','lila'),key=lambda k:metrics[k]['selection_score'])
    qualified=metrics[selected]['selection_score']<=.95 or metrics[selected]['position_xy_rmse_mm']<=3.5
    result={'status':'complete','metrics':metrics,'selected':selected,'control_qualified':qualified,
      'control_frontends':['last',selected] if qualified else [],'fit_domains':768,'validation_domains':256,
      'diagnostic_only':True,'no_closed_loop_success_measured':True,
      'PCA_fit_includes_validation':False,'additional_geometry_labels_used_only_by_offline_readouts':True,
      'limits':'Complete frontend comparison; differing external supervision and ROI usage. Probe ranking is not control performance.',
      'protocol_sha256':digest(R/'PROTOCOL.json')}
    save_json(out/'RESULT.json',result)
    # Same runtime/training contract as P20, with new held-out seeds.
    p=json.loads((R/'PROTOCOL.json').read_text())
    runtime={'dev':p['dev'],'final':p['final']}
    for name in FRONTENDS:save_json(R/'frontends'/name/'PROTOCOL.json',runtime)
    print('PROBE_COMPLETE',selected,'qualified',qualified,flush=True)
if __name__=='__main__':main()
