"""P22 v02 independent geometry renderers; geometry truth only scores trajectories."""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.geometry_insert import GeometryInsert,geometry,initial_position,domain,HALF,INNER
from prove.fast_adapter.visual_proprio_adapter import VisualBase,ProprioAdapter
from prove.fast_adapter.insert_adapter import PROBE_ACTIONS,support_actions
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL
from tools.fast_adapter.p22_data import RGBEncoder
R=Path('evaluations/fast_adapter_p22_geometry_v02')
from tools.fast_adapter.p18_run import save_json,digest

def weight_hash(net):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()

def evaluate(stage,stratum,seed,dino):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    protocol=json.loads((R/'PROTOCOL.json').read_text());settings=protocol[stage]
    if stage=='final':assert (R/'FINAL_FREEZE.json').exists()
    n=settings['domains_per_stratum'];ss=settings['seed']+1000*('id','position','scale').index(stratum)
    out=R/stage/stratum/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    configs=[geometry(ss+i,stratum) for i in range(n)]
    envs=[GeometryInsert(c,render=True) for c in configs]
    assert all(e.renderer._model is e.model for e in envs), 'renderer/model ownership mismatch'
    for i,e in enumerate(envs):e.set_domain(*domain(ss+100000+i))
    encoder=RGBEncoder(dino)
    pca={k:v.cuda() if torch.is_tensor(v) else v for k,v in torch.load(R/'bank/PCA.pt',weights_only=True).items()}
    last_rgb=None
    def observe():
        nonlocal last_rgb
        last_rgb=np.stack([e.rgb() for e in envs])
        raw=encoder.encode(last_rgb).half().float()
        v=(raw-pca['mean'])@pca['basis']/pca['sd']
        p=torch.tensor(np.stack([e.state() for e in envs]),dtype=torch.float32,device='cuda')
        return v,p
    base=VisualBase().cuda().eval();base.load_state_dict(torch.load(R/'models/base/model.pt',weights_only=True)['state'])
    base.requires_grad_(False);hashes={'base':weight_hash(base)}
    nets={};latents={};curve=None;began=time.time()
    if seed:
        for name in ('history','hybrid'):
            net=ProprioAdapter().cuda().eval();net.load_state_dict(torch.load(R/'models'/f'seed{seed}'/f'{name}.pt',weights_only=True)['state'])
            net.requires_grad_(False);nets[name]=net;hashes[name]=weight_hash(net)
    for i,e in enumerate(envs):e.reset(initial_position(ss+200000+i))
    vs=[];ps=[]
    v,p=observe();vs.append(v);ps.append(p)
    for a in PROBE_ACTIONS:
        for e in envs:e.step(a)
        v,p=observe();vs.append(v);ps.append(p)
    sv=torch.stack(vs,1);sp=torch.stack(ps,1);sa=support_actions(n,'cuda')
    if seed:
        with torch.no_grad():
            latents['history']=nets['history'].initial(sv,sp,sa)
            latents['u0']=nets['hybrid'].initial(sv,sp,sa)
        latents['u3'],curve=nets['hybrid'].adapt(sv,sp,sa)
        latents['swapped']=latents['u3'].roll(1,0)
    torch.save({'visual':sv.cpu(),'proprio':sp.cpu(),'actions':sa.cpu(),
      'latents':{k:v.cpu() for k,v in latents.items()},'loss_curve':None if curve is None else curve.cpu(),
      'weights':hashes,'seed_base':ss},out/'SUPPORT_SEALED.pt')
    save_json(out/'SUPPORT_SEAL.json',{'sha256':digest(out/'SUPPORT_SEALED.pt'),'query_not_started':True,'time':time.time()})
    arms=['history','u0','u3','swapped'] if seed else ['base','oracle']
    metrics={}
    for arm in arms:
        for i,e in enumerate(envs):e.reset(initial_position(ss+300000+i))
        positions=[];velocities=[];actions=[];visuals=[];bases=[];audit_rgb=[]
        for t in range(49):
            positions.append(np.stack([e.state() for e in envs]));velocities.append(np.stack([e.data.qvel.copy() for e in envs]))
            if t==48:break
            if arm=='oracle':act=np.stack([e.expert_action() for e in envs])
            else:
                v,p=observe();visuals.append(v.cpu().half())
                if t in (0,24,47):audit_rgb.append(last_rgb[:2].copy())
                with torch.no_grad():
                    ba=base(v,p);bases.append(ba.cpu())
                    if arm=='base':act=ba.cpu().numpy()
                    else:
                        net=nets['history' if arm=='history' else 'hybrid']
                        act=net.action(v,p,ba,latents[arm]).cpu().numpy()
            actions.append(act)
            for e,a in zip(envs,act):e.step(a)
        raw={'positions':np.stack(positions,1),'velocities':np.stack(velocities,1),
             'actions':np.stack(actions,1),'visual':torch.stack(visuals,1) if visuals else None,
             'base':torch.stack(bases,1) if bases else None,'arm':arm,'stratum':stratum,'seed_base':ss,
             'audit_rgb':np.stack(audit_rgb) if audit_rgb else None,'audit_rgb_steps':[0,24,47]}
        torch.save(raw,out/f'{arm}_RAW.pt')
        raw_sha=digest(out/f'{arm}_RAW.pt')
        # Outcomes opened only after all actions/states are saved and hashed.
        success=[];checks=[]
        for i,e in enumerate(envs):
            last=[]
            for t in (46,47,48):
                e.data.qpos[:]=raw['positions'][i,t];e.data.qvel[:]=raw['velocities'][i,t]
                import mujoco
                mujoco.mj_forward(e.model,e.data);last.append(e.facts())
            success.append(all(x['inserted'] for x in last));checks.append(last)
        metrics[arm]={'successes':sum(success),'n':n,'rate':sum(success)/n,'success':success,
                      'raw_sha256':raw_sha,'final_three_checks':checks}
        print(json.dumps({'stage':stage,'stratum':stratum,'seed':seed,'arm':arm,'successes':sum(success),'n':n,
                          'seconds':time.time()-began}),flush=True)
    assert weight_hash(base)==hashes['base']
    assert all(weight_hash(net)==hashes[name] for name,net in nets.items())
    save_json(out/'RESULT.json',{'metrics':metrics,'model_weights_unchanged':True,'model_hashes':hashes,
      'runtime_inputs':['RGB','controlled_object_xyz','past_issued_actions'],
      'slot_geometry_runtime_access':False,'seconds':time.time()-began,
      'gradient':None if curve is None else {'initial_loss':float(curve[:,0].mean()),'final_loss':float(curve[:,-1].mean()),
        'z_update_rms':float((latents['u3']-latents['u0']).square().mean().sqrt()),'alpha':nets['hybrid'].alpha.item()}})
    for e in envs:e.close()
    print('EVAL_COMPLETE',stage,stratum,seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['dev','final'],required=True)
    p.add_argument('--stratum',choices=['id','position','scale'],required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--dino',default=DEFAULT_MODEL);a=p.parse_args();evaluate(a.stage,a.stratum,a.seed,a.dino)
