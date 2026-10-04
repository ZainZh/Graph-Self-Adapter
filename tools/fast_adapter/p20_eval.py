"""P20 real-contact closed-loop evaluation. Only the reference oracle reads dynamics."""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from prove.fast_adapter.insert3d import Insert3D,domain,start_position,nominal_action,oracle_action,PROBES,STEP,GOAL
from prove.fast_adapter.insert_adapter import InteractionAdapter,support_actions,project
from prove.fast_adapter.visual_alignment import FrozenDinoROI,DEFAULT_MODEL
from tools.fast_adapter.p18_run import save_json,digest
R=Path('evaluations/fast_adapter_p20_insert3d_v01')

def model_hash(net):
    h=hashlib.sha256()
    for k,v in net.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def setting(stage,stratum):
    p=json.loads((R/'PROTOCOL.json').read_text())
    if stage=='dev':return 32,p['dev']['domain_seed'],p['dev']['start_seed'],'train'
    return 64,p['final']['domain_seeds'][stratum],p['final']['start_seeds'][stratum],stratum

def main(stage,stratum,seed,dino):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False
    n,ds,ss,level=setting(stage,stratum);began=time.time()
    out=R/stage/stratum/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
    learned=seed!=0
    envs=[Insert3D(render=(i==0)) for i in range(n)]
    for e in envs[1:]:e.renderer=envs[0].renderer
    params=[domain(ds+i,level) for i in range(n)]
    starts=[start_position(ss+i) for i in range(n)]
    for e,p in zip(envs,params):e.set_domain(*p)
    encoder=FrozenDinoROI(dino,microbatch=16) if learned else None
    pcas=[{k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in p.items()} for p in torch.load(R/'bank/PCA.pt',weights_only=False)] if learned else None
    visibility=[]
    def observe():
        obs=[e.observe() for e in envs];visibility.extend([o[2] for o in obs])
        full=encoder.encode(np.stack([o[0] for o in obs]),np.stack([o[1] for o in obs]))
        # Match half-precision cache quantization used during training.
        return project(full.half(),pcas),full.half().cpu()
    models={};hashes={};latents={};curves={}
    if learned:
        for name in ['history','hybrid']:
            net=InteractionAdapter().cuda().eval();net.load_state_dict(torch.load(R/'models'/f'seed{seed}'/f'{name}.pt',weights_only=False,map_location='cuda')['state']);net.requires_grad_(False)
            models[name]=net;hashes[name]=model_hash(net)
        # Support and task query use distinct starts; both depend on the same domain.
        for i,e in enumerate(envs):e.reset(start_position(ss+1000000+i))
        raw_support=[];support=[]
        v,f=observe();support.append(v);raw_support.append(f)
        for a in PROBES:
            for e in envs:e.step(a)
            v,f=observe();support.append(v);raw_support.append(f)
        support=torch.stack(support,1);acts=support_actions(n,'cuda')
        with torch.no_grad():
            latents['history']=models['history'].initial(support,acts)
            latents['hybrid_u0']=models['hybrid'].initial(support,acts)
        latents['hybrid_u3'],curves['hybrid_u3']=models['hybrid'].adapt(support,acts)
        latents['hybrid_swapped']=latents['hybrid_u3'].roll(1,0)
        seal={'latents':{k:v.cpu() for k,v in latents.items()},'loss_curves':{k:v.cpu() for k,v in curves.items()},'support_visual':support.cpu(),'support_raw_dino':torch.stack(raw_support,1),'domain_seeds':list(range(ds,ds+n)),'support_start_seeds':list(range(ss+1000000,ss+1000000+n)),'query_start_seeds':list(range(ss,ss+n)),'model_hashes':hashes}
        torch.save(seal,out/'LATENTS_BEFORE_QUERY.pt')
        save_json(out/'LATENTS_BEFORE_QUERY.json',{'sha256':digest(out/'LATENTS_BEFORE_QUERY.pt'),'timestamp':time.time(),'outcomes_opened':False,'per_query_z_frozen':True})
        arms=['history','hybrid_u0','hybrid_u3','hybrid_swapped']
    else:arms=['base','strong_fixed','oracle']
    metrics={}
    for arm in arms:
        for e,pos in zip(envs,starts):e.reset(pos)
        positions=[];velocities=[];actions=[];bases=[];features=[];facts=[];actor_deltas=[]
        for t in range(46):
            positions.append(np.stack([e.state() for e in envs]));velocities.append(np.stack([e.data.qvel.copy() for e in envs]));facts.append([e.facts() for e in envs])
            if t==45:break
            base=np.stack([nominal_action(e.state(),t) for e in envs]);bases.append(base)
            if learned:
                v,_=observe();features.append(v.cpu().half());net=models['history' if arm=='history' else 'hybrid']
                with torch.no_grad():
                    act=net.action(v,torch.tensor(base,dtype=torch.float32,device='cuda'),latents[arm]).cpu().numpy()
                actions.append(act);actor_deltas.append(act-base)
            elif arm=='oracle':act=np.stack([oracle_action(e.state(),t,p[0],p[1]) for e,p in zip(envs,params)]);actions.append(act)
            elif arm=='strong_fixed':act=np.clip(2*base,-1,1);actions.append(act)
            else:act=base;actions.append(act)
            for e,a in zip(envs,act):e.step(a)
        result={'positions':np.stack(positions,1),'velocities':np.stack(velocities,1),'actions':np.stack(actions,1),'base_actions':np.stack(bases,1),'success':np.array([e.streak>=3 for e in envs]),'final_streak':np.array([e.streak for e in envs]),'facts':facts,'domain_seeds':list(range(ds,ds+n)),'query_start_seeds':list(range(ss,ss+n)),'arm':arm,'seed':seed,'stratum':stratum,'stage':stage}
        if learned:result['visuals']=torch.stack(features,1);result['actor_residual']=np.stack(actor_deltas,1)
        torch.save(result,out/f'{arm}_RAW.pt')
        metrics[arm]={'successes':int(result['success'].sum()),'n':n,'rate':float(result['success'].mean()),'max_penetration':max(f['max_penetration'] for row in facts for f in row),'raw_sha256':digest(out/f'{arm}_RAW.pt')}
        print(json.dumps({'stage':stage,'stratum':stratum,'seed':seed,'arm':arm,**metrics[arm],'seconds':time.time()-began}),flush=True)
    if learned:
        assert all(model_hash(models[k])==h for k,h in hashes.items()),'network weights changed during test'
        diagnostics={'support_initial_loss':curves['hybrid_u3'][:,0].mean().item(),'support_updated_loss':curves['hybrid_u3'][:,-1].mean().item(),'z_update_rms':(latents['hybrid_u3']-latents['hybrid_u0']).square().mean().sqrt().item(),'alpha':models['hybrid'].alpha.item(),'model_weights_unchanged':True,'min_visible_pixels':np.array(visibility).min(0).tolist()}
    else:diagnostics={'reference_only':True}
    save_json(out/'RESULT.json',{'stage':stage,'stratum':stratum,'seed':seed,'metrics':metrics,'diagnostics':diagnostics,'seconds':time.time()-began})
    envs[0].close();print('EVAL_COMPLETE',stage,stratum,seed,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['dev','final'],required=True);p.add_argument('--stratum',choices=['train','ood'],default='train');p.add_argument('--seed',type=int,required=True);p.add_argument('--dino',default=DEFAULT_MODEL);a=p.parse_args();main(a.stage,a.stratum,a.seed,a.dino)
