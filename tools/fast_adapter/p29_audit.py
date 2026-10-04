"""Recompute policy, latent, masked inputs and all physical success states from seals."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from tools.fast_adapter.p29_common import *
from tools.fast_adapter.p29_train import support_inputs,previous
from tools.fast_adapter.p29_eval import STRATA

def err(a,b):return float((a-b).abs().max())
def audit(phase='final',round_name='r1',tasks=None):
    import mujoco
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    tasks=tasks or list(TASKS);errors={'action':0.,'latent':0.,'support':0.};counts={'queries':0,'support_episodes':0,'physical_states':0,'source_hashes':0};details=[]
    for b in json.loads((R/'EVALUATION_SOURCE_FREEZE.json').read_text())['bindings']:
        assert digest(b['path'])==b['sha256'],b['path'];counts['source_hashes']+=1
    for task in tasks:
        root=task_root(task);base=load_base(task);nets={(s,v):load_adapter(task,s,v,round_name) for s in SEEDS for v in ('fast','history','robot')}
        fp=root/('FINAL_FREEZE.json' if phase=='final' else f'DEV_{round_name}_FREEZE.json')
        for b in json.loads(fp.read_text())['bindings']:assert digest(b['path'])==b['sha256'],b['path']
        for seed in SEEDS:
            net,blob=nets[(seed,'fast')];zs=[]
            for p in sorted((root/'data/meta').glob('support_[0-9]*.pt')):
                b=torch.load(p,weights_only=True);z,_=net.adapt(support_inputs({k:b[k].cuda() for k in ('visual','proprio','actions')}),3);zs.append(z.cpu())
            errors['latent']=max(errors['latent'],err(torch.cat(zs),blob['training_z']),err(torch.cat(zs).mean(0),blob['mean_z']))
        for si,st in enumerate(STRATA):
            folder=root/phase/round_name/st;result=json.loads((folder/'RESULT.json').read_text());n=result['conditions']
            assert result['networks_unchanged'] and result['initial_states_matched']
            origin=(29910000 if phase=='final' else 4100000)+TASKS.index(task)*100000+si*10000
            for off in range(0,n,BLOCK):
                nb=min(BLOCK,n-off);ids=np.arange(off,off+nb);seal=json.loads((folder/f'QUERY_START_SEAL_{off:03d}.json').read_text())
                for b in [*seal['support_bindings'],seal['latents']]:assert digest(b['path'])==b['sha256'],b['path'];counts['source_hashes']+=1
                zs=torch.load(seal['latents']['path'],weights_only=True);support={}
                for source,delta in [('correct',2000000),('donor',3000000)]:
                    b=torch.load(folder/f'SUPPORT_{source}_{off:03d}.pt',weights_only=True);assert b['reset_seeds']==(origin+delta+ids).tolist();assert 'teacher' not in b
                    with torch.no_grad():a=base(b['visual'][:,:-1].cuda(),b['proprio'][:,:-1].cuda(),previous(b)[:,:-1].cuda())
                    errors['support']=max(errors['support'],err(a.cpu(),b['actions']));counts['support_episodes']+=nb
                    support[source]=support_inputs({k:b[k].cuda() for k in ('visual','proprio','actions')})
                for seed in SEEDS:
                    net,blob=nets[(seed,'fast')]
                    for source in ('correct','donor'):
                        z,_=net.adapt(support[source],3);errors['latent']=max(errors['latent'],err(z.cpu(),zs[f'fast_{source}_seed{seed}']))
                    errors['latent']=max(errors['latent'],err(blob['mean_z'][None].expand(nb,-1),zs[f'fast_mean_seed{seed}']))
                    for variant in ('history','robot'):
                        z,_=nets[(seed,variant)][0].adapt(support['correct'],3);errors['latent']=max(errors['latent'],err(z.cpu(),zs[f'{variant}_seed{seed}']))
                baseline=torch.load(folder/f'base_{off:03d}.pt',weights_only=True);initial=baseline['states']['qpos'][:,0]
                for row in [x for x in result['rows'] if x['offset']==off]:
                    assert digest(row['path'])==row['sha256'];b=torch.load(row['path'],weights_only=True);assert 'teacher' not in b and b['updates_in_query']==0
                    assert b['reset_seeds']==(origin+4000000+ids).tolist();assert torch.equal(b['states']['qpos'][:,0],initial)
                    if task==TASKS[0]:assert (b['visual'].reshape(nb,HORIZON+1,2,71)[:,:,1]==0).all()
                    with torch.no_grad():
                        v,p,prev=b['visual'][:,:-1].cuda(),b['proprio'][:,:-1].cuda(),previous(b)[:,:-1].cuda();ba=base(v,p,prev)
                        if row['arm']=='base':a=ba
                        else:
                            seed=int(row['arm'].rsplit('seed',1)[1]);variant=row['arm'].split('_')[0];variant=variant if variant in ('history','robot') else 'fast';net=nets[(seed,variant)][0]
                            z=zs[row['arm']].cuda()[:,None].expand(-1,HORIZON,-1);assert torch.equal(b['latent'],zs[row['arm']])
                            a=net.action(v,p,prev,ba,z);assert torch.equal(net.action(v,p,prev,ba,net.z0.expand_as(z)),ba)
                        errors['action']=max(errors['action'],err(a.cpu(),b['actions']))
                    side=json.loads(Path(row['path']).with_suffix('.json').read_text());assert side['sha256']==row['sha256']
                    assert row['success']==[bool(any(x['success'] for x in m)) for m in side['metrics']]
                    assert row['terminal_success']==[bool(m[-1]['success']) for m in side['metrics']]
                    for j in range(nb):
                        expected=object_config(task,int(origin+ids[j]),st);assert expected==side['configs'][j]
                        e=ObjectEnv(expected,render=False);e.reset(b['reset_seeds'][j]);seq=[]
                        for t in range(1,HORIZON+1):
                            for key in ('qpos','qvel','mocap_pos','mocap_quat'):getattr(e.env.data,key)[:]=b['states'][key][j,t].numpy()
                            mujoco.mj_forward(e.env.model,e.env.data);_,m=e.env.evaluate_state(e.env._get_obs(),b['actions'][j,t-1].numpy());seq.append(bool(m['success']));counts['physical_states']+=1
                        assert seq==[bool(x['success']) for x in side['metrics'][j]],(task,st,row['arm'],off,j)
                        e.close()
                    counts['queries']+=nb
                print('AUDIT_BLOCK',phase,task,st,off,errors,flush=True)
    assert max(errors.values())<.001,errors
    out=R/'summary'/phase/round_name;out.mkdir(parents=True,exist_ok=True)
    save_json(out/'AUDIT.json',{'passed':True,'phase':phase,'round':round_name,'tasks':tasks,'counts':counts,'max_errors':errors,'tolerance':.001,'every_query_success_state_replayed':True,'scope':'float32 policy inputs and latents, zero absent target, source bindings, paired resets, native actuation and custom contact-based evaluator; no bitwise DINO re-encoding claim'})
    print('AUDIT_PASSED',counts,errors,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='final',choices=['final','development']);p.add_argument('--round',default='r1');p.add_argument('--task',choices=TASKS);a=p.parse_args();audit(a.phase,a.round,[a.task] if a.task else None)
