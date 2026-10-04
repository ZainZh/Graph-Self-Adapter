"""P29 independent block partition of the sealed evaluator; scientific body unchanged."""
import argparse,time
import numpy as np
import torch
from tools.fast_adapter.p29_common import *
from tools.fast_adapter.p29_data import collect_block
from tools.fast_adapter.p29_train import support_inputs
STRATA=('canonical','train','geometry_ood')

def execute(task,configs,seeds,visual,base,path,net=None,z=None):
    # No teacher actions, native contacts, rewards or outcome flags enter the policy.
    blob,metrics=collect_block(task,configs,seeds,HORIZON,'base',visual,base,net,z,labels=False)
    blob['latent']=None if z is None else z.detach().cpu();blob['updates_in_query']=0
    if net is not None:
        with torch.no_grad():
            v=blob['visual'][:,:-1].cuda();p=blob['proprio'][:,:-1].cuda();prev=torch.cat((torch.zeros_like(blob['actions'][:,:1]),blob['actions'][:,:-1]),1).cuda()
            ba=base(v,p,prev);k0=net.action(v,p,prev,ba,net.z0.expand(len(v),HORIZON,-1));assert torch.equal(ba,k0)
    torch.save(blob,path);h=digest(path)
    result={'path':str(path),'sha256':h,'success':[bool(any(x['success'] for x in m)) for m in metrics],
            'terminal_success':[bool(m[-1]['success']) for m in metrics],
            'first_success':[next((i+1 for i,x in enumerate(m) if x['success']),None) for m in metrics],
            'configs':configs,'reset_seeds':list(map(int,seeds)),'metrics':metrics}
    save_json(path.with_suffix('.json'),result)
    return blob['states']['qpos'][:,0],result

def evaluate(task,stratum,dino,phase='final',round_name='r1',blocks=None,worker='worker'):
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    root=task_root(task);out=root/phase/round_name/stratum;out.mkdir(parents=True,exist_ok=True)
    fpath=root/('FINAL_FREEZE.json' if phase=='final' else f'DEV_{round_name}_FREEZE.json');freeze=json.loads(fpath.read_text())
    for b in freeze['bindings']:assert digest(b['path'])==b['sha256'],b['path']
    grid=json.loads((root/'FINAL_GRID_FREEZE.json').read_text())
    assert grid['final_freeze_sha256']==digest(fpath)
    for binding in grid['bindings']:assert digest(binding['path'])==binding['sha256'],binding['path']
    assignment=next(x for x in grid['workers'] if x['worker']==worker)
    assert assignment['task']==task and assignment['stratum']==stratum and assignment['blocks']==blocks
    assert not (out/f'WORKER_{worker}.json').exists()
    base=load_base(task);visual=Visual(dino,task,torch.load(root/'data/PCA.pt',weights_only=True))
    nets={(s,v):load_adapter(task,s,v,round_name) for s in SEEDS for v in ('fast','history','robot')}
    before={'base':weight_hash(base),**{f'{s}_{v}':weight_hash(n[0]) for (s,v),n in nets.items()}}
    n=(32 if stratum=='canonical' else 64) if phase=='final' else 8
    base_origin=29910000 if phase=='final' else 4100000
    origin=base_origin+TASKS.index(task)*100000+STRATA.index(stratum)*10000
    rows=[];began=time.time()
    assert blocks and len(set(blocks))==len(blocks) and all(x in range(0,n,BLOCK) for x in blocks)
    for off in blocks:
        assert not any(out.glob(f'*_{off:03d}.pt')), 'An assigned block already exists; do not overwrite'
        ids=np.arange(off,min(off+BLOCK,n));cfgs=[object_config(task,int(origin+i),stratum) for i in ids]
        donors=[object_config(task,int(origin+1000000+i),stratum) for i in ids]
        supports={};support_bindings=[]
        for source,cs,delta in [('correct',cfgs,2000000),('donor',donors,3000000)]:
            blob,metrics=collect_block(task,cs,origin+delta+ids,SUPPORT,'base',visual,base,labels=False)
            p=out/f'SUPPORT_{source}_{off:03d}.pt';torch.save(blob,p);h=digest(p);support_bindings.append({'path':str(p),'sha256':h})
            save_json(p.with_suffix('.json'),{'source_sha256':h,'configs':cs,'metrics':metrics})
            supports[source]=support_inputs({k:blob[k].cuda() for k in ('visual','proprio','actions')})
        latents={}
        for seed in SEEDS:
            f,fb=nets[(seed,'fast')]
            latents[f'fast_correct_seed{seed}']=f.adapt(supports['correct'],3)[0]
            latents[f'fast_donor_seed{seed}']=f.adapt(supports['donor'],3)[0]
            latents[f'fast_mean_seed{seed}']=fb['mean_z'].cuda()[None].expand(len(ids),-1)
            for variant in ('history','robot'):latents[f'{variant}_seed{seed}']=nets[(seed,variant)][0].adapt(supports['correct'],3)[0]
        lp=out/f'LATENTS_{off:03d}.pt';torch.save({k:v.cpu() for k,v in latents.items()},lp)
        save_json(out/f'QUERY_START_SEAL_{off:03d}.json',{'time':time.time(),'query_not_started':True,'support_bindings':support_bindings,'latents':{'path':str(lp),'sha256':digest(lp)},'model_freeze_sha256':digest(fpath)})
        initial,result=execute(task,cfgs,origin+4000000+ids,visual,base,out/f'base_{off:03d}.pt')
        rows.append({'arm':'base','offset':off,**{k:v for k,v in result.items() if k not in ('metrics','configs')}})
        for seed in SEEDS:
            for variant in ('fast_correct','fast_mean','fast_donor','history','robot'):
                name=f'{variant}_seed{seed}';net=nets[(seed,variant if variant in ('history','robot') else 'fast')][0]
                reset,result=execute(task,cfgs,origin+4000000+ids,visual,base,out/f'{name}_{off:03d}.pt',net,latents[name]);assert torch.equal(initial,reset)
                rows.append({'arm':name,'offset':off,**{k:v for k,v in result.items() if k not in ('metrics','configs')}})
        print(task,phase,stratum,'complete_conditions',off+len(ids),'seconds',time.time()-began,flush=True)
    after={'base':weight_hash(base),**{f'{s}_{v}':weight_hash(n[0]) for (s,v),n in nets.items()}};assert before==after
    save_json(out/f'WORKER_{worker}.json',{'worker':worker,'blocks':blocks,'planned_total_conditions':n,'task':task,'phase':phase,'round':round_name,'stratum':stratum,'conditions':len(blocks)*BLOCK,'query_episodes':len(blocks)*BLOCK*11,'support_episodes':len(blocks)*BLOCK*2,'seconds':time.time()-began,'networks_unchanged':True,'weights':before,'initial_states_matched':True,'rows':rows})
    print('EVAL_COMPLETE',task,phase,stratum,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=TASKS);p.add_argument('--stratum',required=True,choices=STRATA);p.add_argument('--phase',choices=['final'],default='final');p.add_argument('--round',default='r1');p.add_argument('--dino',default='assets/dinov3');p.add_argument('--blocks',nargs='+',required=True,type=int);p.add_argument('--worker',required=True);a=p.parse_args();evaluate(a.task,a.stratum,a.dino,a.phase,a.round,a.blocks,a.worker)
