"""Three independent support sources per condition; fixed paired queries."""
import argparse,json,time
import numpy as np
import torch
from prove.fast_adapter.geometry_insert import geometry
from tools.fast_adapter.p24_rollout import SceneBatch,visual_assets,score
from tools.fast_adapter.p26_rollout import trajectory
from tools.fast_adapter.p27_common import R,SEEDS,STRATA,N,BLOCK,SOURCES,SOURCE_OFFSETS,ARM_MODES,load_adapter,load_base,adaptation_hash,weight_hash,feedback,concatenate_raw
from tools.fast_adapter.p18_run import save_json,digest


def main(st,dino):
    assert (R/'SOURCE_FREEZE.json').exists()
    out=R/'evaluation'/st;out.mkdir(parents=True,exist_ok=False);began=time.time()
    ss=2740000+10000*STRATA.index(st);base=load_base();bh=weight_hash(base)
    encoder,pca=visual_assets(dino);supports={};weights={'base':bh}
    for source,offset in zip(SOURCES,SOURCE_OFFSETS):
        chunks=[]
        for first in range(0,N,BLOCK):
            ids=np.arange(first,first+BLOCK);gs=ss+offset
            scene=SceneBatch([geometry(gs+int(i),st) for i in ids],gs+100000+ids,encoder,pca)
            raw=trajectory(scene,base,gs+200000+ids,32);scene.close()
            path=out/f'SUPPORT_{source}_{first:03d}_RAW.pt';torch.save(raw,path)
            chunks.append(raw)
        raw=concatenate_raw(chunks);idx,fb=feedback(raw)
        raw.update(seed_base=ss+offset,source=source,selected_indices=idx,feedback=fb)
        path=out/f'SUPPORT_{source}_SEALED.pt';torch.save(raw,path)
        save_json(out/f'SUPPORT_{source}_SEAL.json',{'sha256':digest(path),'queries_not_started':True,'conditions':N,'physical_actions_each':32})
        supports[source]=fb
        print(json.dumps({'support_complete':source,'stratum':st,'conditions':N}),flush=True)
    nets={};latents={};latent_hashes={}
    for seed in SEEDS:
        parent=load_adapter(seed,'parent');ah=adaptation_hash(parent)
        for name in ARM_MODES:
            net=parent if name=='parent' else load_adapter(seed,name)
            assert adaptation_hash(net)==ah
            nets[(name,seed)]=net;weights[f'{name}_seed{seed}']=weight_hash(net)
        paths={};zs={}
        for source in SOURCES:
            path,loss=parent.feedback_path({k:v.cuda() for k,v in supports[source].items()})
            paths[source]={'path':torch.stack(path,1).cpu(),'loss':loss.cpu()}
            zs[source]=path[-1].detach()
        fixed=torch.load(R/'controls'/f'seed{seed}.pt',weights_only=True)
        for mode in ('mean','matchedmean'):zs[mode]=fixed[mode].cuda().expand(N,-1)
        zs['k0']=parent.initial(N).detach()
        lp=out/f'LATENTS_seed{seed}.pt';torch.save({'paths':paths,'latents':{k:v.cpu() for k,v in zs.items()}},lp)
        latent_hashes[str(seed)]=digest(lp);latents[seed]=zs
    save_json(out/'QUERY_START_SEAL.json',{'latents':latent_hashes,'weights':weights,'all_supports_and_latents_sealed':True,'time':time.time()})
    pieces={};metric_parts={}
    for first in range(0,N,BLOCK):
        ids=np.arange(first,first+BLOCK);sl=slice(first,first+BLOCK)
        scene=SceneBatch([geometry(ss+int(i),st) for i in ids],ss+100000+ids,encoder,pca)
        jobs=[('base',None,None)]
        jobs += [(f'{name}_{mode}_seed{seed}',nets[(name,seed)],latents[seed][mode][sl]) for seed in SEEDS for name,modes in ARM_MODES.items() for mode in modes]
        for arm,net,z in jobs:
            raw=trajectory(scene,base,ss+300000+ids,48,net,z)
            raw.update(seed_base=ss+first,stratum=st,arm=arm,condition_offset=first)
            p=out/f'{arm}_block{first:03d}_RAW.pt';torch.save(raw,p);sealed=digest(p)
            metric=score(scene,raw);metric['raw_sha256']=sealed;metric['condition_offset']=first
            pieces.setdefault(arm,[]).append({'path':str(p),'sha256':sealed,'condition_offset':first})
            metric_parts.setdefault(arm,[]).append(metric)
        scene.close()
        print(json.dumps({'query_conditions_complete':first+BLOCK,'stratum':st,'arms':len(jobs)}),flush=True)
    assert weight_hash(base)==bh
    for (name,seed),net in nets.items():assert weight_hash(net)==weights[f'{name}_seed{seed}']
    metrics={}
    for arm,parts in metric_parts.items():
        success=sum([x['success'] for x in parts],[])
        metrics[arm]={'success':success,'successes':sum(success),'n':N,'rate':float(np.mean(success)),
            'final_three_checks':sum([x['final_three_checks'] for x in parts],[]),'raw_pieces':pieces[arm]}
    save_json(out/'RESULT.json',{'metrics':metrics,'weights':weights,'latent_records':latent_hashes,'weights_unchanged':True,
        'latents_fixed_during_query':True,'k0_equals_base':True,'runtime_goal_truth':False,'seconds':time.time()-began,
        'independent_condition_clusters':N,'donor_supports_independent_and_not_reused_between_conditions':True})
    print('EVAL_COMPLETE',st,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stratum',choices=STRATA,required=True);p.add_argument('--dino',default='assets/dinov3');a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;main(a.stratum,a.dino)
