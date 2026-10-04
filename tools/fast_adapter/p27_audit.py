"""Frozen control, action/latent and success reconstruction; prediction diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch,mujoco
from prove.fast_adapter.geometry_insert import GeometryInsert,geometry,domain,initial_position
from tools.fast_adapter.p27_common import R,P25,P26,SEEDS,STRATA,N,BLOCK,SOURCES,SOURCE_OFFSETS,ARM_MODES,load_base,load_adapter,adaptation_hash,weight_hash,feedback
from tools.fast_adapter.p18_run import digest,save_json


def main():
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    assert (R/'CAMPAIGN_DONE.json').exists();verified=0
    for name,key in [('SOURCE_FREEZE.json','bindings'),('ANALYSIS_FREEZE.json','bindings'),('EXPORT_RECEIPT.json','files')]:
        for b in json.loads((R/name).read_text())[key]:
            assert digest(Path(b['path']))==b['sha256'],b['path'];verified+=1
    maxima={'action':0.,'latent':0.,'support':0.,'base':0.};episodes=0;aliases=0;base=load_base('cpu');predict=[]
    def error(key,a,b):
        val=float((torch.as_tensor(a)-torch.as_tensor(b)).abs().max());maxima[key]=max(maxima[key],val)
        assert val<.001,(key,val)
    # Recompute fixed matched-context calibration from TRAIN-only inputs.
    train=torch.load(P26/'training_refresh/SUPPORT.pt',weights_only=False);_,tf=feedback(train)
    for seed in SEEDS:
        net=load_adapter(seed,'parent','cpu');zs=net.feedback_path(tf)[0][-1]
        fixed=torch.load(R/'controls'/f'seed{seed}.pt',weights_only=True)
        error('latent',zs,fixed['calibration_z']);error('latent',zs.mean(0),fixed['matchedmean'])
        old=torch.load(P25/'models'/f'seed{seed}'/'physical.pt',map_location='cpu',weights_only=True)['training_mean_z']
        torch.testing.assert_close(old,fixed['mean'],atol=0,rtol=0)
    for st in STRATA:
        folder=R/'evaluation'/st;result=json.loads((folder/'RESULT.json').read_text());ss=2740000+10000*STRATA.index(st)
        assert result['weights_unchanged'] and result['latents_fixed_during_query'] and not result['runtime_goal_truth']
        assert result['weights']['base']==weight_hash(base)
        fbs={}
        for source,offset in zip(SOURCES,SOURCE_OFFSETS):
            p=folder/f'SUPPORT_{source}_SEALED.pt';assert digest(p)==json.loads((folder/f'SUPPORT_{source}_SEAL.json').read_text())['sha256']
            raw=torch.load(p,weights_only=False);idx,fb=feedback(raw)
            torch.testing.assert_close(idx,raw['selected_indices'],atol=0,rtol=0)
            for k,v in fb.items():torch.testing.assert_close(v,raw['feedback'][k],atol=0,rtol=0)
            assert raw['visual'].dtype==torch.float32 and raw['seed_base']==ss+offset
            expected=np.stack([initial_position(ss+offset+200000+i) for i in range(N)])
            assert np.array_equal(expected,raw['positions'][:,0])
            with torch.no_grad():error('support',base(raw['visual'][:,:32],torch.tensor(raw['positions'][:,:32],dtype=torch.float32)),raw['actions'])
            fbs[source]=fb
        nets={};zs={}
        for seed in SEEDS:
            parent=load_adapter(seed,'parent','cpu');ah=adaptation_hash(parent)
            lp=folder/f'LATENTS_seed{seed}.pt';assert digest(lp)==result['latent_records'][str(seed)]
            cached=torch.load(lp,weights_only=True);zs[seed]=cached['latents']
            for source in SOURCES:
                path,_=parent.feedback_path(fbs[source]);error('latent',torch.stack(path,1),cached['paths'][source]['path']);error('latent',path[-1],cached['latents'][source])
            fixed=torch.load(R/'controls'/f'seed{seed}.pt',weights_only=True)
            for mode in ('mean','matchedmean'):error('latent',fixed[mode].expand(N,-1),cached['latents'][mode])
            error('latent',parent.initial(N),cached['latents']['k0'])
            for name in ARM_MODES:
                net=parent if name=='parent' else load_adapter(seed,name,'cpu')
                assert adaptation_hash(net)==ah and weight_hash(net)==result['weights'][f'{name}_seed{seed}'];nets[(name,seed)]=net
        for arm,metric in result['metrics'].items():
            assert metric['n']==N;labels=[]
            for part in metric['raw_pieces']:
                rp=Path(part['path']);assert digest(rp)==part['sha256'];raw=torch.load(rp,weights_only=False)
                first=part['condition_offset'];sl=slice(first,first+BLOCK)
                assert raw['condition_offset']==first and raw['seed_base']==ss+first and raw['expert_label'] is None
                expected=np.stack([initial_position(ss+300000+i) for i in range(first,first+BLOCK)])
                assert np.array_equal(expected,raw['positions'][:,0]) and raw['visual'].dtype==torch.float32
                v=raw['visual'][:,:48];p=torch.tensor(raw['positions'],dtype=torch.float32);a=torch.tensor(raw['actions'])
                prev=torch.cat((torch.zeros_like(a[:,:1]),a[:,:-1]),1)
                with torch.no_grad():
                    ba=base(v,p[:,:48]);error('base',ba,raw['base'])
                    if arm=='base':expected_action=ba
                    else:
                        name,mode,seed_str=arm.split('_');seed=int(seed_str[4:]);net=nets[(name,seed)]
                        expected_action=net.action(v,p[:,:48],prev,ba,zs[seed][mode][sl,None].expand(-1,48,-1))
                        k0=net.action(v,p[:,:48],prev,ba,net.z0.expand(BLOCK,48,-1))
                        torch.testing.assert_close(k0,ba,atol=0,rtol=0);aliases+=1
                    error('action',expected_action,a)
                    assert torch.isfinite(a).all() and float(a.abs().max())<=1.000001 and float((a-ba).abs().max())<=.350001
                    if arm=='base':
                        target=(p[:,1:]-p[:,:48])/.008
                        for seed in SEEDS:
                            net=nets[('refresh',seed)];u=net.encode(v,p[:,:48],prev)
                            for mode,z in zs[seed].items():
                                pred=net.signal_prediction(u,a,z[sl,None].expand(-1,48,-1))
                                predict.append({'stratum':st,'seed':seed,'mode':mode,'offset':first,'condition_mse':(pred-target).square().mean((1,2)).tolist()})
                for i in range(BLOCK):
                    e=GeometryInsert(geometry(ss+first+i,st),render=False);e.set_domain(*domain(ss+100000+first+i));flags=[]
                    for t in (46,47,48):
                        e.data.qpos[:]=raw['positions'][i,t];e.data.qvel[:]=raw['velocities'][i,t];mujoco.mj_forward(e.model,e.data);flags.append(e.facts()['inserted'])
                    labels.append(all(flags));e.close();episodes+=1
            assert labels==metric['success'] and sum(labels)==metric['successes'],(st,arm)
    assert episodes==6528
    save_json(R/'summary/AUDIT.json',{'passed':True,'bindings_verified':verified,'queries_recomputed':episodes,'support_trajectories_recomputed':1152,
        'max_errors':maxima,'tolerance':.001,'k0_aliases_verified_batches':aliases,'initial_states_verified':True,'all_network_weights_unchanged':True,
        'scope':'Original fp32 model inputs, fixed-context provenance, z updates, actions and physical terminal labels; no assertion of bitwise visual re-encoding.'})
    save_json(R/'summary/PREDICTION_DIAGNOSTIC.json',{'scope':'Post-seal predictor comparison on identical frozen-base query states and actions. Uses observed peg displacement, no geometry/action-teacher labels, no additional updates.','records':predict})
    print(json.dumps({'passed':True,'queries':episodes,'max_errors':maxima}),flush=True)

if __name__=='__main__':main()
