"""P25 independent action/latent/selection/outcome reconstruction, original fp32 inputs."""
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from prove.fast_adapter.geometry_insert import GeometryInsert, geometry, domain
from tools.fast_adapter.p25_common import R, SEEDS, VARIANTS, load_base, load_adapter, weight_hash, select_feedback, transitions
from tools.fast_adapter.p18_run import save_json, digest


def main():
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False
    assert (R/'CAMPAIGN_DONE.json').exists()
    verified = 0
    for name, key in [('SOURCE_FREEZE.json','bindings'),('MODEL_FREEZE.json','bindings'),('EXPORT_RECEIPT.json','files')]:
        for b in json.loads((R/name).read_text())[key]:
            assert digest(Path(b['path'])) == b['sha256'], b['path']; verified += 1
    base = load_base('cpu'); episodes = 0; maxima = dict(action=0., latent=0., support=0., base=0.)
    coverage = {}; identity = {}; diagnostics = []
    for st in ('id','position','scale'):
        root = R/'development'/st; result = json.loads((root/'RESULT.json').read_text())
        assert result['weights_unchanged'] and result['latents_fixed_during_query'] and result['slot_geometry_not_runtime_input']
        assert weight_hash(base) == result['weights']['base']
        support = torch.load(root/'SUPPORT_SEALED.pt', weights_only=False)
        assert digest(root/'SUPPORT_SEALED.pt') == json.loads((root/'SUPPORT_SEAL.json').read_text())['sha256']
        sv = support['visual']; sp = torch.tensor(support['positions'],dtype=torch.float32); sa = torch.tensor(support['actions'])
        assert sv.dtype == torch.float32
        idx = select_feedback(sp,sa); assert torch.equal(idx,support['selected_indices'])
        feedback = transitions(sv,sp,sa,idx); early = transitions(sv,sp,sa,torch.arange(3).expand(len(sv),-1))
        for k,v in feedback.items(): torch.testing.assert_close(v,support['feedback'][k],atol=0,rtol=0)
        with torch.no_grad():
            sup_action = base(sv[:,:32],sp[:,:32])
        error=float((sup_action-sa).abs().max());maxima['support']=max(maxima['support'],error);assert error<.001
        ss = support['seed_base']; contacts = np.zeros((len(sv),3),bool); low = np.zeros_like(contacts)
        for i in range(len(sv)):
            e = GeometryInsert(geometry(ss+i,st),render=False);e.set_domain(*domain(ss+100000+i))
            for j,t in enumerate(idx[i].tolist()):
                e.data.qpos[:]=support['positions'][i,t+1];e.data.qvel[:]=support['velocities'][i,t+1]
                mujoco.mj_forward(e.model,e.data)
                contacts[i,j]=bool(e.data.ncon)
                low[i,j]=e.state()[2]-.030*e.scale <= .068*e.scale
            e.close()
        coverage[st]={'selected_indices':idx.tolist(),'contact_at_selected_next_observation_by_window':contacts.sum(0).tolist(),
            'below_wall_top_by_window':low.sum(0).tolist(),'conditions':len(sv),'evaluator_only_postseal':True}
        nets={};zs={}
        for seed in SEEDS:
            record = torch.load(root/f'LATENTS_seed{seed}.pt',weights_only=True)
            assert digest(root/f'LATENTS_seed{seed}.pt') == result['latent_records'][str(seed)]
            for name in VARIANTS:
                net=load_adapter(seed,name,'cpu');nets[(name,seed)]=net
                assert weight_hash(net)==result['weights'][f'{name}_seed{seed}']
                path,loss=net.feedback_path(feedback)
                error=float((torch.stack(path,1)-record['paths'][name]['path']).abs().max())
                maxima['latent']=max(maxima['latent'],error);assert error<.001,(st,seed,name,error)
                checkpoint=torch.load(R/'models'/f'seed{seed}'/f'{name}.pt',map_location='cpu',weights_only=True)
                expected={'correct':path[-1],'swapped':path[-1].roll(1,0),'mean':checkpoint['training_mean_z'].expand(len(sv),-1)}
                if name=='original':expected['k0']=net.initial(len(sv))
                if name=='physical':expected['early']=net.feedback_path(early)[0][-1]
                for mode,z in expected.items():
                    cached=record['latents'][name][mode]
                    assert torch.isfinite(cached).all()
                    error=float((z-cached).abs().max());maxima['latent']=max(maxima['latent'],error)
                    assert error<.001,(st,seed,name,mode,error)
                    zs[f'{name}_{mode}_seed{seed}']=cached
                dz=path[-1]-path[0]
                diagnostics.append({'stratum':st,'seed':seed,'variant':name,'alpha':float(net.alpha),
                    'z_update_rms':float(dz.square().mean().sqrt()),
                    'common_update_energy_fraction':float(dz.mean(0).square().sum()/dz.square().sum(-1).mean().clamp_min(1e-12)),
                    'loss_before':loss[:,:,0].mean(0).tolist() if loss is not None else None,
                    'loss_after':loss[:,:,1].mean(0).tolist() if loss is not None else None})
        starts = None
        for arm,metric in result['metrics'].items():
            rawpath=root/f'{arm}_RAW.pt';assert digest(rawpath)==metric['raw_sha256']
            raw=torch.load(rawpath,weights_only=False)
            assert raw['expert_label'] is None and np.isfinite(raw['actions']).all()
            assert np.max(np.abs(raw['actions']))<=1.000001
            if starts is None:starts=raw['positions'][:,0].copy()
            else:assert np.array_equal(starts,raw['positions'][:,0])
            if arm not in ('oracle','bounded_oracle'):
                v=raw['visual'][:,:48];assert v.dtype==torch.float32
                p=torch.tensor(raw['positions'][:,:48],dtype=torch.float32)
                prev=torch.tensor(np.concatenate((np.zeros_like(raw['actions'][:,:1]),raw['actions'][:,:-1]),1))
                with torch.no_grad():
                    ba=base(v,p);error=float((ba-raw['base']).abs().max());maxima['base']=max(maxima['base'],error);assert error<.001
                    if arm=='base':
                        action=ba
                        for (name,seed),net in nets.items():
                            if name!='original':
                                z0=net.z0.expand(len(v),48,-1)
                                zero=net.action(v,p,prev,ba,z0)
                                torch.testing.assert_close(zero,ba,atol=0,rtol=0)
                                identity[f'{st}_{name}_seed{seed}']=True
                    else:
                        name=arm.split('_')[0];seed=int(arm.rsplit('seed',1)[1]);net=nets[(name,seed)]
                        z=zs[arm][:,None].expand(-1,48,-1)
                        action=net.action(v,p,prev,ba,z)
                        assert float((action-ba).abs().max())<=.350001
                error=float(np.max(np.abs(action.numpy()-raw['actions'])));maxima['action']=max(maxima['action'],error)
                assert error<.001,(st,arm,error)
            for i in range(metric['n']):
                e=GeometryInsert(geometry(raw['seed_base']+i,st),render=False);e.set_domain(*domain(raw['seed_base']+100000+i))
                if arm in ('oracle','bounded_oracle'):
                    for t in range(48):
                        e.data.qpos[:]=raw['positions'][i,t];e.data.qvel[:]=raw['velocities'][i,t]
                        mujoco.mj_forward(e.model,e.data);a=e.expert_action().astype(np.float32)
                        if arm=='bounded_oracle':
                            b=raw['base'][i,t].numpy();a=np.clip(a,np.maximum(-1,b-.35),np.minimum(1,b+.35))
                        assert np.max(np.abs(a-raw['actions'][i,t]))<1e-6
                flags=[]
                for t in (46,47,48):
                    e.data.qpos[:]=raw['positions'][i,t];e.data.qvel[:]=raw['velocities'][i,t]
                    mujoco.mj_forward(e.model,e.data);flags.append(e.facts()['inserted'])
                assert all(flags)==metric['success'][i],(st,arm,i)
                e.close();episodes+=1
    assert episodes==1116
    out=R/'summary';out.mkdir(exist_ok=True)
    save_json(out/'AUDIT.json',{'passed':True,'source_model_export_bindings':verified,'query_episodes_recomputed':episodes,
        'max_errors':maxima,'tolerance':.001,'shared_initial_states':True,'all_anchored_k0_aliases_verified':identity,
        'selected_support_coverage':coverage,'latent_diagnostics':diagnostics,
        'scope':'Replay of original cached float32 controller inputs, latent updates, selector and terminal physical outcome. No claim of bitwise frontend re-encoding.'})
    print(json.dumps({'passed':True,'episodes':episodes,'max_errors':maxima}),flush=True)


if __name__=='__main__':main()
