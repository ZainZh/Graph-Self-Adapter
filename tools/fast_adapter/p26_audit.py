"""Reconstruct P26 frozen inputs, actions, adaptation and terminal outcomes."""
import json
from pathlib import Path
import numpy as np
import torch,mujoco
from prove.fast_adapter.geometry_insert import GeometryInsert,geometry,domain
from prove.fast_adapter.insertion_teacher import reference_action
from tools.fast_adapter.p26_common import R,PARENT,SEEDS,ARMS,load_base,load_adapter,adaptation_hash,weight_hash
from tools.fast_adapter.p25_common import select_feedback,transitions
from tools.fast_adapter.p18_run import digest,save_json


def main():
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    assert (R/'CAMPAIGN_DONE.json').exists()
    verified=0
    for name,key in [('SOURCE_FREEZE.json','bindings'),('STAGE_A_FREEZE.json','bindings'),('MODEL_FREEZE.json','bindings'),('EXPORT_RECEIPT.json','files')]:
        for b in json.loads((R/name).read_text())[key]:
            assert digest(Path(b['path']))==b['sha256'],b['path'];verified+=1
    base=load_base('cpu');maxima={'action':0.,'latent':0.,'support':0.,'base':0.};episodes=0;identity=0
    def error(name,a,b):
        value=float(torch.max(torch.abs(torch.as_tensor(a)-torch.as_tensor(b))))
        maxima[name]=max(maxima[name],value);assert value<.001,(name,value)
    def check_support(raw):
        v=raw['visual'];p=torch.tensor(raw['positions'],dtype=torch.float32);a=torch.tensor(raw['actions'])
        assert v.dtype==torch.float32
        idx=select_feedback(p,a);torch.testing.assert_close(idx,raw['selected_indices'],atol=0,rtol=0)
        fb=transitions(v,p,a,idx)
        for k,x in fb.items():torch.testing.assert_close(x,raw['feedback'][k],atol=0,rtol=0)
        with torch.no_grad():error('support',base(v[:,:32],p[:,:32]),a)
        return fb
    def check_controller(raw,net=None,z=None):
        v=raw['visual'][:,:48];assert v.dtype==torch.float32 and raw['expert_label'] is None
        p=torch.tensor(raw['positions'][:,:48],dtype=torch.float32)
        a=torch.tensor(raw['actions']);prev=torch.cat((torch.zeros_like(a[:,:1]),a[:,:-1]),1)
        with torch.no_grad():
            ba=base(v,p);error('base',ba,raw['base'])
            expected=ba if net is None else net.action(v,p,prev,ba,z[:,None].expand(-1,48,-1))
            error('action',expected,a)
            assert float((a-ba).abs().max())<=.350001
            if net is not None:
                zero=net.action(v,p,prev,ba,net.z0.expand(len(v),48,-1))
                torch.testing.assert_close(zero,ba,atol=0,rtol=0)
    for st in ('id','position','scale'):
        folder=R/'development'/st;result=json.loads((folder/'RESULT.json').read_text());m=result['metrics']
        assert result['weights_unchanged'] and result['adaptation_modules_unchanged'] and result['latents_fixed_during_query']
        assert result['weights']['base']==weight_hash(base)
        sp=folder/'SUPPORT_SEALED.pt';assert digest(sp)==json.loads((folder/'SUPPORT_SEAL.json').read_text())['sha256']
        fb=check_support(torch.load(sp,weights_only=False));nets={};latents={}
        for seed in SEEDS:
            parent=load_adapter(seed,'parent','cpu');ah=adaptation_hash(parent)
            for arm in ('parent',*ARMS):
                net=parent if arm=='parent' else load_adapter(seed,arm,'cpu')
                assert adaptation_hash(net)==ah
                assert weight_hash(net)==result['weights'][f'{arm}_seed{seed}'];nets[(arm,seed)]=net
            lp=folder/f'LATENTS_seed{seed}.pt';assert digest(lp)==result['latent_records'][str(seed)]
            cached=torch.load(lp,weights_only=True);path,loss=parent.feedback_path(fb)
            error('latent',torch.stack(path,1),cached['path'])
            mean=torch.load(PARENT/'models'/f'seed{seed}'/'physical.pt',weights_only=True,map_location='cpu')['training_mean_z']
            zs={'correct':path[-1],'swapped':path[-1].roll(1,0),'mean':mean.expand(len(path[-1]),-1),'k0':parent.initial(len(path[-1]))}
            for mode,z in zs.items():error('latent',z,cached['latents'][mode])
            latents[seed]=cached['latents']
        starts=None
        for arm,metric in m.items():
            rp=folder/f'{arm}_RAW.pt';assert digest(rp)==metric['raw_sha256'];raw=torch.load(rp,weights_only=False)
            assert raw['expert_label'] is None and np.isfinite(raw['actions']).all() and np.abs(raw['actions']).max()<=1.000001
            if starts is None:starts=raw['positions'][:,0]
            else:assert np.array_equal(starts,raw['positions'][:,0])
            ref=arm in ('repaired_oracle','bounded_repaired_oracle')
            if not ref:
                if arm=='base':check_controller(raw)
                else:
                    name,mode,seed_str=arm.split('_');seed=int(seed_str[4:]);check_controller(raw,nets[(name,seed)],latents[seed][mode]);identity+=1
            for i in range(metric['n']):
                e=GeometryInsert(geometry(raw['seed_base']+i,st),render=False);e.set_domain(*domain(raw['seed_base']+100000+i))
                if ref:
                    for t in range(48):
                        e.data.qpos[:]=raw['positions'][i,t];a=reference_action(e)
                        if arm=='bounded_repaired_oracle':
                            b=raw['base'][i,t].numpy();a=np.clip(a,np.maximum(-1,b-.35),np.minimum(1,b+.35))
                        assert np.max(np.abs(a-raw['actions'][i,t]))<1e-6
                flags=[]
                for t in (46,47,48):
                    e.data.qpos[:]=raw['positions'][i,t];e.data.qvel[:]=raw['velocities'][i,t];mujoco.mj_forward(e.model,e.data);flags.append(e.facts()['inserted'])
                assert all(flags)==metric['success'][i],(st,arm,i)
                e.close();episodes+=1
    training=R/'training_refresh';sp=training/'SUPPORT.pt';assert digest(sp)==json.loads((training/'SUPPORT_SEAL.json').read_text())['sha256']
    fb=check_support(torch.load(sp,weights_only=False))
    for seed in SEEDS:
        net=load_adapter(seed,'relabel','cpu',stage='a');raw=torch.load(training/f'query_seed{seed}_RAW.pt',weights_only=False)
        z=net.feedback_path(fb)[0][-1];error('latent',z,raw['z']);check_controller(raw,net,raw['z'])
        assert raw['training_ids']==list(range(0,384,12))
    assert episodes==540
    save_json(R/'summary/AUDIT.json',{'passed':True,'bindings_verified':verified,'development_queries_recomputed':episodes,'training_queries_actions_recomputed':64,
        'max_errors':maxima,'tolerance':.001,'k0_aliases_verified_on_batches':identity,'adaptation_and_base_unchanged':True,
        'same_initial_states':True,'scope':'Original fp32 controller inputs and post-seal physical states; no claim of bitwise frontend re-encoding.'})
    print(json.dumps({'passed':True,'episodes':episodes,'max_errors':maxima}),flush=True)

if __name__=='__main__':main()
