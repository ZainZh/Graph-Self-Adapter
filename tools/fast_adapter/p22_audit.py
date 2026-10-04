"""Independent post-seal validation of all exported P22 query outcomes/actions."""
from pathlib import Path
import json
import numpy as np
import torch
import mujoco
from prove.fast_adapter.geometry_insert import GeometryInsert,geometry
from prove.fast_adapter.visual_proprio_adapter import VisualBase,ProprioAdapter
from tools.fast_adapter.p22_eval import weight_hash
from tools.fast_adapter.p22_data import R
from tools.fast_adapter.p18_run import save_json,digest

def main():
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    verified=0
    for name,key in [('SOURCE_FREEZE.json','bindings'),('TRAIN_FREEZE.json','bindings'),('FINAL_FREEZE.json','bindings'),('EXPORT_RECEIPT.json','files')]:
        f=R/name
        if not f.exists():continue
        for b in json.loads(f.read_text())[key]:
            p=Path(b['path']);assert p.exists(),str(p)
            assert digest(p)==b['sha256'],str(p);verified+=1
    base=VisualBase().eval();base.load_state_dict(torch.load(R/'models/base/model.pt',weights_only=True)['state'])
    base.requires_grad_(False)
    action_error=0.;latent_error=0.;episodes=0;updates=[]
    for result_path in sorted(R.glob('*/*/seed*/RESULT.json')):
        out=result_path.parent;r=json.loads(result_path.read_text());seed=int(out.name[4:]);st=out.parent.name
        assert r['model_weights_unchanged'] and weight_hash(base)==r['model_hashes']['base']
        support=torch.load(out/'SUPPORT_SEALED.pt',weights_only=True)
        assert digest(out/'SUPPORT_SEALED.pt')==json.loads((out/'SUPPORT_SEAL.json').read_text())['sha256']
        nets={}
        if seed:
            for name in ('history','hybrid'):
                m=ProprioAdapter().eval();m.load_state_dict(torch.load(R/'models'/f'seed{seed}'/f'{name}.pt',weights_only=True)['state'])
                m.requires_grad_(False);assert weight_hash(m)==r['model_hashes'][name];nets[name]=m
            sv,sp,sa=[support[x] for x in ('visual','proprio','actions')]
            with torch.no_grad():z0=nets['hybrid'].initial(sv,sp,sa);zh=nets['history'].initial(sv,sp,sa)
            z3,curve=nets['hybrid'].adapt(sv,sp,sa)
            for name,z in [('u0',z0),('history',zh),('u3',z3),('swapped',z3.roll(1,0))]:
                error=float((z-support['latents'][name]).abs().max());latent_error=max(latent_error,error)
                assert error<.001,(out,name,error)
            updates.append({'path':str(out),'z_rms':float((z3-z0).square().mean().sqrt()),
                            'loss_before':float(curve[:,0].mean()),'loss_after':float(curve[:,-1].mean())})
        for arm,m in r['metrics'].items():
            rawpath=out/f'{arm}_RAW.pt';assert digest(rawpath)==m['raw_sha256']
            raw=torch.load(rawpath,weights_only=False);assert np.max(np.abs(raw['actions']))<=1.000001
            if arm!='oracle':
                v=raw['visual'].float();p=torch.tensor(raw['positions'][:,:48],dtype=torch.float32)
                with torch.no_grad():
                    ba=base(v,p)
                    base_error=float((ba-raw['base']).abs().max())
                    assert base_error<.005,(out,arm,'base',base_error)
                    if arm=='base':pred=ba
                    else:
                        zz=support['latents'][arm][:,None,:].expand(-1,48,-1)
                        pred=nets['history' if arm=='history' else 'hybrid'].action(v,p,raw['base'],zz)
                        assert np.max(np.abs(raw['actions']-raw['base'].numpy()))<=.350001
                error=float(np.max(np.abs(pred.numpy()-raw['actions'])));action_error=max(action_error,error)
                # Queries stored PCA features in fp16; runtime used float32.
                assert error<.005,(out,arm,error)
            for i in range(m['n']):
                env=GeometryInsert(geometry(raw['seed_base']+i,st),render=False)
                passed=[]
                for t in (46,47,48):
                    env.data.qpos[:]=raw['positions'][i,t];env.data.qvel[:]=raw['velocities'][i,t]
                    mujoco.mj_forward(env.model,env.data);passed.append(env.facts()['inserted'])
                assert all(passed)==m['success'][i],(out,arm,i)
                env.close();episodes+=1
    save_json(R/'summary/AUDIT.json',{'passed':True,'hash_bindings':verified,'query_episodes_recomputed':episodes,
      'maximum_action_replay_error':action_error,'action_replay_tolerance':.005,
      'quantization_note':'Queries cached visual64 as fp16; live base/actor saw float32 PCA output.',
      'maximum_latent_replay_error':latent_error,'latent_replay_tolerance':.001,'latent_updates':updates,
      'source_policy_interface':'RGB-derived features, controlled-object XYZ, base predicted from those same inputs and support-only latent; expert functions only used in declared oracle arm or labels.'})
    print(json.dumps({'passed':True,'bindings':verified,'episodes':episodes,'action_error':action_error,'latent_error':latent_error}),flush=True)

if __name__=='__main__':main()

