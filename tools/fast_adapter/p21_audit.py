"""Root acceptance: exported bytes, frozen inputs, latent math and action reconstruction."""
from pathlib import Path
import json,time
import numpy as np
import torch
from prove.fast_adapter.insert_adapter import InteractionAdapter,support_actions
from tools.fast_adapter.p18_run import digest,save_json
R=Path('evaluations/fast_adapter_p21_frontends_v01')

def main():
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False
    verified=[]
    for receipt in ['EXPORT_RECEIPT.json','FEATURE_FREEZE.json','TRAIN_FREEZE.json','FINAL_FREEZE.json']:
        doc=json.loads((R/receipt).read_text())
        for b in doc.get('files',doc.get('bindings',[])):
            p=Path(b['path']);assert p.is_file(),str(p)
            assert digest(p)==b['sha256'],f'changed bytes: {p}'
            verified.append(str(p))
    # Recompute representation validation errors from exported diagnostic predictions and simulator labels.
    meta=[torch.load(R/'REMOTE_CACHE_METADATA'/f'shard{i}/META.pt',weights_only=False) for i in range(4)]
    xyz=torch.cat([m['positions_diagnostic_only'] for m in meta])[768:]
    before=[0,1,2,3,4,6,8,10,12];after=[1,2,3,4,5,7,9,11,13]
    diagnostic=json.loads((R/'diagnostic/RESULT.json').read_text())
    for name,m in diagnostic['metrics'].items():
        z=torch.load(R/'diagnostic'/f'{name}_READOUT.pt',weights_only=False)
        xy=float((z['position_predictions'][:,:,:2]-xyz[:,:,:2]).square().mean().sqrt()*1000)
        movement=float((z['motion_predictions']-(xyz[:,after]-xyz[:,before])).square().mean().sqrt()*1000)
        assert abs(xy-m['position_xy_rmse_mm'])<1e-9
        assert abs(movement-m['support_and_query_displacement_rmse_mm'])<1e-9
    final=json.loads((R/'summary/FINAL_RESULTS.json').read_text());audits=[];count=0
    for rep in diagnostic['control_frontends']:
        for st in ['train','ood']:
            for seed in [11,22,33]:
                folder=R/'frontends'/rep/'final'/st/f'seed{seed}'
                seal=torch.load(folder/'LATENTS_COMPACT.pt',weights_only=False)
                s=seal['support_visual'].cuda();actions=support_actions(len(s),'cuda')
                nets={}
                for kind in ['history','hybrid']:
                    net=InteractionAdapter().cuda().eval()
                    net.load_state_dict(torch.load(R/'frontends'/rep/'models'/f'seed{seed}'/f'{kind}.pt',weights_only=False,map_location='cuda')['state'])
                    net.requires_grad_(False);nets[kind]=net
                with torch.no_grad():
                    hz=nets['history'].initial(s,actions);u0=nets['hybrid'].initial(s,actions)
                u3,curve=nets['hybrid'].adapt(s,actions)
                computed={'history':hz,'hybrid_u0':u0,'hybrid_u3':u3,'hybrid_swapped':u3.roll(1,0)}
                max_z=max(float((v.cpu()-seal['latents'][k]).abs().max()) for k,v in computed.items())
                assert max_z<1e-4, max_z
                assert torch.allclose(curve.cpu(),seal['loss_curves']['hybrid_u3'],atol=1e-4,rtol=1e-5)
                action_error=0
                for arm,z in computed.items():
                    raw=torch.load(folder/f'{arm}_RAW.pt',weights_only=False)
                    v=raw['visuals'].float().cuda();base=torch.tensor(raw['base_actions'],dtype=torch.float32,device='cuda')
                    with torch.no_grad():a=nets['history' if arm=='history' else 'hybrid'].action(v,base,z[:,None].expand(-1,45,-1)).cpu().numpy()
                    err=float(np.max(np.abs(a-raw['actions'])));action_error=max(action_error,err)
                    assert err<.001, err
                    pos=raw['positions'][:,-3:];inside=(np.abs(pos[:,:,:2])+np.array([.014,.0125])<=np.array([.0175,.016])+.0002).all(-1)
                    bottom=pos[:,:,2]-.030;penetration=np.array([[f['max_penetration'] for f in row] for row in raw['facts'][-3:]]).T
                    success=(inside&(bottom>=.0065)&(bottom<=.013)&(penetration<=.0015)).all(1)
                    assert np.array_equal(success,raw['success'])
                    assert np.max(np.abs(raw['actions']))<=1+1e-7
                    assert np.max(np.abs(raw['actions']-raw['base_actions']))<=.350001
                    assert len(raw['success'])==48;count+=48
                report=json.loads((folder/'RESULT.json').read_text());assert report['diagnostics']['model_weights_unchanged']
                audits.append({'frontend':rep,'stratum':st,'seed':seed,'max_recomputed_z_error':max_z,'max_action_error_from_half_precision_visual_trace':action_error,'models_unchanged':True})
    assert count==2304
    audit={'status':'passed','verified_binding_count':len(verified),'final_learned_query_trials':count,'final_unique_domains':96,
           'diagnostic_metrics_recomputed':True,'latent_and_action_reconstruction':audits,
           'all_success_geometry_recomputed':True,'model_and_source_freezes_unchanged':True,
           'action_reconstruction_tolerance':.001,'why_tolerance':'raw visuals are stored float16; controller originally consumed float32 projection',
           'large_raw_dino_retained_remote_manifest':str(R/'EXTERNAL_FEATURE_CACHE.json')}
    save_json(R/'summary/AUDIT.json',audit)
    accepted={'status':'completed_mixed_ood_within_checkpoint_gain_only','selected_frontend':diagnostic['selected'],
      'supported':['LiLa has lower position/displacement linear-readout errors on the predeclared diagnostic split','OOD: same LiLa Hybrid checkpoint U0->U3 success +11.111pp, paired95% interval [2.778,20.833]pp','Correct domain z outperforms swapped z in both strata'],
      'not_established':['Reliable ID same-checkpoint gradient advantage','Reliable advantage over same-frontend History','Reliable success advantage of LiLa frontend over last-layer frontend under the present intervals','Beating Base or fixed high-gain feedback','GT-free or robot/VLA/RSI capability'],
      'decision':'retain LiLa as a qualified visual frontend candidate; no formal robot/harness promotion; close this frozen batch without retuning',
      'audit':{'path':str(R/'summary/AUDIT.json'),'sha256':digest(R/'summary/AUDIT.json')},
      'final_results':{'path':str(R/'summary/FINAL_RESULTS.json'),'sha256':digest(R/'summary/FINAL_RESULTS.json')},
      'limits':['96 unique final domains, repeated across3 correlated model seeds','Known target/proprio base; GT ROIs for the last-layer reference','LiLa has external action/future-image training supervision and is projected to64D','Strong-fixed has same global action cap but does not share the learned residual limit','All final data is now opened; future retuning requires a new independent final']}
    save_json(R/'summary/ROOT_ACCEPTANCE.json',accepted);print(json.dumps({'audit':audit['status'],'bindings':len(verified),'final_learned_queries':count,'status':accepted['status']}),flush=True)
if __name__=='__main__':main()
