"""Read-only optimization diagnostics on already-used training data."""
from pathlib import Path
import json,torch
from tools.fast_adapter.p18_run import model,records_from_features,pack
from tools.fast_adapter.p18_experiment import prepare_latent_for_arm,projected_oracle_action,P18HiddenShift,sample_p18_positions,step_p18_shift

def main():
    torch.set_num_threads(2)
    root=Path('evaluations/fast_adapter_p18_visual_v01');bank=torch.load(root/'train_bank/bank.pt',map_location='cpu',weights_only=False)
    ids=torch.randperm(bank['n'],generator=torch.Generator().manual_seed(90007001))[:256]
    feats=bank['support'][ids,:4].float();qfeat=bank['query'][ids,0].float();base=bank['query_base'][ids,0];expert=bank['query_oracle'][ids,0]
    rows=[]
    for ck in sorted((root/'train').glob('*/*/model.pt')):
        cp=torch.load(ck,map_location='cpu',weights_only=False);sys=model(cp['family']);sys.load_state_dict(cp['state']);sys.freeze_for_eval()
        records=records_from_features(sys,feats)
        z,metrics=prepare_latent_for_arm(sys,cp['family'],records,batch_size=256,k=3,create_graph=False,detach_initial=True)
        zero,_=prepare_latent_for_arm(sys,cp['family'],records,batch_size=256,k=0,create_graph=False,detach_initial=True)
        with torch.no_grad():
            a,_,u=sys.act(pack(qfeat,torch.zeros_like(base),base),z);a0,_,_=sys.act(pack(qfeat,torch.zeros_like(base),base),zero)
            after=records[-1].u_next
            rows.append({'family':cp['family'],'seed':cp['seed'],'projected_u_mean_std_across_domains':float(u.std(0).mean()),'projected_u_total_variance':float(u.var(0).sum()),'raw_dino_mean_std':float(qfeat.flatten(1).std(0).mean()),'z_distance_K3_K0':float((z-zero).norm(dim=-1).mean()),'z_std_across_domains':float(z.std(0).mean()),'mean_action_change_K3_K0':float((a-a0).norm(dim=-1).mean()),'action_MSE_K3':float((a-expert).square().mean()),'action_MSE_K0':float((a0-expert).square().mean()),'action_MSE_base':float((base-expert).square().mean()),'inner_alpha':float(sys.inner_alpha),'adaptation_trace':metrics})
    x,g=sample_p18_positions(1024,90007002);shift=P18HiddenShift(torch.eye(2).repeat(1024,1,1),torch.zeros(1024,2))
    for _ in range(12):x=step_p18_shift(x,shift,.5*(g-x))
    out={'scope':'256 sampled TRAINING-bank domains; no update, no selection, not held-out mechanism evidence','rows':rows,'nominal_base_engineering_success':float((torch.linalg.norm(x-g,dim=-1)<=.05).float().mean())}
    (root/'preflight/TRAINED_OPTIMIZATION_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'models':len(rows),'nominal_base_success':out['nominal_base_engineering_success']}))

if __name__=='__main__':main()
