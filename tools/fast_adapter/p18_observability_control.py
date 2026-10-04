"""Training-data-only diagnostic with EXTRA geometry supervision.

Not a fair primary baseline and not fresh test evidence. The readout receives
training geometry labels; at the diagnostic deployment side, only cached DINO
support features, known executed probe actions and the common geometric base
controller are consumed. True M,b are used solely by the simulator/scoring.
"""
import json,torch
from pathlib import Path
from tools.fast_adapter.p18_run import physics_data
from tools.fast_adapter.p18_experiment import P18HiddenShift,bounded_axis_probe_actions,projected_oracle_action,step_p18_shift

def main():
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 r=Path('evaluations/fast_adapter_p18_visual_v01');b=torch.load(r/'train_bank/bank.pt',map_location='cpu',weights_only=False)
 x=b['query'][:8192].flatten(0,1).flatten(1).float().cuda();y=(2*b['query_base'][:8192]).flatten(0,1).cuda();nfit=4096*4
 mu=x[:nfit].mean(0);sd=x[:nfit].std(0).clamp(min=.05);x=(x-mu)/sd;x=torch.cat([x,torch.ones(x.shape[0],1,device='cuda')],1)
 w=torch.linalg.solve(x[:nfit].T@x[:nfit]+10*torch.eye(x.shape[1],device='cuda'),x[:nfit].T@y[:nfit])
 # Domains4096..8191 are disjoint from the geometry-readout fit but were part
 # of the ordinary neural models' training bank; hence not formal held-out.
 sf=b['support'][4096:8192,:4].float().flatten(2).cuda();xx=(sf-mu)/sd;xx=torch.cat([xx,torch.ones(*xx.shape[:2],1,device='cuda')],-1)
 relative=(xx@w).cpu();observed_displacement=relative[:,:-1]-relative[:,1:]
 design=torch.cat([bounded_axis_probe_actions()[:3],torch.ones(3,1)],1)
 fit=torch.linalg.lstsq(design.expand(observed_displacement.shape[0],-1,-1),observed_displacement).solution
 estimated=P18HiddenShift(fit[:,:2].transpose(1,2),fit[:,2])
 sh,_,_,qx,qg=physics_data(b['n'],b['seed'],'train',train=True)
 true=P18HiddenShift(sh.M[4096:8192],sh.b[4096:8192]);start=qx[4096:8192,0];goal=qg[4096:8192,0];states={'base':start.clone(),'DINO_geometry_plus_LS':start.clone(),'oracle':start.clone()}
 failed={name:torch.zeros(len(start),dtype=torch.bool) for name in states}
 invalid=~torch.isfinite(estimated.M).all((1,2)) | (torch.linalg.det(estimated.M).abs()<.1)
 # Predefined bounded fallback for ill-conditioned estimates; never true M/b.
 estimated=P18HiddenShift(torch.where(invalid[:,None,None],torch.eye(2),estimated.M),torch.where(invalid[:,None],torch.zeros(2),estimated.b))
 for t in range(12):
  for name,state in states.items():
   action=.5*(goal-state) if name=='base' else projected_oracle_action(state,goal,true if name=='oracle' else estimated)
   states[name]=step_p18_shift(state,true,action)
   failed[name] |= (states[name].abs()>1.546875).any(-1)
 result={'scope':'4096 domains from OPENED training bank; 4096 separate geometry-readout-fit domains; EXTRA supervised geometry labels; diagnostic only, NOT main comparison or fresh held-out benefit','K':3,'fit_ridge':10,'ill_conditioned_estimate_fallbacks':int(invalid.sum()),'predicted_bias_rmse':(estimated.b-true.b).square().mean(0).sqrt().tolist(),'results':{name:{'success':float(((torch.linalg.norm(state-goal,dim=-1)<=.05)&~failed[name]).float().mean()),'mean_error':float(torch.linalg.norm(state-goal,dim=-1).mean())} for name,state in states.items()}}
 (r/'preflight/OBSERVABILITY_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 torch.save({'mu':mu.cpu(),'sd':sd.cpu(),'weight':w.cpu(),'supervision':'extra training geometry; diagnostic readout only'},r/'preflight/DIAGNOSTIC_GEOMETRY_READOUT.pt')

if __name__=='__main__':main()
