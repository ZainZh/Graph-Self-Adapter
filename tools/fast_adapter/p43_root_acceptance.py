from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from prove.fast_adapter.geometry3d_env import GeometrySkill,configuration
b=Path('evaluations/fast_adapter_p43_neural_geometry_v01');raw={};max_q=0.;max_v=0.;physical=0;all_outcomes=0;hashes=0
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for task,first in [('cup',433100),('insert',434100)]:
 for model in [11,22]:
  freeze_path=b/'independent_freeze'/f'{task}_seed{model}.json';f=json.loads(freeze_path.read_text());d=Path(f['execution']['out']);audit=json.loads((d/'AUDIT.json').read_text());assert audit['passed'] and audit['cells']==96 and audit['physics_replays']==4
  assert len(list(d.glob('*/RESULT.json')))==96
  for seed,split in zip(f['seeds'],f['splits']):
   for arm in f['arms']:
    p=d/f'{split}_{seed}_{arm}';r=json.loads((p/'RESULT.json').read_text());pre=json.loads((p/'PRE_RUN.json').read_text());seal=json.loads((p/'SEAL.json').read_text());assert (r['task'],r['seed'],r['split'],r['arm'])==(task,seed,split,arm);assert r['scope']=='independent_validation';assert pre['bindings']['freeze']['sha256']==sha(freeze_path);assert pre['bindings']['checkpoint']['sha256']==f['checkpoint_sha256'];assert pre['bindings']['pca']['sha256']==f['pca_sha256'];assert pre['cfg']==configuration(task,seed,split)
    for k,name in [('trace','TRACE.npy'),('observations','OBS.npy'),('terminal','TERMINAL.npy')]:assert sha(p/name)==seal[k];hashes+=1
    for name,h in pre['sources'].items():assert sha(p/'source'/name)==h==f['files'][name]
    assert seal['neural_parameters_before']==seal['neural_parameters_after'] and r['neural_weights_unchanged']
    tr=np.load(p/'TRACE.npy',allow_pickle=True);term=np.load(p/'TERMINAL.npy',allow_pickle=True).item();assert len(tr)==190
    e=GeometrySkill(pre['cfg'],render=False)
    try:
     if model==11 and seed==first and split=='geometry':
      for x in tr:
       max_q=max(max_q,float(np.max(np.abs(e.data.qpos-x['qpos']))));max_v=max(max_v,float(np.max(np.abs(e.data.qvel-x['qvel']))));e.step(x['command_p'],x['command_R'],x['closed'])
      max_q=max(max_q,float(np.max(np.abs(e.data.qpos-term['qpos']))));max_v=max(max_v,float(np.max(np.abs(e.data.qvel-term['qvel']))));physical+=1
     good=[];pen=[]
     for x in list(tr[-9:])+[term]:
      e.data.qpos[:]=x['qpos'];e.data.qvel[:]=x['qvel'];e.mj.mj_forward(e.model,e.data);good.append(e.facts()['success']);pen.append(max([max(0.,-float(c.dist)) for c in e.data.contact]+[0.]))
     assert (all(good) and max(pen)<.002)==r['success'];assert sum(good)==r['last10_successes'];assert abs(max(pen)-r['max_last10_penetration'])<1e-10;all_outcomes+=1
    finally:e.close()
    raw[(task,model,split,seed,arm)]=r['success']
summary=json.loads((b/'training/INDEPENDENT_SUMMARY.json').read_text());rng=np.random.default_rng(430926);checks=[]
for task in ['cup','insert']:
 for split in ['id','geometry']:
  reported=next(s for s in summary['summaries'] if (s['task'],s['split'])==(task,split));seeds=sorted(set(k[3] for k in raw if k[0]==task and k[2]==split));fixed=np.array([[raw[(task,m,split,s,'fixed')] for m in [11,22]] for s in seeds],dtype=float);active=np.array([[raw[(task,m,split,s,'continuous')] for m in [11,22]] for s in seeds],dtype=float);delta=(active-fixed).mean(1)
  assert fixed.mean()==reported['fixed_success'] and active.mean()==reported['continuous_success'] and delta.mean()==reported['paired_difference']
  row=dict(task=task,split=split,conditions=len(seeds),fixed_successes=int(fixed.sum()),continuous_successes=int(active.sum()),episodes_per_arm=int(fixed.size),rescues=int(((active==1)&(fixed==0)).sum()),harms=int(((active==0)&(fixed==1)).sum()))
  if split=='geometry':
   boot=delta[rng.integers(0,len(seeds),size=(100000,len(seeds)))].mean(1);null=(rng.choice([-1.,1.],size=(100000,len(seeds)))*delta).mean(1);p=(1+np.count_nonzero(null>=delta.mean()-1e-12))/100001
   assert np.array_equal(np.quantile(boot,[.0125,.9875]),reported['ci97_5_bonferroni']);assert p==reported['one_sided_sign_flip_p'];row['primary_accepted']=reported['acceptance'];row['delta_pp']=delta.mean()*100;row['ci97_5_pp']=(np.quantile(boot,[.0125,.9875])*100).tolist();row['raw_p']=p
  checks.append(row)
primary=sorted([r for r in checks if 'raw_p' in r],key=lambda r:r['raw_p']);last=0.
for i,row in enumerate(primary):
 last=max(last,min(1.,(2-i)*row['raw_p']));expected=next(x for x in summary['summaries'] if x['task']==row['task'] and x['split']=='geometry');assert last==expected['holm_adjusted_p'];assert row['primary_accepted']==(row['delta_pp']>0 and last<.05)
assert all_outcomes==384 and max(max_q,max_v)<1e-8
record=dict(status='completed_independent_positive_with_limits',accepted_at_utc=datetime.now(timezone.utc).isoformat(),full_numerical_audit_passed=True,root_independent_checks=dict(cells=384,trace_obs_terminal_hashes=hashes,all_source_snapshot_hashes_and_frozen_bindings=True,all_384_successes_recomputed=True,bootstrap_and_sign_flip_and_holm_independently_reproduced=True,root_full_physics_replays=physical,root_physics_qpos_max_error=max_q,root_physics_qvel_max_error=max_v),delegated_audit=dict(all_384_neural_decisions_recomputed=True,latent_and_action_max_error=summary['maximum_replay_errors'],full_physics_replays=summary['full_physics_replays']),results=checks,limits=summary['limits']+['Two frozen trained checkpoints per task, not arbitrary training-seed population','A learned geometry estimator with visible-mask/calibration priors and a fixed skill controller; not the original generic DINO transition predictor or complete VLA','Geometry visibility and substantial remaining OOD failures persist'],sources={str(b/'summary/ROOT_LOCAL_CROSS_HOST_REPLAY.json'):sha(b/'summary/ROOT_LOCAL_CROSS_HOST_REPLAY.json'),str(b/'training/INDEPENDENT_SUMMARY.json'):sha(b/'training/INDEPENDENT_SUMMARY.json'),str(b/'training/DEVELOPMENT_SUMMARY.json'):sha(b/'training/DEVELOPMENT_SUMMARY.json')})
out=b/'summary/ROOT_ACCEPTANCE.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
