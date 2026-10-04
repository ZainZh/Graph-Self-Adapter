from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from prove.fast_adapter.geometry3d_env import GeometrySkill
from prove.fast_adapter.geometry3d_adapter import pose,command
b=Path('evaluations/fast_adapter_p43_neural_geometry_v01');rows=[]
for path in sorted((b/'development').glob('*/*continuous/RESULT.json')):
 r=json.loads(path.read_text());cell=path.parent;pre=json.loads((cell/'PRE_RUN.json').read_text());tr=np.load(cell/'TRACE.npy',allow_pickle=True);obs=np.load(cell/'OBS.npy',allow_pickle=True)
 env=GeometrySkill(pre['cfg'],render=False);updates=[]
 try:
  for x in tr:
   d=x['adaptation'];t=x['t']
   if d is None or d['U']==0 or t<12:continue
   env.data.qpos[:]=x['qpos'];env.data.qvel[:]=x['qvel'];env.mj.mj_forward(env.model,env.data)
   true=env.oracle();before=pose(np.array(d['pre_update_points']));after=x['estimate'];length=r['tool_calibration'].get('length',0.)
   oracle_p,oracle_R,_,_=command(r['task'],t,true,env.start_pos,env.start_R,length)
   before_p,before_R,_,_=command(r['task'],t,before,env.start_pos,env.start_R,length)
   after_p,after_R,_,_=command(r['task'],t,after,env.start_pos,env.start_R,length)
   updates.append(dict(t=t,position_error_before_mm=float(np.linalg.norm(before_p-oracle_p)*1000),position_error_after_mm=float(np.linalg.norm(after_p-oracle_p)*1000),angle_error_before_deg=float(np.linalg.norm(Rotation.from_matrix(before_R@oracle_R.T).as_rotvec())*180/np.pi),angle_error_after_deg=float(np.linalg.norm(Rotation.from_matrix(after_R@oracle_R.T).as_rotvec())*180/np.pi),support_loss_before=d['support_loss_before'],support_loss_after=d['support_loss_after']))
  valid=[i for i,o in enumerate(obs) if o['corners'] is not None and (r['task']!='cup' or i<105)]
  rows.append(dict(task=r['task'],model_seed=pre['bindings']['model_seed'],split=r['split'],seed=r['seed'],success=r['success'],source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),usable_observations=len(valid),last_usable_observation=valid[-1] if valid else None,updates=updates))
 finally:env.close()
summary=[]
for task in ['cup','insert']:
 for split in ['id','geometry']:
  rs=[r for r in rows if r['task']==task and r['split']==split];us=[u for r in rs for u in r['updates']];p=lambda k:float(np.median([u[k] for u in us])) if us else None
  summary.append(dict(task=task,split=split,episodes=len(rs),successes=sum(r['success'] for r in rs),update_steps=len(us),median_position_before_mm=p('position_error_before_mm'),median_position_after_mm=p('position_error_after_mm'),median_angle_before_deg=p('angle_error_before_deg'),median_angle_after_deg=p('angle_error_after_deg'),loss_decreased=sum(u['support_loss_after']<u['support_loss_before'] for u in us),position_improved=sum(u['position_error_after_mm']<u['position_error_before_mm'] for u in us),angle_improved=sum(u['angle_error_after_deg']<u['angle_error_before_deg'] for u in us),failed=[{k:v for k,v in r.items() if k!='updates'} for r in rs if not r['success']]))
out=b/'summary/DEVELOPMENT_MECHANISM_REVIEW.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(dict(scope='Post-hoc diagnosis on the already-opened 96-cell development cohort only; all 32 continuous episodes, no new rollouts or learning',evaluation_truth='Oracle reconstructed from sealed qpos/qvel after observation/action seal, not supplied to candidate; mj_forward may differ slightly from original last-substep derived state',limit='Same-state ideal-command errors are diagnostic proxies, not new success or counterfactual recovery evidence. Insertion ideal control retains the same held-length visual calibration as the candidate.',summaries=summary,rows=rows),indent=2)+'\n')
print(json.dumps(summary,indent=2))
