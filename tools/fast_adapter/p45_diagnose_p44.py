import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prove.fast_adapter.robot_geometry_env_v05 import RobotGeometrySkill
from prove.fast_adapter.geometry3d_adapter import project
root=Path('evaluations/fast_adapter_p44_robot_geometry_v01/development_v05/cup_seed11');out=[]
for p in sorted(root.iterdir()):
 if not (p/'TRACE.npy').exists():continue
 pre=json.loads((p/'PRE_RUN.json').read_text());e=RobotGeometrySkill(pre['cfg'],False);tr=np.load(p/'TRACE.npy',allow_pickle=True);obs=np.load(p/'OBS.npy',allow_pickle=True);rows=[]
 for i in [0,11,30,59,90,104,105,124,189]:
  s=tr[i];e.mj.mj_setState(e.model,e.data,s['integration'],e.mj.mjtState.mjSTATE_INTEGRATION);e.mj.mj_forward(e.model,e.data)
  c,R,size=e.oracle();truthgrip=c+R[:,0]*(size[0]-.003)-R[:,2]*.004;point=np.array([[-size[0],size[1],0],[size[0],size[1],0],[size[0],-size[1],0],[-size[0],-size[1],0]])@R.T+c;uv=project(point,obs[i]);corn=obs[i]['corners'];er=None if corn is None else min(float(np.sqrt(np.square(np.roll(corn,k,axis=0)-uv).sum(-1).mean())) for k in range(4))
  ep,erot,esz=s['estimate'];eg=ep+erot[:,0]*(esz[0]-.003)-erot[:,2]*.004;pose=e.proprio();record=dict(t=i,observed_corners=corn is not None,corner_rms_px=er,true_handle_pixels_diagonal=float(np.linalg.norm(uv.max(0)-uv.min(0))),estimated_center_error_m=float(np.linalg.norm(ep-c)),estimated_grip_error_m=float(np.linalg.norm(eg-truthgrip)),estimated_angle_error_deg=float(np.degrees(np.linalg.norm(Rotation.from_matrix(erot@R.T).as_rotvec()))),tool_to_estimated_grip_m=float(np.linalg.norm(pose[0]-eg)),tool_to_true_grip_m=float(np.linalg.norm(pose[0]-truthgrip)),actual_to_command_position_m=float(np.linalg.norm(pose[0]-s['command_p'])),ik_error=s['robot_tracking'].get('ik_position_error'),estimate_size=np.asarray(esz).tolist(),true_size=size.tolist());rows.append(record)
 out.append(dict(cell=str(p),rows=rows,visible_before_closure=sum(x['corners'] is not None for x in obs[:105])));e.close()
train=np.load('evaluations/fast_adapter_p43_neural_geometry_v01/data/cup/train/observations.npz');ranges={}
for key in ['position','camera_position','local','role_bbox']:
 x=train[key].reshape(-1,train[key].shape[-1]) if key!='role_bbox' else train[key].reshape(-1,12)
 a=np.array([np.load(Path(r['cell'])/'OBS.npy',allow_pickle=True)[0][key].reshape(-1) for r in out]);std=x.std(0);ranges[key]=dict(training_min=x.min(0).tolist(),training_max=x.max(0).tolist(),p44_first=a.tolist(),fraction_p44_outside_training_range=float(np.mean((a<x.min(0))|(a>x.max(0)))))
record=dict(scope='read-only diagnosis of opened sealed P44 v05; evaluator truth only in analysis',cells=out,input_shift=ranges)
Path('evaluations/fast_adapter_p45_robot_mechanism_v01/diagnostics/P44_CUP_FAILURE_DIAGNOSIS.json').write_text(json.dumps(record,indent=2)+'\n')
for r in out:
 print(Path(r['cell']).name, 'preclosevalid',r['visible_before_closure'])
 for x in r['rows']:
  if x['t'] in [0,104,105]:print({k:x[k] for k in ['t','corner_rms_px','estimated_grip_error_m','estimated_angle_error_deg','tool_to_estimated_grip_m','tool_to_true_grip_m','ik_error']})
print('shift',{k:v['fraction_p44_outside_training_range'] for k,v in ranges.items()})
