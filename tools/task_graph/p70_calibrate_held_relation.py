"""Fixed Q2 two-training-state held relation label calibration; no rollout."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 root=Path('evaluations/outer_rsi_p70_object_graph_v01');out=root/'training/held_relation_q2_v01';out.mkdir(exist_ok=False)
 bindings={};samples=[];transforms=[];verified=0
 for index,t,init in [(0,231,20),(1,227,2)]:
  cell=root/f'qualification/Q2/queue/cell_{index:03d}';seal=read(cell/'SEAL.json');result=read(cell/'RESULT.json');prefix=read(cell/'PREFIX_RESULT.json');identity=read(cell/'CELL.json')
  assert identity['init_id']==init and identity['version']=='Q2' and identity['split']=='qualification'
  assert seal['cell']==identity and result['seal_sha256']==sha(cell/'SEAL.json')
  assert prefix['qualified'] and prefix['completed_stage']=='lift_clear' and prefix['actions']==t
  for name,digest in seal['files'].items():
   assert sha(cell/name)==digest,name
   verified+=1
  tracepath=cell/'prefix/controller/TRACE.jsonl';trace=[json.loads(s) for s in tracepath.read_text().splitlines()];row=next(r for r in trace if r['t']==t)
  assert row==trace[-1] and row['facts']['lift_clear'] and row['facts']['attached']
  pub=cell/'prefix'/row['public'];priv=cell/'prefix'/row['private'];assert sha(pub)==row['public_sha256'] and sha(priv)==row['private_sha256']
  names=read(cell/'private_NAMES.json');bi=names['body'].index('black_book_1_main')
  with np.load(pub) as z:
   E=np.eye(4);E[:3,3]=z['eef_pos'];E[:3,:3]=Rotation.from_quat(z['eef_quat_xyzw']).as_matrix()
  H=np.asarray(row['controller_state']['held_T_eef_book']);P=E@H
  with np.load(priv) as z:
   B=np.eye(4);B[:3,3]=z['supervisor_body_xpos'][bi];B[:3,:3]=z['supervisor_body_xmat'][bi].reshape(3,3)
  D=np.linalg.inv(P)@B;transforms.append(D)
  samples.append(dict(init_id=init,cell_index=index,t=t,D_predicted_book_actual_book=D.tolist(),translation_mm=(D[:3,3]*1000).tolist(),rotation_vector_degrees=np.degrees(Rotation.from_matrix(D[:3,:3]).as_rotvec()).tolist()))
  for p in [cell/'SEAL.json',cell/'RESULT.json',cell/'PREFIX_RESULT.json',cell/'CELL.json',cell/'private_NAMES.json',tracepath,pub,priv]:bindings[str(p)]=sha(p)
 poses=np.stack(transforms);rots=Rotation.from_matrix(poses[:,:3,:3]);meanrot=rots.mean();T=np.eye(4);T[:3,:3]=meanrot.as_matrix();T[:3,3]=poses[:,:3,3].mean(0)
 scatter=np.linalg.norm(poses[:,:3,3]-T[:3,3],axis=1)*1000;angular=np.degrees((meanrot.inv()*rots).magnitude())
 bindings[str(Path(__file__))]=sha(__file__)
 calibration=dict(schema='prove.p70-held-relation-calibration.v1',training_GT_supervision=True,runtimeGT=False,n_independent_training_states=2,n_new_physical_steps=0,
  estimator=dict(translation='arithmetic mean in predicted book frame',rotation='scipy Rotation.mean quaternion mean'),
  T_predicted_book_actual_book=T.tolist(),translation_mm=(T[:3,3]*1000).tolist(),rotation_vector_degrees=np.degrees(meanrot.as_rotvec()).tolist(),
  apply=dict(when='once AFTER lift_clear succeeds, at entering orient_clear; never during lift',equation='H_corrected = H_propagated @ T_predicted_book_actual_book; B_runtime = E_public @ H_corrected',maximum_applications_per_episode=1),
  scope='Learned fixed grasp-offset prior from opened native-book training states Q2 init20/init2; not neural training, fresh validation, size generalization or general 6D perception.',
  silhouette_fit_v01_v02_v03_used=False,
  dispersion=dict(translation_population_std_mm=(poses[:,:3,3].std(0)*1000).tolist(),translation_distance_from_mean_mm=scatter.tolist(),rotation_distance_from_mean_degrees=angular.tolist()),samples=samples,bindings=bindings)
 dump(out/'CALIBRATION.json',calibration)
 dump(out/'TRAINING_RECEIPT.json',dict(schema='prove.p70-held-relation-training-receipt.v1',calibration_sha256=sha(out/'CALIBRATION.json'),source_sha256=sha(__file__),all_source_seal_hashes_verified=True,sealed_files_verified=verified,n_independent_training_states=2,selection='Root predeclared Q2 cell000 terminal231 and cell001 terminal227',optimizer_steps=0,physical_steps=0,training_GT_supervision=True,runtimeGT=False,silhouette_fit_used=False,bindings=bindings))
 print(json.dumps(dict(calibration_sha256=sha(out/'CALIBRATION.json'),receipt_sha256=sha(out/'TRAINING_RECEIPT.json'),translation_mm=calibration['translation_mm'],rotation_vector_degrees=calibration['rotation_vector_degrees'],dispersion=calibration['dispersion'],sealed_files_verified=verified),indent=2))
if __name__=='__main__':main()
