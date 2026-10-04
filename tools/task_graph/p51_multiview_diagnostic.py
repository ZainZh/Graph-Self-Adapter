#!/usr/bin/env python3
"""Exactly twenty offline alternating visible-image updates; no simulator."""
from pathlib import Path
import json,sys,hashlib
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prove.task_graph.p51_public_grounding import P51MultiviewFrozenAdapterGrounder
from prove.fast_adapter.p45_neural import load_model
from tools.fast_adapter.p47_evaluate import parameters_sha
cv2.setNumThreads(1);torch.set_num_threads(1)
base=ROOT/'evaluations/fast_adapter_p51_libero_complete_v01';bundle=base/'grounding_multiview_reference_v01';out=base/'MULTIVIEW_DIAGNOSTIC_v01.json'
if out.exists():raise FileExistsError('Preserve the original bounded diagnostic')
checkpoint=ROOT/'evaluations/fast_adapter_p50_precision_policy_v01/input_cache/p45_insert_geometry_seed11.pt'
model,_=load_model(checkpoint,'cpu');before=parameters_sha(model);grounder=P51MultiviewFrozenAdapterGrounder(bundle,model)
f=np.load(ROOT/'evaluations/fast_adapter_p48_task_graph_v01/gt_replay_v01/frame_0000.npz')
public={}
for channel in grounder.channels:
 masks=np.load(bundle/channel/'reference_masks.npz');meta=json.loads((bundle/channel/'RENDER.json').read_text())
 public[channel]=dict(rgb=grounder.channels[channel].rgb,role_masks={role:masks[role] for role in ('actor','object','goal')},
                      **{k:meta[k] for k in ('K','camera_position','camera_rotation')},
                      position=f['observation_robot0_eef_pos'],rotation=Rotation.from_quat(f['observation_robot0_eef_quat']).as_matrix())
# These fixed reference transforms are used only after each public-input update
# to quantify opened-development error, never supplied as runtime observations.
nominal_reference=grounder.T0;nominal_target=nominal_reference@grounder.offset
rows=[]
for t in range(20):
 channel=('agentview','frontview')[t%2];r=grounder.step(t,channel=channel,**public[channel]);q=r['quality']
 row=dict(t=t,channel=channel,valid=r['valid'],inliers=q['inliers'],tracked_features=q['tracked_features'],reason=q['reason'])
 if r['valid']:
  T=r['candidate_T_world_target'];R=r['T_world_reference']
  row.update(target_position_error_m=float(np.linalg.norm(T[:3,3]-nominal_target[:3,3])),
             target_rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(T[:3,:3]@nominal_target[:3,:3].T).as_rotvec())),
             reference_position_error_m=float(np.linalg.norm(R[:3,3]-nominal_reference[:3,3])),
             reprojection_rms_px=q['corrected_reprojection_rms_px'],U=r['adaptation_record']['U'],
             T_world_target=T.tolist(),z=r['adaptation_record']['z'])
 rows.append(row)
report=dict(schema='prove.p51-opened-two-camera-diagnostic.v1',scope='twenty alternating self-reference RGB/mask observations; zero actions, no task-success evidence',
            channels=['agentview','frontview'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            neural_weights_unchanged=before==parameters_sha(model),physical_steps=0,rows=rows,
            metric_reference='fixed opened teacher front-wall frame and target transform, evaluator-only for error reporting',
            limitation='Self-reference images do not qualify tracking under new poses, occlusions, or full-task execution')
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2,allow_nan=False))
