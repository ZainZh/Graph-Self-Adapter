"""Quantify cache-batch8 versus online-batch1 numerics, with weights/z fixed.

Uses already-opened train/development observations only. No physical rollout,
parameter update, truth labels or candidate selection is performed.
"""
import json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from prove.fast_adapter.neural_geometry_visual import FrozenRoleVision, array_observation
from prove.fast_adapter.neural_geometry_adapter import load_model, observation_features, align_observation
from prove.fast_adapter.geometry3d_adapter import initial, pose, command
from tools.fast_adapter.p43_data import ROOT, js, sha


def main():
    vision=FrozenRoleVision(microbatch=1)
    records=[]
    for task in ('cup','insert'):
        vision.set_pca(torch.load(ROOT/'data'/task/'role_pca.pt',map_location='cpu',weights_only=True))
        models={}
        for seed in (11,22):
            path=ROOT/'models'/f'{task}_seed{seed}'/'checkpoint.pt'
            if path.exists():
                model,_=load_model(path);model.eval();model.requires_grad_(False)
                models[seed]=(model,sha(path))
        for split in ('train','dev'):
            pack=np.load(ROOT/'data'/task/split/'observations.npz',allow_pickle=False)
            for episode in (0,len(pack['seeds'])-1):
                seed=int(pack['seeds'][episode]);valid=np.flatnonzero(pack['corners_valid'][episode])
                indices=[int(valid[0]),int(valid[-1])]
                with np.load(ROOT/'data_raw'/task/split/str(seed)/'visible_images.npz') as raw:
                    images=raw['rgb'][indices]
                    masks=np.unpackbits(raw['role_masks_packed'][indices],axis=-1)
                singles=vision.encode(images,masks).cpu().numpy()
                first_obs=array_observation(pack,episode,indices[0]);reference=initial(first_obs,task)
                first_obs=align_observation(first_obs,reference)
                first_cached=torch.from_numpy(observation_features(first_obs))[None]
                first_single=torch.from_numpy(observation_features(dict(first_obs,local=singles[0])))[None]
                tensor_ref=torch.tensor(reference,dtype=torch.float32)[None]
                for j,index in enumerate(indices):
                    obs=align_observation(array_observation(pack,episode,index),reference)
                    cached=pack['local'][episode,index];single=singles[j]
                    error=single-cached
                    record=dict(task=task,split=split,seed=seed,frame_index=index,t=int(pack['t'][episode,index]),
                         cache_feature_rms=float(np.sqrt(np.square(cached).mean())),
                         feature_error_rms=float(np.sqrt(np.square(error).mean())),
                         feature_error_relative_rms=float(np.sqrt(np.square(error).mean()/max(np.square(cached).mean(),1e-12))),
                         feature_absmax=float(np.abs(error).max()),models=[])
                    current_cached=torch.from_numpy(observation_features(obs))[None]
                    current_single=torch.from_numpy(observation_features(dict(obs,local=single)))[None]
                    for model_seed,(model,checkpoint_hash) in models.items():
                        with torch.no_grad():
                            z0,offset=model.encode(first_cached,current_cached,tensor_ref)
                            baseline=model.decode(tensor_ref,offset,z0).numpy()[0]
                            variants=[]
                            for name,first_feature in [('current_only',first_cached),('first_and_current',first_single)]:
                                _,changed=model.encode(first_feature,current_single,tensor_ref)
                                points=model.decode(tensor_ref,changed,z0).numpy()[0]
                                first_position=pack['position'][episode,0];first_rotation=pack['rotation'][episode,0]
                                a0=command(task,60,pose(baseline),first_position,first_rotation,.052)
                                a1=command(task,60,pose(points),first_position,first_rotation,.052)
                                variants.append(dict(variant=name,
                                    corners_rms_m=float(np.sqrt(np.square(points-baseline).mean())),
                                    corners_max_euclidean_m=float(np.linalg.norm(points-baseline,axis=-1).max()),
                                    controller_target_position_delta_m=float(np.linalg.norm(a1[0]-a0[0])),
                                    controller_target_rotation_delta_rad=float(np.linalg.norm(Rotation.from_matrix(a1[1]@a0[1].T).as_rotvec()))))
                        record['models'].append(dict(model_seed=model_seed,checkpoint_sha256=checkpoint_hash,
                             identical_z_for_all_variants=True,variants=variants))
                    records.append(record)
    variants=[v for r in records for m in r['models'] for v in m['variants']]
    result=dict(frames=len(records),no_runtime_change=True,no_new_physical_rollout=True,
        original_data_or_pca_modified=False,all_comparisons_use_same_frozen_model_and_same_z=True,
        action_comparison='Shared P41 command at t60 on identical proprio/start; insertion held length fixed .052m in both branches; not a success or rollout test.',
        feature_relative_rms_median=float(np.median([r['feature_error_relative_rms'] for r in records])),
        feature_relative_rms_max=max(r['feature_error_relative_rms'] for r in records),
        feature_absmax=max(r['feature_absmax'] for r in records),
        max_corner_euclidean_delta_m=max(v['corners_max_euclidean_m'] for v in variants),
        max_controller_target_position_delta_m=max(v['controller_target_position_delta_m'] for v in variants),
        max_controller_target_rotation_delta_rad=max(v['controller_target_rotation_delta_rad'] for v in variants),records=records)
    js(ROOT/'visual/BATCH1_MODEL_IMPACT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
