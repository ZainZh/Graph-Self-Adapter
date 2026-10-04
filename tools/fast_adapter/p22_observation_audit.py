"""Replay saved actual RGB through the frozen encoder, independent of runtime."""
from pathlib import Path
import json
import numpy as np
import torch
from tools.fast_adapter.p22_data import RGBEncoder
from tools.fast_adapter.p18_run import save_json,digest

R=Path('evaluations/fast_adapter_p22_geometry_v02')

def main():
    torch.set_num_threads(2)
    encoder=RGBEncoder()
    pca={k:v.cuda() if torch.is_tensor(v) else v for k,v in torch.load(R/'bank/PCA.pt',weights_only=True).items()}
    records=[]
    for st in ('id','position','scale'):
        path=R/'final'/st/'seed0/base_RAW.pt';raw=torch.load(path,weights_only=False)
        rgb=raw['audit_rgb'];assert rgb.shape==(3,2,224,224,3)
        dense=encoder.encode(rgb.reshape(-1,224,224,3)).half().float()
        projected=((dense-pca['mean'])@pca['basis']/pca['sd']).cpu().reshape(3,2,64)
        reference=raw['visual'][:2,raw['audit_rgb_steps']].permute(1,0,2).float()
        diff=projected-reference
        rms=float(diff.square().mean().sqrt());maximum=float(diff.abs().max())
        # Cross GPU bf16 computation + fp16 query caching. This is an input audit,
        # not an outcome threshold or a means of choosing a checkpoint.
        assert rms<.05,(st,rms,maximum)
        records.append({'stratum':st,'actual_input_frames':6,'feature_rmse':rms,'max_difference':maximum,
                        'source':str(path),'sha256':digest(path)})
    save_json(R/'summary/OBSERVATION_AUDIT.json',{'passed':True,'input_frames_reencoded':18,
      'feature_rms_tolerance':.05,'records':records,
      'geometry_contract':'p22_eval_v02 constructs one renderer per model and asserts renderer._model is env.model before rollout; never reuses across geometries.'})
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()

