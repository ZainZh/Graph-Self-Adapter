"""Cache frozen DINO pre-fusion layers from existing hash-bound P45 pixels."""
import argparse
import json
from pathlib import Path
import time
import torch
from prove.fast_adapter.p61_data import source_episode,offline_labels,sha256,FRAME_INDICES
from prove.fast_adapter.visual_alignment import FrozenDinoROI,DEFAULT_MODEL

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--model',default=DEFAULT_MODEL);ap.add_argument('--device',default='cuda')
    ap.add_argument('--microbatch',type=int,default=4)
    args=ap.parse_args();source=Path(args.source);output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    upstream=json.loads((source/'DATASET_MANIFEST.json').read_text())
    if upstream['split_counts'] != {'train':96,'dev':12}:raise ValueError('cohort mismatch')
    net=FrozenDinoROI(args.model,device=args.device,microbatch=args.microbatch)
    records=[];start=time.time()
    for record in upstream['records']:
        origin=source/'episodes'/record['split']/str(record['seed'])
        if sha256(origin/'MANIFEST.json') != record['manifest_sha256']:raise ValueError('manifest mismatch')
        rgb,public,rows,manifest=source_episode(origin)
        layers=[]
        for index in range(0,len(rgb),args.microbatch):
            with torch.inference_mode():
                pixels=torch.from_numpy(rgb[index:index+args.microbatch].copy()).to(args.device).permute(0,3,1,2).float()/255
                pixels=((pixels-net.mean)/net.std).to(torch.bfloat16)
                hidden=net.model(pixel_values=pixels,output_hidden_states=True,return_dict=True).hidden_states
                if len(hidden)!=25:raise ValueError('DINO depth mismatch')
                value=torch.stack([hidden[i][:,net.prefix:] for i in (-12,-8,-4)],1)
                if value.shape[1:]!=(3,784,1024) or not torch.isfinite(value).all():raise ValueError('DINO layer shape/finite failure')
                layers.append(value.to('cpu',torch.float16))
        public['dino_layers']=torch.cat(layers)[:,None]
        private=offline_labels(origin,rows,public['T_world_anchor'],allow_private=True)
        destination=output/'episodes'/record['split']/str(record['seed']);destination.mkdir(parents=True)
        torch.save(public,destination/'public.pt');torch.save(private,destination/'private.pt')
        records.append(dict(path=str(destination.resolve()),split=record['split'],seed=record['seed'],distribution=manifest['distribution'],
            public_sha256=sha256(destination/'public.pt'),private_sha256=sha256(destination/'private.pt'),source_manifest_sha256=record['manifest_sha256']))
        print(json.dumps(dict(done=len(records),total=108,elapsed_s=round(time.time()-start,1),episode=record['seed'])),flush=True)
    result=dict(schema='prove.p61-native-dino-dataset.v1',records=records,split_counts=upstream['split_counts'],frame_indices=list(FRAME_INDICES),
        source_manifest_sha256=sha256(source/'DATASET_MANIFEST.json'),environment='P45 Panda pre-contact privileged collection; not floating P43 holder',
        dino_model_path=str(Path(args.model).resolve()),dino_weights_sha256=sha256(Path(args.model)/'model.safetensors'),
        hidden_indices=[-12,-8,-4],prefix_stripped=net.prefix,dino_dtype='float16 cache from bfloat16 frozen forward',
        preprocessing='native448 RGB/255; official model image_mean/image_std; no fitting',roles=['actor','object','goal'],
        geometry8='opening in first EEF anchor: center3,rotvec3,log(opening halfsize2); contour renderer adds +0.0006m Rz and +0.006m halfsize',
        geometry12_world='private original opening center3,rotation6D,opening halfsize2,privileged held length1',
        held_length_public_available=False,developer_identity='opened P45 development; not fresh held-out',
        sparse_frame_warning='selected intervals are multi-step; never claim one-step transitions',physics_integration_steps=0,
        new_rendering_steps=0,source_code_sha256={p:sha256(p) for p in ['prove/fast_adapter/p61_data.py','tools/fast_adapter/p61_materialize.py']})
    (output/'MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
