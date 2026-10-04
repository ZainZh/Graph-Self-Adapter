"""Postseal cup pairing, public-runtime invariants, and side-by-side videos.

Reads recorded public predictions/actions and already sealed evaluation facts.
Never reopens evaluator geometry, retrains weights, or re-integrates physics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from prove.fast_adapter.p61_data import sha256


def load_cell(path):
    path=Path(path);result=json.loads((path/'RESULT.json').read_text());seal=json.loads((path/'SEAL.json').read_text())
    for name,digest in seal['files'].items():
        if sha256(path/name)!=digest:raise ValueError('Changed sealed artifact: '+str(path/name))
    traces=np.load(path/'TRACE.npy',allow_pickle=True).tolist()
    records={int(r['adaptation']['t']):r['adaptation'] for r in traces if r['adaptation'] is not None}
    if not traces or not records:raise ValueError('Missing runtime trace')
    fixed=result['cell']['arm']=='fixed'
    rotation_zero=all(np.array_equal(np.asarray(r['z_after'])[3:6],np.zeros(3)) for r in records.values())
    latent_finite=all(np.isfinite(r['z_after']).all() for r in records.values())
    causal=all(max(r['support_times'],default=-1)<=r['t'] for r in records.values())
    neural_match=all(np.array_equal(r['command_p'],r['neural_action']['target']) for r in traces if r['neural_action']['used'])
    fixed_zero=not fixed or all(not np.asarray(r['z_after']).any() for r in records.values())
    weights_fixed=seal['weights_before']==seal['weights_after']
    invariants=dict(rotation_z_zero=rotation_zero,latent_finite=latent_finite,causal_support=causal,
        commanded_position_equals_neural_output=neural_match,fixed_z_zero=fixed_zero,weights_unchanged=weights_fixed)
    if not all(invariants.values()):raise ValueError('Runtime audit failed: '+str(invariants))
    return dict(path=str(path),result=result,invariants=invariants,trace=traces,
        updates=sum(bool(r['U']) for r in records.values()),
        max_latent_norm=max(float(np.linalg.norm(r['z_after'])) for r in records.values()))


def pair_video(left,right,output):
    import cv2
    with np.load(Path(left['path'])/'PUBLIC_IMAGES.npz') as a,np.load(Path(right['path'])/'PUBLIC_IMAGES.npz') as b:
        if not np.array_equal(a['t'],b['t']):raise ValueError('Unmatched video times')
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),8,(896,508))
        if not writer.isOpened():raise RuntimeError('MP4 writer unavailable')
        try:
            for frame,t in enumerate(a['t']):
                canvas=np.zeros((508,896,3),np.uint8)
                for col,(cell,archive) in enumerate(((left,a),(right,b))):
                    x=col*448;canvas[60:,x:x+448]=cv2.cvtColor(archive['rgb'][frame],cv2.COLOR_RGB2BGR)
                    result=cell['result'];trace=cell['trace'][int(t)]
                    label=f"{result['cell']['arm']} | {result['outcome_class']}"
                    cv2.putText(canvas,label,(x+5,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(canvas,f"t={int(t)} {trace['stage'][:47]}",(x+5,37),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(canvas,'Recorded RGB; display every fourth control step',(x+5,54),cv2.FONT_HERSHEY_SIMPLEX,.31,(190,190,190),1,cv2.LINE_AA)
                writer.write(canvas)
        finally:writer.release()


def report(cells,output,videos=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runs=[load_cell(p.parent) for p in sorted(Path(cells).glob('*/RESULT.json'))]
    if len(runs)!=24:raise ValueError('Expected all24 completed physical cells')
    lookup={(r['result']['cell']['split'],r['result']['cell']['seed'],r['result']['cell']['arm']):r for r in runs}
    if len(lookup)!=24:raise ValueError('Duplicate cells')
    groups={};pairs=[]
    for split in ('id','size','position'):
        seeds=sorted({key[1] for key in lookup if key[0]==split})
        if len(seeds)!=4:raise ValueError('Expected four conditions per split')
        for arm in ('fixed','translation_size5'):
            rr=[lookup[split,seed,arm] for seed in seeds]
            groups[f'{split}/{arm}']=dict(n=len(rr),success=sum(r['result']['validated_success'] for r in rr),
                mechanical_anomaly=sum(not r['result']['mechanical_valid'] for r in rr),
                native_sustained_success=sum(r['result']['sustained_native_success'] for r in rr),
                update_events=sum(r['updates'] for r in rr))
        for seed in seeds:
            fixed,online=lookup[split,seed,'fixed'],lookup[split,seed,'translation_size5']
            a,b=bool(fixed['result']['validated_success']),bool(online['result']['validated_success'])
            row=dict(split=split,seed=seed,fixed_success=a,online_success=b,rescue=b and not a,harm=a and not b)
            if videos and seed==640100 and split in ('id','size'):
                name=f'{split}_{seed}_paired.mp4';pair_video(fixed,online,output/name);row['video']=name
            pairs.append(row)
    result=dict(schema='prove.p64-cup-complete-public-runtime-audit.v1',groups=groups,pairs=pairs,
        all_runtime_invariants_passed=True,scope='4 conditions per stratum; descriptive paired candidate validation, not broad generalization',
        cells=[{k:v for k,v in r.items() if k!='trace'} for r in runs])
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cells',required=True);p.add_argument('--output',required=True);p.add_argument('--videos',action='store_true')
    a=p.parse_args();report(a.cells,a.output,a.videos)
