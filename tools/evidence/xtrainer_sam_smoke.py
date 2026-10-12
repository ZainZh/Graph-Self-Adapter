#!/usr/bin/env python3
"""Explicit recorded-frame SAM compatibility smoke; no live camera/robot imports."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageDraw
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from prove.xtrainer.agent import digest,write
from prove.xtrainer.io import load_calibration,SafetyError
from prove.xtrainer.perception import LiveVision,role_geometry

EPISODE=Path('/media/zheng/Extreme SSD/geo_mani_data/one_shot_demo/put_mug_on_box/real_zed_bimanual_raw/episode_20261004142813_579607')
BOXES={'white_mug':[157,245,233,333],'red_mug':[260,256,322,347],'box':[314,139,422,237]}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists():raise FileExistsError('fresh offline smoke output required')
    cfg=json.loads((REPO/'task_packages/xtrainer_v01/CONFIG_PROPOSED.json').read_text())
    T,_=load_calibration(cfg['left_calibration'],cfg['right_calibration'])
    config={**cfg['vision'],'role_boxes':BOXES};vision=LiveVision(None,T,config)
    start=time.monotonic();tracker=vision._tracker();load_sec=time.monotonic()-start
    rows=[];bindings=[];args.output.mkdir(parents=True)
    colors={'white_mug':np.array([0,220,255]),'red_mug':np.array([255,60,70]),'box':np.array([30,220,50])}
    for index in range(6):
        source=EPISODE/f'global_{index:06d}.npz'
        with np.load(source,allow_pickle=False) as data:
            if int(data['serial'])!=38968158:raise SafetyError('wrong recorded camera serial')
            rgb=data['rgb'].copy();depth=data['depth_m'].copy();K=data['camera_matrix'].copy()
            stamp=float(data['timestamp_monotonic_sec'])
        if rgb.shape!=(360,640,3) or depth.shape!=(360,640) or not 300<K[0,0]<400:
            raise SafetyError('wrong actual resized frame/intrinsics')
        started=time.monotonic()
        masks=tracker.initialize(rgb,BOXES) if index==0 else tracker.track(rgb)
        import torch
        torch.cuda.synchronize();sam_sec=time.monotonic()-started
        geometry=role_geometry(depth,masks,K,T);overlay=rgb.astype(float).copy();measurements={}
        for role,mask in masks.items():
            mask=np.asarray(mask,dtype=bool)
            overlay[mask]=.55*overlay[mask]+.45*colors[role]
            valid=mask&np.isfinite(depth)&(depth>0)
            measurements[role]={'mask_pixels':int(mask.sum()),'valid_depth_pixels':int(valid.sum()),
                                'surface_median_left_base_m':None if geometry[role] is None else geometry[role].tolist()}
        image=Image.fromarray(overlay.astype(np.uint8));draw=ImageDraw.Draw(image)
        if index==0:
            for role,box in BOXES.items():draw.rectangle(box,outline=tuple(colors[role]),width=2);draw.text((box[0],box[1]),role,fill='yellow')
        output=args.output/f'overlay_{index:06d}.png';image.save(output)
        rows.append({'frame_index':index,'recorded_monotonic_sec':stamp,'sam_call_wall_sec':sam_sec,
                     'roles':measurements,'overlay':{'path':str(output),'sha256':digest(output)}})
        bindings.append({'path':str(source),'sha256':digest(source)})
    paths=[Path(cfg['vision']['sam_utility']),Path(cfg['vision']['sam_checkpoint']),Path(cfg['left_calibration']),
           Path(cfg['right_calibration']),Path(__file__),REPO/'prove/xtrainer/perception.py',EPISODE/'manifest.json']
    bindings.extend({'path':str(path),'sha256':digest(path)} for path in paths)
    report={'schema':'graph-self-adapter.xtrainer-recorded-SAM-smoke.v1','source_episode':str(EPISODE),
            'source_bindings':bindings,'camera_serial':38968158,'role_boxes_from_actual_Root_RGB':BOXES,
            'tracker_load_wall_sec':load_sec,'frames':rows,'hardware_connections':0,'hardware_motion_commands':0,
            'only_recorded_RGBD_used':True,'success_criterion':'source compatibility, real masks and finite permitted RGBD role geometry',
            'not_measured':['segmentation accuracy','live end-to-end latency','hardware task success','collision safety'],
            'all_roles_geometry_known':all(all(r['surface_median_left_base_m'] is not None for r in row['roles'].values()) for row in rows)}
    write(args.output/'REPORT.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('frames','source_bindings')},indent=2))
    return report
if __name__=='__main__':main()
