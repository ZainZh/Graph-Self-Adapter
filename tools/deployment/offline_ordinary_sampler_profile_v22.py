#!/usr/bin/env python3
"""Offline renderer + recorded Root-grounded SAM timing. Never step/command/API."""
import argparse,importlib.util,json,sys,time,threading
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path[:0]=[str(HERE),str(REPO)]
from prove.xtrainer.agent import digest,write
from prove.xtrainer.perception import role_geometry
from sim_robot_io import SimRobot,SimRGBDCamera
from ordinary_server_bundle_v22 import load_bundle

def timed(fn):
 t=time.perf_counter();x=fn();return x,time.perf_counter()-t

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--recorded-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('fresh output required')
 cfg=json.loads(a.config.read_text());v=cfg['runtime']['vision'];ground=json.loads((a.recorded_run/'INITIAL_BOXES.json').read_text());rgbpath=a.recorded_run/'INITIAL_RGB.png'
 assert digest(rgbpath)==ground['frame_sha256']
 rgb=np.asarray(Image.open(rgbpath).convert('RGB'));spec=importlib.util.spec_from_file_location('_offline_sam',v['sam_utility']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 tracker=m.make_sam2_tracker_factory(placeholders=list(ground['role_boxes']),sam2_root=v['sam_root'],checkpoint=v['sam_checkpoint'],config=v['sam_config'],device=v['sam_device'],autocast_dtype=v['sam_autocast_dtype'])('global')
 _,init=timed(lambda:tracker.initialize(rgb,ground['role_boxes']))
 tracks=[]
 for i in range(6):
  masks,duration=timed(lambda:tracker.track(rgb));tracks.append({'sample':i,'sam_forward_sec':duration,'mask_pixels':{r:int(np.asarray(x,bool).sum()) for r,x in masks.items()}})
 # Static offline rendering only; no robot.start(), mj_step(), commands or evaluator.
 b=load_bundle(cfg['task']);robot=SimRobot(scene_run=cfg['sim']['scene_run'],physics_source=cfg['sim']['physics_source'],bundle=b,T_left_from_right=np.eye(4));camera=SimRGBDCamera(robot);camera.connect()
 captures=[]
 for i in range(4):
  with robot.lock:pass
  sample,capture_sec=timed(camera.capture);stamp,rendered,depth,K=sample
  captures.append({'capture_sec':capture_sec,'physical_time_sec':camera.last_physical_time})
 # Use real rendered RGB/depth together for geometry timing; masks come from the
 # approved recorded first view; label this timing-only, not a controller cue.
 _,projection=timed(lambda:role_geometry(depth,masks,K,camera.T))
 camera.close();robot.close()
 write(a.output,{'schema':'graph-self-adapter.offline-sampler-profile.v1','task':cfg['task'],'recorded_RGB':{'path':str(rgbpath),'sha256':digest(rgbpath)},'source':{'path':str(Path(__file__).resolve()),'sha256':digest(__file__)},'sam_initialize_sec':init,'sam_tracks':tracks,'static_camera_captures':captures,'role_projection_sec':projection,'projection_is_timing_only_not_new_control_geometry':True,'background_camera_lock_contention_measured':False,'historical_failure_component_timing_unavailable':True,'physics_steps':0,'commands':0,'API_calls':0,'actual_hardware_test':False,'SAM_checkpoint_sha256':digest(v['sam_checkpoint'])})
 print(a.output)
if __name__=='__main__':main()
