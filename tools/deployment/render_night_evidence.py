#!/usr/bin/env python3
"""Annotated post-seal playback. No simulator stepping, controllers or robot commands."""
import argparse,json,os,subprocess,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import cv2,mujoco,numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.load(open(a.run/'RESULT.json'));assert 'post_seal_private_evaluation' in r
 tr=np.load(a.run/'STATE_TRACE.npz');records=json.load(open(a.run/'SAM_MASK_EVIDENCE/MASK_EVIDENCE.json'))['frames'];events=[]
 for row in records:
  if row.get('stage_id') and (not events or events[-1][1]!=row['stage_id']):events.append((row['physical_time_sec'],row['stage_id']))
 starts=np.asarray([e[0] for e in events]);m=mujoco.MjModel.from_xml_path(str(a.scene));d=mujoco.MjData(m);renderer=mujoco.Renderer(m,height=360,width=640);a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_name(a.output.stem+'.raw.mp4');writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),10.,(640,360));duration=float(tr['timestamps'][-1]);begin=time.monotonic()
 try:
  for t in np.arange(0,duration,.1):
   i=int(np.argmin(abs(tr['timestamps']-t)));d.qpos[:]=tr['qpos'][i];mujoco.mj_kinematics(m,d);renderer.update_scene(d,camera='main');rgb=renderer.render().copy();frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);sid=events[max(0,int(np.searchsorted(starts,t,side='right')-1))][1] if events and t>=starts[0] else 'initial grounding / SAM setup';cv2.rectangle(frame,(0,0),(640,59),(18,18,18),-1);cv2.putText(frame,r['task']+' | physics t='+f'{t:.1f}/{duration:.1f}s',(10,21),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA);cv2.putText(frame,'node: '+sid,(10,43),cv2.FONT_HERSHEY_SIMPLEX,.43,(100,230,255),1,cv2.LINE_AA);cv2.putText(frame,'Recorded sealed execution - offline playback',(10,348),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA);writer.write(frame)
 finally:writer.release();renderer.close()
 subprocess.run(['ffmpeg','-v','error','-y','-i',str(tmp),'-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],check=True);tmp.unlink();a.output.with_suffix('.json').write_text(json.dumps({'run':str(a.run.resolve()),'source':'post-seal STATE_TRACE; evaluator-only offline replay, no new rollout','display_fps':10,'duration_physics_sec':duration,'node_timing_source':'actual SAM evidence phase boundaries','real_time_performance_claim':False},indent=2));print(json.dumps({'video':str(a.output),'render_wall_sec':time.monotonic()-begin}),flush=True)
if __name__=='__main__':main()
