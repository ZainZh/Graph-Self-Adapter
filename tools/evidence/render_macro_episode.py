#!/usr/bin/env python3
"""Offline H264 state reconstruction after seal; Agent waits optionally compressed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_schedule(start,end,waits,*,fps=10,wait_display_sec=2,action_speed=4):
    cursor=start;times=[];compressed=[]
    for left,right in sorted(waits):
        left=max(cursor,left);right=min(end,right)
        if right<=left:continue
        times.extend(np.arange(cursor,left,action_speed/fps));n=max(2,int(min(right-left,wait_display_sec)*fps))
        times.extend(np.linspace(left,right,n,endpoint=False));compressed.append((left,right));cursor=right
    times.extend(np.arange(cursor,end+1e-8,action_speed/fps));return np.asarray(times),compressed


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fps',type=int,default=10)
    p.add_argument('--wait-display-sec',type=float,default=2.);p.add_argument('--action-speed',type=float,default=4.);a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('fresh video required')
    seal=a.episode/'PREDICTIONS_SEALED.json';result_path=a.episode/'RESULT.json';state_path=a.episode/'STATE_TRACE.npz'
    if not all(x.is_file() for x in (seal,result_path,state_path)):raise ValueError('finished prediction seal/result/state trace required')
    result=json.loads(result_path.read_text());freeze=json.loads(a.freeze.read_text());bound={str(Path(r['path']).resolve()):r['sha256'] for r in freeze['bindings']}
    if bound.get(str(a.scene.resolve()))!=digest(a.scene):raise ValueError('original frozen scene mismatch')
    if result.get('shared_runtime_sha256')!=freeze.get('runtime_sha256') or result.get('post_seal_state_trace',{}).get('sha256')!=digest(state_path):raise ValueError('sealed runtime/state evidence mismatch')
    state=np.load(state_path,allow_pickle=False);frames=json.loads((a.episode/'SAM_MASK_EVIDENCE/MASK_EVIDENCE.json').read_text())['frames']
    usable=[f for f in frames if f.get('physical_time_sec') is not None and not f.get('discarded_due_to_stale')]
    wall=np.array([f.get('observed_monotonic_sec',f.get('stamp')) for f in usable]);physical=np.array([f['physical_time_sec'] for f in usable]);waits=[]
    for record in result.get('Root_decisions',[]):
        path=a.episode/'REAL_AGENT_HANDOFFS'/record['request_id']/'REQUEST.json'
        request=json.loads(path.read_text());left=request['issued_monotonic_sec'];right=left+record['wall_review_sec']
        waits.append((float(np.interp(left,wall,physical)),float(np.interp(right,wall,physical))))
    display_times,compressed=frame_schedule(float(state['timestamps'][0]),float(state['timestamps'][-1]),waits,fps=a.fps,wait_display_sec=a.wait_display_sec,action_speed=a.action_speed)
    indices=np.searchsorted(state['timestamps'],display_times).clip(0,len(state['timestamps'])-1)
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import mujoco,cv2
    model=mujoco.MjModel.from_xml_path(str(a.scene));data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=360,width=640)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r',str(a.fps),'-i','-',
             '-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE);sample_t=np.array([r['physical_time_sec'] for r in usable]);done=sum(s['completed'] for s in result['stage_trace']);outcome='PASS' if result.get('strict_success') else 'FAIL'
    try:
        for index in indices:
            t=float(state['timestamps'][index]);data.qpos[:]=state['qpos'][index];mujoco.mj_forward(model,data)
            renderer.disable_depth_rendering();renderer.disable_segmentation_rendering();renderer.update_scene(data,camera='main');rgb=renderer.render().copy()
            sample=usable[max(0,min(len(usable)-1,np.searchsorted(sample_t,t)-1))];stage=sample.get('stage_id') or 'initial Grounding'
            label=f'{outcome} {done}/{result["graph_stages"]} | {stage} | physical t={t:.1f}s'
            controller='white observed Code' if stage.startswith('white_direct') else 'red frozen reference+Code'
            cv2.putText(rgb,f'{controller}; actions {a.action_speed:g}x offline; unsupported motion has no Root waits',(8,64),cv2.FONT_HERSHEY_SIMPLEX,.37,(255,255,0),1)
            cv2.putText(rgb,label,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,0),1)
            if any(left<=t<right for left,right in compressed):cv2.putText(rgb,'Root supported-boundary wait compressed',(8,42),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,0),1)
            cv2.putText(rgb,'Post-seal reconstruction; actual SAM evidence separate',(8,348),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            process.stdin.write(rgb.tobytes())
    finally:
        process.stdin.close();renderer.close();code=process.wait()
    if code:raise RuntimeError('H264 encoding failed')
    manifest={'video':str(a.output),'sha256':digest(a.output),'codec':'H264','fps':a.fps,'physical_duration_sec':float(state['timestamps'][-1]-state['timestamps'][0]),
        'display_duration_sec':len(indices)/a.fps,'Root_waits_compressed':compressed,'action_speed_offline':a.action_speed,'wait_display_sec':a.wait_display_sec,'strict_success':result.get('strict_success'),'nodes_completed':done,
        'postseal_only_no_simulation_step':True,'result_sha256':digest(result_path),'state_sha256':digest(state_path),'freeze_sha256':digest(a.freeze)}
    a.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');print(a.output)

if __name__=='__main__':main()
