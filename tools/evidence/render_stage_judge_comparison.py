#!/usr/bin/env python3
"""Uniform4x post-seal comparison: mj_forward only, no integration or scene reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(episode,freeze_path,label):
    result_path=episode/'RESULT.json';state_path=episode/'STATE_TRACE.npz';seal=episode/'PREDICTIONS_SEALED.json'
    result=json.loads(result_path.read_text());freeze=json.loads(freeze_path.read_text())
    if result.get('strict_success') is not True or not json.loads(seal.read_text()).get('sealed_before_private_evaluator'):raise ValueError('sealed strict pass required')
    if result['shared_runtime_sha256']!=freeze['runtime_sha256'] or digest(state_path)!=result['post_seal_state_trace']['sha256']:raise ValueError('runtime/state binding mismatch')
    bound={str(Path(row['path']).resolve()):row['sha256'] for row in freeze['bindings']}
    config_path=next(Path(path) for path in bound if Path(path).name=='CONFIG.json' and Path(path).parent.name!='cup')
    config=json.loads(config_path.read_text());scene=Path(config['sim']['scene_run'])/'scene.xml'
    if digest(scene)!=bound.get(str(scene.resolve())):raise ValueError('frozen scene mismatch')
    state=np.load(state_path,allow_pickle=False);times=state['timestamps'];qpos=state['qpos']
    if len(times)!=len(qpos) or np.any(np.diff(times)<0):raise ValueError('state chronology invalid')
    mask_path=episode/'SAM_MASK_EVIDENCE/MASK_EVIDENCE.json';frames=json.loads(mask_path.read_text())['frames']
    frames=sorted((f for f in frames if f.get('physical_time_sec') is not None and not f.get('discarded_due_to_stale')),key=lambda f:f['physical_time_sec'])
    sample_t=np.asarray([f['physical_time_sec'] for f in frames]);wall=np.asarray([f.get('observed_monotonic_sec',f.get('stamp')) for f in frames]);waits=[]
    for record in result.get('Root_decisions',[]):
        req=json.loads((episode/'REAL_AGENT_HANDOFFS'/record['request_id']/'REQUEST.json').read_text());start=req['issued_monotonic_sec']
        waits.append((float(np.interp(start,wall,sample_t)),float(np.interp(start+record['wall_review_sec'],wall,sample_t))))
    return {'label':label,'result':result,'times':times,'qpos':qpos,'scene':scene,'frames':frames,'sample_t':sample_t,'waits':waits,
            'duration':float(times[-1]-times[0]),'bindings':{'result':{'path':str(result_path.resolve()),'sha256':digest(result_path)},'state':{'path':str(state_path.resolve()),'sha256':digest(state_path)},'freeze':{'path':str(freeze_path.resolve()),'sha256':digest(freeze_path)},'seal':{'path':str(seal.resolve()),'sha256':digest(seal)},'mask_evidence':{'path':str(mask_path.resolve()),'sha256':digest(mask_path)},'scene':{'path':str(scene.resolve()),'sha256':digest(scene)}}}


def main(argv=None):
    repo=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=repo/'media/gemini_graph_v19/annotated_comparison_4x.mp4');p.add_argument('--fps',type=int,default=10);a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('fresh comparison video required')
    left=evidence(repo/'evaluations/stage_trajectory_v18/runs/20261007_101013_bab6630b',repo/'evaluations/stage_trajectory_v18/control/20261007_101013_bab6630b/EXECUTION_FREEZE.json','V18 session judge')
    right=evidence(repo/'evaluations/gemini_graph_v19/runs/20261007_111206_7f0994a9',repo/'evaluations/gemini_graph_v19/control/20261007_111206_7f0994a9/EXECUTION_FREEZE.json','V19 Gemini judge')
    if left['bindings']['scene']['sha256']!=right['bindings']['scene']['sha256']:raise ValueError('comparison scenes differ')
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import mujoco,cv2
    model=mujoco.MjModel.from_xml_path(str(left['scene']));renderer=mujoco.Renderer(model,height=360,width=640)
    for run in (left,right):run['data']=mujoco.MjData(model);run['last_frame']=None
    speed=4.;max_duration=max(left['duration'],right['duration']);display_times=np.arange(0,np.ceil(max_duration*a.fps/speed)*speed/a.fps+1e-8,speed/a.fps)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x360','-r',str(a.fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],stdin=subprocess.PIPE)
    def frame(run,elapsed):
        ended=elapsed>=run['duration'];t=min(float(run['times'][0])+elapsed,float(run['times'][-1]))
        if ended and run['last_frame'] is not None:return run['last_frame']
        index=max(0,min(len(run['times'])-1,int(np.searchsorted(run['times'],t,side='right')-1)));actual_t=float(run['times'][index])
        data=run['data'];data.qpos[:]=run['qpos'][index];mujoco.mj_forward(model,data)
        renderer.disable_depth_rendering();renderer.disable_segmentation_rendering();renderer.update_scene(data,camera='main');rgb=renderer.render().copy()
        sample_index=max(0,min(len(run['frames'])-1,int(np.searchsorted(run['sample_t'],actual_t,side='right')-1)));stage=run['frames'][sample_index].get('stage_id') or 'initial manual Grounding'
        duration=int(round(run['duration']));header=f"{run['label']} | {duration//60}m{duration%60:02d}s | final PASS 25/25"
        cv2.rectangle(rgb,(0,0),(639,72),(0,0,0),-1)
        cv2.putText(rgb,header,(6,17),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
        cv2.putText(rgb,f'{stage} | physical t={actual_t:.1f}s',(6,35),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,0),1)
        waiting=any(start<=actual_t<end for start,end in run['waits'])
        status='FINISHED - final frame held' if ended else ('Session visual judgement wait' if run is left else 'OpenRouter Gemini inference wait') if waiting else 'Controller execution'
        cv2.putText(rgb,status,(6,53),cv2.FONT_HERSHEY_SIMPLEX,.43,(100,255,100),1)
        cv2.putText(rgb,'Uniform 4x INCLUDING ALL waits; offline reconstruction',(6,69),cv2.FONT_HERSHEY_SIMPLEX,.39,(255,255,255),1)
        if ended:run['last_frame']=rgb.copy()
        return rgb
    try:
        for i,t in enumerate(display_times):
            process.stdin.write(np.concatenate([frame(left,t),frame(right,t)],axis=1).tobytes())
            if i%200==0:print(f'{i}/{len(display_times)} frames',flush=True)
    finally:process.stdin.close();renderer.close();code=process.wait()
    if code:raise RuntimeError('H264 encoding failed')
    manifest={'schema':'graph-self-adapter.stage-judge-comparison-video.v1','video':str(a.output.resolve()),'sha256':digest(a.output),'codec':'H264','resolution':[1280,360],'fps':a.fps,'display_duration_sec':len(display_times)/a.fps,'uniform_time_compression':4,'waits_compressed_separately':False,'V19_final_frame_held_until_V18_finishes':True,'postseal_reconstruction_not_live_recording':True,'mujoco_mj_forward_only':True,'mujoco_mj_step_calls':0,'renderer_source_sha256':digest(__file__),'runs':[{'label':r['label'],'physical_duration_sec':r['duration'],'strict_success':r['result']['strict_success'],'nodes_completed':r['result']['graph_stages'],'judgement_source':'current session agent' if r is left else 'OpenRouter google/gemini-3.8-flash; program API receipt, not human approval','evidence_bindings':r['bindings']} for r in (left,right)]}
    a.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');print(a.output,flush=True)

if __name__=='__main__':main()
