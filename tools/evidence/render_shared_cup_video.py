#!/usr/bin/env python3
"""Post-seal private state reconstruction only; never used by a controller."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=int,default=10);a=p.parse_args(argv)
    if not (a.episode/'PREDICTIONS_SEALED.json').exists():raise ValueError('predictions must be sealed before reconstruction')
    if a.output.exists():raise FileExistsError('fresh video output required')
    result=json.loads((a.episode/'RESULT.json').read_text());state=np.load(a.episode/'STATE_TRACE.npz',allow_pickle=False)
    masklog=json.loads((a.episode/'SAM_MASK_EVIDENCE/MASK_EVIDENCE.json').read_text())['frames']
    samples=[row for row in masklog if row['stage_id'] is not None];sample_t=np.array([r['physical_time_sec'] for r in samples])
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import mujoco,cv2
    model=mujoco.MjModel.from_xml_path(str(a.scene));data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=360,width=640)
    a.output.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(a.output),cv2.VideoWriter_fourcc(*'mp4v'),a.fps,(640,360))
    if not writer.isOpened():raise RuntimeError('video encoder unavailable')
    times=state['timestamps'];indices=np.searchsorted(times,np.arange(times[0],times[-1]+1e-6,1/a.fps)).clip(0,len(times)-1)
    try:
        for index in indices:
            data.qpos[:]=state['qpos'][index];mujoco.mj_forward(model,data)
            renderer.disable_depth_rendering();renderer.disable_segmentation_rendering();renderer.update_scene(data,camera='main')
            frame=renderer.render().copy();stage=samples[max(0,min(len(samples)-1,np.searchsorted(sample_t,times[index])-1))]['stage_id'] if samples else 'initial'
            frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            cv2.putText(frame,f'{stage}  physical t={times[index]:.2f}s',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.putText(frame,'POST-SEAL RECONSTRUCTION; actual SAM mask evidence is separate',(8,348),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            writer.write(frame)
    finally:writer.release();renderer.close()
    print(a.output);return str(a.output)

if __name__=='__main__':main()
