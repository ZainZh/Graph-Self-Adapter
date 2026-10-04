#!/usr/bin/env python3
"""Chinese annotated video from sealed complete-task online episode only.

Prefix visualization reconstructs saved positions via mj_fwdPosition, never
integrates or executes actions. Insert uses the actual recorded public RGB.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont

STAGES=['移动','接近','抓取','抬起','搬运','对齐','放入']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def choose_font(explicit):
    choices=[explicit] if explicit else [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),Path('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc')]
    for path in choices:
        if path and Path(path).is_file():return Path(path)
    raise FileNotFoundError('Provide --font with an installed Chinese font; do not silently render missing glyphs')


def stage_of(row):
    events=row.get('events',[])
    candidates=[e.get('judge',e) for e in events]
    actionable=[r for r in candidates if r.get('action') is not None]
    return (actionable or candidates)[-1]['stage_id'] if candidates else 's6'


def draw_frame(rgbs,stage,t,total,updates,origin,font_path,result=None):
    canvas=Image.new('RGB',(896,736),(17,23,34));d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(str(font_path),24);small=ImageFont.truetype(str(font_path),19);title=ImageFont.truetype(str(font_path),29)
    d.text((18,10),'原始 LIBERO：抓书 → 收纳盒后格',font=title,fill='white')
    n=int(stage[1:]);backend='冻结小策略' if n in (3,6) else '代码控制' if n<7 else '冻结神经策略 + 在线 z'
    d.text((18,51),f'阶段 {n}/7：{STAGES[n-1]}  |  {backend}',font=font,fill=(130,220,255))
    d.text((18,88),f'观察帧 {t+1}/{total}   |   当前阶段 z 更新：{updates if n==7 else "未启用"}',font=small,fill=(218,223,234))
    for i,name in enumerate(('外部相机 agentview','腕部相机')):
        canvas.paste(Image.fromarray(np.asarray(rgbs[i],dtype=np.uint8)),(i*448,128))
        d.rectangle((i*448,128,i*448+230,157),fill=(20,25,32));d.text((i*448+9,132),name,font=small,fill='white')
    for i,name in enumerate(STAGES):
        color=(50,164,110) if i<n-1 else (52,145,220) if i==n-1 else (61,68,80)
        x=i*128;d.rectangle((x+5,588,x+123,626),fill=color);d.text((x+16,593),f'{i+1} {name}',font=small,fill='white')
    d.text((18,637),origin,font=small,fill=(212,215,220))
    if result is None:
        d.text((18,668),'仅放入阶段在线更新 z；网络权重始终冻结',font=small,fill=(218,223,234))
        d.text((18,698),'开发工程回合；三组均成功，不据此声称 Adapter 带来增益',font=small,fill=(240,201,126))
    else:
        d.text((18,668),f'封存末帧：官方成功={result["official_success"]}  图完成={result["graph_complete"]}',font=small,fill=(111,232,155))
        d.text((18,698),'结果摘要定格；三组均成功，尚无 Adapter 额外收益证据',font=small,fill=(240,201,126))
    return np.asarray(canvas)


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--font',type=Path);p.add_argument('--fps',type=int,default=20);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=a.episode;result=json.loads((root/'RESULT.json').read_text());binding=json.loads((root/'BINDING.json').read_text())
    if result['cell']['arm']!='online':raise ValueError('Only requested online full-task engineering video')
    seal=json.loads((root/'SEAL.json').read_text())
    if sha(root/'SEAL.json')!=result['seal_sha256']:raise ValueError('Result/seal mismatch')
    if sha(root/'BINDING.json')!=seal['binding_sha256']:raise ValueError('Binding changed')
    for relative in ('prefix/TRACE.jsonl','public/ACTIONS.jsonl'):
        if sha(root/relative)!=seal['files'][relative]:raise ValueError('Annotation trace differs from seal')
    prefix=[json.loads(s) for s in (root/'prefix/TRACE.jsonl').read_text().splitlines()]
    inserts=[json.loads(s) for s in (root/'public/ACTIONS.jsonl').read_text().splitlines()]
    # Last prefix observation and first Insert observation are the same handoff.
    if prefix[-1].get('ready'):prefix=prefix[:-1]
    total=len(prefix)+len(inserts)
    if total>450:raise ValueError('Requested bounded video has more than450 source frames')
    for folder,rows,key in [('prefix',prefix,'file'),('public',inserts,'public_observation')]:
        for row in rows:
            path=root/folder/row[key];relative=str(path.relative_to(root))
            if sha(path)!=seal['files'][relative]:raise ValueError('Source frame differs from seal')
    font=choose_font(a.font);ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    model=mujoco.MjModel.from_xml_string(binding['environment']['relocated_xml']);data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,height=448,width=448);option=mujoco.MjvOption();option.geomgroup[0]=0;option.geomgroup[1]=1
    command=[ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','896x736','-r',str(a.fps),'-i','-',
             '-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE);written=0;last=None
    try:
        for row in prefix:
            with np.load(root/'prefix'/row['file'],allow_pickle=False) as state:data.qpos[:]=state['supervisor_qpos']
            mujoco.mj_fwdPosition(model,data)
            rgb=[]
            for camera in binding['public']['cameras']:
                renderer.update_scene(data,camera=camera,scene_option=option);rgb.append(renderer.render().copy())
            last=draw_frame(rgb,stage_of(row),written,total,0,'前六阶段：封存状态重建可视化（无积分、无新动作）',font)
            proc.stdin.write(last.tobytes());written+=1
        for i,row in enumerate(inserts):
            with np.load(root/'public'/row['public_observation'],allow_pickle=False) as public:rgb=public['rgb'].copy()
            final=result if i==len(inserts)-1 else None
            last=draw_frame(rgb,'s7',written,total,row['updates'],'放入阶段：原始记录 RGB；动作与 z 数值来自封存日志',font,final)
            proc.stdin.write(last.tobytes());written+=1
        for _ in range(min(a.fps*2,450-written)):proc.stdin.write(last.tobytes());written+=1
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('Video encoder failed')
    finally:
        renderer.close()
        if proc.poll() is None:proc.kill();proc.wait()
    report=dict(scope='recorded-state visualization only; no new rollout/integration',source_episode=str(root),result_sha256=sha(root/'RESULT.json'),
                source_seal_sha256=sha(root/'SEAL.json'),video_sha256=sha(a.output),source_frames=total,video_frames=written,fps=a.fps,
                prefix='mj_fwdPosition from sealed qpos, native two cameras',insert='original recorded public RGB',
                explicit_limit='Three complete-task engineering arms all succeeded; no incremental adapter benefit claim')
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
