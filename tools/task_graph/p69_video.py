"""P69 sealed full-episode state rendering. No policy inference or physics step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def metadata(root):
    """No MuJoCo import or state-array loading; validates exact recorded timeline."""
    root=Path(root);result=read(root/'RESULT.json');seal=read(root/'SEAL.json');binding=read(root/'BINDING.json');cell=read(root/'CELL.json')
    if result.get('schema')!='prove.p69-full-rsi-result.v1' or not result.get('full_task_success_evidence'):
        raise ValueError('P69 complete task result required, including failed tasks')
    if result['cell']!=cell or cell not in binding['cells'] or seal['cell']!=cell:raise ValueError('Cell identity differs')
    if sha(root/'SEAL.json')!=result['seal_sha256']:raise ValueError('Result seal reference differs')
    bound={name:sha(root/name) for name in ('RESULT.json','SEAL.json')}
    def checked(name):
        path=root/name
        if name not in seal['files'] or sha(path)!=seal['files'][name]:raise ValueError('Unsealed/changed source: '+name)
        bound[name]=seal['files'][name];return path
    for name in ('CELL.json','BINDING.json','PREFIX_RESULT.json','CODE_RESULT.json','STAGE_TRANSITIONS.json'):checked(name)
    timeline=[]
    for folder,count in (('prefix',result['prefix_actions']),('code',result['code_actions'])):
        trace=root/folder/'TRACE.jsonl'
        if not trace.exists():
            if count:raise ValueError('Missing executed trace')
            continue
        records=lines(checked(f'{folder}/TRACE.jsonl'))
        if len(records)!=count+1:raise ValueError('Expected recorded action states plus terminal observation')
        for i,row in enumerate(records):
            if (row.get('action') is not None)!=(i<count):raise ValueError('Trace execution boundary differs')
            path=f"{folder}/{row['file']}";checked(path)
            if row.get('sha256') and row['sha256']!=bound[path]:raise ValueError('Trace frame digest differs')
            events=row.get('events') or []
            candidates=[e.get('judge',e) for e in events]
            actionable=[r for r in candidates if r.get('action') is not None]
            stage=row.get('stage_id') or ((actionable or candidates)[-1].get('stage_id') if candidates else row.get('phase','unknown'))
            actual_controller=next((event.get('controller',{}).get('controller') for event in reversed(events) if event.get('controller',{}).get('controller')),None)
            if 'settle' in str(row.get('phase','')):actual_controller='公开观测沉降：静止末端、打开夹爪'
            timeline.append(dict(state=path,stage=stage,origin=folder,local_t=i,executed=i<count,
                                 controller=actual_controller or row.get('controller',row.get('route'))))
        # If another stage exists, its first state owns this shared handoff time.
        if folder=='prefix' and ((root/'code/TRACE.jsonl').exists() or (root/'public/EXECUTED.jsonl').exists()):timeline.pop()
        if folder=='code' and (root/'public/EXECUTED.jsonl').exists():timeline.pop()
    if (root/'public/EXECUTED.jsonl').exists():
        execution=lines(checked('public/EXECUTED.jsonl'));predictions=lines(checked('public/PREDICTIONS.jsonl'))
        if len(execution)!=result['actions']+1 or len(predictions)!=len(execution):raise ValueError('Micro trace count differs')
        for i,(row,pred) in enumerate(zip(execution,predictions)):
            if row['t']!=i or pred['t']!=i or bool(row['executed'])!=(i<result['actions']):raise ValueError('Micro order differs')
            state=f'private/frame_{i:04d}.npz';checked(state)
            timeline.append(dict(state=state,stage='s9',origin='private',local_t=i,executed=bool(row['executed']),controller='冻结小动作模型 + 阶段局部z',event=pred.get('telemetry',{}).get('event')))
    if len(timeline)!=result['total_actions']+1 or sum(r['executed'] for r in timeline)!=result['total_actions']:
        raise ValueError('Full timeline must contain every executed action and one terminal state')
    if not 0<=result['total_actions']<=544:raise ValueError('P69 budget exceeded')
    return dict(root=str(root),result=result,binding=binding,cell=cell,timeline=timeline,source_hashes=bound)


def scene_binding(binding):
    pointer=binding['p55_binding'];receipts=[]
    for _ in range(7):
        if sha(pointer['path'])!=pointer['sha256']:raise ValueError('Scene parent hash differs')
        receipts.append(pointer);value=read(pointer['path'])
        if value.get('schema')=='prove.p53-full-original-task.v1':return value,receipts
        pointer=value['parent']
    raise ValueError('Scene parent chain too deep')


def render(episodes,output,fps,font_path,asset_map=None):
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from tools.task_graph.p52_render_full_online import choose_font
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('PYOPENGL_PLATFORM','egl')
    import mujoco
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype(str(choose_font(font_path)),21);backends=[]
    mapping=read(asset_map) if asset_map else {}
    for episode in episodes:
        scene,chain=scene_binding(episode['binding']);episode['scene_chain']=chain
        for asset,digest in scene['environment'].get('assets',{}).items():
            relocated=asset
            for old,new in sorted(mapping.items(),key=lambda p:-len(p[0])):relocated=relocated.replace(old,new)
            if sha(relocated)!=digest:raise ValueError('Scene asset bytes changed: '+relocated)
        xml=scene['environment']['relocated_xml']
        for old,new in sorted(mapping.items(),key=lambda p:-len(p[0])):xml=xml.replace(old,new)
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
        renderer=mujoco.Renderer(model,height=448,width=448);option=mujoco.MjvOption();option.geomgroup[0]=0;option.geomgroup[1]=1
        backends.append((model,data,renderer,option,scene['public']['cameras']))
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:raise FileNotFoundError('ffmpeg')
    width,height=896*len(episodes),600
    command=[ffmpeg,'-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','22',str(output/'full_task.mp4')]
    process=subprocess.Popen(command,stdin=subprocess.PIPE);annotations=[]
    try:
        for t in range(max(len(e['timeline']) for e in episodes)):
            canvas=Image.new('RGB',(width,height),(17,23,34));draw=ImageDraw.Draw(canvas);frame_sources=[]
            for column,(episode,backend) in enumerate(zip(episodes,backends)):
                model,data,renderer,option,cameras=backend;index=min(t,len(episode['timeline'])-1);row=episode['timeline'][index];x=column*896
                with np.load(Path(episode['root'])/row['state'],allow_pickle=False) as arrays:
                    qpos=arrays['supervisor_qpos'];data.qpos[:]=qpos
                mujoco.mj_fwdPosition(model,data)  # Position-only rendering; never integrates actions.
                for view,camera in enumerate(cameras):
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render().copy()),(x+view*448,108))
                cell=episode['cell'];controller=row['controller'] or ('Code' if row['origin']=='code' else '前缀Code／精细模型（由阶段日志定位）')
                draw.text((x+12,8),f"P69 {cell['version']} | {cell['split']} 初态 {cell['init_id']} | 阶段 {row['stage']}",font=font,fill='white')
                controller={'r2_retargeted_code_eef_feedback':'Code：公开参考、本体反馈','r2_public_retargeted_frozen_neural_action':'冻结精细动作模型：公开参考重定位'}.get(controller,controller)
                draw.text((x+12,40),f"控制器：{controller}",font=font,fill='white')
                draw.text((x+12,72),f"执行步 {index}/{episode['result']['total_actions']}"+(' | 结束后定格' if t>index else ''),font=font,fill='white')
                terminal=index==len(episode['timeline'])-1
                label='实际终态：'+('成功' if episode['result']['graph_complete'] and episode['result']['official_success'] else '失败') if terminal else '封存真实机械臂状态重绘；只forward/render，无新动作'
                draw.text((x+12,568),label,font=font,fill=(150,225,180))
                frame_sources.append(dict(episode=episode['root'],source=row,elapsed_step=index,terminal=terminal,held=t>index))
            process.stdin.write(np.asarray(canvas).tobytes());annotations.append(dict(frame=t,columns=frame_sources))
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Video encoding failed')
    finally:
        for _,_,renderer,_,_ in backends:renderer.close()
        if process.poll() is None:process.kill();process.wait()
    (output/'ANNOTATIONS.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(schema='prove.p69-sealed-video.v1',sources=episodes,fps=fps,frames=len(annotations),physics_steps=0,policy_calls=0,
                  alignment='Original elapsed action index; shorter episode terminal freezes. No reordered/reconstructed actions.',
                  asset_map=None if asset_map is None else dict(path=str(asset_map),sha256=sha(asset_map)),files={n:sha(output/n) for n in ('full_task.mp4','ANNOTATIONS.json')})
    (output/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path)
    p.add_argument('--check-metadata-only',action='store_true');p.add_argument('--font',type=Path);p.add_argument('--asset-map',type=Path);p.add_argument('--fps',type=int,default=20)
    a=p.parse_args()
    if not 1<=len(a.episode)<=2 or a.fps<=0:p.error('One or two episodes and positive FPS required')
    episodes=[metadata(path) for path in a.episode]
    if len(episodes)==2 and any(episodes[0]['cell'].get(k)!=episodes[1]['cell'].get(k) for k in ('split','init_id','state_sha256')):
        raise ValueError('Side-by-side requires matched split and initial state')
    if a.check_metadata_only:
        print(json.dumps([dict(cell=e['cell'],source_states=len(e['timeline']),total_actions=e['result']['total_actions'],metadata_ok=True) for e in episodes]));return
    if a.output is None:p.error('--output required for rendering')
    render(episodes,a.output,a.fps,a.font,a.asset_map)


if __name__=='__main__':main()
