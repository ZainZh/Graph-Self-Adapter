"""Independent first-OOD displays from sealed states, never physics rollouts."""
import json,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
import mujoco
from prove.fast_adapter.geometry3d_env import GeometrySkill,quat
from tools.fast_adapter.p43_visual_media import font,sha

ROOT=Path('evaluations/fast_adapter_p43_neural_geometry_v01')
OUT=ROOT/'summary/final_media'
ARMS=('fixed','continuous')
LABELS={'fixed':'同一模型：固定 z','continuous':'同一模型：持续更新 z'}
TASKS={'cup':('杯柄抓持与抬升',433100),'insert':('倾斜三维插入',434100)}


def main():
    OUT.mkdir(parents=True,exist_ok=False);large,medium,small,tiny=font(25),font(23),font(18),font(16);entries=[]
    for task,(title,seed) in TASKS.items():
        rows=[];environments=[];suppressed=[0];original_step=mujoco.mj_step
        def suppress_settling(*args,**kwargs):suppressed[0]+=1
        try:
            for arm in ARMS:
                cell=ROOT/'independent'/f'{task}_seed11'/f'geometry_{seed}_{arm}';pre=json.loads((cell/'PRE_RUN.json').read_text());seal=json.loads((cell/'SEAL.json').read_text())
                assert pre['scope']=='independent_validation' and pre['bindings']['model_seed']==11
                for key,name in [('trace','TRACE.npy'),('terminal','TERMINAL.npy')]:assert sha(cell/name)==seal[key]
                assert sha('prove/fast_adapter/geometry3d_env.py')==pre['sources']['prove/fast_adapter/geometry3d_env.py']
                # The legacy constructor attempts 100 settle steps. Suppress
                # those calls so every displayed state comes only from the seal.
                with patch.object(mujoco,'mj_step',suppress_settling):env=GeometrySkill(pre['cfg'])
                environments.append(env)
                rows.append(dict(arm=arm,cell=cell,env=env,trace=np.load(cell/'TRACE.npy',allow_pickle=True),
                                 terminal=np.load(cell/'TERMINAL.npy',allow_pickle=True).item(),result=json.loads((cell/'RESULT.json').read_text())))
            def reject_step(*args,**kwargs):raise RuntimeError('Physics stepping is forbidden during media reconstruction')
            mujoco.mj_step=reject_step
            width,height=896,1330;fps=15;output=OUT/f'{task}_first_ood_seed11_state_reconstruction.mp4'
            encoder=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
                '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
            try:
                for frame in range(191):
                    canvas=Image.new('RGB',(width,height),'#111a26');draw=ImageDraw.Draw(canvas)
                    draw.text((18,8),f'P43 独立验证｜{title}｜预定首个范围外条件',font=large,fill='#eef6ff')
                    draw.text((18,44),f'条件 {seed} · 模型种子11 · 同一checkpoint · 同初态、同190步预算',font=small,fill='#becfe1')
                    draw.text((18,73),'封存状态重建视角｜非逐像素原始RGB｜未新增物理执行或神经推理',font=tiny,fill='#f2cf86')
                    for column,record in enumerate(rows):
                        x=column*448;env=record['env'];state=record['terminal'] if frame==190 else record['trace'][frame];row=record['trace'][min(frame,189)]
                        env.data.qpos[:]=state['qpos'];env.data.qvel[:]=state['qvel'];env.data.mocap_pos[0]=row['command_p'];env.data.mocap_quat[0]=quat(row['command_R'])
                        if task=='cup':env.data.ctrl[:]=.022 if row['closed'] else 0.
                        mujoco.mj_forward(env.model,env.data)
                        global_rgb=env.render_global();env.renderer.update_scene(env.data,camera='wrist');wrist=env.renderer.render().copy()
                        draw.rectangle((x+2,108,x+446,145),fill='#223248');draw.text((x+12,111),LABELS[record['arm']],font=medium,fill='#eef7ff')
                        canvas.paste(Image.fromarray(global_rgb),(x,151));draw.rectangle((x+4,155,x+120,182),fill='#172333');draw.text((x+10,157),'全景重建',font=small,fill='#ffffff')
                        draw.text((x+12,607),'工具相机重建 · 原视觉输入的同一机位',font=tiny,fill='#c5d9ed');canvas.paste(Image.fromarray(wrist),(x,637))
                        stage='执行190步后的封存终态' if frame==190 else f'步 {frame+1:03d}/190 · {row["stage"]}'
                        draw.text((x+12,1095),stage,font=small,fill='#f0f4f8');adapt=row.get('adaptation')
                        info='角点不可用，保留已有估计' if adapt is None else f'已纳入观察 {adapt["S"]} 帧｜本帧更新 {adapt["U"]} 次'
                        if task=='cup' and frame>=105:info='闭合后保持估计，按本体位姿抬升'
                        draw.text((x+12,1125),info,font=tiny,fill='#aebfd1')
                        if frame==190:
                            result=record['result'];passed=result['success'];color='#76e3be' if passed else '#ffb09f'
                            draw.text((x+12,1160),'末10状态判据：'+('通过' if passed else '未通过'),font=medium,fill=color)
                            detail=(f'最终抬升 {1000*result["lift"]:.1f}mm｜双侧接触：'+('是' if result['bilateral'] else '否')) if task=='cup' else f'深度 {1000*result["depth"]:.1f}mm｜横向 {1000*result["lateral"]:.1f}mm'
                            draw.text((x+12,1199),detail,font=tiny,fill='#dce7f2')
                            draw.text((x+12,1228),f'成功状态 {result["last10_successes"]}/10｜最大穿透 {1000*result["max_last10_penetration"]:.2f}mm',font=tiny,fill='#dce7f2')
                        else:
                            detail='判据：双侧抓持，抬升超过40mm' if task=='cup' else '判据：深度>34mm，每轴偏差<3mm'
                            draw.text((x+12,1168),detail,font=tiny,fill='#a9bed3')
                            if task=='insert':draw.text((x+12,1199),'角差<0.1rad；最后10状态持续满足',font=tiny,fill='#a9bed3')
                            else:draw.text((x+12,1199),'最后10状态持续满足，穿透<2mm',font=tiny,fill='#a9bed3')
                        draw.text((x+12,1265),'单个预定例子不代表整体成功率',font=tiny,fill='#c5d9ed')
                    encoder.stdin.write(np.asarray(canvas).tobytes())
                    if frame==190:
                        canvas.save(OUT/f'{task}_first_ood_cover.jpg',quality=94)
                        for _ in range(44):encoder.stdin.write(np.asarray(canvas).tobytes())
            finally:encoder.stdin.close()
            if encoder.wait()!=0:raise RuntimeError('Video encoding failed')
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames:format=duration','-of','json',str(output)],text=True))
            assert int(probe['streams'][0]['nb_frames'])==235
            entries.append(dict(task=task,seed=seed,model_seed=11,scope='independent_validation',selection='Predetermined first OOD condition, not chosen by result',
                video=str(output),video_sha256=sha(output),cover=str(OUT/f'{task}_first_ood_cover.jpg'),probe=probe,
                constructor_settle_calls_suppressed=suppressed[0],actual_physics_steps=0,network_calls=0,rendered_states=191,
                view='Saved qpos/qvel and commands -> mj_forward -> render; not original pixel RGB',
                sources=[dict(arm=r['arm'],cell=str(r['cell']),hashes={f:sha(r['cell']/f) for f in ('TRACE.npy','TERMINAL.npy','SEAL.json','PRE_RUN.json','RESULT.json')}) for r in rows]))
        finally:
            mujoco.mj_step=original_step
            for env in environments:env.close()
    manifest=dict(generator=__file__,generator_sha256=sha(__file__),videos=entries,no_new_trials=True)
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P43 独立首例重建</title><style>body{max-width:896px;margin:30px auto;background:#111a26;color:#eef6ff;font:18px/1.7 sans-serif}video{width:100%}a{color:#8ddbc7}</style><h1>P43 独立验证：预定首例</h1><p>根据封存状态重建全景与工具相机，不是逐像素原始RGB。只执行mj_forward与渲染，没有新增物理执行或神经推理。每任务模型种子11的预定第一个范围外条件，同一checkpoint固定z与持续z并列。</p>']
    for e in entries:html.append(f'<h2>{TASKS[e["task"]][0]} · {e["seed"]}</h2><video controls preload="metadata" src="{Path(e["video"]).name}" poster="{Path(e["cover"]).name}"></video>')
    html.append('<p><a href="MANIFEST.json">来源、渲染边界与哈希</a></p></html>');(OUT/'index.html').write_text('\n'.join(html));print(json.dumps(dict(completed=True,videos=2,actual_physics_steps=0)))

if __name__=='__main__':main()
