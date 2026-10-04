"""Chinese P43 first-OOD comparisons, derived only from sealed RGB/trace files."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('evaluations/fast_adapter_p43_neural_geometry_v01')
OUT=ROOT/'visual/media'
ARMS=('fixed','continuous','prefix')
LABELS={'fixed':'同一模型：固定 z','continuous':'同一模型：持续更新 z','prefix':'前12步更新，此后固定 z'}
TASKS={'cup':('杯柄抓持与抬升',430104),'insert':('倾斜三维插入',431104)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def js(path,record):
    Path(path).write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')


def font(size):
    path=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True).strip()
    return ImageFont.truetype(path,size)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    large,medium,small,tiny=font(28),font(23),font(19),font(17)
    entries=[]
    for task,(title,seed) in TASKS.items():
        records=[];bindings=[]
        for arm in ARMS:
            source=ROOT/'development'/f'{task}_seed11'/f'geometry_{seed}_{arm}'
            seal=json.loads((source/'SEAL.json').read_text())
            pre=json.loads((source/'PRE_RUN.json').read_text())
            result=json.loads((source/'RESULT.json').read_text())
            assert sha(source/'TRACE.npy')==seal['trace']
            assert pre['scope']=='development' and result['scope']=='development'
            assert result['seed']==seed and result['arm']==arm and result['split']=='geometry'
            assert pre['bindings']['model_seed']==11
            frames=np.load(source/'FRAMES.npz',allow_pickle=False)['frames']
            trace=np.load(source/'TRACE.npy',allow_pickle=True)
            assert frames.shape==(190,448,896,3) and len(trace)==190
            records.append(dict(arm=arm,frames=frames,trace=trace,result=result))
            bindings.append(dict(arm=arm,source=str(source),
                sha256={name:sha(source/name) for name in ('FRAMES.npz','TRACE.npy','RESULT.json','SEAL.json','PRE_RUN.json')},
                recorded_trace_seal_matched=True,model_bindings=pre['bindings']))
        W,H=1344,1260;fps=15;hold=45
        output=OUT/f'{task}_first_ood_seed11_comparison.mp4'
        process=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2',
             '-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
        try:
            for sequence_index,t in enumerate(list(range(190))+[189]*hold):
                final=t==189
                canvas=Image.new('RGB',(W,H),'#111a26');draw=ImageDraw.Draw(canvas)
                draw.text((18,10),f'P43 神经几何适配｜{title}｜预定首个范围外开发样本',font=large,fill='#f2f7fc')
                draw.text((18,52),f'条件 {seed} · 模型种子11 · 相同初态与190步预算 · 固定物理参数 · 此例不代表整体成功率',font=small,fill='#b5c8dc')
                for column,record in enumerate(records):
                    x=column*448;arm=record['arm'];src=record['frames'][t]
                    draw.rectangle((x+2,89,x+446,144),fill='#223248')
                    draw.text((x+12,94),LABELS[arm],font=medium,fill='#ecf5ff')
                    input_note=('闭合前读取当前可见观测；仅z权限不同' if task=='cup'
                                else '角点可用时读当前观测；仅z权限不同')
                    draw.text((x+12,124),input_note,font=tiny,fill='#b7c9dd')
                    canvas.paste(Image.fromarray(src[:,:448]),(x,150))
                    draw.rectangle((x+5,155,x+119,182),fill='#172333')
                    draw.text((x+10,157),'斜视全景',font=small,fill='#ffffff')
                    draw.text((x+12,605),'工具相机 · 运行视觉输入',font=small,fill='#dce9f7')
                    canvas.paste(Image.fromarray(src[:,448:]),(x,637))
                    row=record['trace'][t]
                    draw.text((x+12,1092),f'步 {t+1:03d}/190 · {row["stage"]}',font=small,fill='#f0f4f8')
                    adaptation=row.get('adaptation')
                    information='角点不可用，沿用已有估计' if adaptation is None else f'已纳入观察 {adaptation["S"]} 帧｜本帧梯度更新 {adaptation["U"]} 次'
                    if task=='cup' and t>=105:information='闭合后保持估计，使用本体位姿抬升'
                    draw.text((x+12,1122),information,font=tiny,fill='#a9bed3')
                    if final:
                        result=record['result'];passed=bool(result['success'])
                        label='末10状态判据：通过' if passed else '末10状态判据：未通过'
                        color='#76e3be' if passed else '#ffb09f'
                        draw.text((x+12,1151),label,font=medium,fill=color)
                        if task=='cup':
                            detail=f'最终抬升 {1000*result["lift"]:.1f}mm｜双侧接触：'+('是' if result['bilateral'] else '否')
                        else:
                            detail=f'深度 {1000*result["depth"]:.1f}mm｜横向 {1000*result["lateral"]:.1f}mm｜角差 {result["angle"]:.3f}rad'
                        draw.text((x+12,1184),detail,font=tiny,fill='#dce7f2')
                        draw.text((x+12,1210),f'成功状态 {result["last10_successes"]}/10｜最大穿透 {1000*result["max_last10_penetration"]:.2f}mm',font=tiny,fill='#dce7f2')
                    else:
                        criterion=('判据：双侧抓持，抬升超过40mm' if task=='cup' else '判据：深度>34mm，每轴偏差<3mm，角差<0.1rad')
                        draw.text((x+12,1160),criterion,font=tiny,fill='#a9bed3')
                        draw.text((x+12,1190),'最后10状态持续满足，且穿透小于2mm',font=tiny,fill='#a9bed3')
                        draw.text((x+12,1217),'结果在执行完成后显示；未新增机器人运行',font=tiny,fill='#a9bed3')
                process.stdin.write(np.asarray(canvas).tobytes())
                if sequence_index==189:canvas.save(OUT/f'{task}_first_ood_cover.jpg',quality=94)
        finally:
            process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Video encoding failed')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height,nb_frames:format=duration','-of','json',str(output)],text=True))
        stream=probe['streams'][0]
        assert stream['width']==W and stream['height']==H and int(stream['nb_frames'])==190+hold
        entry=dict(task=task,title=title,seed=seed,model_seed=11,split='geometry',scope='development',
            selection='Predetermined first geometry development seed; no outcome-based selection',
            arms=list(ARMS),video=str(output),video_sha256=sha(output),cover=str(OUT/f'{task}_first_ood_cover.jpg'),
            frames=190+hold,fps=fps,video_probe=probe,no_new_environment_or_neural_run=True,
            sources=bindings,results={r['arm']:r['result'] for r in records},
            image_timing='Source images are pre-action states t0..189; final outcome labels use sealed terminal and last10-state evaluator results.')
        entries.append(entry);print(json.dumps(dict(task=task,video=str(output),successes={r['arm']:r['result']['success'] for r in records}),ensure_ascii=False),flush=True)
    manifest=dict(scope='post-seal derived development media',generator=str(Path(__file__)),
        generator_sha256=sha(__file__),videos=entries)
    js(ROOT/'visual/P43_DEV_VIDEOS.json',manifest)
    html=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P43 开发首例对比</title><style>body{max-width:1344px;margin:28px auto;padding:0 18px;background:#111a26;color:#edf4fc;font:18px/1.7 sans-serif}video{width:100%}a{color:#8ddbc7}</style>',
          '<h1>P43：神经几何适配的开发首例</h1><p>预定第一个范围外条件，模型种子11；同一checkpoint的固定z、持续z更新、前12步后固定三臂。全景与工具相机同时展示，保留全部正负结果。视频是已有封存RGB的派生，不是新增运行，也不能代替整批结果。</p>']
    for entry in entries:
        html.append(f'<h2>{entry["title"]} · 条件{entry["seed"]}</h2><video controls preload="metadata" poster="{Path(entry["cover"]).name}" src="{Path(entry["video"]).name}"></video>')
    html.append('<p><a href="../P43_DEV_VIDEOS.json">完整来源、哈希与判据</a> · <a href="../DATA_REPORT_ZH.md">视觉数据与输入说明</a></p></html>')
    (OUT/'index.html').write_text('\n'.join(html))


if __name__=='__main__':main()
