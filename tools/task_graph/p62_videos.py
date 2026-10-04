#!/usr/bin/env python3
"""Render only P62's two prespecified pairs from sealed saved video arrays.

No simulator imports, replay, new observations, or outcome-based selection.
Requires both complete fixed/TS5 pairs before creating any output artifact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ARMS=('fixed','translation_size5')
PAIRS=(('id',611100),('geometry',611200))
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
WIDTH,HEIGHT,FPS=1920,1080,20
PHASES={'shared observation motion':'共同观察 / Shared observation',
        'neural align':'对齐 / Align','neural approach':'接近 / Approach',
        'neural insert':'插入 / Insert','neural hold':'保持 / Hold',
        'valid estimated geometry unavailable: hold measured pose':'几何不可用时保持 / Hold: unavailable geometry'}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()

def dump_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)

def verdict(result):
    if not result['mechanical_valid']:return '机械异常 / Mechanical anomaly','#ffb0a3'
    if result['validated_success']:return '严格成功 / Strict success','#92e6b8'
    return '严格失败 / Strict failure','#ffd396'

def load_pairs(freeze_path):
    freeze_path=Path(freeze_path);frozen=json.loads(freeze_path.read_text());freeze_sha=sha(freeze_path)
    if frozen.get('schema')!='prove.p62-physical-development-freeze.v1' or len(frozen.get('cells',[]))!=16:
        raise ValueError('expected bounded sixteen-cell P62 freeze')
    pairs=[]
    for split,seed in PAIRS:
        records={}
        for index,cell in enumerate(frozen['cells']):
            if (cell['split'],cell['condition_seed'])!=(split,seed):continue
            if cell['arm'] not in ARMS or cell.get('encoder')!='task_trained_lila' or cell.get('video') is not True:
                raise ValueError('prespecified video/model/arm identity changed')
            directory=freeze_path.parent/'cells'/f"{index:02d}_{split}_{seed}_{cell['encoder']}_{cell['arm']}"
            if (directory/'ABORT.json').exists() or not (directory/'RESULT.json').is_file():
                raise ValueError(f'complete pair required; no rendering for incomplete/aborted {directory.name}')
            result=json.loads((directory/'RESULT.json').read_text());seal=json.loads((directory/'SEAL.json').read_text())
            if (result.get('schema')!='prove.p62-physical-result.v1' or result.get('cell')!=cell or result.get('cell_index')!=index
                or result.get('steps')!=240 or result.get('seal_sha256')!=sha(directory/'SEAL.json')
                or seal.get('freeze',{}).get('sha256')!=freeze_sha
                or seal.get('neural_parameters_before')!=seal.get('neural_parameters_after')):
                raise ValueError('result/freeze/model seal mismatch')
            files={}
            for name in ('FRAMES.npz','DETAIL_FRAMES.npz','TRACE.npy','OBS.npy','PRE_RUN.json','TERMINAL.npy'):
                files[name]=sha(directory/name)
                if seal['files'].get(name)!=files[name]:raise ValueError(f'sealed presentation source changed: {name}')
            with np.load(directory/'FRAMES.npz',allow_pickle=False) as a:frames=a['frames'].copy()
            with np.load(directory/'DETAIL_FRAMES.npz',allow_pickle=False) as a:details=a['frames'].copy()
            trace=np.load(directory/'TRACE.npy',allow_pickle=True).tolist()
            if len(frames)!=240 or len(details)!=240 or len(trace)!=240 or [r['t'] for r in trace]!=list(range(240)):
                raise ValueError('expected exactly240 saved pre-action frames and trace rows')
            for array in (frames,details):
                if array.ndim!=4 or array.shape[-1]!=3 or array.dtype!=np.uint8:raise ValueError('saved frames must be uint8 RGB')
            updates=np.array([int(r['adaptation']['U']) for r in trace])
            if int(updates.sum())!=int(result['gradient_steps']):raise ValueError('trace/result update count mismatch')
            if cell['arm']=='fixed' and np.any(updates):raise ValueError('fixed arm updated latent')
            if any(not np.array_equal(r['adaptation']['z_after'][3:6],np.zeros(3)) for r in trace):raise ValueError('rotation latent changed')
            if cell['arm'] in records:raise ValueError('duplicate presentation arm')
            records[cell['arm']]=dict(cell=cell,result=result,frames=frames,details=details,trace=trace,
                cumulative_updates=np.cumsum(updates),directory=str(directory),
                provenance=dict(result_sha256=sha(directory/'RESULT.json'),seal_sha256=sha(directory/'SEAL.json'),files=files))
        if set(records)!=set(ARMS):raise ValueError('both prespecified arms are required')
        pairs.append([records[arm] for arm in ARMS])
    return pairs

def fit(canvas,array,box):
    image=Image.fromarray(array);image.thumbnail((box[2],box[3]),Image.Resampling.LANCZOS)
    canvas.paste(image,(box[0]+(box[2]-image.width)//2,box[1]+(box[3]-image.height)//2))

def render_frame(records,t,*,result_card=False,font_path=FONT):
    """Pure layout function; input images are saved pre-action frames only."""
    canvas=Image.new('RGB',(WIDTH,HEIGHT),'#101823');draw=ImageDraw.Draw(canvas)
    title=ImageFont.truetype(font_path,27);text=ImageFont.truetype(font_path,21);small=ImageFont.truetype(font_path,18)
    split='ID' if records[0]['cell']['split']=='id' else '几何 OOD / Geometry OOD'
    seed=records[0]['cell']['condition_seed']
    draw.text((22,12),f'P62 · {split} · 条件 / Condition {seed}',font=title,fill='white')
    draw.text((22,53),'有界开发验证 / Bounded development · 同一冻结视觉与 P50 动作模型 / Same frozen models',font=text,fill='#bccddd')
    for column,record in enumerate(records):
        x=column*(WIDTH//2);row=record['trace'][t];adapt=row['adaptation']
        name='固定 z / Fixed' if record['cell']['arm']=='fixed' else '位置＋尺寸更新 / Position + size adaptation'
        draw.text((x+18,96),name,font=title,fill='#a8daff')
        caption='结果卡：最后保存的动作前画面 / Result card: last saved pre-action view' if result_card else f'保存画面 t={t:03d}/239；240 次动作预算 / Saved pre-action frame'
        draw.text((x+18,137),caption,font=small,fill='#bdccdb')
        fit(canvas,record['frames'][t],(x+16,166,928,464))
        fit(canvas,record['details'][t],(x+16,646,352,290))
        draw.text((x+24,938),'近景 / Detail',font=small,fill='#bdccdb')
        y=658;tx=x+388
        phase=PHASES.get(row['stage'],'几何不可用时保持 / Hold: unavailable geometry')
        lines=[phase,f'本帧 GN 步 / GN now: {adapt["U"]}',
            f'累计 GN 步 / GN total: {int(record["cumulative_updates"][t])}',
            f'公开轮廓历史 / Public views: {adapt["S"]}',
            '朝向 z 恒为 0 / Rotation z = 0',
            '网络权重冻结 / All network weights frozen']
        for line in lines:draw.text((tx,y),line,font=text,fill='white');y+=34
        if result_card:
            outcome,color=verdict(record['result'])
            draw.text((tx,y+7),'封存终态评价 / Sealed final result:',font=small,fill='#bdccdb')
            draw.text((tx,y+37),outcome,font=text,fill=color)
        else:
            draw.text((tx,y+7),'执行中；终态见结尾 / Running; verdict at end',font=text,fill='#bdccdb')
    draw.line((WIDTH//2,92,WIDTH//2,958),fill='#34465a',width=2)
    draw.text((22,982),'预先指定每分布首条件；完整结果见报告 / Prespecified first condition per split; see report for all16 attempts.',font=text,fill='#bdccdb')
    draw.text((22,1015),'新开发条件，非独立确认集 / Development, not held-out confirmation. 20fps 仅播放速度 / playback only.',font=text,fill='#bdccdb')
    draw.text((22,1046),'仅合成封存画面，无重放或新物理；终态结果不表示当前画面已成功 / Sealed frames only; final verdict is post-trajectory.',font=small,fill='#bdccdb')
    return canvas

def render_pair(records,output,font_path=FONT):
    command=['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}',
        '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-threads','1',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for i in range(260):
            frame=render_frame(records,min(i,239),result_card=i>=240,font_path=font_path)
            process.stdin.write(frame.tobytes())
        process.stdin.close();error=process.stderr.read().decode()
        if process.wait()!=0:raise RuntimeError(error)
    except BaseException:
        process.kill();process.wait();raise
    return dict(path=str(output),sha256=sha(output),fps=FPS,frames=260,recorded_pre_action_frames=240,
        final_card_repeated_frames=20,terminal_frame_recorded=False,playback_timing_is_not_physics_timing=True,
        conditions=[r['cell'] for r in records],sources=[r['provenance'] for r in records],
        final_strict_results={r['cell']['arm']:bool(r['result']['validated_success']) for r in records})

def present(freeze_path,output,font_path=FONT):
    pairs=load_pairs(freeze_path)  # Validate both complete pairs before creating output.
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    videos=[]
    for pair in pairs:
        cell=pair[0]['cell'];target=output/f"{cell['split']}_{cell['condition_seed']}_fixed_vs_ts5_zh_en.mp4"
        videos.append(render_pair(pair,target,font_path))
    manifest=dict(schema='prove.p62-prespecified-pair-videos.v1',freeze_sha256=sha(freeze_path),renderer_sha256=sha(__file__),
        font=dict(path=font_path,sha256=sha(font_path)),selection='exact preselected ID611100 and geometry611200; never selected by outcome',
        physics_integration_steps=0,new_rendering_steps=0,independent_test_claim=False,videos=videos)
    dump_new(output/'MANIFEST.json',manifest);return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--font',default=FONT)
    args=parser.parse_args();manifest=present(args.freeze,args.output,args.font)
    print(json.dumps({'videos':[v['path'] for v in manifest['videos']]},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
