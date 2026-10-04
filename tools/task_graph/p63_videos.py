#!/usr/bin/env python3
"""Three-column privileged diagnosis from prespecified sealed saved frames only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.task_graph.p62_videos import sha,dump_new,fit,verdict,PHASES,FONT

PAIRS=(('id',611100),('geometry',611200))
ARMS=('translation_size5','orientation_oracle','full_geometry_oracle')
WIDTH,HEIGHT,FPS=1920,1080,20
WARNING='真值诊断，不是学习方法成绩'
NAMES={'translation_size5':'P62 位置＋尺寸更新（封存参照）',
       'orientation_oracle':'只替换真实朝向', 'full_geometry_oracle':'替换完整真实槽几何'}


def binding(path):
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def load_cell(freeze_path,index,cell):
    directory=Path(freeze_path).parent/'cells'/f"{index:02d}_{cell['split']}_{cell['condition_seed']}_{cell['encoder']}_{cell['arm']}"
    if cell['video'] is not True or cell['encoder']!='task_trained_lila':raise ValueError('prespecified video/model identity changed')
    if (directory/'ABORT.json').exists():raise ValueError('preselected cell aborted; cannot substitute another video')
    result=json.loads((directory/'RESULT.json').read_text());seal=json.loads((directory/'SEAL.json').read_text())
    if (result['cell']!=cell or result['cell_index']!=index or result['steps']!=240
        or result['seal_sha256']!=sha(directory/'SEAL.json') or seal['freeze']!=binding(freeze_path)
        or seal['neural_parameters_before']!=seal['neural_parameters_after']):raise ValueError('result/freeze/model seal mismatch')
    needed=('FRAMES.npz','DETAIL_FRAMES.npz','TRACE.npy')
    for name in needed:
        if seal['files'].get(name)!=sha(directory/name):raise ValueError('saved presentation source changed: '+name)
    with np.load(directory/'FRAMES.npz',allow_pickle=False) as data:frames=data['frames'].copy()
    with np.load(directory/'DETAIL_FRAMES.npz',allow_pickle=False) as data:details=data['frames'].copy()
    trace=np.load(directory/'TRACE.npy',allow_pickle=True).tolist()
    for array in (frames,details):
        if array.ndim!=4 or array.shape[0]!=240 or array.shape[-1]!=3 or array.dtype!=np.uint8:
            raise ValueError('exactly240 saved RGB frames required')
    if len(trace)!=240 or [r['t'] for r in trace]!=list(range(240)):raise ValueError('trace time identity changed')
    updates=np.array([int(r['adaptation']['U']) for r in trace])
    if updates.sum()!=result['gradient_steps']:raise ValueError('update counts changed')
    return dict(cell=cell,result=result,frames=frames,details=details,trace=trace,cumulative_updates=updates.cumsum(),
        provenance=dict(directory=str(directory.resolve()),freeze=binding(freeze_path),result=binding(directory/'RESULT.json'),
            seal=binding(directory/'SEAL.json'),files={name:binding(directory/name) for name in needed}))


def load_pairs(freeze_path,summary_path):
    # The canonical reader refuses to open any outcome until all16 attempts end,
    # verifies trajectory seals, and replays actual P50 commands without physics.
    from tools.task_graph.p63_oracle import summarize
    summary=summarize(freeze_path)
    supplied=json.loads(Path(summary_path).read_text())
    if summary!=supplied:raise ValueError('provided summary differs from current sealed/audited results')
    frozen=json.loads(Path(freeze_path).read_text())
    baseline=frozen['files']['p62_physical_freeze'];acceptance=frozen['files']['p62_acceptance']
    for item in (baseline,acceptance):
        if binding(item['path'])!=item:raise ValueError('P62 baseline source binding changed')
    old=json.loads(Path(baseline['path']).read_text())
    accepted=json.loads(Path(acceptance['path']).read_text())
    if accepted['freeze']!=baseline or accepted['all_cells_engineering_valid'] is not True:
        raise ValueError('P62 accepted baseline required')
    pairs=[]
    for split,seed in PAIRS:
        records={}
        for path,data,arms in ((baseline['path'],old,('translation_size5',)),(freeze_path,frozen,ARMS[1:])):
            for index,cell in enumerate(data['cells']):
                if (cell['split'],cell['condition_seed'])!=(split,seed) or cell['arm'] not in arms:continue
                if cell['arm'] in records:raise ValueError('duplicate comparison arm')
                records[cell['arm']]=load_cell(path,index,cell)
        if set(records)!=set(ARMS):raise ValueError('all three preselected comparison arms required')
        pairs.append([records[arm] for arm in ARMS])
    return pairs


def render_frame(records,t,*,result_card=False,font_path=FONT):
    canvas=Image.new('RGB',(WIDTH,HEIGHT),'#101823');draw=ImageDraw.Draw(canvas)
    title=ImageFont.truetype(font_path,30);text=ImageFont.truetype(font_path,24);small=ImageFont.truetype(font_path,21)
    split='ID' if records[0]['cell']['split']=='id' else '几何 OOD'
    seed=records[0]['cell']['condition_seed']
    draw.text((22,12),f'P63 · {split} · 条件 {seed} · {WARNING}',font=title,fill='#ffd396')
    draw.text((22,58),'相同冻结视觉与 P50 动作模型；公共物长校准不变；右侧两臂仅在控制器输入处使用真值',font=text,fill='#bccddd')
    for column,record in enumerate(records):
        x=column*640;row=record['trace'][t];adapt=row['adaptation']
        draw.text((x+14,106),NAMES[record['cell']['arm']],font=text,fill='#a8daff')
        caption='结尾结果卡：沿用最后一张动作前画面' if result_card else f'封存动作前画面 t={t:03d}/239'
        draw.text((x+14,144),caption,font=small,fill='#bdccdb')
        fit(canvas,record['frames'][t],(x+10,179,620,310))
        fit(canvas,record['details'][t],(x+10,502,620,310))
        draw.text((x+15,508),'近景',font=small,fill='white')
        phase=PHASES.get(row['stage'],'几何不可用时保持').split(' / ')[0]
        lines=[f'阶段：{phase}',f'本帧更新 {adapt["U"]} 步 · 累计 {int(record["cumulative_updates"][t])} 步',
               f'公开轮廓历史 {adapt["S"]} 条 · 公共朝向 z = 0']
        for n,line in enumerate(lines):draw.text((x+14,824+n*32),line,font=small,fill='white')
        if result_card:
            outcome,color=verdict(record['result'])
            draw.text((x+14,925),'封存终态：'+outcome.split(' / ')[0],font=text,fill=color)
        else:draw.text((x+14,925),'执行中；终态评价仅在结尾显示',font=small,fill='#bdccdb')
        if column:draw.line((x,101,x,970),fill='#34465a',width=2)
    draw.text((22,979),'预定每分布首条件；同八个已开放开发条件；完整16次诊断尝试见报告，不按结果选例',font=small,fill='#bdccdb')
    draw.text((22,1013),'仅合成原封存画面，无模拟或新动作；20fps 是播放速度，不是物理时钟',font=small,fill='#bdccdb')
    draw.text((22,1045),'结尾是完整轨迹之后的评价，不代表所展示的动作前画面已成功。'+WARNING,font=small,fill='#ffd396')
    return canvas


def render_pair(records,target,font_path=FONT):
    command=['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}',
        '-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','21','-threads','1',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for i in range(260):process.stdin.write(render_frame(records,min(i,239),result_card=i>=240,font_path=font_path).tobytes())
        process.stdin.close();error=process.stderr.read().decode()
        if process.wait()!=0:raise RuntimeError(error)
    except BaseException:
        process.kill();process.wait();raise
    return dict(file=binding(target),fps=FPS,frames=260,recorded_pre_action_frames=240,final_card_repeated_frames=20,
        terminal_frame_recorded=False,playback_timing_is_not_physics_timing=True,
        conditions=[r['cell'] for r in records],sources=[r['provenance'] for r in records],
        final_strict_results={r['cell']['arm']:bool(r['result']['validated_success']) for r in records})


def present(freeze_path,summary_path,output,font_path=FONT):
    pairs=load_pairs(freeze_path,summary_path)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    videos=[]
    for pair in pairs:
        cell=pair[0]['cell'];target=output/f"{cell['split']}_{cell['condition_seed']}_ts5_orientation_full_oracle_zh.mp4"
        videos.append(render_pair(pair,target,font_path))
    root=Path(__file__).resolve().parents[2]
    manifest=dict(schema='prove.p63-prespecified-diagnostic-videos.v1',freeze=binding(freeze_path),summary=binding(summary_path),
        renderer_sources={name:binding(root/name) for name in ('tools/task_graph/p63_videos.py','tools/task_graph/p62_videos.py')},
        audit_sources='P63 frozen source closure binds unchanged oracle summary and P50 audit helpers',
        font=binding(font_path),selection='exact ID611100 and geometry611200, TS5 baseline plus two privileged arms',
        diagnostic_only=True,learned_method_score=False,warning=WARNING,physics_integration_steps=0,
        new_environment_rendering_steps=0,independent_test_claim=False,videos=videos)
    dump_new(output/'MANIFEST.json',manifest);return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('freeze','summary','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--font',default=FONT);a=parser.parse_args()
    manifest=present(a.freeze,a.summary,a.output,a.font)
    print(json.dumps({'videos':[v['file']['path'] for v in manifest['videos']]},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
