#!/usr/bin/env python3
"""P67 Chinese four-column presentation from completed sealed report only."""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.fast_adapter.p64_cup_presentation import put
from tools.fast_adapter.p65_presentation import checked_ref, faststart
from tools.fast_adapter.p65_report import FONT, need, read, ref
from tools.fast_adapter.p67_report import ARMS, SPLITS, COMPARISONS, SEEDS
from prove.fast_adapter.p61_data import sha256

ARM_NAMES={'frame_fixed':'A 逐帧固定 z','reference_fixed':'B 固定参考、不更新',
           'entry_once':'C 固定参考、入口一次','continuous':'D 固定参考、持续更新'}


def caption(row,arm):
    phase=row['phase_before'];r=row['adaptation']
    names={'observe':'共同观察运动','clearance':'Code 抬手到安全高度','align':'Code 在杯柄外上方对齐',
           'descend':'Code 在杯柄外下降','approach':'冻结动作 MLP 接近杯柄',
           'close':'Code 闭合真实夹指','transport':'Code 维持抓持并抬升','hold':'Code 保持抬升'}
    purpose=names[phase]
    if phase=='approach' and row['command_mode']=='code':purpose='公开几何暂不可用，Code 保持位置'
    fresh=r['t']==row['t'];u=r['U'] if fresh else 0
    if phase in ('close','transport','hold'):detail='已退出局部阶段；活动 z 清零，沿用冻结目标。'
    elif arm=='frame_fixed':detail='闭合前逐帧读取视觉，z 始终为零。'
    elif phase!='approach':detail='积累因果公开历史，尚未锁定局部参考。'
    elif not r['reference_available']:detail='无已接纳历史参考，等待资格。'
    elif r['event']=='entry':detail=f"固定参考 t={r['ref_time']}；完成入口初始化。"
    elif r['event']=='continuous':detail=f"固定参考 t={r['ref_time']}；新合格反馈继续更新。"
    elif arm=='reference_fixed':detail=f"固定参考 t={r['ref_time']}；z 为零，无梯度更新。"
    elif not r['entry_initialized']:detail=f"保持参考 t={r['ref_time']}；等待至少三帧合格支持。"
    elif arm=='entry_once':detail='入口更新已完成；保持参考与 z，不再更新。'
    else:detail='无新合格反馈；保持参考与 z，估计不漂移。'
    return purpose,detail+f' 本步真实 U={u}。'


def load_selected(summary,split):
    selected=[]
    for arm in ARMS:
        item=next(r for r in summary['cells'] if r['result']['cell']==dict(split=split,seed=SEEDS[split][0],arm=arm))
        root=Path(item['path']);checked_ref(item['result_binding']);checked_ref(item['seal_binding'])
        seal=read(root/'SEAL.json')
        for name in ('TRACE.npy','PUBLIC_IMAGES.npz'):
            need(sha256(root/name)==seal['files'][name],'Changed sealed render source')
        trace=np.load(root/'TRACE.npy',allow_pickle=True).tolist()
        with np.load(root/'PUBLIC_IMAGES.npz',allow_pickle=False) as archive:
            frames=dict(t=archive['t'],rgb=archive['rgb'])
        need(np.array_equal(frames['t'],np.arange(0,240,4)),'Unexpected source RGB timeline')
        selected.append(dict(item=item,trace=trace,frames=frames))
    return selected


def probe(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames,nb_read_frames,r_frame_rate,duration',
        '-of','json',str(path)],text=True))['streams'][0]
    need(data['codec_name']=='h264' and data['pix_fmt']=='yuv420p' and data['width']==2240 and data['height']==800
         and int(data['nb_read_frames'])==84 and int(data['nb_frames'])==84 and data['r_frame_rate']=='8/1'
         and abs(float(data['duration'])-10.5)<1e-6 and faststart(path),'Unexpected video codec/timing/layout')
    return data


def render(split,selected,video,font):
    title=ImageFont.truetype(font,23);text=ImageFont.truetype(font,18)
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
             '-s','2240x800','-r','8','-i','pipe:0','-an','-c:v','libx264','-preset','fast',
             '-crf','18','-profile:v','high','-pix_fmt','yuv420p','-movflags','+faststart',
             '-threads','2',str(video)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for index,t in enumerate(selected[0]['frames']['t']):
            picture=Image.new('RGB',(2240,800),(15,22,34));draw=ImageDraw.Draw(picture)
            draw.text((18,10),f'P67 杯柄｜{split}｜新条件配对确认；单任务、单checkpoint；无朝向 OOD',font=title,fill='white')
            for column,(arm,run) in enumerate(zip(ARMS,selected)):
                x=column*560;row=run['trace'][int(t)];purpose,feedback=caption(row,arm)
                draw.text((x+12,52),ARM_NAMES[arm],font=title,fill=(180,210,255))
                draw.text((x+12,85),f'控制步 {int(t)}/239；实际命令：'+row['command_mode'],font=text,fill='white')
                put(draw,(x+12,115),purpose,text,width=535)
                put(draw,(x+12,150),feedback,text,width=535)
                picture.paste(Image.fromarray(run['frames']['rgb'][index]).resize((548,548)),(x+6,214))
            draw.text((18,773),'封存动作前 RGB，每 4 步一帧；播放压缩，不代表实际墙钟耗时。',font=text,fill=(185,198,215))
            if index==0:picture.save(video.with_suffix('.poster.png'))
            process.stdin.write(np.asarray(picture).tobytes())
        picture=Image.new('RGB',(2240,800),(15,22,34));draw=ImageDraw.Draw(picture)
        draw.text((24,35),'240 步执行完毕：封存后的原 P47 标准评估（文字终态卡）',font=title,fill='white')
        for column,(arm,run) in enumerate(zip(ARMS,selected)):
            x=column*560;item=run['item'];result=item['result']
            draw.text((x+18,115),ARM_NAMES[arm],font=title,fill=(180,210,255))
            label='严格成功' if result['validated_success'] else '机械异常，判失败' if not result['mechanical_valid'] else '未完成抓取'
            put(draw,(x+18,190),'结果：'+label,title,width=520)
            put(draw,(x+18,270),f"实际 NN 命令：{item['nn_command_steps']} 步；Code 命令：{item['code_command_steps']} 步",text,width=520)
            put(draw,(x+18,360),f"终止阶段：{item['terminal_phase']}；z 更新事件：{item['update_events']}",text,width=520)
        put(draw,(24,510),'这是预选的同条件四臂展示，不替代 768 条完整记录。成功标准、机械门与 240 步预算沿用冻结合同；全程异常不因视觉结果看似成功而忽略。',text,width=2180)
        for _ in range(24):process.stdin.write(np.asarray(picture).tobytes())
    finally:
        process.stdin.close()
        need(process.wait()==0,'H.264 encoding failed')
    return command


def write_index(summary,report,output,manifest):
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P67 新条件配对确认</title>',
        '<style>body{font:16px system-ui;max-width:1450px;margin:32px auto;padding:0 18px;line-height:1.6}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;text-align:left}video{width:100%}</style>',
        '<h1>P67：固定参考与入口校准的新条件配对确认</h1>',
        '<p>192 个预声明新条件 × 四臂，共 768 条；ID／尺寸／位置各64条件。单杯柄任务、单冻结checkpoint，无新训练和朝向OOD。全部失败、机械异常、未进入条件均保留分母。</p>',
        '<p>成功率区间为95% Wilson。主检验D−A三分组各做双侧精确McNemar，并在三项内Holm校正；次检验D−C使用独立三项Holm。其他比较仅描述；配对差区间为探索性bootstrap，不能替代精确检验。</p>',
        '<table><tr><th>分组</th><th>臂</th><th>严格成功／95% Wilson</th><th>native成功</th><th>机械异常</th><th>进入approach/close</th><th>终止阶段</th></tr>']
    for split in SPLITS:
        for arm in ARMS:
            g=summary['groups'][split+'/'+arm];lo,hi=g['wilson95']
            page.append(f'<tr><td>{split}</td><td>{ARM_NAMES[arm]}</td><td>{g["success"]}/64 · {lo:.1%}–{hi:.1%}</td>'
                f'<td>{g["sustained_native_success"]}/64</td><td>{g["mechanical_anomalies"]}/64</td>'
                f'<td>{g["phase_entered"]["approach"]}/64 · {g["phase_entered"]["close"]}/64</td>'
                f'<td>{html.escape(str(g["terminal_phases"]))}</td></tr>')
    page.append('</table><h2>六种配对比较</h2><table><tr><th>分组／比较</th><th>救回／误伤／平局</th><th>差值／探索性95%区间</th><th>双侧精确p／Holm p</th></tr>')
    for split in SPLITS:
        for label in COMPARISONS:
            c=summary['comparisons'][split+'/'+label];lo,hi=c['delta_exploratory_bootstrap95']
            ptext=(f'{c["exact_mcnemar_two_sided_p"]:.5g} ／ {c["holm_three_strata_p"]:.5g}' if 'exact_mcnemar_two_sided_p' in c else '仅描述')
            page.append(f'<tr><td>{split} / {label}</td><td>{c["rescues"]}／{c["harms"]}／{c["ties"]}</td>'
                f'<td>{c["delta"]:.1%} · {lo:.1%}–{hi:.1%}</td><td>{ptext}</td></tr>')
    page.append('</table><p>B−A只描述参考保持，不算梯度收益。192条件合并数仅描述，不覆盖尺寸分组，也不证明同算力优化器优越性。</p><h2>预先选定的三个条件</h2>')
    for video in manifest['videos']:
        href=html.escape(os.path.relpath(video['video']['path'],output),quote=True)
        page.append(f'<h3>{video["split"]} · seed {SEEDS[video["split"]][0]}</h3><video controls preload="metadata" src="{href}"></video>')
    href=html.escape(os.path.relpath(report,output),quote=True)
    page.append(f'<p>每4步保存的动作前RGB；仅终末卡显示结果，不替代768条统计。<a href="{href}">唯一原始结果JSON与审计</a> · <a href="MANIFEST.json">展示来源绑定</a></p></html>')
    (output/'index.html').write_text('\n'.join(page)+'\n')


def build(report,output,font=FONT):
    report,output=Path(report).resolve(),Path(output).resolve();summary=read(report)
    need(summary.get('schema')=='prove.p67-reference-confirmation-complete-audit.v1' and summary.get('complete')
         and summary.get('all_runtime_invariants_passed') and len(summary['cells'])==768,'Completed P67 audited report required')
    need(not output.exists(),'Refusing to overwrite presentation');output.mkdir(parents=True)
    manifest=dict(schema='prove.p67-h264-four-arm-presentation.v1',source_report=ref(report),source_generator=ref(__file__),
        font=ref(font),physics_steps=0,optimizer_updates=0,source_of_metrics='only source_report; no copied results JSON',
        selected_seeds={s:SEEDS[s][0] for s in SPLITS},selected_splits=list(SPLITS),videos=[])
    for split in SPLITS:
        selected=load_selected(summary,split);video=output/f'{split}_{SEEDS[split][0]}_four_arms.mp4'
        command=render(split,selected,video,font);notes=[]
        for arm,run in zip(ARMS,selected):
            root=Path(run['item']['path']);public=output/f'{split}_{arm}_PUBLIC_TRACE.jsonl'
            with public.open('w') as stream:
                for row in run['trace']:
                    r=row['adaptation'];fresh=r['t']==row['t']
                    stream.write(json.dumps(dict(t=row['t'],phase_before=row['phase_before'],command_mode=row['command_mode'],
                        adaptation=dict(t=r['t'],U=r['U'] if fresh else 0,event=r['event'] if fresh else None,
                            ref_time=r['ref_time'],admitted=r['admitted'] if fresh else False,
                            reference_available=r['reference_available'],entry_initialized=r['entry_initialized'])))+'\n')
            segments=[]
            for t in run['frames']['t']:
                purpose,feedback=caption(run['trace'][int(t)],arm)
                segments.append(dict(start_step=int(t),end_step=int(t),stage=purpose,detail=feedback,
                    sources=[dict(file=public.name,line=int(t)+1,pointer='/phase_before'),
                             dict(file=public.name,line=int(t)+1,pointer='/adaptation')]))
            notes.append(dict(arm=arm,segments=segments,source_bindings={name:ref(root/name) for name in
                ('TRACE.npy','PUBLIC_IMAGES.npz','SEAL.json','RESULT.json')},derived_public_trace=ref(public)))
        annotations=output/f'{split}_ANNOTATIONS.json';annotations.write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
        manifest['videos'].append(dict(split=split,video=ref(video),annotations=ref(annotations),probe=probe(video),encoding_command=command))
    write_index(summary,report,output,manifest)
    manifest['index']=ref(output/'index.html')
    (output/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--font',default=FONT)
    args=parser.parse_args();build(args.report,args.output,args.font)
