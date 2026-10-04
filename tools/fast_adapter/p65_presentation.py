#!/usr/bin/env python3
"""Corrected P65 H.264 display copies from sealed RGB; original bytes preserved."""
from __future__ import annotations

import argparse
import copy
import html
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from tools.fast_adapter.p65_report import ARMS,SPLITS,FONT,caption as original_caption,read,ref,need,write_index
from tools.fast_adapter.p64_cup_presentation import put
from prove.fast_adapter.p61_data import sha256

ARM_NAMES={'fixed':'固定 z','online_local':'阶段内在线 z（主比较）','online_history':'闭合前持续更新（诊断）'}


def caption(row,arm):
    purpose,feedback=original_caption(row,arm)
    if arm=='fixed':
        feedback=('z 始终为零；闭合后沿用冻结目标。' if row['phase_before'] in ('close','transport','hold')
                  else 'z 始终为零；闭合前读取当前视觉。')
    return purpose,feedback


def probe(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=codec_name,pix_fmt,width,height,nb_frames,nb_read_frames,r_frame_rate,duration',
        '-of','json',str(path)],text=True))['streams'][0]
    need(data['width']==1680 and data['height']==800 and int(data['nb_read_frames'])==84
         and int(data['nb_frames'])==84 and data['r_frame_rate']=='8/1'
         and abs(float(data['duration'])-10.5)<1e-6,'Video dimensions/frame count/timing changed')
    return data


def faststart(path):
    atoms=[]
    with Path(path).open('rb') as stream:
        while True:
            head=stream.read(8)
            if not head:break
            need(len(head)==8,'Truncated MP4 atom')
            size,kind=struct.unpack('>I4s',head);header=8
            if size==1:size=struct.unpack('>Q',stream.read(8))[0];header=16
            if size==0:break
            need(size>=header,'Invalid MP4 atom')
            atoms.append(kind.decode('ascii'));stream.seek(size-header,1)
    return 'moov' in atoms and 'mdat' in atoms and atoms.index('moov')<atoms.index('mdat')


def checked_ref(value):
    need(sha256(value['path'])==value['sha256'],'Changed display source: '+value['path'])


def load_selected(summary,split):
    selected=[]
    for arm in ARMS:
        item=next(r for r in summary['cells'] if r['result']['cell']==dict(split=split,seed=640100,arm=arm))
        root=Path(item['path']);seal=read(root/'SEAL.json')
        checked_ref(item['result_binding']);checked_ref(item['seal_binding'])
        for name in ('TRACE.npy','PUBLIC_IMAGES.npz'):
            need(sha256(root/name)==seal['files'][name],'Changed sealed render source')
        trace=np.load(root/'TRACE.npy',allow_pickle=True).tolist()
        with np.load(root/'PUBLIC_IMAGES.npz',allow_pickle=False) as archive:
            frames=dict(t=archive['t'],rgb=archive['rgb'])
        need(np.array_equal(frames['t'],np.arange(0,240,4)),'Unexpected source RGB timeline')
        selected.append(dict(item=item,trace=trace,frames=frames))
    return selected


def render(split,selected,video,font):
    title=ImageFont.truetype(font,23);text=ImageFont.truetype(font,18)
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
             '-s','1680x800','-r','8','-i','pipe:0','-an','-c:v','libx264','-preset','fast',
             '-crf','18','-profile:v','high','-pix_fmt','yuv420p','-movflags','+faststart',
             '-threads','2',str(video)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for index,t in enumerate(selected[0]['frames']['t']):
            picture=Image.new('RGB',(1680,800),(15,22,34));draw=ImageDraw.Draw(picture)
            draw.text((18,10),f'P65 杯柄｜{split}｜仅位置／尺寸变化，无朝向 OOD',font=title,fill='white')
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
        picture=Image.new('RGB',(1680,800),(15,22,34));draw=ImageDraw.Draw(picture)
        draw.text((24,35),'240 步执行完毕：封存后的原 P47 标准评估（文字终态卡）',font=title,fill='white')
        for column,(arm,run) in enumerate(zip(ARMS,selected)):
            x=column*560;item=run['item'];result=item['result']
            draw.text((x+18,115),ARM_NAMES[arm],font=title,fill=(180,210,255))
            label='严格成功' if result['validated_success'] else '机械异常，判失败' if not result['mechanical_valid'] else '未完成抓取'
            put(draw,(x+18,190),'结果：'+label,title,width=520)
            put(draw,(x+18,270),f"实际 NN 命令：{item['nn_command_steps']} 步；Code 命令：{item['code_command_steps']} 步",text,width=520)
            put(draw,(x+18,360),f"终止阶段：{item['terminal_phase']}；z 更新事件：{item['update_events']}",text,width=520)
        put(draw,(24,510),'这是预选的同条件三臂展示，不替代 36 条完整记录。成功标准、机械门与 240 步预算沿用冻结合同；全程异常不因视觉结果看似成功而忽略。',text,width=1620)
        for _ in range(24):process.stdin.write(np.asarray(picture).tobytes())
    finally:
        process.stdin.close()
        need(process.wait()==0,'H.264 encoding failed')
    return command


def build(report,output,font=FONT):
    report,output=Path(report).resolve(),Path(output).resolve()
    summary=read(report)
    need(summary.get('schema')=='prove.p65-stage-local-cup-complete-audit.v1'
         and summary['complete'] and summary['all_runtime_invariants_passed'] and len(summary['cells'])==36,
         'Completed P65 audited source required')
    original_manifest=report.parent/'videos/MANIFEST.json'
    old=read(original_manifest)
    old_files={str(p.relative_to(report.parent)):sha256(p) for p in report.parent.rglob('*') if p.is_file()}
    need(not output.exists(),'Refusing to overwrite presentation')
    output.mkdir(parents=True)
    manifest=dict(schema='prove.p65-h264-presentation.v1',source_report=ref(report),
        supersedes_manifest=ref(original_manifest),source_generator=ref(__file__),font=ref(font),
        original_report_directory=str(report.parent),original_report_files=old_files,
        revisions=['Fixed arm captions distinguish preclose current vision from postclose frozen target.',
                   'History arm labeled preclose continuous updates, not whole-episode updates.',
                   'H.264 High/yuv420p/faststart display encoding; same 84-frame source timeline.'],
        physics_steps=0,optimizer_updates=0,source_of_metrics='only source_report; no copied results JSON',videos=[])
    for split in SPLITS:
        previous=next(v for v in old['videos'] if v['split']==split)
        checked_ref(previous['video']);checked_ref(previous['annotations'])
        original_probe=probe(previous['video']['path'])
        selected=load_selected(summary,split)
        video=output/f'{split}_640100_three_arms.mp4'
        command=render(split,selected,video,font)
        annotations=read(previous['annotations']['path'])
        for arm,run,notes in zip(ARMS,selected,annotations):
            need(notes['arm']==arm,'Original annotation arm order changed')
            for binding in notes['source_bindings'].values():checked_ref(binding)
            for segment in notes['segments']:
                t=segment['start_step'];purpose,feedback=caption(run['trace'][t],arm)
                segment.update(stage=purpose,detail=feedback)
            notes['supersedes_annotations']=previous['annotations']
            notes['display_revision']='preclose current vision / postclose retained target; source trace unchanged'
        annotation_path=output/f'{split}_ANNOTATIONS.json'
        annotation_path.write_text(json.dumps(annotations,ensure_ascii=False,indent=2)+'\n')
        verified=probe(video)
        need(verified['codec_name']=='h264' and verified['pix_fmt']=='yuv420p' and faststart(video),
             'Browser-compatible H.264/faststart verification failed')
        manifest['videos'].append(dict(split=split,seed=640100,video=ref(video),annotations=ref(annotation_path),
            poster=ref(video.with_suffix('.poster.png')),original_video=previous['video'],
            original_annotations=previous['annotations'],original_probe=original_probe,probe=verified,
            faststart=True,encoding_command=command,sources=previous['sources']))
    display=copy.deepcopy(summary);display['videos']=manifest
    write_index(display,output)
    page=(output/'index.html').read_text()
    href=html.escape(os.path.relpath(report,output),quote=True)
    page=page.replace('href="RESULTS.json"','href="'+href+'"')
    for item in manifest['videos']:
        name=Path(item['video']['path']).name
        anchor=f'href="{name}"'
        end=page.index('</a>',page.index(anchor))+4
        player=f'<br><video controls preload="metadata" style="width:100%;max-width:1200px" poster="{Path(item["poster"]["path"]).name}"><source src="{name}" type="video/mp4"></video>'
        page=page[:end]+player+page[end:]
    page=page.replace('</html>','<p><a href="MANIFEST.json">展示修订及来源绑定</a>；原报告与视频全部保留。</p></html>')
    (output/'index.html').write_text(page)
    current={str(p.relative_to(report.parent)):sha256(p) for p in report.parent.rglob('*') if p.is_file()}
    need(current==old_files,'Original report directory changed during display generation')
    manifest['original_bytes_unchanged']=True
    manifest['index']=ref(output/'index.html')
    (output/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--font',default=FONT)
    args=parser.parse_args()
    result=build(args.report,args.output,args.font)
    print(json.dumps(dict(videos=[dict(split=x['split'],probe=x['probe'],faststart=x['faststart']) for x in result['videos']],
                         original_bytes_unchanged=result['original_bytes_unchanged']),indent=2))


if __name__=='__main__':main()
