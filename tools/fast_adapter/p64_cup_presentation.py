"""Chinese postsealed cup videos; no outcomes before a separate terminal card.

The sealed renderer evidence is sampled every four control steps. The final
result is text on an end card, never represented as an unrecorded terminal RGB.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from prove.fast_adapter.p61_data import sha256
from tools.fast_adapter.p64_cup_report import load_cell

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
PURPOSES={
    'shared observation motion':'共同观察：移动腕部相机，获取视角变化',
    'valid target geometry unavailable: hold':'保持位置：等待可用的杯柄几何估计',
    'raise hand to shared clearance':'抬手避障：先到达共同安全高度',
    'align above handle outside approach':'柄外对齐：在杯柄上方调整接近位置',
    'descend outside handle':'柄外下降：保持张开手指，下降到抓持高度',
    'approach handle with open fingers':'接近杯柄：张开手指到达估计抓持位置',
    'close physical fingers':'闭合夹爪：尝试建立双侧真实接触',
    'lift with finger contacts':'保持夹持并抬升：检验能否持续抓起杯子',
}


def put(draw,xy,text,font,fill=(233,238,247),width=510):
    x,y=xy;line=''
    for char in text:
        trial=line+char
        if draw.textbbox((0,0),trial,font=font)[2]>width and line:
            draw.text((x,y),line,font=font,fill=fill);y+=26;line=char
        else:line=trial
    if line:draw.text((x,y),line,font=font,fill=fill)
    return y+26


def feedback(row,arm):
    record=row.get('adaptation')
    if arm=='fixed':return '反馈状态：持续读取视觉；z 保持固定'
    if row['stage'] in ('close physical fingers','lift with finger contacts'):
        if record and record['t']==row['t'] and record['U']:
            return f"反馈状态：本步 z 更新 {record['U']} 次；闭合后冻结"
        return '反馈状态：z 已冻结，闭合和抬升时不再更新'
    if record is None:return '反馈状态：尚无视觉估计'
    if record['t']==row['t'] and record['U']:
        return f"反馈状态：z 本步更新 {record['U']} 次；支持观测 {record['S']} 帧"
    if not record['admitted']:return '反馈状态：当前闭合轮廓不可用，不更新 z'
    return f"反馈状态：积累可见轮廓；支持观测 {record['S']} 帧"


def render_pair(left,right,destination,font_path):
    import cv2
    title_font=ImageFont.truetype(font_path,24);body_font=ImageFont.truetype(font_path,18)
    small_font=ImageFont.truetype(font_path,16)
    split=left['result']['cell']['split'];split_zh={'id':'训练范围内','size':'尺寸范围外'}[split]
    writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),8,(1120,800))
    if not writer.isOpened():raise RuntimeError('MP4 writer unavailable')
    try:
        with np.load(Path(left['path'])/'PUBLIC_IMAGES.npz') as a,np.load(Path(right['path'])/'PUBLIC_IMAGES.npz') as b:
            if not np.array_equal(a['t'],b['t']):raise ValueError('Unmatched recorded sample times')
            for index,t in enumerate(a['t']):
                canvas=Image.new('RGB',(1120,800),(15,22,34));draw=ImageDraw.Draw(canvas)
                draw.text((18,10),f'杯柄抓取｜{split_zh}｜同一模型：固定 z 与在线 z',font=title_font,fill=(245,245,245))
                draw.text((18,45),'仅位置／尺寸变化；朝向固定。逐帧说明动作目的，不提前显示结果。',font=small_font,fill=(184,194,210))
                for panel,(cell,archive) in enumerate(((left,a),(right,b))):
                    x=panel*560;row=cell['trace'][int(t)];arm=cell['result']['cell']['arm']
                    draw.text((x+16,77),'固定 z' if arm=='fixed' else '在线 z',font=title_font,fill=(143,192,255) if panel==0 else (120,226,189))
                    draw.text((x+138,83),f'控制步 {int(t)} / 239（动作前观测）',font=small_font,fill=(213,221,233))
                    put(draw,(x+16,113),'目的：'+PURPOSES.get(row['stage'],row['stage']),body_font)
                    put(draw,(x+16,166),feedback(row,arm),small_font,fill=(205,217,231))
                    rgb=Image.fromarray(archive['rgb'][index]).resize((548,548),Image.Resampling.LANCZOS)
                    canvas.paste(rgb,(x+6,215))
                draw.text((18,774),'原始封存 RGB；每 4 个控制步展示一帧，播放速度经过压缩。',font=small_font,fill=(181,192,208))
                writer.write(cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR))
        # An explicit end card avoids pretending the t=236 sample is t=240.
        canvas=Image.new('RGB',(1120,800),(15,22,34));draw=ImageDraw.Draw(canvas)
        draw.text((30,54),'240 步执行完毕：封存后的终态评估',font=title_font,fill=(245,245,245))
        put(draw,(30,106),'此页为评估文字，不是未保存的终态画面。全部物理步骤和机械异常均计入。',body_font,width=1030)
        for panel,cell in enumerate((left,right)):
            x=panel*560;r=cell['result'];arm=r['cell']['arm']
            draw.text((x+30,202),'固定 z' if arm=='fixed' else '在线 z',font=title_font,fill=(220,233,250))
            label='严格成功' if r['validated_success'] else ('机械异常，判失败' if not r['mechanical_valid'] else '未完成抓取')
            put(draw,(x+30,262),f'结果：{label}',title_font)
            put(draw,(x+30,332),f"末 10 状态持续抓起：{'是' if r['sustained_native_success'] else '否'}",body_font)
            penetration=r.get('max_full_trajectory_contact_penetration')
            put(draw,(x+30,386),'全程最大穿透：'+('非有限值' if penetration is None else f'{penetration*1000:.2f} 毫米'),body_font)
            put(draw,(x+30,440),f"神经动作步数：{r['neural_action_steps']}；z 更新事件：{r['update_events']}",body_font)
        put(draw,(30,566),'成功要求：末 10 状态均有双侧接触且抬升超过 4 厘米；全程穿透小于 2 毫米，并满足关节限制与有限值要求。',body_font,width=1030)
        put(draw,(30,654),'这是预先选定条件的演示，不替代三个分布下全部 24 个回合的统计。',body_font,width=1030)
        frame=cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
        for _ in range(24):writer.write(frame)
    finally:writer.release()


def build(report,output,font=FONT):
    report=Path(report);output=Path(output)
    data=json.loads(report.read_text())
    if len(data['cells'])!=24 or not data.get('all_runtime_invariants_passed'):raise ValueError('Completed audited 24-cell report required')
    output.mkdir(parents=True,exist_ok=False)
    lookup={(r['result']['cell']['split'],r['result']['cell']['seed'],r['result']['cell']['arm']):r for r in data['cells']}
    records=[]
    for split in ('id','size'):
        pair=[load_cell(lookup[split,640100,arm]['path']) for arm in ('fixed','translation_size5')]
        name=f'{split}_640100_paired.mp4';render_pair(*pair,output/name,font)
        records.append(dict(split=split,seed=640100,path=name,sha256=sha256(output/name),
            sources=[dict(path=r['path'],seal_sha256=sha256(Path(r['path'])/'SEAL.json'),
                result_sha256=sha256(Path(r['path'])/'RESULT.json')) for r in pair]))
    manifest=dict(schema='prove.p64-cup-chinese-presentation.v1',report_sha256=sha256(report),
        source_code_sha256=sha256(__file__),font=dict(path=font,sha256=sha256(font)),videos=records,
        outcome_disclosure='only after all recorded frames, separate terminal text card',physics_steps=0,
        frame_semantics='recorded preaction RGB at steps0,4,...236; no synthetic terminal RGB',fps=8,
        annotations='Chinese stage purpose, public feedback availability, actual latent update, closing freeze, control step')
    (output/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);parser.add_argument('--output',required=True);parser.add_argument('--font',default=FONT)
    args=parser.parse_args();print(json.dumps(build(args.report,args.output,args.font),ensure_ascii=False,indent=2))
