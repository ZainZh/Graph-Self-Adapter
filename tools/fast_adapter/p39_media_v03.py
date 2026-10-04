"""Selected annotated P39 videos, replayed only from sealed physical actions."""
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from prove.fast_adapter.planar_contact import PlanarPush,STEPS
from tools.fast_adapter.p39_tokens import ROOT,sha,js

OUT=ROOT/'summary/media_v02'

def font(size):
    p=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True)
    return ImageFont.truetype(p,size)

def source(split,kind,arm):
    folder=ROOT/'final'/kind/('seed0' if kind=='references' else 'seed11')/split
    raw=folder/(arm+'_RUNTIME.pt');res=folder/(arm+'_RESULT.json');ev=folder/(arm+'_EVALUATOR.json')
    return dict(kind=kind,arm=arm,raw=torch.load(raw,weights_only=True,map_location='cpu'),result=json.loads(res.read_text()),ev=json.loads(ev.read_text()),files=[raw,res,ev])

def render(name,split,index,settings,explanation):
    sources=[source(split,k,a) for k,a,_ in settings];envs=[PlanarPush(s['ev']['configs'][index]) for s in sources]
    cols=3;rows=(len(sources)+2)//3;w=cols*448;rh=540;h=110+rows*rh+70
    big=font(28);mid=font(22);small=font(18);events=[];video=OUT/(name+'.mp4')
    proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','6','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    try:
        for t in range(STEPS+1):
            canvas=Image.new('RGB',(w,h),(245,248,252));draw=ImageDraw.Draw(canvas)
            draw.text((18,8),'二维接触推移：不同Token如何改变持续适配',font=big,fill='#183746')
            draw.text((18,46),explanation,font=small,fill='#3f5969')
            stage='共同探测' if t<3 else ('持续反馈控制' if t<STEPS else '封存后的最终评价')
            draw.text((18,76),f'{split} · 初态 {index} · 已执行 {t}/{STEPS} 步 · {stage}',font=small,fill='#3f5969')
            for j,(s,e,setting) in enumerate(zip(sources,envs,settings)):
                o,rgb,masks=e.public();state=e.state();raw=s['raw'];col=j%cols;row=j//cols;x=col*448;y=110+row*rh
                pose_error=float(np.max(np.abs(state['qpos']-raw['qpos'][index,t].numpy())));assert pose_error<1e-9
                rgb_match=__import__('hashlib').sha256(rgb.tobytes()).hexdigest()==raw['rgb_hashes'][t][index]
                mask_match=__import__('hashlib').sha256(masks.tobytes()).hexdigest()==raw['mask_hashes'][t][index]
                draw.text((x+10,y+4),setting[2],font=mid,fill='#183746')
                panel=Image.fromarray(rgb);pd=ImageDraw.Draw(panel)
                for box,label,color in zip(o['boxes'],('Actor','Object','Goal'),('#116d93','#9b3575','#976719')):
                    b=list(map(int,box));pd.rectangle(b,outline=color,width=2)
                    cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
                    pd.line((cx-4,cy,cx+4,cy),fill=color,width=2);pd.line((cx,cy-4,cx,cy+4),fill=color,width=2)
                canvas.paste(panel,(x,y+34))
                distance=float(torch.linalg.vector_norm(raw['positions'][index,t,2]-raw['positions'][index,t,1]))*1000
                updates=t if s['arm'] in ('continuous','donor') else min(t,3) if s['arm']=='first3' else 0
                text=f'可见距离 {distance:.1f} mm · 更新 {updates} 次'
                if t==STEPS:
                    end=s['result']['rows'][index]
                    status='成功' if end['success'] else ('观测中止' if not end['observation_valid'] else ('未对齐' if end['error']>=.0025 else '未停稳'))
                    text=status+f' · 中心误差 {end["error"]*1000:.2f} mm'
                draw.text((x+10,y+486),text,font=small,fill='#235b4b' if t==STEPS and s['result']['rows'][index]['success'] else '#3f5969')
                if t<STEPS:draw.text((x+10,y+511),f'下一步推力 ({raw["actions"][index,t,0]:.2f}, {raw["actions"][index,t,1]:.2f}) N',font=small,fill='#3f5969')
                else:draw.text((x+10,y+511),f'末步速度 {end["speed"]*1000:.1f} mm/s · 阈值 8 mm/s',font=small,fill='#3f5969')
                events.append(dict(step=t,panel=j,source=str(s['files'][0]),index=index,stage=stage,display=text,pose_error=pose_error,rgb_match=rgb_match,mask_match=mask_match))
                if t<STEPS:e.step(raw['actions'][index,t].numpy())
            draw.text((18,h-57),'蓝框：工具；紫框：物块；黄框：目标。目标：中心误差＜2.5 mm，且速度＜8 mm/s。',font=small,fill='#526878')
            draw.text((18,h-30),'GT mask辅助的两轴圆柱接触，非机械臂；视频为封存示例，完整统计见报告。',font=small,fill='#526878')
            payload=np.asarray(canvas).tobytes();proc.stdin.write(payload)
            if t in (0,20,40):canvas.save(OUT/f'{name}_step{t}.jpg',quality=92)
            if t==STEPS:
                canvas.save(OUT/(name+'.jpg'),quality=92)
                for _ in range(12):proc.stdin.write(payload)
        proc.stdin.close();assert proc.wait()==0
    finally:
        for e in envs:e.close()
    bindings=[dict(path=str(p),sha256=sha(p)) for s in sources for p in s['files']]
    js(OUT/(name+'_ANNOTATIONS.json'),dict(schema='prove.episode-annotations.v1',role='post-seal selected illustration, not metric evidence',split=split,index=index,explanation=explanation,source_bindings=bindings,segments=[dict(start_step=0,end_step=2,stage='共同探测',detail='相同条件下各方法前三次探测相同',sources=bindings),dict(start_step=3,end_step=39,stage='反馈执行',detail='只读取已完成动作的反馈；固定经验和错配反馈分别标明',sources=bindings),dict(start_step=40,end_step=40,stage='最终评价',detail='只在最后展示封存后的物理成功判据',sources=bindings)],frames=events,video=str(video),video_sha256=sha(video),rgb_exact=sum(e['rgb_match'] for e in events),mask_exact=sum(e['mask_match'] for e in events),total_frames=len(events),supersedes_manifest=str(ROOT/'summary/media_v01'/(name+'_ANNOTATIONS.json')),revision='Show position and speed criteria separately; add center markers and prefer a case where mean and donor both fail. Original videos and all trial artifacts preserved.'))

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    settings=[('references','base','冻结基座'),('global','continuous','整图DINO反馈'),('local','continuous','局部空间DINO反馈'),('relation','continuous','显式相对关系'),('fused','continuous','局部DINO＋关系'),('history','continuous','History对照')]
    split='joint';base=source(split,'references','base');fused=source(split,'fused','continuous');glob=source(split,'global','continuous')
    candidates=[i for i,x in enumerate(base['result']['rows']) if not x['success'] and fused['result']['rows'][i]['success'] and not glob['result']['rows'][i]['success']]
    if not candidates:candidates=[i for i,x in enumerate(base['result']['rows']) if not x['success'] and fused['result']['rows'][i]['success']]
    index=candidates[0] if candidates else 0
    render('representation_comparison',split,index,settings,'同一初态、相同控制器和40步预算；比较用于更新z的表示。')
    failures=[i for i,r in enumerate(fused['result']['rows']) if not r['success']]
    if failures:render('retained_failure',split,failures[0],settings,'保留的融合方法失败例；不只展示成功案例。')
    if (ROOT/'final/fused/seed11/paired/continuous_RESULT.json').exists():
        split='paired';f=source(split,'fused','continuous');d=source(split,'fused','donor')
        m=source(split,'fused','mean')
        ids=[i for i,x in enumerate(f['result']['rows']) if x['success'] and not d['result']['rows'][i]['success'] and not m['result']['rows'][i]['success']]
        if not ids:ids=[i for i,x in enumerate(f['result']['rows']) if x['success'] and not d['result']['rows'][i]['success']]
        render('feedback_causality',split,ids[0] if ids else 0,[('fused','mean','固定训练平均z'),('fused','donor','相反阻力物块的反馈'),('fused','continuous','当前物块的反馈')],'两物块初始外观相同、隐藏阻力不同；交换已完成交互的反馈。')

if __name__=='__main__':torch.set_num_threads(1);main()
