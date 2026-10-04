"""Annotated post-seal MuJoCo replay; selected illustrations never enter metrics."""
import json
import subprocess
from pathlib import Path
import hashlib
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from prove.fast_adapter.near_contact_push import NearPush, rgb_measurements

R=Path('evaluations/fast_adapter_p38_causal_v01');OUT=R/'summary/media_v03'


def font(size):
    path=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True)
    return ImageFont.truetype(path,size)


def source(kind,arm):
    folder=R/'push_contrast/final'/kind/'seed11'
    raw=folder/(arm+'_RUNTIME.pt');result=folder/(arm+'_RESULT.json');evaluator=folder/(arm+'_EVALUATOR.json')
    return {'kind':kind,'arm':arm,'raw_path':raw,'result_path':result,'evaluator_path':evaluator,
            'raw':torch.load(raw,weights_only=True),'result':json.loads(result.read_text()),
            'private':json.loads(evaluator.read_text())}


def render_video(name,sources,indices,labels,explanation):
    OUT.mkdir(parents=True,exist_ok=True)
    rows=len(indices);cols=len(sources);w=448*cols;row_h=528;h=112+row_h*rows+74
    video=OUT/(name+'.mp4');poster=OUT/(name+'.jpg')
    command=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
             '-r','5','-i','-','-an','-c:v','libx264','-crf','20','-preset','medium','-pix_fmt','yuv420p',str(video)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    environments=[[NearPush(s['private']['configs'][i]) for s in sources] for i in indices]
    events=[];titles=font(23);small=font(19);large=font(29);frames_hash=[];raw_rgb_checks=[]
    for step in range(31):
        canvas=Image.new('RGB',(w,h),(244,247,251));draw=ImageDraw.Draw(canvas)
        draw.text((20,10),'交互反馈如何改变下一次推力',font=large,fill=(20,40,65))
        draw.text((20,51),explanation,font=small,fill=(40,60,80))
        stage='共同轻推探测' if step<3 else ('连续反馈执行' if step<30 else '封存后的最终评价')
        draw.text((20,79),f'已执行 {step}/30 步 · {stage} · 绿色工具推动蓝色物块，横向对齐红标记',font=small,fill=(40,60,80))
        for row,index in enumerate(indices):
            regime='较低阻力' if index%2==0 else '较高阻力'
            y0=112+row*row_h
            for col,s in enumerate(sources):
                env=environments[row][col];rgb=env.rgb();raw_hash=hashlib.sha256(rgb.tobytes()).hexdigest()
                expected=s['raw']['rgb_sha256'][step][index]
                raw_rgb_checks.append(raw_hash==expected)
                rgb_geometry_error=float(np.max(np.abs(rgb_measurements(rgb)['positions']-s['raw']['positions'][index,step].numpy())))
                pose_error=float(np.max(np.abs(env.state()['qpos']-s['raw']['qpos'][index,step].numpy())))
                assert pose_error<1e-10
                panel=Image.fromarray(rgb);canvas.paste(panel,(col*448,y0+36))
                draw.text((col*448+12,y0+5),labels[col]+' · '+regime,font=titles,fill=(30,55,75))
                error=abs(float(s['raw']['positions'][index,step,2]-s['raw']['positions'][index,step,1]))*1000
                updates=step if s['arm'] in ('continuous','donor') else (min(step,3) if s['arm']=='first3' else 0)
                text=f'反馈更新 {updates} 次 · RGB估计距离 {error:.1f} mm'
                if step==30:
                    ok=s['result']['rows'][index]['success'];text=('最终成功' if ok else '最终未完成')+f' · 物理误差 {s["result"]["rows"][index]["error"]*1000:.1f} mm'
                draw.text((col*448+12,y0+487),text,font=small,fill=(20,90,65) if step==30 and s['result']['rows'][index]['success'] else (45,60,75))
                events.append({'step':step,'row':row,'column':col,'condition_index':index,'stage':stage,
                               'label':labels[col],'feedback_updates':updates,'display':text,
                               'runtime_source':str(s['raw_path']),'runtime_step':step,'rgb_matches_record':raw_hash==expected,
                               'rgb_geometry_error_m':rgb_geometry_error,'physical_pose_error':pose_error})
                if step<30:env.step(float(s['raw']['actions'][index,step]))
        draw.text((20,h-60),'真实MuJoCo接触 · 单轴短推 · 非机械臂任务；阻力标签仅用于事后展示，执行模型看不到。',font=small,fill=(50,65,85))
        draw.text((20,h-33),'演示从已封存结果中选取；整体成功率以完整独立验证表格为准。',font=small,fill=(50,65,85))
        payload=np.asarray(canvas).tobytes();proc.stdin.write(payload)
        frames_hash.append(hashlib.sha256(payload).hexdigest())
        if step in (0,15,30):canvas.save(OUT/f'{name}_step{step}.jpg',quality=92)
        if step==30:
            canvas.save(poster,quality=92)
            for _ in range(10):proc.stdin.write(payload)
    proc.stdin.close();assert proc.wait()==0
    for row in environments:
        for env in row:env.close()
    (OUT/(name+'_RGB_REPLAY_CHECK.json')).write_text(json.dumps({'passed':all(raw_rgb_checks),
          'matched':sum(raw_rgb_checks),'total':len(raw_rgb_checks),
          'mismatches':[e for e in events if not e['rgb_matches_record']]},ensure_ascii=False,indent=2)+'\n')
    bindings=[]
    for s in sources:
        for key in ('raw_path','result_path','evaluator_path'):
            path=s[key];bindings.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    data={'schema':'prove.episode-annotations.v1','name':name,'role':'post-seal selected illustration,not a metric source',
          'selection':{'indices':indices,'rule':'first pair showing an appearance-feedback failure while motion succeeds' if len(indices)>1 else 'first high-resistance case where own feedback succeeds and trainmean/donor fail'},
          'source_bindings':bindings,'segments':[{'start_step':0,'end_step':2,'stage':'共同探测','detail':'各方法前三步动作一致','sources':bindings},
             {'start_step':3,'end_step':29,'stage':'反馈执行','detail':'仅已完成交互用于下一次修正','sources':bindings},
             {'start_step':30,'end_step':30,'stage':'最终结果','detail':'只在最后显示封存后物理评价','sources':bindings}],
          'frames':events,'raw_rgb_checks':{'passed':all(raw_rgb_checks),'frames_checked':len(raw_rgb_checks)},
          'rendering_limit':'Re-rendered from exact physical replay; any nonidentical RGB raster is recorded, not used to change outcomes.',
          'supersedes_manifest':str(R/'summary/media_v02'/(name+'_ANNOTATIONS.json')) if (R/'summary/media_v02'/(name+'_ANNOTATIONS.json')).exists() else None,
          'video':str(video),'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),'poster':str(poster),'frame_hashes':frames_hash}
    (OUT/(name+'_ANNOTATIONS.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(name,'complete',sum(raw_rgb_checks),'/',len(raw_rgb_checks),'RGB frames exact; all physical poses exact; max detector difference',max(e['rgb_geometry_error_m'] for e in events),flush=True)


def main():
    base=source('references','base');appearance=source('appearance','continuous');motion=source('motion','continuous')
    candidates=[i for i in range(0,64,2) if any(not appearance['result']['rows'][j]['success'] and motion['result']['rows'][j]['success'] for j in (i,i+1))]
    pair=candidates[0]
    render_video('representation_pair',[base,appearance,motion],[pair,pair+1],['冻结基座','DINO特征反馈','可见运动反馈'],
                 '两行初始外观分别相同，隐藏阻力不同；比较用于更新z的反馈表示。')
    donor=source('motion','donor');mean=source('motion','train_mean')
    ids=[i for i in range(1,64,2) if motion['result']['rows'][i]['success'] and not donor['result']['rows'][i]['success'] and not mean['result']['rows'][i]['success']]
    render_video('correct_vs_wrong_feedback',[mean,donor,motion],[ids[0]],['固定训练平均经验','错误物体的反馈','当前物体的反馈'],
                 '同一初态、同一动作接口与预算；只有反馈来源或是否更新不同。')


if __name__=='__main__':
    torch.set_num_threads(1);main()
