"""Annotated P40 illustrative videos replay sealed actions; no new trials."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from prove.fast_adapter.contact_skills import ContactSkill,STEPS
from tools.fast_adapter.p40_evaluate_v02 import ROOT,js,sha

NAMES={'cup':'杯柄抓持并抬升','hammer':'锤头接触推进物块','insert':'物块直线插入槽口'}
LABELS={'base':'冻结基础控制','relation':'关系后果反馈','local':'局部DINO反馈','motion':'初态参照的变化DINO','fused':'局部DINO＋关系','history':'History'}

def font(size):
    path=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK SC'],text=True);return ImageFont.truetype(path,size)

def source(task,split,kind,arm):
    p=ROOT/task/'final'/kind/('seed0' if kind=='references' else 'seed11')/split
    paths=[p/(arm+'_RUNTIME.pt'),p/(arm+'_SEAL.json'),p/(arm+'_RESULT.json')]
    return dict(raw=torch.load(paths[0],weights_only=True),seal=json.loads(paths[1].read_text()),result=json.loads(paths[2].read_text()),paths=paths,kind=kind,arm=arm)

def progress(task,env,q):
    if task=='cup':return float(q[2]-env.initial_object[2])
    if task=='hammer':return float(q[env.model.joint('object_slide').qposadr[0]])
    return -float(q[env.zj])

def final_label(task,row):
    if not row['observation_valid']:return '必需观测缺失，中止'
    if row['success']:return '成功'
    if task=='cup':return '抬升不足' if row['progress']<=.028 else '未保持完整抓持条件'
    if task=='hammer':
        if row['progress']<=.015:return '推进不足'
        if row['progress']>=.021:return '推进过量'
        if row['speed']>=.008:return '尚未停稳'
        return '未满足完整接触判据'
    return '插入深度不足' if row['progress']<=.042 else '横向未对齐'

def render(task,split,index,panels,name,explanation,out):
    sources=[source(task,split,k,a) for k,a,_ in panels];envs=[ContactSkill(s['seal']['configs'][index]) for s in sources]
    cols=3;rows=(len(panels)+2)//3;w=cols*448;rh=545;h=116+rh*rows+66;h+=h%2;events=[]
    video=out/(name+'.mp4');big,mid,small=font(28),font(22),font(18)
    proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','10','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    try:
        for t in range(STEPS+1):
            canvas=Image.new('RGB',(w,h),(245,248,252));draw=ImageDraw.Draw(canvas)
            draw.text((16,8),NAMES[task]+'：交互反馈如何改变用力',font=big,fill='#183746')
            draw.text((16,48),explanation,font=small,fill='#415e6f')
            stage='共同准备与探测' if t<12 else ('使用已观察反馈控制' if t<STEPS else '封存后的最终评价')
            draw.text((16,79),f'{split} · 初态 {index} · 已执行 {t}/{STEPS} 步 · 仿真 {t*.04:.2f}s · {stage}',font=small,fill='#415e6f')
            for j,(s,e,panel) in enumerate(zip(sources,envs,panels)):
                o,rgb,masks=e.public();raw=s['raw'];x=(j%cols)*448;y=116+(j//cols)*rh
                error=float(np.max(np.abs(e.data.qpos-raw['qpos'][index,t].numpy())));assert error<1e-8
                rgb_match=hashlib.sha256(rgb.tobytes()).hexdigest()==raw['rgb_hashes'][t][index]
                mask_match=hashlib.sha256(masks.tobytes()).hexdigest()==raw['mask_hashes'][t][index]
                draw.text((x+10,y+3),panel[2],font=mid,fill='#183746')
                im=Image.fromarray(rgb);pd=ImageDraw.Draw(im)
                for box,visible,color in zip(o['boxes'],o['visible'],('#3a9fd2','#c756a6','#e2af4f')):
                    if visible:pd.rectangle(tuple(map(int,box)),outline=color,width=2)
                command=raw['actions'][index,min(t,STEPS-1)].numpy()
                if task=='cup':activity='对准把手并夹紧' if t<8 else '试提：观察杯子是否跟随' if t<75 else '保持抓持，检查是否滑落'
                elif task=='hammer':activity='接触推进：观察木块响应' if command[-1]>0 else '停止加力，等待停稳'
                else:activity='对准槽口或停止推进' if command[-1]==0 else '接触插入：观察推进与卡滞'
                pd.rectangle((0,420,448,448),fill='#183746');pd.text((8,423),activity,font=small,fill='white')
                canvas.paste(im,(x,y+34));metric=progress(task,e,raw['qpos'][index,t].numpy())*1000
                text=('抬升' if task=='cup' else '推进' if task=='hammer' else '插入')+f' {metric:.1f} mm'
                if t==STEPS:
                    row=s['result']['rows'][index];status=final_label(task,row)
                    text=status+' · '+text
                draw.text((x+10,y+487),text,font=small,fill='#235b4b' if t==STEPS and s['result']['rows'][index]['success'] else '#415e6f')
                force=float(raw['actions'][index,min(t,STEPS-1),-1]);label='单侧夹持力' if task=='cup' else '向下作用力'
                draw.text((x+10,y+512),f'{label} {force:.2f} N',font=small,fill='#415e6f')
                events.append(dict(step=t,panel=j,pose_error=error,rgb_match=rgb_match,mask_match=mask_match,stage=stage,activity=activity,display=text,force=force))
                if t<STEPS and not raw['aborted'][index,t]:e.step(raw['actions'][index,t].numpy())
            note='自由刚体杯子；真实双指摩擦抓持。' if task=='cup' else '已持工具由平移台驱动；物块沿导向运动。' if task=='hammer' else '平移台持有物块；被槽遮挡后仍可读取工具本体位移。'
            draw.text((16,h-56),note+' 采用模拟器可见mask，非完整机械臂任务。',font=small,fill='#526878')
            draw.text((16,h-29),'前12步动作相同；成功率以全量测试为准。框：蓝 Actor／紫 Object／黄 Goal 或功能部位。',font=small,fill='#526878')
            payload=np.asarray(canvas).tobytes();proc.stdin.write(payload)
            if t in (0,25,60,100):canvas.save(out/f'{name}_step{t}.jpg',quality=92)
            if t==STEPS:
                canvas.save(out/(name+'.jpg'),quality=92)
                for _ in range(15):proc.stdin.write(payload)
        proc.stdin.close();assert proc.wait()==0
    finally:
        for e in envs:e.close()
    bindings=[dict(path=str(p),sha256=sha(p)) for s in sources for p in s['paths']]
    js(out/(name+'_ANNOTATIONS.json'),dict(schema='prove.episode-annotations.v1',task=task,split=split,index=index,selection='First deterministic qualifying illustration, seed11; not a success-rate sample selection',source_bindings=bindings,panels=panels,
        segments=[dict(start_step=0,end_step=11,stage='共同探测',detail='同初态各方法控制动作一致',sources=bindings),dict(start_step=12,end_step=99,stage='反馈执行',detail=explanation,sources=bindings),dict(start_step=100,end_step=100,stage='最终评价',detail='只在最后展示封存后的成功标签',sources=bindings)],frames=events,video=str(video),video_sha256=sha(video),renderer_source=dict(path=__file__,sha256=sha(__file__)),rgb_exact=sum(x['rgb_match'] for x in events),mask_exact=sum(x['mask_match'] for x in events),frames_checked=len(events)))

def main():
    out=ROOT/'summary/media_v01';out.mkdir(parents=True,exist_ok=False);freeze=json.loads((ROOT/'FINAL_FREEZE.json').read_text());catalog=[]
    for task in NAMES:
        fixed=freeze['strong_fixed'][task];selected=None
        for split in ('joint','physical','paired','train'):
            relation=source(task,split,'relation','continuous')['result']['rows'];base=source(task,split,'references','base')['result']['rows'];strong=source(task,split,'references',fixed)['result']['rows']
            ids=[i for i,r in enumerate(relation) if r['success'] and not strong[i]['success']]
            if not ids:ids=[i for i,r in enumerate(relation) if r['success'] and not base[i]['success']]
            if ids:selected=(split,ids[0]);break
        split,index=selected or ('joint',0)
        panels=[('references','base','冻结基础控制'),('references',fixed,'开发选定固定力度'),('relation','continuous','关系后果反馈'),('local','continuous','局部DINO反馈'),('motion','continuous','初态参照的变化DINO'),('fused','continuous','局部DINO＋关系')]
        name=task+'_representations';render(task,split,index,panels,name,'同一初态、相同动作预算；只比较如何从已发生的交互调整后续用力。',out);catalog.append(dict(task=task,name=name,split=split,index=index))
        rel=source(task,'paired','relation','continuous')['result']['rows'];donor=source(task,'paired','relation','donor')['result']['rows'];mean=source(task,'paired','relation','mean')['result']['rows']
        ids=[i for i,r in enumerate(rel) if r['success'] and not donor[i]['success'] and not mean[i]['success']]
        if not ids:ids=[i for i,r in enumerate(rel) if r['success'] and not donor[i]['success']]
        if ids:
            name=task+'_feedback';panels=[('relation','mean','固定训练平均z'),('relation','donor','相反物理条件的反馈'),('relation','continuous','当前物体自己的反馈')]
            render(task,'paired',ids[0],panels,name,'外观相同、隐藏物理条件不同；检查用错交互经验会怎样。',out);catalog.append(dict(task=task,name=name,split='paired',index=ids[0]))
        if task=='cup':
            rows=source(task,'joint','relation','continuous')['result']['rows'];failures=[i for i,r in enumerate(rows) if not r['success']]
            if failures:
                name=task+'_retained_failure';render(task,'joint',failures[0],[('references','base','冻结基础控制'),('relation','continuous','关系后果反馈'),('motion','continuous','变化DINO反馈')],name,'保留的关系适配失败例；不只展示成功案例。',out);catalog.append(dict(task=task,name=name,split='joint',index=failures[0]))
    js(out/'VIDEO_INDEX.json',dict(videos=catalog,scope='Selected post-seal illustrations only'))

if __name__=='__main__':torch.set_num_threads(1);main()
