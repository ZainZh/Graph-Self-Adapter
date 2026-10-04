"""Two annotated matched examples from already sealed P31 state records."""
import subprocess
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from tools.fast_adapter.p31v2_common import *
from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco

ARMS=['base','old_full','old_stage','own_full','own_stage','own_stage_first3']
LABELS=['冻结基座','旧材料 · 全程','旧材料 · 关键阶段','自身轨迹 · 全程','自身轨迹 · 关键阶段','自身轨迹 · 仅3次']


def main():
    out=ROOT/'summary/media';out.mkdir(parents=True,exist_ok=False)
    allrows=[]
    for seed in (11,22):allrows+=json.loads((ROOT/'development'/f'seed{seed}'/'RESULT.json').read_text())['rows']
    lookup={(r['seed'],r['split'],r['condition'],r['arm']):r for r in allrows};choices={};fallback=[]
    for split in ('geometry_ood','train'):
        for seed in (11,22):
            for c in range(16):
                b=lookup[seed,split,c,'base']['success'];g=lookup[seed,split,c,'own_stage']['success'];item=(seed,split,c)
                if g and not b:choices.setdefault('rescue',item)
                if b and not g:choices.setdefault('harm',item)
                fallback.append(('success' if g else 'failure',item))
    selected=list(choices.items())
    for x in fallback:
        if len(selected)>=2:break
        if x[1] not in [y[1] for y in selected]:selected.append(x)
    fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title=ImageFont.truetype(fontpath,23);font=ImageFont.truetype(fontpath,17);videos=[]
    for kind,(seed,split,c) in selected:
        rr={a:lookup[seed,split,c,a] for a in ARMS};raw={a:torch.load(rr[a]['source']['path'],weights_only=True) for a in ARMS}
        meta={a:json.loads(Path(rr[a]['source']['path']).with_suffix('.json').read_text()) for a in ARMS}
        j=c%8;env=ObjectEnv(rr['base']['config']);env.reset(rr['base']['reset_seed'])
        file=out/f'cup_{split}_{kind}_s{seed}_c{c}.mp4';w,h=1680,590
        proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}',
             '-framerate','12','-i','-','-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(file)],stdin=subprocess.PIPE)
        try:
            for t in range(201):
                im=Image.new('RGB',(w,h),(21,30,39));d=ImageDraw.Draw(im)
                d.text((12,9),f'P31 杯子把手抓取｜'+('OOD新几何' if split=='geometry_ood' else 'ID新物体')+f'｜执行步 {t}/200｜种子{seed} · 条件{c}',font=title,fill='white')
                d.text((12,48),'同一初态配对；接触/成功注释由封存后评价提供，不进入控制。演示由已存状态重绘，未新增rollout。',font=font,fill=(190,210,220))
                for col,a in enumerate(ARMS):
                    x=col*280;d.text((x+8,89),LABELS[col],font=font,fill=(146,227,207))
                    for key,states in raw[a]['states'].items():getattr(env.env.data,key)[:]=states[j,t].numpy()
                    mujoco.mj_forward(env.env.model,env.env.data);im.paste(Image.fromarray(env.rgb()).resize((280,280)),(x,120))
                    count=0 if a=='base' else int(raw[a]['counts_before_action'][j,t]) if t<200 else int(raw[a]['updates'][j])
                    d.text((x+8,411),f'z已更新 {count} 次',font=font,fill='white')
                    idx=min(t,199);eligible=bool(raw[a]['active'][j,idx])
                    phase='基础动作' if a=='base' else '全程允许适配' if '_stage' not in a else '识别到关键交互' if eligible else '非关键阶段：暂停修正'
                    if a.endswith('_first3') and count>=3:phase='已更新3次：保持当前z'
                    d.text((x+8,440),phase,font=font,fill=(190,213,233))
                    if t:
                        m=meta[a]['metrics'][j][t-1]
                        stage='抬升保持达标' if m['success'] else '已夹住把手，抬升/保持' if m['bilateral_handle'] else '尚未双侧夹住把手'
                        d.text((x+8,469),stage,font=font,fill='white')
                    if t==200:d.text((x+8,508),'回合'+('成功' if rr[a]['success'] else '失败'),font=title,fill=(128,229,160) if rr[a]['success'] else (255,160,120))
                proc.stdin.write(np.asarray(im).tobytes())
                if t==200:im.save(file.with_suffix('.png'))
            proc.stdin.close();assert proc.wait()==0
        finally:
            env.close()
            if proc.poll() is None:proc.terminate();proc.wait()
        videos.append({'task':TASK,'kind':kind,'seed':seed,'split':split,'condition':c,'file':str(file),'sha256':digest(file),
             'sources':bindings(sorted({Path(rr[a]['source']['path']) for a in ARMS})),
             'outcomes':{a:rr[a]['success'] for a in ARMS},'frames':201})
        print('VIDEO_COMPLETE',file,flush=True)
    dump(out/'VIDEO_INDEX.json',{'selection':'first own_stage rescue and harm vs base, OOD then ID; fill first success/failure if needed; illustrative only','videos':videos})


if __name__=='__main__':main()
