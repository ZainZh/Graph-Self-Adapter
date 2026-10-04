"""At most two annotated, state-reconstructed examples; no new rollouts."""
import subprocess
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from tools.fast_adapter.p32_common import *
from prove.fast_adapter.object_interaction_env import ObjectEnv,mujoco

ARMS=['base','predictive','learned','learned_first3','learned_donor']
LABELS=['冻结基座','预测损失 · 持续','学到损失 · 持续','学到损失 · 仅3次','学到损失 · 其他物体']


def main():
    out=ROOT/'summary/media';out.mkdir(exist_ok=False)
    rr=json.loads((ROOT/'development/RESULT.json').read_text())['rows']
    lookup={(r['seed'],r['split'],r['condition'],r['arm']):r for r in rr};choices={};fallback=[]
    for split in ('geometry_ood','train'):
        for seed in (11,22):
            for c in range(8):
                b=lookup[None,split,c,'base']['success'];a=lookup[seed,split,c,'learned']['success'];item=(seed,split,c)
                if a and not b:choices.setdefault('rescue',item)
                if b and not a:choices.setdefault('harm',item)
                fallback.append(('success' if a else 'failure',item))
    selected=list(choices.items())
    for item in fallback:
        if len(selected)>=2:break
        if item[1] not in [x[1] for x in selected]:selected.append(item)
    fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';title=ImageFont.truetype(fp,22);font=ImageFont.truetype(fp,17)
    videos=[]
    for kind,(seed,split,c) in selected:
        rows={a:lookup[None if a=='base' else seed,split,c,a] for a in ARMS}
        raw={a:torch.load(rows[a]['source']['path'],weights_only=True) for a in ARMS}
        meta={a:json.loads(Path(rows[a]['source']['path']).with_suffix('.json').read_text()) for a in ARMS}
        env=ObjectEnv(rows['base']['config']);env.reset(rows['base']['reset_seed'])
        file=out/f'cup_{split}_{kind}_s{seed}_c{c}.mp4';w,h=1600,590
        proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}',
                  '-framerate','12','-i','-','-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(file)],stdin=subprocess.PIPE)
        try:
            for t in range(201):
                im=Image.new('RGB',(w,h),(21,30,39));draw=ImageDraw.Draw(im)
                draw.text((12,9),f'P32 杯柄抓取｜'+('几何范围外' if split=='geometry_ood' else '同分布新物体')+f'｜执行步 {t}/200｜种子{seed} · 条件{c}',font=title,fill='white')
                draw.text((12,48),'同初态配对；接触/成功注释在记录封存后添加，不进入控制。由已存状态重绘，无新增任务回合。',font=font,fill=(190,210,220))
                for col,a in enumerate(ARMS):
                    x=col*320;draw.text((x+8,87),LABELS[col],font=font,fill=(146,227,207))
                    for key,states in raw[a]['states'].items():getattr(env.env.data,key)[:]=states[c,t].numpy()
                    mujoco.mj_forward(env.env.model,env.env.data);im.paste(Image.fromarray(env.rgb()).resize((296,296)),(x+12,115))
                    count=0 if a=='base' else int(raw[a]['counts_before_action'][c,t]) if t<200 else int(raw[a]['updates'][c])
                    draw.text((x+8,419),f'z已更新 {count} 次',font=font,fill='white')
                    active=bool(raw[a]['active'][c,min(t,199)])
                    phase='基座执行' if a=='base' else '识别到交互：允许修正' if active else '阶段外：基座执行，暂停更新'
                    if a=='learned_first3' and count>=3:phase='已完成3次：z保持固定'
                    draw.text((x+8,448),phase,font=font,fill=(190,213,233))
                    if t:
                        m=meta[a]['metrics'][c][t-1]
                        stage='抬升保持达标' if m['success'] else '双侧把手接触，抬升/保持中' if m['bilateral_handle'] else '尚未双侧夹住把手'
                        draw.text((x+8,477),stage,font=font,fill='white')
                    if t==200:draw.text((x+8,517),'回合'+('成功' if rows[a]['success'] else '失败'),font=title,fill=(128,229,160) if rows[a]['success'] else (255,160,120))
                proc.stdin.write(np.asarray(im).tobytes())
                if t==200:im.save(file.with_suffix('.png'))
            proc.stdin.close();assert proc.wait()==0
        finally:
            env.close()
            if proc.poll() is None:proc.terminate();proc.wait()
        label={'rescue':'救回个例','harm':'误伤个例','success':'成功个例','failure':'失败个例'}[kind]
        videos.append({'kind':kind,'caption':f'{label} · {"几何范围外" if split=="geometry_ood" else "同分布"} · 种子{seed} 条件{c}',
             'seed':seed,'split':split,'condition':c,'file':str(file),'sha256':digest(file),'frames':201,
             'outcomes':{a:rows[a]['success'] for a in ARMS},'sources':bindings(sorted({Path(r['source']['path']) for r in rows.values()}))})
        print('VIDEO_COMPLETE',file,flush=True)
    dump(out/'VIDEO_INDEX.json',{'selection':'First learned/base rescue and harm, scanning OOD then ID,11 then22; fill success/failure if absent. Illustrative only.',
         'videos':videos,'generator_sha256':digest(Path(__file__))})


if __name__=='__main__':main()
