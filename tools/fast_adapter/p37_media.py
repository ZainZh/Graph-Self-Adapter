"""A few annotated P37 examples, reconstructed after full trajectory sealing."""
from pathlib import Path
import json
import subprocess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tools.fast_adapter.p33_common import dump, digest
from prove.fast_adapter.object_interaction_env import ObjectEnv, mujoco
from prove.fast_adapter.held_hammer_env import HeldHammerEnv

ROOT=Path('evaluations/fast_adapter_p37_full_success_v01')


def main():
    torch.set_num_threads(2)
    data=json.loads((ROOT/'summary/SUMMARY.json').read_text());rows=data['rows']
    lookup={(r['task'],r['split'],r['kind'],r['representation'],r['seed'],r['mode'],r['condition']):r for r in rows}
    candidates=[r for r in rows if r['kind']=='p35' and r['representation']=='g448w' and r['seed']==11 and r['mode']=='continuous' and r['source_row'] is not None]
    out=ROOT/'summary/media';out.mkdir(parents=True,exist_ok=True)
    fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title=ImageFont.truetype(fp,24);font=ImageFont.truetype(fp,20);small=ImageFont.truetype(fp,17)
    videos=[]
    for task in ('mug-handle-lift','hammer-block'):
        cc=sorted([r for r in candidates if r['task']==task],key=lambda r:(r['split']!='geometry_ood',r['condition']))
        chosen=[]
        for target in ('rescue','harm','base_harm','success','failure'):
            for r in cc:
                fixed=lookup[task,r['split'],'p35','g448w',11,'fixed',r['condition']]
                baseline=lookup[task,r['split'],'base','base',None,'fixed',r['condition']]
                qualifies={'rescue':r['success'] and not fixed['success'],'harm':fixed['success'] and not r['success'],
                           'base_harm':baseline['success'] and not r['success'],'success':r['success'],'failure':not r['success']}[target]
                if qualifies and (r['split'],r['condition']) not in [(x['split'],x['condition']) for _,x in chosen]:
                    chosen.append((target,r));break
            if len(chosen)==2:break
        for reason,r in chosen:
            split=r['split'];c=r['condition']
            rr=[lookup[task,split,'base','base',None,'fixed',c],lookup[task,split,'p35','g448w',11,'fixed',c],r]
            blobs=[torch.load(x['source']['path'],weights_only=True,map_location='cpu') for x in rr]
            metrics=[json.loads((Path(x['source']['path']).parent/'OUTCOMES.json').read_text())['metrics'][x['source_row']] for x in rr]
            assert all(x['source_row'] is not None for x in rr)
            cls=ObjectEnv if task=='mug-handle-lift' else HeldHammerEnv
            env=cls(r['config']);env.reset(r['reset_seed'])
            path=out/f'{task}_{split}_{reason}_s11_c{c}.mp4';w,h=1440,730
            proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
                '-video_size',f'{w}x{h}','-framerate','15','-i','-','-c:v','libx264','-threads','2','-crf','22',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
            try:
                for t in range(201):
                    im=Image.new('RGB',(w,h),(20,28,39));d=ImageDraw.Draw(im)
                    name='抓杯柄并提起' if task=='mug-handle-lift' else '持锤推击木块'
                    setting='范围内新物体' if split=='train' else '几何范围外新物体'
                    d.text((14,10),f'P37 {name}｜{setting}｜条件{c}｜执行步 {t}/200',font=title,fill='white')
                    d.text((14,48),'同初态完整回合；固定z与持续更新使用相同的448双视角关系模型。原基座始终用全局224。',font=small,fill=(202,218,234))
                    d.text((14,77),'以下为统一全局视角的封存状态重绘；接触／成功注释只来自事后评价，不输入控制器。',font=small,fill=(226,209,167))
                    for col,(row,b,mm) in enumerate(zip(rr,blobs,metrics)):
                        x=col*480;j=row['source_row']
                        d.text((x+14,114),['原冻结视觉基座','448双视角：固定z','448双视角：持续更新z'][col],font=title,fill=(122,222,191))
                        for k,v in b['states'].items():getattr(env.env.data,k)[:]=v[j,t].numpy()
                        mujoco.mj_forward(env.env.model,env.env.data)
                        im.paste(Image.fromarray(env.rgb()).resize((448,448)),(x+16,153))
                        count=b.get('update_counts',[0]*200)[t] if t<200 else b.get('updates',0)
                        d.text((x+14,610),f'梯度更新：{count} 次',font=font,fill='white')
                        if t:
                            m=mm[t-1]
                            if task=='mug-handle-lift':
                                stage='保持成功条件' if m['success'] else '双侧抓持，正在抬升／保持' if m['bilateral_handle'] else '接近／调整，未建立双侧抓持'
                            else:
                                stage='保持成功条件' if m['success'] else '锤头接触木块' if m['head_target_contact'] else '持锤接近／调整'
                                if not m['bilateral_handle']:stage='工具双侧抓持未保持'
                            d.text((x+14,642),stage,font=font,fill=(210,222,233))
                        if t==200:
                            msg='回合成功，末步达标' if row['terminal_success'] else '回合曾成功，末步未达标' if row['success'] else '本回合未成功'
                            d.text((x+14,681),msg,font=font,fill=(136,228,159) if row['success'] else (255,155,122))
                    proc.stdin.write(np.asarray(im).tobytes())
                    if t==200:im.save(path.with_suffix('.jpg'),quality=85)
                proc.stdin.close();assert proc.wait()==0
            finally:
                env.close()
                if proc.poll() is None:proc.terminate();proc.wait()
            videos.append(dict(task=task,split=split,condition=c,reason=reason,file=str(path),cover=str(path.with_suffix('.jpg')),
                sha256=digest(path),outcomes=[x['success'] for x in rr],sources=[x['source'] for x in rr]))
            print('VIDEO_COMPLETE',path,flush=True)
    dump(out/'VIDEO_INDEX.json',dict(videos=videos,state_reconstruction_no_new_trials=True,
        selection='P35 g448w seed11 only. Up to two/task: first fixed-to-continuous rescue, harm, base-to-continuous harm, success, failure; OOD then ID. Illustrative post-outcome selection, not average evidence.',
        generator=dict(path=str(Path(__file__)),sha256=digest(Path(__file__))),summary_sha256=digest(ROOT/'summary/SUMMARY.json')))


if __name__=='__main__':main()
