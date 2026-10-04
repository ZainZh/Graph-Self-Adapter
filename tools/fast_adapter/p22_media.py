"""Representative annotated videos, replaying sealed qpos/qvel without new rollouts."""
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,subprocess
import numpy as np
import torch,mujoco
from PIL import Image,ImageDraw,ImageFont
from prove.fast_adapter.geometry_insert import GeometryInsert,geometry
from tools.fast_adapter.p22_data import R
from tools.fast_adapter.p18_run import save_json,digest

def main():
    stage='final' if json.loads((R/'CAMPAIGN_DONE.json').read_text())['final_run'] else 'dev'
    out=R/'summary/media';out.mkdir(parents=True,exist_ok=False)
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    small=ImageFont.truetype(font_path,17);font=ImageFont.truetype(font_path,21)
    names={'base':'视觉＋本体基座','history':'History','u0':'同模型：不更新 z','u3':'同模型：更新 z'} if stage=='final' else {'base':'视觉＋本体基座','oracle':'真值参考控制'}
    strata={'id':'同分布','position':'槽位置变化','scale':'整体尺度变化'};index=[]
    for st,label in strata.items():
        root=R/stage/st;ref=json.loads((root/'seed0/RESULT.json').read_text())['metrics']
        choices=[]
        if stage=='final':
            for seed in (11,22,33):
                m=json.loads((root/f'seed{seed}/RESULT.json').read_text())['metrics']
                u0=np.array(m['u0']['success']);u3=np.array(m['u3']['success'])
                for kind,mask in [('update_win',~u0&u3),('update_harm',u0&~u3)]:
                    ids=np.flatnonzero(mask)
                    if len(ids) and not any(c[0]==kind for c in choices):choices.append((kind,seed,int(ids[0])))
            if not choices:
                m=json.loads((root/'seed11/RESULT.json').read_text())['metrics'];u3=np.array(m['u3']['success'])
                for kind,mask in [('success',u3),('failure',~u3)]:
                    ids=np.flatnonzero(mask)
                    if len(ids):choices.append((kind,11,int(ids[0])))
        else:
            for kind,mask in [('base_success',np.array(ref['base']['success'])),('base_failure',~np.array(ref['base']['success']))]:
                ids=np.flatnonzero(mask)
                if len(ids):choices.append((kind,0,int(ids[0])))
        for kind,seed,i in choices[:2]:
            paths={a:root/f'seed{0 if a in ("base","oracle") else seed}'/f'{a}_RAW.pt' for a in names}
            data={a:torch.load(p,weights_only=False) for a,p in paths.items()}
            results={a:json.loads(p.parent.joinpath('RESULT.json').read_text())['metrics'][a]['success'][i] for a,p in paths.items()}
            raw=next(iter(data.values()));cfg=geometry(raw['seed_base']+i,st);env=GeometryInsert(cfg)
            file=out/f'{st}_{kind}_seed{seed}_domain{i}.mp4';width=336*len(names);height=574
            proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
              '-framerate','8','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(file)],stdin=subprocess.PIPE)
            for t in range(49):
                im=Image.new('RGB',(width,height),(23,31,38));d=ImageDraw.Draw(im)
                d.text((12,8),f'{label}｜{"独立测试" if stage=="final" else "开发资格"}｜样本{i}｜步骤{t}/48',font=font,fill='white')
                d.text((12,41),'先共同探测5次，再从新初态执行；执行中 z 固定。误差为事后标注，模型未读取槽坐标。',font=small,fill=(185,205,220))
                for col,(a,title) in enumerate(names.items()):
                    r=data[a];pos=r['positions'][i,t];env.data.qpos[:]=pos;env.data.qvel[:]=r['velocities'][i,t]
                    mujoco.mj_forward(env.model,env.data);f=env.facts();x=col*336
                    im.paste(Image.fromarray(env.rgb()).resize((336,336)),(x,110));d.text((x+10,80),title,font=font,fill=(180,235,215))
                    error=np.linalg.norm(pos[:2]-np.asarray(cfg['xy']))*1000
                    state='已进入槽内' if f['inserted'] else '接触中、尚未插入' if f['contact_count'] else '接近或对齐中'
                    d.text((x+10,451),state,font=small,fill='white')
                    d.text((x+10,479),f'水平偏差 {error:.1f} mm｜接触 {f["contact_count"]}',font=small,fill=(205,216,226))
                    if t==48:d.text((x+10,514),'最终：'+('成功' if results[a] else '失败'),font=font,fill=(120,230,165) if results[a] else (255,160,125))
                    elif a not in ('base','oracle'):
                        delta=np.linalg.norm(r['actions'][i,t]-r['base'][i,t].numpy())
                        d.text((x+10,514),f'动作修正幅度 {delta:.3f}',font=small,fill=(205,216,226))
                proc.stdin.write(np.asarray(im).tobytes())
                if t==48:im.save(file.with_suffix('.png'))
            proc.stdin.close();assert proc.wait()==0;env.close()
            index.append({'stratum':st,'kind':kind,'seed':seed,'domain':i,'file':str(file),'sha256':digest(file),
              'outcomes':results,'sources':[{'path':str(p),'sha256':digest(p)} for p in paths.values()],
              'display_only':'No physics stepping or model inference; qpos/qvel replay only.'})
    save_json(out/'VIDEO_INDEX.json',{'selection':'At most two per stratum; first update win and harm across ordered seeds, otherwise first success/failure. All episodes in statistics.','videos':index})
    print('VIDEOS_COMPLETE',len(index),flush=True)

if __name__=='__main__':main()
