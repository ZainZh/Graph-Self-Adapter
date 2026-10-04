"""Two descriptive annotated episodes; statistics retain all preregistered runs."""
import json,subprocess
from pathlib import Path
import numpy as np
import torch,mujoco
from PIL import Image,ImageDraw,ImageFont
from prove.fast_adapter.geometry_insert import GeometryInsert,geometry,domain
from tools.fast_adapter.p27_common import R,SEEDS,STRATA,BLOCK
from tools.fast_adapter.p18_run import save_json,digest


def main():
    out=R/'summary/media';out.mkdir(parents=True,exist_ok=False);choices={};fallback=[]
    for st in STRATA:
        m=json.loads((R/'evaluation'/st/'RESULT.json').read_text())['metrics'];base=np.array(m['base']['success'])
        for seed in SEEDS:
            c=np.array(m[f'refresh_correct_seed{seed}']['success']);fixed=np.array(m[f'refresh_matchedmean_seed{seed}']['success'])
            for kind,mask in [('feedback_rescue',c&~fixed),('base_harm',base&~c)]:
                ids=np.flatnonzero(mask)
                if len(ids) and kind not in choices:choices[kind]=(st,seed,int(ids[0]),kind)
            for kind,mask in [('success',c),('failure',~c)]:
                ids=np.flatnonzero(mask)
                if len(ids):fallback.append((st,seed,int(ids[0]),kind))
    selected=list(choices.values())
    for item in fallback:
        if len(selected)>=2:break
        if item[:3] not in [v[:3] for v in selected]:selected.append(item)
    names={'id':'同分布','position':'槽位置变化','scale':'整体尺度变化'};index=[]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',21);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
    for st,seed,i,kind in selected:
        folder=R/'evaluation'/st;m=json.loads((folder/'RESULT.json').read_text())['metrics'];block=i//BLOCK*BLOCK;j=i%BLOCK
        arms={'base':'基座／不更新z',f'refresh_matchedmean_seed{seed}':'固定训练平均z',f'refresh_donor1_seed{seed}':'错配反馈z',f'refresh_correct_seed{seed}':'本环境梯度更新z'}
        paths={arm:folder/f'{arm}_block{block:03d}_RAW.pt' for arm in arms};raw={arm:torch.load(p,weights_only=False) for arm,p in paths.items()}
        sp=folder/'SUPPORT_correct_SEALED.pt';support=torch.load(sp,weights_only=False);ss=support['seed_base'];idx=support['selected_indices'][i].tolist()
        cfg=geometry(ss+i,st);e=GeometryInsert(cfg);e.set_domain(*domain(ss+100000+i))
        file=out/f'{st}_{kind}_seed{seed}_condition{i}.mp4';width,height=1344,574
        proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','8','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(file)],stdin=subprocess.PIPE)
        for t in range(33):
            im=Image.new('RGB',(width,height),(23,31,38));d=ImageDraw.Draw(im)
            d.text((12,8),f'{names[st]}｜条件{i}｜本环境支持采集 {t}/32',font=font,fill='white')
            d.text((12,42),'主方法读取本环境3条反馈；固定平均z来自训练材料，错配z来自独立同档环境。普通控制不读槽真值。',font=small,fill=(190,215,230))
            e.data.qpos[:]=support['positions'][i,t];e.data.qvel[:]=support['velocities'][i,t];mujoco.mj_forward(e.model,e.data)
            im.paste(Image.fromarray(e.rgb()).resize((448,448)),(448,100))
            for k,step in enumerate(idx):
                d.text((16,150+k*92),f'窗口{k+1}：动作{step} → 反馈{step+1}',font=small,fill='white')
                d.text((16,181+k*92),'可更新z' if t>=(8,24,32)[k] else '仍在采集',font=small,fill=(175,232,212))
            d.text((930,160),'网络权重始终冻结',font=font,fill='white');d.text((930,205),'主方法用当前环境z',font=font,fill='white')
            proc.stdin.write(np.asarray(im).tobytes())
        for t in range(49):
            im=Image.new('RGB',(width,height),(23,31,38));d=ImageDraw.Draw(im)
            d.text((12,8),f'{names[st]}｜条件{i}｜相同独立起点执行 {t}/48',font=font,fill='white')
            d.text((12,42),'本阶段z固定；后三列使用同一个动作头，只改变z的来源。水平差和底高为事后注释；末三帧合格才成功。',font=small,fill=(190,215,230))
            for col,(arm,label) in enumerate(arms.items()):
                r=raw[arm];e.data.qpos[:]=r['positions'][j,t];e.data.qvel[:]=r['velocities'][j,t];mujoco.mj_forward(e.model,e.data);f=e.facts();x=col*336
                d.text((x+10,80),label,font=font,fill=(175,232,212));im.paste(Image.fromarray(e.rgb()).resize((336,336)),(x,110))
                state='已插入，正在保持' if f['inserted'] else '已对齐，下插未到位' if f['inside_xy'] and f['bottom_z']>.013*cfg['scale'] else '已对齐，深度或接触超限' if f['inside_xy'] else '槽壁接触，尚未对齐' if f['contact_count'] else '接近或对齐'
                err=np.linalg.norm(r['positions'][j,t,:2]-np.array(cfg['xy']))*1000
                d.text((x+10,450),state,font=small,fill='white');d.text((x+10,480),f'水平差 {err:.1f} mm｜底高 {1000*f["bottom_z"]:.1f} mm',font=small,fill=(190,215,230))
                d.text((x+10,506),f'底高合格范围 {6.5*cfg["scale"]:.1f}–{13*cfg["scale"]:.1f} mm',font=small,fill=(190,215,230))
                if t==48:
                    ok=m[arm]['success'][i];d.text((x+10,538),'最终：'+('成功' if ok else '失败'),font=font,fill=(120,230,165) if ok else (255,160,125))
            proc.stdin.write(np.asarray(im).tobytes())
            if t==48:im.save(file.with_suffix('.png'))
        proc.stdin.close();assert proc.wait()==0;e.close()
        index.append({'stratum':st,'seed':seed,'condition':i,'kind':kind,'file':str(file),'sha256':digest(file),
            'sources':[{'path':str(p),'sha256':digest(p)} for p in [sp,*paths.values()]],'outcomes':{arm:m[arm]['success'][i] for arm in arms},
            'display_only':'Re-rendered sealed states,82 frames; no new physics or model inference. Descriptive selected cases, not success-rate evidence.'})
    save_json(out/'VIDEO_INDEX.json',{'selection':'First correct-feedback rescue vs matchedmean and first harm vs base in ordered strata/seeds; missing categories filled by first success/failure. At most2 episodes; all runs in statistics.','videos':index})
    print('VIDEOS_COMPLETE',len(index),flush=True)

if __name__=='__main__':main()
