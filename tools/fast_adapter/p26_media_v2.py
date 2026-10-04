"""Display-only annotation correction; same two episodes as sealed P26 media.
Original renderer and media remain unchanged. No new rollout or case selection.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw, ImageFont
from prove.fast_adapter.geometry_insert import GeometryInsert, geometry, domain
from tools.fast_adapter.p26_common import R, SEEDS
from tools.fast_adapter.p18_run import save_json, digest


def main():
    out=R/'summary/media_v2';out.mkdir(parents=True,exist_ok=False)
    choices={}; fallback=[]; names={'id':'同分布','position':'槽位置变化','scale':'整体尺度变化'}
    for st in names:
        m=json.loads((R/'development'/st/'RESULT.json').read_text())['metrics']
        base=np.array(m['base']['success'])
        for seed in SEEDS:
            physical=np.array(m[f'refresh_correct_seed{seed}']['success'])
            for kind,mask in [('rescue',~base&physical),('harm',base&~physical)]:
                ids=np.flatnonzero(mask)
                if len(ids) and kind not in choices:choices[kind]=(st,seed,int(ids[0]),kind)
            for kind,mask in [('success',physical),('failure',~physical)]:
                ids=np.flatnonzero(mask)
                if len(ids):fallback.append((st,seed,int(ids[0]),kind))
    selected=list(choices.values())
    for item in fallback:
        if len(selected)>=2:break
        if item[:3] not in [x[:3] for x in selected]:selected.append(item)
    fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=ImageFont.truetype(fontpath,21);small=ImageFont.truetype(fontpath,17)
    width,height=1344,574;index=[]
    for st,seed,i,kind in selected:
        root=R/'development'/st;metrics=json.loads((root/'RESULT.json').read_text())['metrics']
        arms={'base':'冻结视觉基座',f'parent_correct_seed{seed}':'原P25位移版本',
              f'relabel_correct_seed{seed}':'修正训练标签',f'refresh_correct_seed{seed}':'修正标签＋当前轨迹'}
        paths={a:root/f'{a}_RAW.pt' for a in arms}
        raw={a:torch.load(p,weights_only=False) for a,p in paths.items()}
        support=torch.load(root/'SUPPORT_SEALED.pt',weights_only=False)
        ss=support['seed_base'];cfg=geometry(ss+i,st);env=GeometryInsert(cfg);env.set_domain(*domain(ss+100000+i))
        filename=out/f'{st}_{kind}_seed{seed}_condition{i}.mp4'
        proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
             '-framerate','8','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(filename)],stdin=subprocess.PIPE)
        indices=support['selected_indices'][i].tolist()
        for t in range(33):
            im=Image.new('RGB',(width,height),(23,31,38));d=ImageDraw.Draw(im)
            d.text((12,8),f'{names[st]}｜开发样本{i}｜共同支持采集 {t}/32 步',font=font,fill='white')
            d.text((12,42),'由同一个冻结基座执行；窗口结束后才使用该窗口选中的反馈。槽位置和接触真值不提供给选择器。',font=small,fill=(190,215,230))
            env.data.qpos[:]=support['positions'][i,t];env.data.qvel[:]=support['velocities'][i,t]
            mujoco.mj_forward(env.model,env.data);im.paste(Image.fromarray(env.rgb()).resize((448,448)),(448,100))
            d.text((16,130),'三个反馈来源（事后标记）',font=font,fill=(175,232,212))
            for j,step in enumerate(indices):
                end=(8,24,32)[j];text=f'窗口{j+1}：动作{step} → 反馈{step+1}'
                d.text((16,183+j*85),text,font=small,fill='white')
                status='已到窗口末，可更新' if t>=end else '仍在采集'
                d.text((16,211+j*85),status,font=small,fill=(165,195,211))
            if t in [step+1 for step in indices]:d.text((920,160),'这条反馈被选中',font=font,fill=(255,209,105))
            d.text((920,230),'按实际位移与动作',font=font,fill='white')
            d.text((920,267),'的响应偏差选择',font=font,fill='white')
            proc.stdin.write(np.asarray(im).tobytes())
        for t in range(49):
            im=Image.new('RGB',(width,height),(23,31,38));d=ImageDraw.Draw(im)
            d.text((12,8),f'{names[st]}｜开发样本{i}｜独立初态执行 {t}/48 步',font=font,fill='white')
            d.text((12,42),'已用32步采集、3条反馈更新；本阶段 z 固定。位置误差和接触为事后注释。末尾连续三帧合格才成功。',font=small,fill=(190,215,230))
            for col,(arm,title) in enumerate(arms.items()):
                r=raw[arm];env.data.qpos[:]=r['positions'][i,t];env.data.qvel[:]=r['velocities'][i,t]
                mujoco.mj_forward(env.model,env.data);facts=env.facts();x=col*336
                d.text((x+10,80),title,font=font,fill=(175,232,212));im.paste(Image.fromarray(env.rgb()).resize((336,336)),(x,110))
                state='已插入，正在保持' if facts['inserted'] else '已对齐，下插未到位' if facts['inside_xy'] and facts['bottom_z']>.013*cfg['scale'] else '已对齐，深度或接触超限' if facts['inside_xy'] else '槽壁接触，尚未对齐' if facts['contact_count'] else '接近或对齐'
                d.text((x+10,450),state,font=small,fill='white')
                error=np.linalg.norm(r['positions'][i,t,:2]-np.asarray(cfg['xy']))*1000
                d.text((x+10,480),f'水平差 {error:.1f} mm｜底高 {facts["bottom_z"]*1000:.1f} mm',font=small,fill=(190,215,230))
                d.text((x+10,506),f'底高合格范围 {6.5*cfg["scale"]:.1f}–{13*cfg["scale"]:.1f} mm',font=small,fill=(190,215,230))
                if t==48:
                    ok=metrics[arm]['success'][i]
                    d.text((x+10,538),'最终：'+('成功' if ok else '失败'),font=font,fill=(120,230,165) if ok else (255,160,125))
            proc.stdin.write(np.asarray(im).tobytes())
            if t==48:im.save(filename.with_suffix('.png'))
        proc.stdin.close();assert proc.wait()==0;env.close()
        sources=[root/'SUPPORT_SEALED.pt',*paths.values()]
        index.append({'stratum':st,'seed':seed,'condition':i,'kind':kind,'file':str(filename),'sha256':digest(filename),
            'sources':[{'path':str(p),'sha256':digest(p)} for p in sources],
            'outcomes':{a:metrics[a]['success'][i] for a in arms},'selected_indices':indices,
            'display_only':'Re-rendering saved states; no new physics steps or inference. Both support and query shown.'})
    save_json(out/'VIDEO_INDEX.json',{'supersedes_index':str(R/'summary/media/VIDEO_INDEX.json'),'supersedes_sha256':digest(R/'summary/media/VIDEO_INDEX.json'),'annotation_revision':'Explain insertion depth and holding, unchanged episodes/frames/outcomes. Display only.','selection':'At most two total: first physical rescue and harm versus base, across ordered strata/seeds; fill missing categories with first success/failure. All trials included in statistics.','videos':index})
    print('VIDEOS_COMPLETE',len(index),flush=True)


if __name__=='__main__':main()
