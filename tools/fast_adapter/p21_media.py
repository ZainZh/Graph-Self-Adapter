"""Annotated re-rendering of sealed query states, with no policy or physics rollout."""
from pathlib import Path
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import subprocess,json
import torch,numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from prove.fast_adapter.insert3d import Insert3D
from tools.fast_adapter.p18_run import save_json,digest
R=Path('evaluations/fast_adapter_p21_frontends_v01');SEEDS=[11,22,33]
def main():
    selected=json.loads((R/'diagnostic/RESULT.json').read_text())['selected']
    out=R/'summary/media';out.mkdir(parents=True,exist_ok=False)
    def path(st,seed,arm):return R/'frontends'/('last' if arm=='base' else selected)/'final'/st/f'seed{seed}'/f'{arm}_RAW.pt'
    def read(st,seed,arm):return torch.load(path(st,seed,arm),weights_only=False)
    samples=[]
    for kind in ['gradient_win','gradient_harm','base_win']:
        found=None
        for st in ['train','ood']:
            for seed in SEEDS:
                u0=read(st,seed,'hybrid_u0')['success'];u3=read(st,seed,'hybrid_u3')['success'];base=read(st,0,'base')['success']
                mask=(~u0&u3) if kind=='gradient_win' else (u0&~u3) if kind=='gradient_harm' else (base&~u3)
                ids=np.flatnonzero(mask)
                if len(ids):found=(kind,st,seed,int(ids[0]));break
            if found:break
        if found:samples.append(found)
    env=Insert3D();fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(fontpath,18);big=ImageFont.truetype(fontpath,22)
    names={'base':'基础控制','history':'LiLa + History','hybrid_u0':'LiLa + 冻结 z','hybrid_u3':'LiLa + 更新 z'}
    titles={'gradient_win':'梯度更新救回：冻结z失败，更新z成功','gradient_harm':'梯度更新反例：冻结z成功，更新z失败','base_win':'仍存在的差距：基础控制成功，学习修正失败'}
    index=[]
    for kind,st,seed,i in samples:
        data={arm:read(st,0 if arm=='base' else seed,arm) for arm in names}
        file=out/f'{kind}_{st}_seed{seed}_domain{i}.mp4'
        proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1344x560','-framerate','8','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(file)],stdin=subprocess.PIPE)
        for t in range(46):
            im=Image.new('RGB',(1344,560),(23,31,38));d=ImageDraw.Draw(im)
            d.text((14,8),f'{titles[kind]}｜{st.upper()} · 模型种子{seed} · 域{i}',font=big,fill='white')
            d.text((14,43),'更新z发生在前序5次共同探测之后；以下为新初态的完整插入，所有方法在回合内固定z。',font=font,fill=(182,207,222))
            for col,(arm,label) in enumerate(names.items()):
                r=data[arm];env.reset(r['positions'][i,t]);env.data.qvel[:]=r['velocities'][i,t];mujoco.mj_forward(env.model,env.data)
                rgb=env.observe()[0];x=col*336;im.paste(Image.fromarray(rgb).resize((336,336)),(x,115))
                d.text((x+12,82),label,font=big,fill=(184,235,212));f=r['facts'][t][i]
                d.text((x+10,455),f'步骤{t}/45 · '+('对齐槽口' if t<14 else '下降、接触与稳定'),font=font,fill='white')
                d.text((x+10,486),('已满足插入关系' if f['inserted'] else '尚未满足插入关系')+f" · 接触{f['contact_count']}",font=font,fill=(193,203,212))
                if t==45:d.text((x+10,517),'最终：'+('成功' if r['success'][i] else '失败'),font=big,fill=(120,230,170) if r['success'][i] else (255,155,120))
                elif arm!='base':d.text((x+10,517),f"当前修正幅度 {np.linalg.norm(r['actions'][i,t]-r['base_actions'][i,t]):.3f}",font=font,fill=(204,214,225))
            proc.stdin.write(np.array(im).tobytes())
            if t==45:im.save(file.with_suffix('.png'))
        proc.stdin.close();assert proc.wait()==0
        sources=[{'path':str(path(st,0 if arm=='base' else seed,arm)),'sha256':digest(path(st,0 if arm=='base' else seed,arm))} for arm in names]
        index.append({'title':titles[kind],'kind':kind,'stratum':st,'seed':seed,'domain_index':i,'path':str(file),'sha256':digest(file),'sources':sources,'display_only':'Re-render sealed qpos/qvel; no controller inference or physics stepping; no changed success labels.','success':{k:bool(v['success'][i]) for k,v in data.items()}})
    env.close();save_json(out/'VIDEO_INDEX.json',{'selection':'First matching domain; ID before OOD, seeds11/22/33. Win, harm and base-win when available. All trials retained in statistics.','videos':index});print('ANNOTATED_VIDEOS',len(index))
if __name__=='__main__':main()
