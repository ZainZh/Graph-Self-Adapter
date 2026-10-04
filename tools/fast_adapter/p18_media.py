"""Two explicitly selected, annotated illustrative replays; not added to metrics."""
import json,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from tools.fast_adapter.p18_run import *
from prove.fast_adapter.visual_alignment import render_batch,appearance

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    root=ROOT;out=root/'summary/media';out.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype(FONT,20);big=ImageFont.truetype(FONT,26)
    enc=FrozenDinoROI();systems={}
    for f in FAMILIES:
        cp=torch.load(root/'train'/f/'seed11/model.pt',map_location='cpu',weights_only=False)
        m=model(f).cuda();m.load_state_dict(cp['state']);m.freeze_for_eval();systems[f]=m
    names=['Base','Fixed-z','Gradient-z','History','Hybrid','Oracle']
    entries=[]
    for stratum,wanted in [('id',True),('strong',False)]:
        raw=[json.loads(l) for l in (root/'final/gradient_z/seed11/raw.jsonl').read_text().splitlines()]
        candidates=[r for r in raw if r['arm']=='base' and r['stratum']==stratum]
        d,q=next((r['domain'],qi) for r in candidates for qi,s in enumerate(r['query_success']) if s==wanted)
        stream=2026092231+STRATA.index(stratum)*10000
        sh,support,g,qx,qg=physics_data(128,stream,stratum);sstyle=appearance(stream+4,128);qstyle=appearance(stream+5,512)
        feats=torch.stack([enc.observe(support[d:d+1,k].numpy(),g[d:d+1].numpy(),sstyle[d:d+1]) for k in range(4)],1)
        zs={}
        for f,m in systems.items():
            with torch.no_grad():rec=records_from_features(m,feats)
            z,_=prepare_latent_for_arm(m,f,rec,batch_size=1,k=3,create_graph=False,detach_initial=True);zs[f]=z.detach()
        zs['fixed_z']=systems['gradient_z'].z0.detach()[None]
        init=qx[d,q][None];goal=qg[d,q][None];shift=P18HiddenShift(sh.M[d:d+1],sh.b[d:d+1]);style=qstyle[d*4+q:d*4+q+1]
        x={name:init.clone() for name in names};prev={name:torch.zeros(1,2,device='cuda') for name in names};frames=[];end={}
        def canvas(images,head,subtitle,labels):
            c=Image.new('RGB',(1120,950),(17,23,32));dr=ImageDraw.Draw(c)
            dr.text((20,12),head,font=big,fill='white');dr.text((20,52),subtitle,font=font,fill=(185,205,215))
            for i,(im,label) in enumerate(zip(images,labels)):
                ox=20+(i%3)*365;oy=105+(i//3)*400
                c.paste(Image.fromarray(im).resize((336,336)),(ox,oy+34));dr.text((ox,oy),names[i],font=big,fill=(225,232,239));dr.text((ox,oy+373),label,font=font,fill=(145,224,200))
            dr.text((20,916),'带注释的案例复现；正式成功率由完整原始记录计算，不由视频估计。',font=font,fill=(200,205,210))
            return c
        for k in range(4):
            rgb,_=render_batch(support[d:d+1,k],g[d:d+1],sstyle[d:d+1])
            c=canvas([rgb[0]]*6,f'P18 · {stratum.upper()} · 同一物理环境 / 六种方法',f'阶段1：共享校准交互 {k}/3；K表示交互次数，各方法看到相同证据',['校准后将重置到新的起点']*6)
            frames.extend([c]*3)
        for t in range(13):
            ims=[];labels=[]
            for name in names:
                rgb,_=render_batch(x[name],goal,style);ims.append(rgb[0]);err=float(torch.linalg.norm(x[name]-goal));labels.append(f'误差 {err:.4f}'+((' · 成功' if err<=.05 else ' · 未成功') if t==12 else ''))
            c=canvas(ims,f'P18 · {stratum.upper()} · 域 {d} / 新起点 {q}',f'阶段2：新起点闭环控制 {t}/12；所有权重和z已冻结；成功阈值0.05',labels);frames.extend([c]*(6 if t==12 else 3))
            if t==12:break
            with torch.no_grad():
                # Encode all actual model observations together, without synthetic features.
                learned=['Fixed-z','Gradient-z','History','Hybrid']
                xx=torch.cat([x[name] for name in learned]);gg=goal.repeat(4,1);ss=np.repeat(style,4,axis=0)
                ft=enc.observe(xx.numpy(),gg.numpy(),ss)
                for name in names:
                    base=nominal_base_action_from_roi(x[name],goal)
                    if name=='Base':a=base
                    elif name=='Oracle':a=projected_oracle_action(x[name],goal,shift)
                    else:
                        fam={'Fixed-z':'gradient_z','Gradient-z':'gradient_z','History':'history','Hybrid':'hybrid'}[name];z=zs['fixed_z' if name=='Fixed-z' else fam]
                        j=learned.index(name);a,_,_=systems[fam].act(pack(ft[j:j+1],prev[name],base.cuda()),z);a=a.cpu().clamp(-1,1)
                    x[name]=step_p18_shift(x[name],shift,a);prev[name]=a.cuda()
        path=out/f'{stratum}_six_methods_annotated.mp4'
        proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1120x950','-framerate','4','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-crf','22',str(path)],stdin=subprocess.PIPE)
        for frame in frames:proc.stdin.write(np.asarray(frame).tobytes())
        proc.stdin.close();assert proc.wait()==0
        frames[-1].save(out/f'{stratum}_final_comparison.png')
        entries.append({'path':str(path),'stratum':stratum,'model_seed':11,'domain':d,'query':q,'selection':'first base '+('success' if wanted else 'failure')+' in fixed record order','replay_only':True,'batch_note':'single-case render/inference replay; formal raw metrics unchanged','replay_final_errors':{name:float(torch.linalg.norm(x[name]-goal)) for name in names}})
    (out/'VIDEO_INDEX.json').write_text(json.dumps(entries,indent=2,ensure_ascii=False)+'\n');print(json.dumps(entries))

if __name__=='__main__':main()
