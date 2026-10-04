"""Annotated view-ablation examples from already sealed physical branch frames."""
import subprocess
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from tools.fast_adapter.p35_common import *


def main():
    out=ROOT/'summary/media';out.mkdir(parents=True,exist_ok=False)
    pred=torch.load(ROOT/'branches/PREDICTIONS.pt',weights_only=True)
    outcomes=torch.load(ROOT/'branches/OUTCOMES.pt',weights_only=True)['outcomes'];videos=[]
    fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title=ImageFont.truetype(fp,23);font=ImageFont.truetype(fp,18)
    for task in TASKS:
        oo=[o for o in outcomes if o['task']==task];cases=[]
        for seed in (11,22):
            rr={v:next(r for r in pred['rows'] if (r['task'],r['view'],r['seed'],r['family'],r['mode'])==
                       (task,v,seed,'relation_attention','continuous')) for v in VIEWS}
            for i,o in enumerate(oo):
                vals=np.array(o['progress']);choices=[int(rr[v]['choice'][i]) for v in VIEWS]
                cases.append({'seed':seed,'snapshot_index':i,'choices':choices,'best':int(vals.argmax()),
                    'view_gain_m':float(vals[choices[3]]-vals[choices[0]]),
                    'remaining_gap_m':float(vals.max()-vals[choices[3]]),'updates':rr['g448w']['updates'][i]})
        picks=[('largest_view_gain',max(cases,key=lambda c:c['view_gain_m'])),
               ('largest_remaining_gap',max(cases,key=lambda c:c['remaining_gap_m']))]
        for kind,pick in picks:
            o=oo[pick['snapshot_index']];path=out/f'{task}_{kind}.mp4';width,height=1440,1100
            proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24',
                '-video_size',f'{width}x{height}','-framerate','5','-i','-','-c:v','libx264','-crf','22',
                '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
            labels=['原冻结基座动作','224 全局','448 全局','224 全局＋腕部','448 全局＋腕部','最佳候选（事后评价）']
            indices=[0]+pick['choices']+[pick['best']]
            for frame in range(25):
                t=min(frame//4,4);im=Image.new('RGB',(width,height),(20,29,40));d=ImageDraw.Draw(im)
                name='杯柄抓取' if task==TASKS[0] else '持锤推击木块'
                d.text((14,12),f'P35 {name}｜四步局部动作比较 {t}/4｜源{o["role"]} 物体{o["object_id"]} 第{o["t"]}步｜种子{pick["seed"]}',font=title,fill='white')
                d.text((14,50),'关系注意力＋源轨迹历史适配；同状态选一次动作，保持执行四步。各格播放统一全局相机记录。',font=font,fill=(202,218,234))
                d.text((14,79),'这是按差异或剩余差距挑选的示例，不代表平均结果；不是整任务成功率。',font=font,fill=(239,208,135))
                for j,idx in enumerate(indices):
                    x=(j%3)*480;y=115+(j//3)*480
                    d.text((x+14,y),labels[j],font=title,fill=(116,227,196))
                    im.paste(Image.fromarray(o['rgb'][idx,t].numpy()).resize((370,370)),(x+52,y+38))
                    value=(o['traces'][idx]['values'][t]-o['traces'][idx]['values'][0])*1000
                    metric='抓持中的抬升变化' if task==TASKS[0] else '木块位移变化'
                    d.text((x+14,y+418),f'候选{idx}｜{metric} {value:+.3f} mm',font=font,fill='white')
                    if 1<=j<=4:d.text((x+14,y+446),f'此前完成z更新 {pick["updates"]} 次',font=font,fill=(195,205,218))
                    if j==5:d.text((x+14,y+446),'运行时看不到这一评价答案',font=font,fill=(195,205,218))
                proc.stdin.write(np.asarray(im).tobytes())
                if frame==24:im.save(path.with_suffix('.jpg'),quality=85)
            proc.stdin.close();assert proc.wait()==0
            videos.append({'task':task,'kind':kind,'file':str(path),'cover':str(path.with_suffix('.jpg')),
                'selection':pick,'source_role':o['role'],'source_object_id':o['object_id'],'source_t':o['t'],
                'scope':'four held-action steps; continuous adaptation only along the preceding source prefix',
                'sha256':digest(path)})
    dump(out/'VIDEO_INDEX.json',{'videos':videos,'no_new_simulation':True,
         'selection':'relation_attention continuous only, both seeds eligible; maximum 448dual-vs224global progress difference and maximum 448dual regret',
         'bindings':bindings([ROOT/'branches/PREDICTIONS.pt',ROOT/'branches/OUTCOMES.pt',Path(__file__)])})
    print('P35_MEDIA_COMPLETE',flush=True)


if __name__=='__main__':main()
