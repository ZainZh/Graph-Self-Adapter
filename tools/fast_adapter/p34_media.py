"""Annotated short branches from sealed images; never described as full successes."""
import subprocess
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from tools.fast_adapter.p34_common import *

def main():
    out=ROOT/'summary/media';out.mkdir(parents=True,exist_ok=False)
    pred=torch.load(ROOT/'branches/PREDICTIONS.pt',weights_only=True)
    outcomes=torch.load(ROOT/'branches/OUTCOMES.pt',weights_only=True)['outcomes']
    selection=json.loads((ROOT/'validation/SELECTION_SEAL.json').read_text())['selected'];videos=[]
    fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';title=ImageFont.truetype(fp,22);font=ImageFont.truetype(fp,18)
    for task in TASKS:
        oc=[o for o in outcomes if o['task']==task];cases=[]
        for seed in (11,22):
            chosen=next(r for r in pred['rows'] if r['task']==task and r['family']==selection[task] and r['seed']==seed and r['mode']=='continuous')
            flat=next(r for r in pred['rows'] if r['task']==task and r['family']=='flat' and r['seed']==seed and r['mode']=='continuous')
            for i,o in enumerate(oc):
                v=np.array(o['progress']);c=int(chosen['choice'][i]);f=int(flat['choice'][i])
                cases.append({'seed':seed,'i':i,'chosen':c,'flat':f,'best':int(v.argmax()),'gain':float(v[c]-v[0]),
                              'regret':float(v.max()-v[c]),'updates':chosen['updates'][i]})
        picks=[('local_gain',max(cases,key=lambda x:x['gain'])),('remaining_gap',max(cases,key=lambda x:x['regret']))]
        for kind,pick in picks:
            o=oc[pick['i']];path=out/f'{task}_{kind}.mp4';w,h=1600,620
            proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}',
                '-framerate','5','-i','-','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
            labels=['原冻结基座动作','直接拼接 + 历史适配','选中关系表示 + 历史适配','九候选中最佳（事后评价）']
            indices=[0,pick['flat'],pick['chosen'],pick['best']]
            for frame in range(25):
                t=min(frame//4,4);im=Image.new('RGB',(w,h),(20,29,40));d=ImageDraw.Draw(im)
                name='杯柄抓取' if task==TASKS[0] else '持锤推击木块'
                d.text((12,10),f'P34 {name}｜局部四步比较 {t}/4｜源 {o["role"]} 物体{o["object_id"]} 第{o["t"]}步｜种子{pick["seed"]}',font=title,fill='white')
                d.text((12,47),'相同真实状态，历史已适配后只选一次动作，保持执行四步。不是整任务成功率，也不是四步内持续重新选动作。',font=font,fill=(202,218,234))
                for j,idx in enumerate(indices):
                    x=j*400;d.text((x+10,85),labels[j],font=font,fill=(116,227,196))
                    im.paste(Image.fromarray(o['rgb'][idx,t].numpy()).resize((380,380)),(x+10,118))
                    change=(o['traces'][idx]['values'][t]-o['traces'][idx]['values'][0])*1000
                    metric='抓持中的抬升变化' if task==TASKS[0] else '木块位移变化'
                    d.text((x+10,506),f'候选 {idx}｜{metric}',font=font,fill='white')
                    d.text((x+10,537),f'当前 {change:+.3f} mm',font=font,fill=(230,211,133))
                    if j in (1,2):d.text((x+10,570),f'此前可用梯度更新 {pick["updates"]} 次',font=font,fill=(195,205,218))
                    if j==3:d.text((x+10,570),'只作评价，不提供给运行时',font=font,fill=(195,205,218))
                proc.stdin.write(np.asarray(im).tobytes())
                if frame==24:im.save(path.with_suffix('.jpg'),quality=85)
            proc.stdin.close();assert proc.wait()==0
            videos.append({'task':task,'kind':kind,'file':str(path),'cover':str(path.with_suffix('.jpg')),
                           'scope':'four held-action execution steps, not full task success','selection':pick,
                           'source_role':o['role'],'source_object_id':o['object_id'],'source_t':o['t'],
                           'sha256':digest(path)})
    dump(out/'VIDEO_INDEX.json',{'selection':'largest selected-versus-base local gain and largest remaining regret per task, both seeds eligible',
         'no_new_simulation':True,'videos':videos,'sources':bindings([ROOT/'branches/OUTCOMES.pt',ROOT/'branches/PREDICTIONS.pt',Path(__file__)])})

if __name__=='__main__':main()
