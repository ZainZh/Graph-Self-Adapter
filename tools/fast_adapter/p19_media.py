"""Annotated visualization of sealed trajectories, not rerun or metric source."""
import json,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from tools.fast_adapter.p19_run import ROOT,physics_data,digest,save_json
from prove.fast_adapter.visual_alignment import render_batch,appearance

NAMES={'base':'Base 基础控制','gradient_s3_u0':'Frozen z 同模型不更新','gradient_s3_u3':'Gradient S3 / U3','history_s3':'History 历史编码','least_squares':'LS 同观测线性校准','oracle':'Oracle 真动力学'}
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def main():
 torch.set_num_threads(2)
 out=ROOT/'summary/media';out.mkdir(parents=True,exist_ok=True)
 font=ImageFont.truetype(FONT,19);big=ImageFont.truetype(FONT,24);entries=[]
 for stratum,mode in [('id','rescue'),('strong','failure')]:
  raw_path=ROOT/'final'/stratum/'dino_geometry_seed11_RAW.pt'
  if not raw_path.exists():continue
  raw=torch.load(raw_path,weights_only=False)
  cand=(~raw['base']['success'] & raw['gradient_s3_u3']['success']) if mode=='rescue' else ~raw['gradient_s3_u3']['success']
  options=torch.nonzero(cand)
  if not len(options):continue
  d,q=map(int,options[0]);seed=2026092341+{'id':0,'strong':20000}[stratum]
  sh,support,sg,qx,qg=physics_data(256,seed,stratum)
  sstyle=appearance(seed+4,256)[d:d+1];qstyle=appearance(seed+5,1024)[d*4+q:d*4+q+1]
  frames=[]
  def canvas(images,stage,labels):
   im=Image.new('RGB',(1140,930),(18,26,34));draw=ImageDraw.Draw(im)
   draw.text((20,12),f'P19 · DINO校准 · {stratum.upper()} 域{d} / 起点{q}',font=big,fill='white')
   draw.text((20,49),stage,font=font,fill=(185,220,225))
   for i,(arm,label) in enumerate(zip(NAMES,labels)):
    x=20+375*(i%3);y=95+395*(i//3)
    draw.text((x,y),NAMES[arm],font=big,fill=(220,235,238))
    im.paste(Image.fromarray(images[i]).resize((330,330)),(x,y+36))
    draw.text((x,y+369),label,font=font,fill=(160,230,197))
   draw.text((20,898),'封存轨迹可视化；额外几何监督，query使用GT定位；视频不参与统计。',font=font,fill=(195,206,215))
   return im
  for k in range(4):
   rgb,_=render_batch(support[d:d+1,k],sg[d:d+1],sstyle)
   im=canvas([rgb[0]]*6,f'阶段1：共同校准观察 {k}/3；收齐后做3次梯度更新，再重置起点。',['共同输入，尚未进入新起点测试']*6)
   frames.extend([im]*3)
  for t in range(13):
   images=[];labels=[]
   for arm in NAMES:
    x=raw[arm]['frames'][t,d,q][None];g=qg[d,q][None]
    rgb,_=render_batch(x,g,qstyle);images.append(rgb[0])
    label=f'目标误差 {float((x-g).norm()):.4f}'
    if t==12:label+=' · '+('成功' if raw[arm]['success'][d,q] else '失败')
    labels.append(label)
   im=canvas(images,f'阶段2：新起点控制 {t}/12；z已固定；成功阈值0.05。',labels)
   frames.extend([im]*(8 if t==12 else 3))
  path=out/f'{stratum}_{mode}_annotated.mp4'
  proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1140x930','-framerate','4','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-crf','22',str(path)],stdin=subprocess.PIPE)
  for f in frames:proc.stdin.write(np.asarray(f).tobytes())
  proc.stdin.close();assert proc.wait()==0
  frames[-1].save(out/f'{stratum}_{mode}_final.png')
  entries.append({'path':str(path),'sha256':digest(path),'raw_path':str(raw_path),'raw_sha256':digest(raw_path),'stratum':stratum,'seed':11,'domain':d,'query':q,'selection':'first matching case in fixed order: '+mode,'scope':'visualization of exact sealed positions; no new inference, no contribution to aggregate outcomes','outcomes':{arm:bool(raw[arm]['success'][d,q]) for arm in NAMES}})
 save_json(out/'VIDEO_INDEX.json',entries)
 p=ROOT/'summary/index.html'
 if p.exists():
  body='<div id="videos"><h2>代表轨迹（含中文阶段注释）</h2>'
  for e in entries:
   name=Path(e['path']).name;body+=f'<p>{e["stratum"]} · {e["selection"]}</p><video controls preload="metadata" src="media/{name}"></video>'
  body+='</div>'
  p.write_text(p.read_text().replace('<div id="videos"></div>',body))
 print(json.dumps(entries,ensure_ascii=False))
if __name__=='__main__':main()
