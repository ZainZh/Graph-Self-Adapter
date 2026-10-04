import json,os,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from prove.fast_adapter.insert3d import *
R=Path('evaluations/fast_adapter_p20_insert3d_v01');O=R/'engineering';O.mkdir(exist_ok=True)
env=Insert3D();rows=[];videos={}
for tag,offset,oracle,shifted in [('nominal',0,False,False),('shifted_base',100,False,True),('shifted_oracle',100,True,True),('blocked',0,False,False)]:
 for i in range(12):
  M,b,fr=domain(910000000+offset+i) if shifted else (np.eye(3),np.zeros(3),.7)
  env.set_domain(M,b,fr);env.reset(start_position(910000500+i));frames=[];facts=[];visibility=[]
  for t in range(46):
   if i==0:
    rgb,box,count=env.observe();frames.append(rgb);visibility.append(count)
    if t in [0,14,30,45]:Image.fromarray(rgb).save(O/f'{tag}_step{t}.png')
   f=env.facts();facts.append(f)
   if t==45:break
   if tag=='blocked':
    target=np.array([.032,0,.0385]);a=np.clip(.5*(target-env.state())/STEP,-1,1)
   else:a=oracle_action(env.state(),t,M,b) if oracle else nominal_action(env.state(),t)
   env.step(a)
  row={'tag':tag,'seed':i,'success':env.streak>=3,'final':facts[-1],'min_object_pixels':min((min(x) for x in visibility),default=None),'max_penetration':max(f['max_penetration'] for f in facts)};rows.append(row)
  if i==0:videos[tag]=(frames,facts)
print(json.dumps(rows,indent=2));(O/'SCENE_QUALIFICATION.json').write_text(json.dumps(rows,indent=2)+'\n')
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',19)
for tag,(frames,facts) in videos.items():
 out=O/(tag+'_annotated.mp4');proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','672x752','-framerate','8','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE)
 for t,rgb in enumerate(frames):
  im=Image.new('RGB',(672,752),(22,29,34));im.paste(Image.fromarray(rgb).resize((672,672)),(0,80));draw=ImageDraw.Draw(im);draw.text((12,8),f'P20 三维插入工程检查 · {tag} · {t}/45',font=font,fill='white');draw.text((12,39),f"{'对齐' if t<14 else '下降与接触'}；接触数 {facts[t]['contact_count']}；仅末帧判定结果",font=font,fill=(170,230,200));proc.stdin.write(np.asarray(im).tobytes())
 proc.stdin.close();assert proc.wait()==0
(R/'scene.xml').write_text(scene_xml());env.close()
