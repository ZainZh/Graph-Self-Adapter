"""Annotated display replays of sealed P20 query states; never changes outcomes."""
from pathlib import Path
import subprocess,json
import torch,numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
from prove.fast_adapter.insert3d import Insert3D
from tools.fast_adapter.p20_report import R,raw,SEEDS
from tools.fast_adapter.p18_run import save_json,digest

def main():
 out=R/'summary/media';out.mkdir(parents=True,exist_ok=True);samples=[]
 # Select by a disclosed mechanical rule; every trial remains in the table.
 for target in ['gradient_win','gradient_harm','adapter_failure']:
  found=None
  for st in ['train','ood']:
   for s in SEEDS:
    u0=raw('final',st,s,'hybrid_u0')['success'];u3=raw('final',st,s,'hybrid_u3')['success'];base=raw('final',st,0,'base')['success']
    mask=(~u0&u3) if target=='gradient_win' else (u0&~u3) if target=='gradient_harm' else (base&~u3)
    idx=np.flatnonzero(mask)
    if len(idx):found=(target,st,s,int(idx[0]));break
   if found:break
  if found:samples.append(found)
 if not samples:samples=[('first_example','train',11,0)]
 env=Insert3D();fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(fontpath,18);big=ImageFont.truetype(fontpath,23)
 names={'base':'基础控制','history':'纯历史编码','hybrid_u0':'同模型：冻结 z','hybrid_u3':'同模型：梯度更新 z'};index=[]
 for target,st,seed,i in samples:
  arms=list(names);data={a:raw('final',st,0 if a=='base' else seed,a) for a in arms};file=out/f'{target}_{st}_seed{seed}_domain{i}.mp4';width=1344;height=560
  proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','8','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(file)],stdin=subprocess.PIPE)
  title={'gradient_win':'梯度更新救回：U0 失败，U3 成功','gradient_harm':'梯度更新反例：U0 成功，U3 失败','adapter_failure':'学习控制失败：基础控制成功，U3 失败','first_example':'首个预定测试样本'}[target]
  for t in range(46):
   im=Image.new('RGB',(width,height),(23,31,38));d=ImageDraw.Draw(im);d.text((14,8),f'{title}｜{st.upper()} · 模型种子{seed} · 测试域{i}',font=big,fill='white')
   d.text((14,45),'相同初态、相同动作预算；学习方法先用5次交互形成z，之后完整插入回合中固定z。',font=font,fill=(182,207,222))
   for col,a in enumerate(arms):
    r=data[a];env.reset(r['positions'][i,t]);env.data.qvel[:]=r['velocities'][i,t];mujoco.mj_forward(env.model,env.data);rgb=env.observe()[0]
    x=col*336;im.paste(Image.fromarray(rgb).resize((336,336)),(x,115));d.text((x+12,81),names[a],font=big,fill=(184,235,212))
    f=r['facts'][t][i];phase='对齐槽口' if t<14 else '下降、接触与稳定'
    d.text((x+10,457),f'步骤 {t}/45 · {phase}',font=font,fill='white')
    d.text((x+10,487),f"{'已在槽内' if f['inserted'] else '尚未达到插入判据'} · 接触数 {f['contact_count']}",font=font,fill=(193,203,212))
    if t==45:d.text((x+10,519),'最终：'+('成功' if r['success'][i] else '失败'),font=big,fill=(120,230,170) if r['success'][i] else (255,155,120))
    elif a!='base':
     norm=float(np.linalg.norm(r['actions'][i,t]-r['base_actions'][i,t]));d.text((x+10,519),f'当前学习修正幅度 {norm:.3f}',font=font,fill=(204,214,225))
   proc.stdin.write(np.array(im).tobytes())
   if t==45:im.save(file.with_suffix('.png'))
  proc.stdin.close();assert proc.wait()==0
  bindings=[{'path':str(R/'final'/st/f'seed{0 if a=="base" else seed}'/f'{a}_RAW.pt'),'sha256':digest(R/'final'/st/f'seed{0 if a=="base" else seed}'/f'{a}_RAW.pt')} for a in arms]
  index.append({'title':title,'kind':target,'stratum':st,'seed':seed,'domain_index':i,'path':str(file),'sha256':digest(file),'sources':bindings,'display_only':'rendered from sealed positions and velocities; no rerun or outcome changes; query steps0..45','success':{a:bool(data[a]['success'][i]) for a in arms}})
 env.close();save_json(out/'VIDEO_INDEX.json',{'selection':'first matching domain, train before ood, seeds11/22/33; retain win/harm/failure when available','videos':index})
 page=R/'summary/index.html';text=page.read_text();markup=''.join('<h2>'+e['title']+'</h2><video controls preload="metadata" poster="media/'+Path(e['path']).with_suffix('.png').name+'" src="media/'+Path(e['path']).name+'"></video>' for e in index);page.write_text(text.replace('<div id="videos"></div>','<div id="videos">'+markup+'</div>'))
 report=R/'summary/REPORT_ZH.md';report.write_text(report.read_text()+'\n### 注释视频\n\n'+'\n'.join('- ['+e['title']+'](media/'+Path(e['path']).name+')' for e in index)+'\n\n视频由封存位置/速度重绘，展示完整query控制步0–45；不用于统计或模型选择。\n')
 print(json.dumps(index,ensure_ascii=False))
if __name__=='__main__':main()
